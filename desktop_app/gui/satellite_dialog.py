"""Satellite region selection and video generation dialog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.qt_compat import (
    Qt,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QImage,
    QLabel,
    QMessageBox,
    QPixmap,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTimer,
    QVBoxLayout,
    QWidget,
    qt_exec,
)

_SEL_FILENAME = "last_selection.json"
from desktop_app.gui.archive_picker import (
    ArchivePickerDialog,
    EarthdataLoginDialog,
    load_credentials,
    save_credentials,
)
from desktop_app.gui.imagery_services import ImageryServicesPanel
from desktop_app.gui.landmark_editor import LandmarkEditorDialog, TargetPickerDialog
from desktop_app.gui.satellite_minis import SatelliteMiniStrip
from desktop_app.gui.service_preview import ServicePreviewDialog
from desktop_app.gui.workers import SatelliteWorker, StageWorker
from desktop_app.satellite import (
    prepare_scene,
    prepare_scene_from_image,
    read_world_file_mpp,
    render_scene_videos,
)
from desktop_app.satellite.archives import ArchiveScene, attach_archive_extras
from desktop_app.satellite.objects import sam_available
from desktop_app.satellite.map_widget import SatelliteMapPanel
from desktop_app.satellite.tiles import layer_max_zoom


class _ClickWrapLabel(QLabel):
    """Word-wrapped label that toggles a sibling checkbox on click."""

    def __init__(self, text: str, checkbox: QCheckBox, parent=None):
        super().__init__(text, parent)
        self._cb = checkbox
        self.setWordWrap(True)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._cb.toggle()
        super().mouseReleaseEvent(event)


def _wrap_check(text: str, *, checked: bool = False, tip: str = "") -> Tuple[QCheckBox, QWidget]:
    """Checkbox + wrapping label so long captions never force horizontal scroll."""
    cb = QCheckBox()
    cb.setChecked(checked)
    expand = getattr(QSizePolicy, "Expanding", None) or QSizePolicy.Policy.Expanding
    preferred = getattr(QSizePolicy, "Preferred", None) or QSizePolicy.Policy.Preferred
    lbl = _ClickWrapLabel(text, cb)
    lbl.setSizePolicy(expand, preferred)
    if tip:
        cb.setToolTip(tip)
        lbl.setToolTip(tip)
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    lay.addWidget(cb, 0, Qt.AlignTop)
    lay.addWidget(lbl, 1)
    return cb, row


class SatelliteVideoDialog(QDialog):
    """Interactive satellite map: select region, then generate zoom/pan videos."""

    def __init__(self, out_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Satellite Video")
        # Map min (~520) + sidebar (~380) + chrome; avoid left-clipping the form
        self.setMinimumSize(980, 620)
        self.resize(1200, 720)
        self.out_dir = Path(out_dir)
        self.result: Optional[Dict[str, Any]] = None
        self._worker: Optional[SatelliteWorker] = None
        self._busy = False
        self._discard_result = False
        self._archive_scenes: List[ArchiveScene] = []
        self._main_layer_key: Optional[str] = None

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.map_panel = SatelliteMapPanel()
        self.map_panel.refresh_overview_offline()
        self.mini_strip = SatelliteMiniStrip()
        left.addWidget(self.map_panel, stretch=1)
        left.addWidget(self.mini_strip)
        root.addLayout(left, stretch=3)

        side = QWidget()
        form = QVBoxLayout(side)
        form.setContentsMargins(8, 8, 10, 8)
        form.setSpacing(6)

        def _lab(text: str) -> QLabel:
            w = QLabel(text)
            w.setWordWrap(True)
            return w

        tip = QLabel(
            "Tick satellites · first = main map · others = mini previews · Select Region → Generate"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#aaa;")
        form.addWidget(tip)
        from desktop_app.satellite.offline import offline_enabled, packages

        self._offline_jump_i = 0
        if offline_enabled():
            off = QLabel(
                f"● OFFLINE MODE — imagery from {len(packages())} offline map package(s) only "
                "(no internet). Select a region inside a downloaded area."
            )
            off.setWordWrap(True)
            off.setStyleSheet("color:#4ade80;font-weight:bold;")
            form.addWidget(off)
            self.btn_goto_offline = QPushButton("Go to offline area / برو به منطقه آفلاین")
            self.btn_goto_offline.setMinimumHeight(32)
            self.btn_goto_offline.setStyleSheet(
                "QPushButton { background:#14532d; color:#bbf7d0; font-weight:bold; }"
                "QPushButton:hover { background:#166534; }"
            )
            self.btn_goto_offline.setToolTip(
                "Jump the map to a downloaded offline region. "
                "Click again to cycle through multiple areas.\n"
                "نقشه را روی منطقهٔ دانلودشده می‌برد؛ با چند منطقه، هر کلیک بعدی را نشان می‌دهد."
            )
            self.btn_goto_offline.clicked.connect(self._goto_offline_area)
            form.addWidget(self.btn_goto_offline)

        form.addWidget(_lab("Jump lat/lon"))
        jump_row = QHBoxLayout()
        jump_row.setSpacing(4)
        self.jump_lat = QDoubleSpinBox()
        self.jump_lat.setDecimals(5)
        self.jump_lat.setRange(-85.0, 85.0)
        self.jump_lat.setValue(35.68920)
        self.jump_lat.setMinimumWidth(0)
        self.jump_lon = QDoubleSpinBox()
        self.jump_lon.setDecimals(5)
        self.jump_lon.setRange(-180.0, 180.0)
        self.jump_lon.setValue(51.38900)
        self.jump_lon.setMinimumWidth(0)
        btn_go = QPushButton("Go")
        btn_go.setFixedWidth(36)
        btn_go.clicked.connect(self._jump)
        jump_row.addWidget(self.jump_lat, 1)
        jump_row.addWidget(self.jump_lon, 1)
        jump_row.addWidget(btn_go, 0)
        form.addLayout(jump_row)

        # Multi-select which services feed the dataset (+ live mini previews)
        self.services = ImageryServicesPanel()
        self.services.btn_preview.clicked.connect(self._preview_selected_services)
        self.services.btn_pick_archive.clicked.connect(self._search_archives)
        self.services.btn_login.clicked.connect(self._earthdata_login)
        self.services.btn_own.clicked.connect(self._start_own_image)
        self.services.services_changed.connect(self._on_services_changed)
        self.mini_strip.promote_layer.connect(self._promote_main_layer)
        self.map_panel.map.view_changed.connect(self._on_map_view_for_minis)
        # Toolbar Satellite/Map must not show tiles unless a service is ticked
        try:
            self.map_panel.btn_satellite.clicked.disconnect()
            self.map_panel.btn_map.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.map_panel.btn_satellite.clicked.connect(self._toolbar_show_satellite)
        self.map_panel.btn_map.clicked.connect(self._toolbar_show_map)
        form.addWidget(self.services)
        QTimer.singleShot(0, self._on_services_changed)

        form.addWidget(_lab("Download zoom"))
        self.tile_zoom = QSpinBox()
        self.tile_zoom.setRange(12, 19)
        self.tile_zoom.setValue(17)
        self.tile_zoom.setToolTip("Mosaic download zoom (detail). Map zoom is independent.")
        form.addWidget(self.tile_zoom)

        form.addWidget(_lab("Zoom depth (×)"))
        self.zoom_out = QSpinBox()
        self.zoom_out.setRange(4, 40)
        self.zoom_out.setValue(20)
        self.zoom_out.setToolTip(
            "How many times wider the video starts vs your selection. "
            "Higher = starts more zoomed out, then flies into the box."
        )
        form.addWidget(self.zoom_out)

        form.addWidget(_lab("Duration (s)"))
        self.duration = QDoubleSpinBox()
        self.duration.setRange(2.0, 60.0)
        self.duration.setValue(8.0)
        form.addWidget(self.duration)

        form.addWidget(_lab("FPS"))
        self.fps = QSpinBox()
        self.fps.setRange(10, 60)
        self.fps.setValue(24)
        form.addWidget(self.fps)

        form.addWidget(_lab("Modes"))
        self.mode_zoom, mode_zoom_row = _wrap_check("Zoom flyover", checked=True)
        self.mode_pan, mode_pan_row = _wrap_check("Horizontal pan", checked=True)
        form.addWidget(mode_zoom_row)
        form.addWidget(mode_pan_row)

        self.auto_label, auto_row = _wrap_check("Auto-label target + landmarks", checked=True)
        form.addWidget(auto_row)

        form.addWidget(_lab("Landmarks"))
        self.n_landmarks = QSpinBox()
        self.n_landmarks.setRange(0, 20)
        self.n_landmarks.setValue(8)
        self.n_landmarks.setToolTip(
            "Distinctive objects around the target found automatically. The model learns them too, "
            "so the target can be located from its surroundings when it is small or out of view."
        )
        form.addWidget(self.n_landmarks)

        form.addWidget(_lab("Training videos"))
        self.n_videos = QSpinBox()
        self.n_videos.setRange(1, 16)
        self.n_videos.setValue(6)
        self.n_videos.setToolTip("Varied clips: rotated/off-centre zooms, low passes, lighting changes")
        form.addWidget(self.n_videos)

        self.use_sam, sam_row = _wrap_check(
            "Detect real objects with FastSAM",
            checked=sam_available(),
            tip="Needs FastSAM-s.pt in the project folder; otherwise colour/texture detection is used",
        )
        self.use_sam.setEnabled(sam_available())
        form.addWidget(sam_row)

        self.review, review_row = _wrap_check(
            "Review / edit landmarks before building videos",
            checked=True,
        )
        form.addWidget(review_row)

        self.use_night, night_row = _wrap_check(
            "Night videos / شب",
            tip="Simulate night (streetlights, low light, noise) in a share of training videos",
        )
        form.addWidget(night_row)
        self.night_pct = QSpinBox()
        self.night_pct.setRange(1, 80)
        self.night_pct.setValue(25)
        self.night_pct.setSuffix(" %")
        self.night_pct.setEnabled(False)
        self.night_pct.setToolTip("Share of training videos rendered as simulated night")
        form.addWidget(self.night_pct)
        self.use_night.toggled.connect(self.night_pct.setEnabled)

        self.use_ir, ir_row = _wrap_check(
            "Thermal / IR videos",
            tip="Simulate thermal-like (IR) imagery for night / IR cameras",
        )
        form.addWidget(ir_row)
        self.ir_pct = QSpinBox()
        self.ir_pct.setRange(1, 60)
        self.ir_pct.setValue(15)
        self.ir_pct.setSuffix(" %")
        self.ir_pct.setEnabled(False)
        self.ir_pct.setToolTip("Share rendered as thermal-like (IR) imagery")
        form.addWidget(self.ir_pct)
        self.use_ir.toggled.connect(self.ir_pct.setEnabled)

        self.run_pipeline, pipe_row = _wrap_check(
            "Then run full pipeline (dataset → train → ONNX → package)",
            checked=True,
        )
        form.addWidget(pipe_row)
        self.auto_label.toggled.connect(self.n_landmarks.setEnabled)
        self.auto_label.toggled.connect(self.n_videos.setEnabled)
        self.auto_label.toggled.connect(self.run_pipeline.setEnabled)

        self.sel_label = QLabel("No region selected")
        self.sel_label.setWordWrap(True)
        form.addWidget(self.sel_label)
        self.map_panel.map.selection_changed.connect(self._sync_selection_label)
        self.map_panel.map.view_changed.connect(self._sync_selection_label)

        self.preview = QLabel("Video preview")
        self.preview.setMinimumHeight(120)
        self.preview.setMaximumHeight(180)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#1a1a1a;color:#aaa;border:1px solid #333;")
        self.preview.setScaledContents(False)
        form.addWidget(self.preview)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        form.addWidget(self.progress)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        form.addWidget(self.progress_label)

        buttons = QDialogButtonBox()
        self.btn_generate = buttons.addButton("Generate Video", QDialogButtonBox.AcceptRole)
        self.btn_cancel = buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self._on_cancel)
        form.addWidget(buttons)
        form.addStretch(0)

        # Vertical-only scroll; Ignored width lets viewport constrain wrapping labels
        ignored = getattr(QSizePolicy, "Ignored", None) or QSizePolicy.Policy.Ignored
        preferred = getattr(QSizePolicy, "Preferred", None) or QSizePolicy.Policy.Preferred
        side.setMinimumWidth(0)
        side.setSizePolicy(ignored, preferred)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setWidget(side)
        side_scroll.setMinimumWidth(320)
        side_scroll.setMaximumWidth(480)
        h_off = getattr(Qt, "ScrollBarAlwaysOff", None) or Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        v_on = getattr(Qt, "ScrollBarAsNeeded", None) or Qt.ScrollBarPolicy.ScrollBarAsNeeded
        side_scroll.setHorizontalScrollBarPolicy(h_off)
        side_scroll.setVerticalScrollBarPolicy(v_on)
        root.addWidget(side_scroll, stretch=1)
        self._side_scroll = side_scroll

        # Persist region: restore on open; Clear wipes saved region
        self.map_panel.map.selection_changed.connect(self._persist_selection)
        try:
            self.map_panel.btn_clear.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self.map_panel.btn_clear.clicked.connect(self._clear_selection)
        QTimer.singleShot(100, self._restore_selection)

    def _selection_path(self) -> Path:
        """Per-project selection file under exports/satellite (not global settings)."""
        return self.out_dir / _SEL_FILENAME

    def _persist_selection(self) -> None:
        poly = self.map_panel.map.selection_polygon()
        if not poly or len(poly) < 3:
            return
        m = self.map_panel.map
        payload = {
            "polygon": [[float(lon), float(lat)] for lon, lat in poly],
            "center_lat": float(m.center_lat),
            "center_lon": float(m.center_lon),
            "zoom": int(m.zoom),
        }
        path = self._selection_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass

    def _clear_saved_selection(self) -> None:
        path = self._selection_path()
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass

    def _clear_selection(self) -> None:
        self.map_panel.map.clear_selection()
        self._clear_saved_selection()

    def _restore_selection(self) -> None:
        path = self._selection_path()
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pts = data.get("polygon") or []
            poly: List[Tuple[float, float]] = [
                (float(p[0]), float(p[1])) for p in pts if len(p) >= 2
            ]
            if len(poly) < 3:
                return
            m = self.map_panel.map
            zoom = data.get("zoom")
            lat = data.get("center_lat")
            lon = data.get("center_lon")
            if lat is not None and lon is not None:
                m.go_to(float(lat), float(lon), zoom=int(zoom) if zoom is not None else None)
            else:
                lons = [p[0] for p in poly]
                lats = [p[1] for p in poly]
                m.fit_bbox(min(lats), min(lons), max(lats), max(lons))
            m.set_selection_polygon(poly)
            self._sync_selection_label()
        except (TypeError, ValueError, KeyError, OSError, json.JSONDecodeError):
            self._clear_saved_selection()

    def _jump(self) -> None:
        self.map_panel.map.go_to(self.jump_lat.value(), self.jump_lon.value())

    def _goto_offline_area(self) -> None:
        """Fit the map to the next downloaded offline region (cycles if several)."""
        from desktop_app.satellite.offline import high_zoom_region_boxes, region_boxes

        boxes = high_zoom_region_boxes() or region_boxes(imagery_only=False, min_zoom=0)
        if not boxes:
            QMessageBox.information(
                self,
                "Offline",
                "No offline map regions found.\n"
                "Use Offline Mode → Offline Maps to download or import a package.",
            )
            return
        i = self._offline_jump_i % len(boxes)
        self._offline_jump_i = i + 1
        south, west, north, east = boxes[i]
        self.map_panel.map.fit_bbox(south, west, north, east)
        self.map_panel.refresh_overview_offline()
        self.jump_lat.setValue(0.5 * (south + north))
        self.jump_lon.setValue(0.5 * (west + east))
        n = len(boxes)
        label = f"Offline area {i + 1}/{n}" if n > 1 else "Offline area"
        self.map_panel.status.setText(
            f"{label}: {south:.5f}…{north:.5f} N, {west:.5f}…{east:.5f} E"
        )

    def _sync_selection_label(self) -> None:
        from desktop_app.satellite.map_widget import (
            format_area_m2,
            format_length_m,
            polygon_area_m2,
            polygon_edge_lengths_m,
        )

        poly = self.map_panel.map.selection_polygon()
        draft = self.map_panel.map._draft
        if draft:
            lines = [f"Drawing… {len(draft)} point(s)"]
            for i, d in enumerate(polygon_edge_lengths_m(draft, closed=False)):
                lines.append(f"  {i + 1}→{i + 2}: {format_length_m(d)}")
            if len(draft) >= 3:
                lines.append(f"Area ≈ {format_area_m2(polygon_area_m2(draft))}")
                lines.append("Finish when ready (≥3)")
            self.sel_label.setText("\n".join(lines))
            return
        if not poly:
            self.sel_label.setText("No region — Select Region, click vertices (or Box)")
            return
        m = self.map_panel.map.selection_metrics()
        assert m is not None
        lines = [
            f"{len(poly)} nodes",
            f"Area: {format_area_m2(float(m['area_m2']))}",
            f"Perimeter: {format_length_m(float(m['perimeter_m']))}",
            "Sides:",
        ]
        edges = m["edges_m"]
        for i, d in enumerate(edges):  # type: ignore[arg-type]
            a, b = i + 1, (i + 1) % len(poly) + 1
            lines.append(f"  {a}→{b}: {format_length_m(float(d))}")  # type: ignore[arg-type]
        bbox = self.map_panel.map.selection_bbox()
        if bbox:
            s, w, n, e = bbox
            lines.append(f"S={s:.6f} W={w:.6f}")
            lines.append(f"N={n:.6f} E={e:.6f}")
        self.sel_label.setText("\n".join(lines))

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (
            self.jump_lat,
            self.jump_lon,
            self.tile_zoom,
            self.zoom_out,
            self.duration,
            self.fps,
            self.mode_zoom,
            self.mode_pan,
            self.auto_label,
            self.n_landmarks,
            self.n_videos,
            self.use_night,
            self.use_ir,
            self.use_sam,
            self.review,
            self.run_pipeline,
            self.btn_generate,
            self.map_panel,
            *self.services.widgets_for_enable(),
        ):
            w.setEnabled(enabled)
        if enabled:
            self.night_pct.setEnabled(self.use_night.isChecked())
            self.ir_pct.setEnabled(self.use_ir.isChecked())
            self.btn_cancel.setText("Cancel")
            self.btn_cancel.setEnabled(True)
        else:
            self.night_pct.setEnabled(False)
            self.ir_pct.setEnabled(False)

    def _earthdata_login(self) -> None:
        dlg = EarthdataLoginDialog(self)
        if qt_exec(dlg) == QDialog.Accepted:
            creds = dlg.credentials()
            if not creds.has_auth():
                QMessageBox.warning(self, "Earthdata Login", "Enter username+password or a token.")
                return
            save_credentials(creds)
            self.progress_label.setText("Earthdata credentials saved")

    def _on_services_changed(self) -> None:
        """Main map = primary satellite; other checked tiles → synced mini strip."""
        try:
            layers = self.services.resolve_xyz_layers(preview_only=True)
        except Exception:  # noqa: BLE001
            layers = []
        if not layers:
            self._main_layer_key = None
            self.mini_strip.set_secondary_layers([])
            self.map_panel.map.set_tiles_enabled(False)
            self.map_panel.status.setText("No imagery — tick a satellite service")
            # Uncheck Satellite/Map toggles so they don't imply tiles are on
            self.map_panel.btn_satellite.setChecked(False)
            self.map_panel.btn_map.setChecked(False)
            return
        keys = [k for k, _ in layers]
        if self._main_layer_key not in keys:
            self._main_layer_key = keys[0]
        main = self._main_layer_key or keys[0]
        try:
            self.map_panel.map.set_basemap(main)
            mz = layer_max_zoom(main)
            if self.tile_zoom.value() > mz:
                self.tile_zoom.setValue(mz)
            title = next((t for k, t in layers if k == main), main)
            self.map_panel.status.setText(f"Main map: {title}")
            self.map_panel.btn_satellite.setChecked(main == "satellite")
            self.map_panel.btn_map.setChecked(main == "map")
        except Exception:  # noqa: BLE001
            pass
        secondary = [(k, t) for k, t in layers if k != main]
        self.mini_strip.set_secondary_layers(secondary)
        self._on_map_view_for_minis()

    def _promote_main_layer(self, layer_key: str) -> None:
        self._main_layer_key = layer_key
        self._on_services_changed()

    def _toolbar_show_satellite(self) -> None:
        """Satellite toolbar: ensure Esri is ticked, then make it the main map."""
        if not self.services.chk_esri.isChecked():
            self.services.chk_esri.setChecked(True)
        self._main_layer_key = "satellite"
        self._on_services_changed()

    def _toolbar_show_map(self) -> None:
        """Street map only if some imagery service is already selected."""
        try:
            layers = self.services.resolve_xyz_layers(preview_only=True)
        except Exception:  # noqa: BLE001
            layers = []
        if not layers:
            self.map_panel.btn_map.setChecked(False)
            self.map_panel.map.set_tiles_enabled(False)
            self.map_panel.status.setText("Tick a satellite service first")
            return
        self.map_panel.map.set_basemap("map")
        self.map_panel.btn_map.setChecked(True)
        self.map_panel.btn_satellite.setChecked(False)
        self.map_panel.status.setText("Basemap: map (street)")

    def _on_map_view_for_minis(self) -> None:
        try:
            bbox = self.map_panel.map.visible_bbox()
        except Exception:  # noqa: BLE001
            return
        self.mini_strip.schedule_refresh(bbox)

    def _primary_and_extras(self) -> tuple:
        layer, extra = self.services.primary_and_extras()
        ordered = [layer] + list(extra)
        if self._main_layer_key and self._main_layer_key in ordered:
            rest = [k for k in ordered if k != self._main_layer_key]
            return self._main_layer_key, rest
        return layer, extra

    def _preview_selected_services(self) -> None:
        bbox = self.map_panel.map.selection_bbox()
        if not bbox:
            QMessageBox.warning(
                self,
                "Select Region",
                "Select a region on the map first, then show satellite images.",
            )
            return
        try:
            layers = self.services.resolve_xyz_layers(preview_only=False)
        except ValueError as exc:
            QMessageBox.warning(self, "Services", str(exc))
            return
        if not layers and not self._archive_scenes:
            # Still allow preview of XYZ even without archives; if nothing at all:
            if not self.services.selected_archive_sensors() and not layers:
                QMessageBox.warning(self, "Services", "Tick at least one service.")
                return
        south, west, north, east = bbox
        if layers:
            try:
                self.map_panel.map.set_basemap(layers[0][0])
                mz = layer_max_zoom(layers[0][0])
                if self.tile_zoom.value() > mz:
                    self.tile_zoom.setValue(mz)
            except Exception:  # noqa: BLE001
                pass
        dlg = ServicePreviewDialog(
            south, west, north, east,
            basemap_layers=layers,
            archive_scenes=self._archive_scenes,
            parent=self,
        )
        qt_exec(dlg)

    def _search_archives(self) -> None:
        bbox = self.map_panel.map.selection_bbox()
        if not bbox:
            QMessageBox.warning(
                self,
                "Select Region",
                "Select a region on the map first, then pick archive scenes.",
            )
            return
        sensors = self.services.selected_archive_sensors()
        if not sensors:
            QMessageBox.warning(
                self,
                "Archives",
                "Tick HLS Sentinel-2, HLS Landsat, and/or Sentinel-1 under Archives.",
            )
            return
        south, west, north, east = bbox
        start, end = self.services.archive_date_range()
        dlg = ArchivePickerDialog(
            south, west, north, east,
            sensors=sensors,
            start_date=start,
            end_date=end,
            max_per_sensor=int(self.services.arch_max.value()),
            preselected=[s for s in self._archive_scenes if s.sensor in sensors],
            parent=self,
        )
        if qt_exec(dlg) == QDialog.Accepted:
            self._archive_scenes = list(dlg.selected)
            self.services.status.setStyleSheet("color:#6c6;")
            self.services.status.setText(
                f"{self.services.summary_text()} · {len(self._archive_scenes)} archive scene(s)"
            )
        elif not self._archive_scenes:
            self.services.status.setStyleSheet("color:#888;")
            self.services.status.setText(self.services.summary_text())

    def _start_own_image(self) -> None:
        if not self.auto_label.isChecked():
            QMessageBox.warning(self, "Own image", "Own images require 'Auto-label target + landmarks'.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Satellite / aerial image", "", "Images (*.tif *.tiff *.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Image", f"Cannot read {path}")
            return
        mpp = read_world_file_mpp(path)
        from shared.qt_compat import QInputDialog

        mpp, ok = QInputDialog.getDouble(
            self,
            "Ground resolution",
            "Metres per pixel (ground sample distance) of this image:\n"
            "(read from the world file when present; e.g. 0.3–1.0 for high-res satellites)",
            float(mpp or 0.5),
            0.01,
            100.0,
            3,
        )
        if not ok:
            return
        picker = TargetPickerDialog(img, self)
        if qt_exec(picker) != QDialog.Accepted:
            return
        params = self.params(require_map=False)
        self._params = params
        rect = picker.rect
        out_dir = self.out_dir
        self._busy = True
        self._set_controls_enabled(False)

        def job(progress, frame):
            return prepare_scene_from_image(
                img,
                rect,
                meters_per_px=mpp,
                out_dir=out_dir,
                num_landmarks=params["num_landmarks"],
                use_sam=params["use_sam"],
                source_name=Path(path).name,
                progress_cb=progress,
                frame_cb=frame,
            )

        self._worker = StageWorker(job)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished_ok.connect(self._on_prepared)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _try_accept(self) -> None:
        if self._busy:
            return
        if not self.map_panel.map.selection_polygon():
            draft_n = len(self.map_panel.map._draft)
            tip = (
                f"Polygon not finished yet ({draft_n} point(s)).\n"
                "Click Finish / double-click / right-click to close (≥3 points)."
                if draft_n
                else "Select a region on the satellite map first.\n"
                "Click “Select Region”, then click polygon vertices\n"
                "(or switch to Box and drag a rectangle)."
            )
            QMessageBox.warning(self, "Select Region", tip)
            return
        modes: List[str] = []
        if self.mode_zoom.isChecked():
            modes.append("zoom")
        if self.mode_pan.isChecked():
            modes.append("pan")
        if not modes:
            QMessageBox.warning(self, "Modes", "Select at least one mode (zoom or pan)")
            return
        try:
            self._primary_and_extras()
        except ValueError as exc:
            QMessageBox.warning(self, "Services", str(exc))
            return
        if self.services.has_archive_sensors() and not self._archive_scenes:
            QMessageBox.warning(
                self,
                "Pick archive scenes",
                "Archive sensors are checked but no scenes were picked.\n"
                "Click “Pick archive scenes…” after selecting the region.",
            )
            return

        self._busy = True
        self._discard_result = False
        self._set_controls_enabled(False)
        self.progress.setValue(0)
        self.progress_label.setText("Starting…")
        self.preview.setText("Downloading…")
        self.btn_cancel.setText("Cancel")
        self.btn_cancel.setEnabled(True)

        params = self.params()
        if params["auto_label"]:
            self._start_prepare(params)
            return
        self._worker = SatelliteWorker(params, self.out_dir)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished_ok.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    # ---- auto-label flow: prepare (imagery + landmarks) → review → render
    def _start_prepare(self, params: Dict[str, Any]) -> None:
        out_dir = self.out_dir
        try:
            layer, extra = self._primary_and_extras()
        except ValueError as exc:
            self._busy = False
            self._set_controls_enabled(True)
            QMessageBox.warning(self, "Services", str(exc))
            return
        archive_scenes = list(self._archive_scenes) if self.services.has_archive_sensors() else []
        creds = load_credentials()

        def job(progress, frame):
            prepared = prepare_scene(
                south=params["south"],
                west=params["west"],
                north=params["north"],
                east=params["east"],
                zoom=params["zoom"],
                zoom_out_factor=params["zoom_out_factor"],
                out_dir=out_dir,
                cache_dir=Path("data") / "satellite_cache",
                num_landmarks=params["num_landmarks"],
                use_sam=params["use_sam"],
                layer=layer,
                extra_layers=extra,
                target_lonlat=params.get("target_lonlat"),
                progress_cb=progress,
                frame_cb=frame,
            )
            if archive_scenes:
                if not creds.has_auth():
                    raise RuntimeError(
                        "Earthdata Login required to download archive scenes.\n"
                        "Use “Earthdata Login…”."
                    )

                def arch_progress(done: int, total: int, msg: str) -> None:
                    if progress and total > 0:
                        progress(70 + int(25 * done / total), 100, msg)

                n = attach_archive_extras(
                    prepared,
                    archive_scenes,
                    creds,
                    cache_dir=Path("data") / "satellite_cache" / "archives",
                    progress_cb=arch_progress,
                )
                prepared.save_extras()
                if progress:
                    progress(100, 100, f"Prepared with {len(extra)} tile + {n} archive view(s)")
            return prepared

        self._params = params
        self._worker = StageWorker(job)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished_ok.connect(self._on_prepared)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_prepared(self, prepared: object) -> None:
        if not self._busy:
            return
        if self._discard_result:
            self._busy = False
            self._set_controls_enabled(True)
            self.progress_label.setText("Cancelled")
            self.reject()
            return
        layout = prepared.layout
        if self._params.get("review"):
            dlg = LandmarkEditorDialog(
                prepared.mosaic, layout, prepared.candidates, self,
                extra_mosaics=prepared.extra_mosaics, extra_masks=prepared.extra_masks,
            )
            if qt_exec(dlg) != QDialog.Accepted:
                self._busy = False
                self._set_controls_enabled(True)
                self.progress_label.setText("Cancelled at landmark review")
                return
            layout = dlg.result_layout()
            prepared.candidates = dlg.result_candidates()
            prepared.extra_mosaics, prepared.extra_masks = dlg.result_extras()
            from desktop_app.satellite.scene import save_candidates

            save_candidates(prepared.candidates_path, prepared.candidates)
        prepared.save_layout(layout)
        prepared.save_extras()
        self._on_frame(cv2.imread(str(prepared.preview_path)))
        params = self._params
        out_dir = self.out_dir

        def job(progress, frame):
            res = render_scene_videos(
                prepared.all_mosaics(),
                layout,
                out_dir,
                n_videos=params["n_videos"],
                night_fraction=params.get("night_fraction", 0.0),
                ir_fraction=params.get("ir_fraction", 0.0),
                render_modes=params["modes"],
                duration_s=params["duration_s"],
                fps=params["fps"],
                progress_cb=progress,
                frame_cb=frame,
            )
            res.update(prepared.paths())
            res["run_pipeline"] = bool(params.get("run_pipeline"))
            return res

        self._worker = StageWorker(job)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished_ok.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.progress.setValue(max(0, min(100, int(100 * done / total))))
        self.progress_label.setText(message)

    def _on_frame(self, frame: object) -> None:
        if frame is None or not isinstance(frame, np.ndarray):
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy()).scaled(
            self.preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    def _on_worker_done(self, result: object) -> None:
        self._busy = False
        self.result = result if isinstance(result, dict) else {}
        self.progress.setValue(100)
        # Generation finished on disk — always import, even if the user hit Cancel
        # while the worker was still running (otherwise exports/satellite fills up
        # but the project stays empty and Pipeline fails with "no annotations").
        discarded = bool(getattr(self, "_discard_result", False))
        self._discard_result = False
        has_out = bool(self.result.get("videos") or self.result.get("zoom") or self.result.get("pan"))
        if discarded and not has_out:
            self.progress_label.setText("Cancelled")
            self._set_controls_enabled(True)
            self.reject()
            return
        self.progress_label.setText(
            "Done (importing despite cancel)" if discarded and has_out else "Done"
        )
        self.accept()

    def _on_worker_failed(self, err: str) -> None:
        self._busy = False
        self._set_controls_enabled(True)
        if getattr(self, "_discard_result", False):
            self.progress_label.setText("Cancelled")
            self.reject()
            return
        self.progress_label.setText(f"Failed: {err}")
        QMessageBox.critical(self, "Satellite Video", err)

    def _disconnect_worker_ui(self) -> None:
        w = self._worker
        if w is None:
            return
        for sig, slot in (
            (w.progress, self._on_progress),
            (w.frame_ready, self._on_frame),
            (w.finished_ok, self._on_prepared),
            (w.finished_ok, self._on_worker_done),
            (w.failed, self._on_worker_failed),
        ):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def _force_stop_worker(self) -> None:
        """Detach UI from a stuck job and stop the thread so the dialog can close."""
        w = self._worker
        self._disconnect_worker_ui()
        self._busy = False
        self._discard_result = False
        self._worker = None
        if w is None:
            return
        if w.isRunning():
            w.finished.connect(w.deleteLater)
            if not w.wait(2000):
                w.terminate()
                w.wait(500)
        else:
            w.deleteLater()

    def _on_cancel(self) -> None:
        # Soft-cancel first so a finished run can still be imported; second Cancel
        # (or a second window-close) force-stops a stuck worker.
        if self._busy and self._worker and self._worker.isRunning():
            if self._discard_result:
                self.progress_label.setText("Force closing…")
                self._force_stop_worker()
                self.reject()
                return
            self._discard_result = True
            self.progress_label.setText(
                "Cancelling… wait for current job, or click Cancel again to force close"
            )
            self.btn_cancel.setText("Force Close")
            self.btn_cancel.setEnabled(True)
            self.btn_generate.setEnabled(False)
            try:
                self._worker.progress.disconnect(self._on_progress)
                self._worker.frame_ready.disconnect(self._on_frame)
            except (TypeError, RuntimeError):
                pass
            return
        self.reject()

    def closeEvent(self, event):  # noqa: N802
        if self._busy and self._worker and self._worker.isRunning():
            if self._discard_result:
                self._force_stop_worker()
            else:
                self._on_cancel()
                event.ignore()
                return
        self._persist_selection()
        self.mini_strip.shutdown()
        self.map_panel.shutdown()
        super().closeEvent(event)

    def params(self, require_map: bool = True) -> Dict[str, Any]:
        poly = self.map_panel.map.selection_polygon()
        bbox = self.map_panel.map.selection_bbox()
        if bbox is None or poly is None:
            if require_map:
                raise RuntimeError("No region selected")
            bbox = (0.0, 0.0, 0.0, 0.0)
            poly = []
        south, west, north, east = bbox
        modes: List[str] = []
        if self.mode_zoom.isChecked():
            modes.append("zoom")
        if self.mode_pan.isChecked():
            modes.append("pan")
        lat = (south + north) / 2.0
        lon = (west + east) / 2.0
        span = max(north - south, east - west)
        return {
            "lat": lat,
            "lon": lon,
            "span": span,
            "south": south,
            "west": west,
            "north": north,
            "east": east,
            "target_lonlat": [[float(lon_), float(lat_)] for lon_, lat_ in poly],
            "zoom": int(self.tile_zoom.value()),
            "zoom_out_factor": float(self.zoom_out.value()),
            "duration_s": float(self.duration.value()),
            "fps": float(self.fps.value()),
            "modes": modes,
            "auto_label": self.auto_label.isChecked(),
            "num_landmarks": int(self.n_landmarks.value()),
            "n_videos": int(self.n_videos.value()),
            "run_pipeline": self.auto_label.isChecked() and self.run_pipeline.isChecked(),
            "use_sam": self.use_sam.isChecked(),
            "review": self.review.isChecked(),
            "night_fraction": (self.night_pct.value() / 100.0) if self.use_night.isChecked() else 0.0,
            "ir_fraction": (self.ir_pct.value() / 100.0) if self.use_ir.isChecked() else 0.0,
        }

