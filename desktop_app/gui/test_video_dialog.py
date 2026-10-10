"""Dialog: build varied test videos — scene target and/or any place on the world map."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from desktop_app.gui.landmark_editor import MosaicView, TARGET
from desktop_app.satellite.map_widget import SatelliteMapPanel
from shared.landmarks import SceneLayout
from shared.qt_compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    Qt,
)

Swne = Tuple[float, float, float, float]


class TestVideoDialog(QDialog):
    """
    With-target clips use the project scene (green target, view-only).
    Without-target clips: random on the scene mosaic, or pick any place on a
    real world map (search / pan / Select Region) — imagery is downloaded then.
    """

    def __init__(self, mosaic: np.ndarray, layout: SceneLayout, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build Test Videos / ساخت ویدیو تست")
        self.setMinimumSize(1100, 680)
        self.resize(1280, 820)
        self.mosaic = mosaic
        self.layout_ = layout
        self.params: Optional[Dict[str, Any]] = None
        self.world_regions: List[Swne] = []

        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#9ca3af;")
        left.addWidget(self.hint)

        self.stack = QStackedWidget()
        self.view = MosaicView(mosaic)
        self.view.set_painter(self._paint_overlays)
        self.stack.addWidget(self.view)

        self.map_panel = SatelliteMapPanel()
        self.stack.addWidget(self.map_panel)
        left.addWidget(self.stack, stretch=1)

        self.view_info = QLabel("")
        self.view_info.setWordWrap(True)
        left.addWidget(self.view_info)
        root.addLayout(left, stretch=3)

        side_col = QWidget()
        side_outer = QVBoxLayout(side_col)
        side_outer.setContentsMargins(0, 0, 0, 0)
        side_outer.setSpacing(6)

        side = QWidget()
        form = QVBoxLayout(side)
        form.setContentsMargins(8, 8, 10, 8)
        form.setSpacing(8)

        def _lab(text: str) -> QLabel:
            w = QLabel(text)
            w.setWordWrap(True)
            return w

        pos = QGroupBox("With target / روی منطقه هدف")
        pv = QVBoxLayout(pos)
        pv.addWidget(_lab("Videos"))
        self.n_with = QSpinBox()
        self.n_with.setRange(0, 24)
        self.n_with.setValue(4)
        self.n_with.setToolTip("Zoom/pan clips over the project's green target (scene mosaic)")
        pv.addWidget(self.n_with)
        form.addWidget(pos)

        neg = QGroupBox("Without target / مناطق بدون هدف")
        nv = QVBoxLayout(neg)
        nv.addWidget(_lab("Videos"))
        self.n_without = QSpinBox()
        self.n_without.setRange(0, 24)
        self.n_without.setValue(4)
        self.n_without.setToolTip("Hard-negative clips (no project target in view)")
        nv.addWidget(self.n_without)

        nv.addWidget(_lab("Region source / منبع منطقه"))
        self.empty_mode = QComboBox()
        self.empty_mode.addItem("Random on scene / تصادفی روی صحنه", "random")
        self.empty_mode.addItem("World map / نقشه جهان", "world")
        self.empty_mode.setToolTip(
            "Random: empty areas on the current scene mosaic.\n"
            "World map: search/pan anywhere on Earth, Select Region, Add place — "
            "tiles are downloaded for those places."
        )
        self.empty_mode.currentIndexChanged.connect(self._on_empty_mode_changed)
        nv.addWidget(self.empty_mode)

        row = QHBoxLayout()
        self.btn_add = QPushButton("＋ Add place / افزودن")
        self.btn_add.setToolTip("Add the current world-map selection as a film location")
        self.btn_add.clicked.connect(self._add_world_region)
        self.btn_undo = QPushButton("Undo / برگشت")
        self.btn_undo.clicked.connect(self._undo_region)
        self.btn_clear = QPushButton("Clear / پاک")
        self.btn_clear.clicked.connect(self._clear_regions)
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_clear)
        nv.addLayout(row)
        self.map_hint = QLabel("")
        self.map_hint.setWordWrap(True)
        self.map_hint.setStyleSheet("color:#aaa;")
        nv.addWidget(self.map_hint)
        form.addWidget(neg)

        form.addWidget(_lab("Download zoom (world)"))
        self.tile_zoom = QSpinBox()
        self.tile_zoom.setRange(12, 19)
        self.tile_zoom.setValue(16)
        self.tile_zoom.setToolTip("Tile zoom when downloading world-map places (lower if tiles fail)")
        form.addWidget(self.tile_zoom)

        form.addWidget(_lab("Modes"))
        self.mode_zoom = QCheckBox("Zoom flyover")
        self.mode_zoom.setChecked(True)
        self.mode_pan = QCheckBox("Horizontal pan")
        self.mode_pan.setChecked(True)
        form.addWidget(self.mode_zoom)
        form.addWidget(self.mode_pan)

        zoom_box = QGroupBox("Zoom range & speed / بازه و سرعت زوم")
        zv = QVBoxLayout(zoom_box)
        zv.addWidget(_lab("Start zoom (×) — بزرگ‌تر = دورتر"))
        self.zoom_start = QDoubleSpinBox()
        self.zoom_start.setRange(1.5, 40.0)
        self.zoom_start.setDecimals(1)
        self.zoom_start.setSingleStep(0.5)
        self.zoom_start.setValue(12.0)
        zv.addWidget(self.zoom_start)
        zv.addWidget(_lab("End zoom (×) — کوچک‌تر = نزدیک‌تر"))
        self.zoom_end = QDoubleSpinBox()
        self.zoom_end.setRange(0.5, 20.0)
        self.zoom_end.setDecimals(1)
        self.zoom_end.setSingleStep(0.1)
        self.zoom_end.setValue(1.6)
        zv.addWidget(self.zoom_end)
        zv.addWidget(_lab("Zoom speed (×)"))
        self.zoom_speed = QDoubleSpinBox()
        self.zoom_speed.setRange(0.25, 4.0)
        self.zoom_speed.setDecimals(2)
        self.zoom_speed.setSingleStep(0.25)
        self.zoom_speed.setValue(1.0)
        zv.addWidget(self.zoom_speed)
        form.addWidget(zoom_box)

        pan_box = QGroupBox("Pan altitude & speed / ارتفاع و سرعت پن")
        pn = QVBoxLayout(pan_box)
        pn.addWidget(_lab("Pan view (×)"))
        self.pan_view = QDoubleSpinBox()
        self.pan_view.setRange(1.5, 12.0)
        self.pan_view.setDecimals(1)
        self.pan_view.setSingleStep(0.5)
        self.pan_view.setValue(4.0)
        pn.addWidget(self.pan_view)
        pn.addWidget(_lab("Pan speed (×)"))
        self.pan_speed = QDoubleSpinBox()
        self.pan_speed.setRange(0.25, 4.0)
        self.pan_speed.setDecimals(2)
        self.pan_speed.setSingleStep(0.25)
        self.pan_speed.setValue(1.0)
        pn.addWidget(self.pan_speed)
        form.addWidget(pan_box)

        self.mode_zoom.toggled.connect(zoom_box.setEnabled)
        self.mode_pan.toggled.connect(pan_box.setEnabled)

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

        self.use_night = QCheckBox("Night videos / شب")
        self.use_night.setChecked(False)
        form.addWidget(self.use_night)
        self.night_pct = QSpinBox()
        self.night_pct.setRange(1, 80)
        self.night_pct.setValue(25)
        self.night_pct.setSuffix(" %")
        self.night_pct.setEnabled(False)
        form.addWidget(self.night_pct)
        self.use_night.toggled.connect(self.night_pct.setEnabled)

        self.use_ir = QCheckBox("Thermal / IR videos")
        self.use_ir.setChecked(False)
        form.addWidget(self.use_ir)
        self.ir_pct = QSpinBox()
        self.ir_pct.setRange(1, 60)
        self.ir_pct.setValue(15)
        self.ir_pct.setSuffix(" %")
        self.ir_pct.setEnabled(False)
        form.addWidget(self.ir_pct)
        self.use_ir.toggled.connect(self.ir_pct.setEnabled)

        tip = QLabel(
            "هدف پروژه روی صحنه؛ برای منفی سخت از نقشه جهان هر جا را جستجو/انتخاب کنید."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#aaa;")
        form.addWidget(tip)
        form.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(side)
        side_outer.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Generate / ساخت")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        side_outer.addWidget(buttons)
        root.addWidget(side_col, stretch=1)

        self._go_to_scene()
        self._on_empty_mode_changed()
        self.view_info.setText(self._status_line())

    def _empty_mode_key(self) -> str:
        return str(self.empty_mode.currentData() or "random")

    def _go_to_scene(self) -> None:
        meta = self.layout_.meta or {}
        swne = meta.get("selection_swne") or meta.get("fetch_swne")
        if swne and len(swne) == 4:
            s, w, n, e = (float(v) for v in swne)
            self.map_panel.map.fit_bbox(s, w, n, e)
            return
        poly = self.layout_.polygon(0)
        # No geo — leave default map view
        _ = poly

    def _on_empty_mode_changed(self) -> None:
        world = self._empty_mode_key() == "world"
        self.stack.setCurrentIndex(1 if world else 0)
        for w in (self.btn_add, self.btn_undo, self.btn_clear, self.tile_zoom):
            w.setEnabled(world)
        if world:
            self.hint.setText(
                "نقشه جهان: جستجو کنید، Select Region بزنید، منطقه را بکشید، بعد Add place.\n"
                "World map: Search → Select Region → draw → Add place. Anywhere on Earth."
            )
            self.map_hint.setText(
                "چند مکان می‌توانید اضافه کنید؛ ویدیوهای بدون‌هدف بین آن‌ها می‌چرخند."
            )
            self.map_panel._set_tool("box")
        else:
            self.hint.setText(
                "هدف (سبز) فقط نمایش است. مناطق بدون‌هدف تصادفی از همین صحنه انتخاب می‌شوند.\n"
                "Target (green) is view-only. Without-target: random empty areas on this scene."
            )
            self.map_hint.setText("مناطق دور از هدف به‌صورت تصادفی از روی موزاییک صحنه.")
            self.view.refresh()
        self.view_info.setText(self._status_line())

    def _status_line(self) -> str:
        poly = self.layout_.polygon(0)
        name = self.layout_.classes[0] if self.layout_.classes else "target"
        (x0, y0), (x1, y1) = poly.min(axis=0), poly.max(axis=0)
        base = (
            f"Target (read-only): {name}  ·  "
            f"bbox ≈ {int(x1 - x0)}×{int(y1 - y0)} px  ·  "
            f"{self.layout_.num_landmarks} landmarks"
        )
        if self._empty_mode_key() == "world":
            base += f"  ·  world places: {len(self.world_regions)}"
        return base

    def _add_world_region(self) -> None:
        bbox = self.map_panel.map.selection_bbox()
        if not bbox:
            QMessageBox.information(
                self,
                "World map",
                "Select a region on the map first (Select Region → Box or Polygon).",
            )
            return
        s, w, n, e = bbox
        if n - s < 1e-5 or e - w < 1e-5:
            QMessageBox.warning(self, "World map", "Selection is too small.")
            return
        self.world_regions.append((float(s), float(w), float(n), float(e)))
        self.view_info.setText(self._status_line())
        self.map_hint.setText(f"Added place #{len(self.world_regions)}  ({s:.4f},{w:.4f})–({n:.4f},{e:.4f})")

    def _undo_region(self) -> None:
        if self.world_regions:
            self.world_regions.pop()
            self.view_info.setText(self._status_line())

    def _clear_regions(self) -> None:
        self.world_regions.clear()
        self.view_info.setText(self._status_line())

    def _paint_overlays(self, img: np.ndarray, s: float) -> None:
        thick = max(2, int(round(3 * max(1.0, img.shape[1] / 1600))))
        font = max(0.45, 0.55 * img.shape[1] / 1600)
        poly = self.layout_.polygon(0)
        pts = np.int32(poly * s).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, TARGET, thick, cv2.LINE_AA)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], TARGET)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        c = poly.mean(axis=0) * s
        cv2.putText(
            img,
            "TARGET (view only)",
            (int(c[0]) + 8, int(c[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            TARGET,
            max(1, thick // 2),
            cv2.LINE_AA,
        )

    def _accept(self) -> None:
        modes: List[str] = []
        if self.mode_zoom.isChecked():
            modes.append("zoom")
        if self.mode_pan.isChecked():
            modes.append("pan")
        if not modes:
            modes = ["zoom", "pan"]
        n_with = int(self.n_with.value())
        n_without = int(self.n_without.value())
        if n_with + n_without < 1:
            return
        mode = self._empty_mode_key()
        if mode == "world" and n_without > 0 and not self.world_regions:
            QMessageBox.warning(
                self,
                "World map",
                "Add at least one place on the world map (Select Region → Add place), "
                "or switch Region source to Random.",
            )
            return
        z0 = float(self.zoom_start.value())
        z1 = float(self.zoom_end.value())
        if "zoom" in modes and abs(z0 - z1) < 0.05:
            QMessageBox.warning(self, "Zoom range", "Start zoom and end zoom must be different.")
            return
        self.params = {
            "n_with_target": n_with,
            "n_without_target": n_without,
            "modes": modes,
            "duration_s": float(self.duration.value()),
            "fps": float(self.fps.value()),
            "night_fraction": (self.night_pct.value() / 100.0) if self.use_night.isChecked() else 0.0,
            "ir_fraction": (self.ir_pct.value() / 100.0) if self.use_ir.isChecked() else 0.0,
            "empty_region_mode": mode,
            "empty_regions": [],
            "world_regions": list(self.world_regions) if mode == "world" else [],
            "tile_zoom": int(self.tile_zoom.value()),
            "zoom_start": z0,
            "zoom_end": z1,
            "zoom_speed": float(self.zoom_speed.value()),
            "pan_view": float(self.pan_view.value()),
            "pan_speed": float(self.pan_speed.value()),
        }
        self.accept()
