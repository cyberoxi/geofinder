"""Dialog: select a large area and build a region atlas (base model for the whole area)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.qt_compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSettings,
    QSpinBox,
    QTimer,
    QVBoxLayout,
    QWidget,
)
from desktop_app.atlas import ATLAS_FILE, AtlasBuildOptions, RegionAtlas, estimate, new_atlas
from desktop_app.satellite.map_widget import SatelliteMapPanel
from desktop_app.satellite.tiles import mosaic_layers

RECENT_KEY = "atlas/recent"
VIEW_KEY = "atlas/last_view"


def recent_atlases() -> List[Path]:
    raw = QSettings().value(RECENT_KEY, "")
    try:
        items = json.loads(str(raw)) if raw else []
    except (TypeError, ValueError):
        items = []
    return [Path(p) for p in items if (Path(p) / ATLAS_FILE).exists()]


def remember_atlas(root: str | Path) -> None:
    root = str(Path(root).resolve())
    items = [str(p) for p in recent_atlases() if str(p) != root]
    QSettings().setValue(RECENT_KEY, json.dumps([root] + items[:9]))


def save_map_view(panel: SatelliteMapPanel) -> None:
    m = panel.map
    QSettings().setValue(VIEW_KEY, json.dumps({"lat": m.center_lat, "lon": m.center_lon, "zoom": m.zoom}))


def restore_map_view(panel: SatelliteMapPanel) -> None:
    try:
        d = json.loads(str(QSettings().value(VIEW_KEY, "") or "{}"))
        if "lat" in d:
            panel.map.go_to(float(d["lat"]), float(d["lon"]), zoom=int(d.get("zoom", 13)))
    except (TypeError, ValueError):
        pass


class AtlasBuildDialog(QDialog):
    """Select a large region → atlas under the open project. ``result`` = {"atlas", "epochs", ...}."""

    def __init__(
        self,
        parent=None,
        *,
        project_dir: str | Path,
        model_type: str = "segmentation",
        imgsz: int = 640,
    ):
        super().__init__(parent)
        self.setWindowTitle("Build Region Base Model / ساخت مدل پایه منطقه")
        self.setMinimumSize(980, 620)
        self.resize(1240, 760)
        self.result: Optional[Dict[str, Any]] = None
        self._resume: Optional[RegionAtlas] = None
        self._imgsz = imgsz
        self._project_dir = Path(project_dir).resolve()

        root = QHBoxLayout(self)
        self.map_panel = SatelliteMapPanel()
        root.addWidget(self.map_panel, stretch=3)

        side = QWidget()
        side.setMinimumWidth(360)
        sv = QVBoxLayout(side)
        intro = QLabel(
            "Draw a <b>large area</b> (polygon or box). The app downloads it in chunks, finds "
            "landmarks across the whole area and trains one base model. Later, "
            "<b>Fine-tune Sub-region</b> trains any target inside this area in a fraction of the time.<br>"
            "<span style='color:#aaa'>یک منطقه بزرگ انتخاب کنید؛ مدل پایه کل منطقه را یاد می‌گیرد و "
            "بعداً هر هدف داخل آن سریع ری‌فاین می‌شود.</span><br>"
            f"<span style='color:#8af'>Output under project: {self._project_dir}</span>"
        )
        intro.setWordWrap(True)
        sv.addWidget(intro)

        form = QFormLayout()
        self.name = QLineEdit("region_atlas")
        form.addRow("Atlas name", self.name)

        self.layer = QComboBox()
        for key, title in mosaic_layers().items():
            self.layer.addItem(title, key)
        form.addRow("Imagery", self.layer)
        self.zoom = QSpinBox()
        self.zoom.setRange(13, 19)
        self.zoom.setValue(16)
        self.zoom.setToolTip("Tile zoom of the atlas. Higher = sharper but many more tiles and chunks.")
        form.addRow("Download zoom", self.zoom)
        self.lm_per_chunk = QSpinBox()
        self.lm_per_chunk.setRange(3, 30)
        self.lm_per_chunk.setValue(10)
        form.addRow("Landmarks / chunk", self.lm_per_chunk)
        self.max_lm = QSpinBox()
        self.max_lm.setRange(8, 250)
        self.max_lm.setValue(120)
        self.max_lm.setToolTip("Total landmark classes of the atlas (model classes = 1 + this).")
        form.addRow("Max landmarks", self.max_lm)
        self.videos = QSpinBox()
        self.videos.setRange(1, 16)
        self.videos.setValue(4)
        form.addRow("Videos / chunk", self.videos)
        self.night = QCheckBox("Night / thermal variants")
        self.night.setChecked(True)
        form.addRow("", self.night)
        self.use_sam = QCheckBox("FastSAM object proposals")
        try:
            from desktop_app.satellite.objects import sam_available

            self.use_sam.setChecked(sam_available())
        except Exception:  # noqa: BLE001
            self.use_sam.setChecked(False)
        form.addRow("", self.use_sam)
        self.model_type = QComboBox()
        self.model_type.addItems(["segmentation", "detection"])
        self.model_type.setCurrentText(model_type)
        form.addRow("Model type", self.model_type)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(60)
        form.addRow("Base epochs", self.epochs)
        sv.addLayout(form)

        self.estimate_label = QLabel("Select a region on the map.")
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setStyleSheet("color:#ffcc66;")
        sv.addWidget(self.estimate_label)

        self.b_resume = QPushButton("Resume / retrain existing atlas…")
        self.b_resume.clicked.connect(self._pick_resume)
        sv.addWidget(self.b_resume)
        sv.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Build atlas")
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)
        sv.addWidget(buttons)
        root.addWidget(side, stretch=1)

        self._est_timer = QTimer(self)
        self._est_timer.setSingleShot(True)
        self._est_timer.timeout.connect(self._update_estimate)
        self.map_panel.map.selection_changed.connect(lambda: self._est_timer.start(250))
        for sb in (self.zoom, self.videos):
            sb.valueChanged.connect(lambda _v: self._est_timer.start(250))
        restore_map_view(self.map_panel)

    # ------------------------------------------------------------ helpers
    def _options(self) -> AtlasBuildOptions:
        night = self.night.isChecked()
        return AtlasBuildOptions(
            landmarks_per_chunk=self.lm_per_chunk.value(),
            max_landmarks=self.max_lm.value(),
            videos_per_chunk=self.videos.value(),
            use_sam=self.use_sam.isChecked(),
            night_fraction=0.2 if night else 0.0,
            ir_fraction=0.1 if night else 0.0,
        )

    def _update_estimate(self) -> None:
        poly = self.map_panel.map.selection_polygon()
        if not poly or len(poly) < 3:
            self.estimate_label.setText("Select a region on the map.")
            return
        lons = [p[0] for p in poly]
        lats = [p[1] for p in poly]
        try:
            est = estimate((min(lats), min(lons), max(lats), max(lons)), poly, self.zoom.value(), self._options())
        except Exception as exc:  # noqa: BLE001
            self.estimate_label.setText(f"Estimate failed: {exc}")
            return
        self.estimate_label.setText(
            f"Area ≈ {est['area_km2']} km² · {est['meters_per_px']} m/px\n"
            f"{est['tiles']} tiles (≈{est['download_mb']} MB) · {est['chunks']} chunks · "
            f"{est['videos']} training videos"
        )

    def _pick_resume(self) -> None:
        start = str(recent_atlases()[0]) if recent_atlases() else str(self._project_dir)
        path, _ = QFileDialog.getOpenFileName(self, "Open atlas", start, f"Atlas ({ATLAS_FILE})")
        if not path:
            return
        try:
            atlas = RegionAtlas.load(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Atlas", str(exc))
            return
        self._resume = atlas
        self.name.setText(atlas.name)
        self.zoom.setValue(atlas.zoom)
        self.model_type.setCurrentText(atlas.model_type)
        for w in (self.name, self.zoom, self.layer, self.lm_per_chunk, self.max_lm, self.videos,
                  self.night, self.use_sam, self.model_type):
            w.setEnabled(False)
        s, w_, n, e = atlas.swne
        self.map_panel.map.fit_bbox(s, w_, n, e)
        self.map_panel.map.set_selection_polygon([tuple(p) for p in atlas.polygon_lonlat])
        self.estimate_label.setText("Resuming: " + atlas.summary())

    def _try_accept(self) -> None:
        save_map_view(self.map_panel)
        if self._resume is not None:
            self.result = {"atlas": self._resume, "epochs": self.epochs.value()}
            self.accept()
            return
        poly = self.map_panel.map.selection_polygon()
        if not poly or len(poly) < 3:
            QMessageBox.warning(self, "Atlas", "Select the large area on the map first (Select Region).")
            return
        name = re.sub(r"[^\w.-]+", "_", self.name.text().strip(), flags=re.UNICODE) or "region_atlas"
        root = self._project_dir / name
        if (root / ATLAS_FILE).exists():
            QMessageBox.warning(
                self, "Atlas", f"An atlas already exists in\n{root}\n\nChoose another name or use Resume."
            )
            return
        try:
            atlas = new_atlas(
                root, name, poly, zoom=self.zoom.value(), layer=str(self.layer.currentData()),
                model_type=self.model_type.currentText(), imgsz=self._imgsz, opts=self._options(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Atlas", str(exc))
            return
        self.result = {"atlas": atlas, "epochs": self.epochs.value()}
        self.accept()
