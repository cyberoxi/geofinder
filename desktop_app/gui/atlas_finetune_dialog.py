"""Dialog: pick a target inside a region atlas → new project (instant or fine-tuned from the base model)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from shared.qt_compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from desktop_app.atlas import ATLAS_FILE, RegionAtlas, validate_target
from desktop_app.gui.atlas_build_dialog import recent_atlases, remember_atlas
from desktop_app.satellite.map_widget import SatelliteMapPanel


class AtlasFinetuneDialog(QDialog):
    """``result`` = {"atlas", "mode", "target_lonlat", "project_dir", "project_name", "n_videos", "epochs", ...}."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fine-tune Sub-region from Atlas / ری‌فاین زیرمنطقه")
        self.setMinimumSize(980, 620)
        self.resize(1240, 760)
        self.result: Optional[Dict[str, Any]] = None
        self.atlas: Optional[RegionAtlas] = None

        root = QHBoxLayout(self)
        self.map_panel = SatelliteMapPanel()
        root.addWidget(self.map_panel, stretch=3)

        side = QWidget()
        side.setMinimumWidth(360)
        sv = QVBoxLayout(side)
        intro = QLabel(
            "Choose an atlas, then draw the <b>target</b> inside its dashed orange boundary.<br>"
            "<b>Instant</b>: no training — the base model finds the atlas landmarks and the target is "
            "placed from them (seconds).<br>"
            "<b>Fine-tune</b>: also trains the target itself (better when the camera is so close that "
            "no landmark is visible).<br>"
            "<span style='color:#aaa'>هدف را داخل محدوده نارنجی اطلس رسم کنید. حالت فوری بدون آموزش است "
            "و هدف از روی لندمارک‌های مدل پایه پیدا می‌شود.</span>"
        )
        intro.setWordWrap(True)
        sv.addWidget(intro)

        atlas_row = QHBoxLayout()
        self.atlas_combo = QComboBox()
        for p in recent_atlases():
            self.atlas_combo.addItem(p.name, str(p))
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._browse_atlas)
        atlas_row.addWidget(self.atlas_combo, stretch=1)
        atlas_row.addWidget(b_browse)
        sv.addLayout(atlas_row)
        self.atlas_info = QLabel("No atlas selected.")
        self.atlas_info.setWordWrap(True)
        self.atlas_info.setStyleSheet("color:#ffcc66;")
        sv.addWidget(self.atlas_info)

        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Instant — no training (seconds) / فوری", "instant")
        self.mode.addItem("Fine-tune — train the target (minutes) / ری‌فاین", "finetune")
        form.addRow("Mode", self.mode)
        self.proj_name = QLineEdit("target_1")
        form.addRow("Project name", self.proj_name)
        folder_row = QHBoxLayout()
        self.proj_parent = QLineEdit("")
        self.proj_parent.setPlaceholderText("<atlas>/subregions")
        b_folder = QPushButton("…")
        b_folder.setFixedWidth(32)
        b_folder.clicked.connect(self._browse_parent)
        folder_row.addWidget(self.proj_parent, stretch=1)
        folder_row.addWidget(b_folder)
        form.addRow("Projects folder", folder_row)
        self.n_videos = QSpinBox()
        self.n_videos.setRange(1, 12)
        self.n_videos.setValue(6)
        form.addRow("Videos", self.n_videos)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 300)
        self.epochs.setValue(15)
        form.addRow("Fine-tune epochs", self.epochs)
        self.freeze = QCheckBox("Freeze backbone (faster)")
        self.freeze.setChecked(True)
        self.freeze.setToolTip("Freeze the first 10 YOLOv8 layers — the base model already knows this area.")
        form.addRow("", self.freeze)
        self.zoom_out = QDoubleSpinBox()
        self.zoom_out.setRange(2.0, 40.0)
        self.zoom_out.setValue(10.0)
        self.zoom_out.setToolTip("View around the target (auto-widened until ≥3 atlas landmarks are visible).")
        form.addRow("Zoom-out factor", self.zoom_out)
        self.night = QCheckBox("Night / thermal variants")
        self.night.setChecked(True)
        form.addRow("", self.night)
        from desktop_app.gui.camera_sim_dialog import load_camera_sim

        self.camsim_label = QLabel(load_camera_sim().describe())
        self.camsim_label.setWordWrap(True)
        self.camsim_label.setStyleSheet("color:#8af;")
        self.b_camsim = QPushButton("Camera simulation…")
        self.b_camsim.clicked.connect(self._edit_camera_sim)
        form.addRow(self.b_camsim, self.camsim_label)
        self.package = QCheckBox("Build Jetson package")
        self.package.setChecked(True)
        form.addRow("", self.package)
        sv.addLayout(form)
        sv.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._ok = buttons.button(QDialogButtonBox.Ok)
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)
        sv.addWidget(buttons)
        root.addWidget(side, stretch=1)

        self.mode.currentIndexChanged.connect(lambda _i: self._mode_changed())
        self._mode_changed()
        self.atlas_combo.currentIndexChanged.connect(lambda _i: self._load_atlas(self.atlas_combo.currentData()))
        if self.atlas_combo.count():
            self._load_atlas(self.atlas_combo.currentData())

    def _edit_camera_sim(self) -> None:
        from desktop_app.gui.camera_sim_dialog import edit_camera_sim

        self.camsim_label.setText(edit_camera_sim(self).describe())

    def _mode_changed(self) -> None:
        instant = self.mode.currentData() == "instant"
        for w in (self.n_videos, self.epochs, self.freeze, self.night, self.b_camsim):
            w.setEnabled(not instant)
        self.package.setEnabled(instant)  # fine-tune: the training pipeline builds the package
        self._ok.setText("Create (instant)" if instant else "Create && fine-tune")

    # ------------------------------------------------------------ atlas
    def _browse_atlas(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open atlas", "", f"Atlas ({ATLAS_FILE})")
        if not path:
            return
        root = str(Path(path).parent.resolve())
        idx = self.atlas_combo.findData(root)
        if idx < 0:
            self.atlas_combo.addItem(Path(root).name, root)
            idx = self.atlas_combo.count() - 1
        if idx == self.atlas_combo.currentIndex():
            self._load_atlas(root)
        else:
            self.atlas_combo.setCurrentIndex(idx)

    def _browse_parent(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Folder for the new project", self.proj_parent.text())
        if d:
            self.proj_parent.setText(d)

    def _load_atlas(self, root: Optional[str]) -> None:
        if not root:
            return
        try:
            atlas = RegionAtlas.load(root)
        except Exception as exc:  # noqa: BLE001
            self.atlas = None
            self.atlas_info.setText(f"Cannot open atlas: {exc}")
            return
        self.atlas = atlas
        remember_atlas(atlas.root)
        ok = atlas.base_weights_path() is not None
        self.atlas_info.setText(
            atlas.summary() + ("" if ok else "\n⚠ Base model not trained — run Build Region Base Model first.")
        )
        if not self.proj_parent.text().strip() or "subregions" in self.proj_parent.text():
            self.proj_parent.setText(str(atlas.subregions_dir))
        m = self.map_panel.map
        m.set_reference_polygon([tuple(p) for p in atlas.polygon_lonlat])
        s, w, n, e = atlas.swne
        m.fit_bbox(s, w, n, e)
        m.clear_selection()

    # ------------------------------------------------------------ accept
    def _try_accept(self) -> None:
        if self.atlas is None:
            QMessageBox.warning(self, "Atlas", "Choose an atlas first.")
            return
        if self.atlas.base_weights_path() is None:
            QMessageBox.warning(self, "Atlas", "This atlas has no trained base model yet.")
            return
        poly = self.map_panel.map.selection_polygon()
        try:
            validate_target(self.atlas, [list(p) for p in (poly or [])])
        except ValueError as exc:
            QMessageBox.warning(self, "Target", str(exc))
            return
        name = re.sub(r"[^\w.-]+", "_", self.proj_name.text().strip(), flags=re.UNICODE) or "target"
        parent = Path(self.proj_parent.text().strip() or self.atlas.subregions_dir)
        project_dir = parent / name
        if (project_dir / "project.json").exists():
            QMessageBox.warning(self, "Project", f"A project already exists in\n{project_dir}\nChoose another name.")
            return
        night = self.night.isChecked()
        self.result = {
            "atlas": self.atlas,
            "mode": self.mode.currentData(),
            "build_package": self.package.isChecked(),
            "target_lonlat": [[float(lon), float(lat)] for lon, lat in poly],
            "project_dir": project_dir,
            "project_name": name,
            "n_videos": self.n_videos.value(),
            "epochs": self.epochs.value(),
            "freeze": 10 if self.freeze.isChecked() else 0,
            "zoom_out_factor": self.zoom_out.value(),
            "night_fraction": 0.2 if night else 0.0,
            "ir_fraction": 0.1 if night else 0.0,
        }
        self.accept()
