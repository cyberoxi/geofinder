"""Camera simulation settings for training videos: camera + lens, tilt, sensor look, altitude range.

Stored in QSettings; every video generator (satellite videos, landmark rebuild, camera paths,
atlas build, sub-region fine-tune) reads them via :func:`load_camera_sim`.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from desktop_app.satellite.camera_model import CAMERAS, LENSES, CameraSim
from shared.qt_compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSettings,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    qt_exec,
)

ORG = "GeoFinder"
APP = "Desktop"
_KEY = "render/camera_sim"


def load_camera_sim() -> CameraSim:
    raw = QSettings(ORG, APP).value(_KEY, "")
    try:
        d = json.loads(str(raw)) if raw else {}
    except (TypeError, ValueError):
        d = {}
    return CameraSim.from_dict(d if isinstance(d, dict) else {})


def save_camera_sim(sim: CameraSim) -> None:
    QSettings(ORG, APP).setValue(_KEY, json.dumps(sim.to_dict()))


def active_camera_sim() -> Optional[CameraSim]:
    """The stored settings when at least one option is on, else None (plain satellite views)."""
    sim = load_camera_sim()
    return sim if sim.any_enabled else None


def _zoom_for_mpp(mpp: float, lat: float = 35.0) -> int:
    """Smallest web-map zoom whose ground resolution is ≤ ``mpp`` m/px."""
    base = 156543.03 * math.cos(math.radians(lat))
    return int(min(21, max(10, math.ceil(math.log2(base / max(mpp, 1e-3))))))


class CameraSimDialog(QDialog):
    def __init__(self, parent=None, sim: Optional[CameraSim] = None):
        super().__init__(parent)
        self.setWindowTitle("Camera Simulation / شبیه‌سازی دوربین")
        self.setMinimumWidth(520)
        sim = sim or load_camera_sim()
        root = QVBoxLayout(self)
        intro = QLabel(
            "Make training videos look like the real Jetson camera. Each option is independent; "
            "with all off, videos stay straight-down satellite views.<br>"
            "<span style='color:#aaa'>ویدیوهای آموزشی شبیه دوربین واقعی جتسون ساخته می‌شوند. "
            "هر گزینه جداگانه فعال/غیرفعال می‌شود. بعد از تغییر، ویدیوها را دوباره بسازید و دوباره آموزش دهید.</span>"
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # ---- camera + lens (used by all three options)
        g_cam = QGroupBox("Camera && lens / دوربین و لنز")
        f = QFormLayout(g_cam)
        self.camera = QComboBox()
        for k, c in CAMERAS.items():
            self.camera.addItem(c["title"], k)
        self.camera.setCurrentIndex(max(0, self.camera.findData(sim.camera)))
        f.addRow("Camera", self.camera)
        self.lens = QComboBox()
        for k, l in LENSES.items():
            self.lens.addItem(l["title"], k)
        self.lens.setCurrentIndex(max(0, self.lens.findData(sim.lens)))
        f.addRow("Lens", self.lens)
        self.focal = QDoubleSpinBox()
        self.focal.setRange(1.0, 200.0)
        self.focal.setDecimals(1)
        self.focal.setSuffix(" mm")
        self.focal.setValue(sim.custom_focal_mm)
        f.addRow("Custom focal", self.focal)
        self.fov_label = QLabel()
        self.fov_label.setStyleSheet("color:#8af;")
        f.addRow("Field of view", self.fov_label)
        root.addWidget(g_cam)

        # ---- 1. oblique
        self.g_obl = QGroupBox("1. Oblique camera (tilt) / زاویه مایل")
        self.g_obl.setCheckable(True)
        self.g_obl.setChecked(sim.oblique)
        f = QFormLayout(self.g_obl)
        self.max_pitch = QSpinBox()
        self.max_pitch.setRange(5, 70)
        self.max_pitch.setSuffix("°")
        self.max_pitch.setValue(int(sim.max_pitch))
        self.max_pitch.setToolTip("Largest forward tilt from straight down (limited automatically by the lens FOV).")
        f.addRow("Max tilt", self.max_pitch)
        self.max_roll = QSpinBox()
        self.max_roll.setRange(0, 30)
        self.max_roll.setSuffix("°")
        self.max_roll.setValue(int(sim.max_roll))
        f.addRow("Max roll", self.max_roll)
        self.obl_pct = QSpinBox()
        self.obl_pct.setRange(10, 100)
        self.obl_pct.setSuffix(" %")
        self.obl_pct.setValue(int(round(sim.oblique_fraction * 100)))
        self.obl_pct.setToolTip("Share of videos rendered tilted (the first video always stays straight down).")
        f.addRow("Videos tilted", self.obl_pct)
        root.addWidget(self.g_obl)

        # ---- 2. sensor
        self.g_sen = QGroupBox("2. Real sensor + lens look / شبیه‌سازی سنسور و لنز")
        self.g_sen.setCheckable(True)
        self.g_sen.setChecked(sim.sensor)
        f = QFormLayout(self.g_sen)
        f.addRow(QLabel("Lens distortion (labels follow it), vignetting, colour fringes, motion blur, "
                        "focus, white balance, exposure, noise, sharpening, compression."))
        self.sen_pct = QSpinBox()
        self.sen_pct.setRange(10, 100)
        self.sen_pct.setSuffix(" %")
        self.sen_pct.setValue(int(round(sim.sensor_fraction * 100)))
        f.addRow("Videos with sensor look", self.sen_pct)
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.3, 2.0)
        self.strength.setSingleStep(0.1)
        self.strength.setValue(sim.sensor_strength)
        self.strength.setToolTip("0.5 = mild, 1 = typical, 1.5+ = harsh (cheap lens, bad light)")
        f.addRow("Strength", self.strength)
        root.addWidget(self.g_sen)

        # ---- 3. altitude
        self.g_alt = QGroupBox("3. Flight altitude range / بازه ارتفاع پرواز")
        self.g_alt.setCheckable(True)
        self.g_alt.setChecked(sim.altitude)
        f = QFormLayout(self.g_alt)
        row = QHBoxLayout()
        self.alt_min = QSpinBox()
        self.alt_max = QSpinBox()
        for sb, v in ((self.alt_min, sim.alt_min_m), (self.alt_max, sim.alt_max_m)):
            sb.setRange(5, 10000)
            sb.setSuffix(" m")
            sb.setValue(int(v))
            row.addWidget(sb)
        w = QWidget()
        w.setLayout(row)
        f.addRow("Min / max altitude", w)
        self.alt_label = QLabel()
        self.alt_label.setWordWrap(True)
        self.alt_label.setStyleSheet("color:#ffcc66;")
        f.addRow(self.alt_label)
        root.addWidget(self.g_alt)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        b_off = QPushButton("All off")
        b_off.clicked.connect(self._all_off)
        buttons.addButton(b_off, QDialogButtonBox.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        for sig in (self.camera.currentIndexChanged, self.lens.currentIndexChanged, self.focal.valueChanged,
                    self.alt_min.valueChanged, self.alt_max.valueChanged):
            sig.connect(lambda *_a: self._refresh())
        self._refresh()

    def _all_off(self) -> None:
        for g in (self.g_obl, self.g_sen, self.g_alt):
            g.setChecked(False)

    def values(self) -> CameraSim:
        return CameraSim(
            camera=self.camera.currentData(),
            lens=self.lens.currentData(),
            custom_focal_mm=self.focal.value(),
            oblique=self.g_obl.isChecked(),
            max_pitch=float(self.max_pitch.value()),
            max_roll=float(self.max_roll.value()),
            oblique_fraction=self.obl_pct.value() / 100.0,
            sensor=self.g_sen.isChecked(),
            sensor_fraction=self.sen_pct.value() / 100.0,
            sensor_strength=self.strength.value(),
            altitude=self.g_alt.isChecked(),
            alt_min_m=float(min(self.alt_min.value(), self.alt_max.value())),
            alt_max_m=float(max(self.alt_min.value(), self.alt_max.value())),
        )

    def _refresh(self) -> None:
        sim = self.values()
        interchangeable = CAMERAS[sim.camera].get("interchangeable", True)
        self.lens.setEnabled(interchangeable)
        self.focal.setEnabled(interchangeable and sim.lens == "custom")
        fh, fv = sim.fov_deg()
        self.fov_label.setText(f"{fh:.1f}° × {fv:.1f}°  (focal {sim.lens_params()['focal_mm']:g} mm)")
        t = math.tan(math.radians(fv) / 2.0)
        g_lo, g_hi = 2 * sim.alt_min_m * t, 2 * sim.alt_max_m * t
        # Imagery should not be upscaled more than ~3× at the lowest altitude (720 px frame height)
        need = g_lo / 240.0
        self.alt_label.setText(
            f"Ground seen (frame height): {g_lo:.0f} m … {g_hi:.0f} m.\n"
            f"Low altitude needs imagery of ≈{need:.2f} m/px → download zoom ≥ {_zoom_for_mpp(need)}."
        )


def edit_camera_sim(parent=None) -> CameraSim:
    """Open the dialog; on OK save and return the new settings, else the stored ones."""
    dlg = CameraSimDialog(parent)
    if qt_exec(dlg) == QDialog.Accepted:
        sim = dlg.values()
        save_camera_sim(sim)
        return sim
    return load_camera_sim()
