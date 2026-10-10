"""Jetson camera image controls.

* Camera ISP (CSI / Argus): white balance, exposure, gain, noise reduction, sharpening — baked into
  the camera pipeline (re-opens the camera). See shared.camera_isp.
* Image adjust (live, any source): colour normalisation towards the training imagery, manual
  brightness / contrast / gamma / saturation, CLAHE. See shared.image_adjust.

Both are stored in QSettings and are off by default.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from shared.camera_isp import NR_MODES, PRESETS, WB_MODES, describe_isp, merge_isp
from shared.image_adjust import describe_adjust, merge_adjust
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
APP = "JetsonRuntime"
_ISP_KEY = "camera/isp"
_ADJ_KEY = "image/adjust"


def _load(key: str) -> Dict[str, Any]:
    raw = QSettings(ORG, APP).value(key, "")
    try:
        d = json.loads(str(raw)) if raw else {}
    except (TypeError, ValueError):
        d = {}
    return d if isinstance(d, dict) else {}


def load_isp_settings() -> Dict[str, Any]:
    return merge_isp(_load(_ISP_KEY))


def save_isp_settings(cfg: Dict[str, Any]) -> None:
    QSettings(ORG, APP).setValue(_ISP_KEY, json.dumps(merge_isp(cfg)))


def load_adjust_settings() -> Dict[str, Any]:
    return merge_adjust(_load(_ADJ_KEY))


def save_adjust_settings(cfg: Dict[str, Any]) -> None:
    QSettings(ORG, APP).setValue(_ADJ_KEY, json.dumps(merge_adjust(cfg)))


def _dspin(v: float, lo: float, hi: float, step: float, dec: int = 2, suffix: str = "") -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(dec)
    sb.setValue(float(v))
    if suffix:
        sb.setSuffix(suffix)
    return sb


def _combo(items: Dict[int, str], value: int) -> QComboBox:
    cb = QComboBox()
    for k, t in items.items():
        cb.addItem(t, k)
    cb.setCurrentIndex(max(0, cb.findData(int(value))))
    return cb


# ---------------------------------------------------------------- camera ISP (CSI)


class CameraIspDialog(QDialog):
    """Argus image settings for the CSI camera (applied when the camera is (re)opened)."""

    def __init__(self, parent=None, cfg: Optional[Dict[str, Any]] = None):
        super().__init__(parent)
        self.setWindowTitle("Camera image settings (CSI) / تنظیمات تصویر دوربین")
        self.setMinimumWidth(460)
        c = merge_isp(cfg if cfg is not None else load_isp_settings())
        root = QVBoxLayout(self)
        tip = QLabel(
            "Sensor / ISP settings of the IMX477 (Argus). Short exposure reduces motion blur; "
            "fixed white balance keeps colours stable. The camera re-opens to apply them.\n"
            "تنظیمات سنسور دوربین CSI. نوردهی کوتاه تاری حرکت را کم می‌کند."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#94a3b8;")
        root.addWidget(tip)

        self.enabled = QCheckBox("Use these settings (off = Argus defaults) / فعال")
        self.enabled.setChecked(c["enabled"])
        root.addWidget(self.enabled)
        self.preset = QComboBox()
        self.preset.addItem("— preset —", "")
        for name in PRESETS:
            self.preset.addItem(name, name)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        root.addWidget(self.preset)

        form = QFormLayout()
        self.wbmode = _combo(WB_MODES, c["wbmode"])
        form.addRow("White balance", self.wbmode)
        self.awblock = QCheckBox("Lock white balance")
        self.awblock.setChecked(c["awblock"])
        form.addRow("", self.awblock)
        self.saturation = _dspin(c["saturation"], 0.0, 2.0, 0.1)
        form.addRow("Saturation", self.saturation)
        self.exp_auto = QCheckBox("Auto exposure time")
        self.exp_auto.setChecked(c["exposure_auto"])
        form.addRow("", self.exp_auto)
        self.exp_min = QSpinBox()
        self.exp_min.setRange(13, 100000)
        self.exp_min.setSuffix(" µs")
        self.exp_min.setValue(int(c["exposure_min_us"]))
        self.exp_max = QSpinBox()
        self.exp_max.setRange(20, 400000)
        self.exp_max.setSuffix(" µs")
        self.exp_max.setValue(int(c["exposure_max_us"]))
        self.exp_max.setToolTip("Upper limit of the exposure time. 1000–2000 µs keeps a moving camera sharp.")
        form.addRow("Exposure min", self.exp_min)
        form.addRow("Exposure max", self.exp_max)
        self.gain_max = _dspin(c["gain_max"], 1.0, 22.0, 1.0, 1)
        form.addRow("Max analog gain", self.gain_max)
        self.exp_comp = _dspin(c["exposure_comp"], -2.0, 2.0, 0.1, 1, " EV")
        form.addRow("Exposure compensation", self.exp_comp)
        self.aelock = QCheckBox("Lock auto exposure")
        self.aelock.setChecked(c["aelock"])
        form.addRow("", self.aelock)
        self.tnr_mode = _combo(NR_MODES, c["tnr_mode"])
        self.tnr_strength = _dspin(c["tnr_strength"], -1.0, 1.0, 0.1, 1)
        self.tnr_strength.setToolTip("−1 = default")
        form.addRow("Noise reduction", self.tnr_mode)
        form.addRow("NR strength", self.tnr_strength)
        self.ee_mode = _combo(NR_MODES, c["ee_mode"])
        self.ee_strength = _dspin(c["ee_strength"], -1.0, 1.0, 0.1, 1)
        self.ee_strength.setToolTip("−1 = default")
        form.addRow("Edge enhancement", self.ee_mode)
        form.addRow("Edge strength", self.ee_strength)
        root.addLayout(form)

        self.exp_auto.toggled.connect(self._sync)
        self.enabled.toggled.connect(self._sync)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._sync()

    def _sync(self, *_a) -> None:
        on = self.enabled.isChecked()
        for w in (self.preset, self.wbmode, self.awblock, self.saturation, self.exp_auto, self.gain_max,
                  self.exp_comp, self.aelock, self.tnr_mode, self.tnr_strength, self.ee_mode, self.ee_strength):
            w.setEnabled(on)
        for w in (self.exp_min, self.exp_max):
            w.setEnabled(on and not self.exp_auto.isChecked())

    def _apply_preset(self, *_a) -> None:
        name = self.preset.currentData()
        if not name:
            return
        c = merge_isp({**PRESETS[name], "enabled": True})
        self.enabled.setChecked(True)
        self.wbmode.setCurrentIndex(max(0, self.wbmode.findData(c["wbmode"])))
        self.awblock.setChecked(c["awblock"])
        self.saturation.setValue(c["saturation"])
        self.exp_auto.setChecked(c["exposure_auto"])
        self.exp_min.setValue(int(c["exposure_min_us"]))
        self.exp_max.setValue(int(c["exposure_max_us"]))
        self.gain_max.setValue(c["gain_max"])
        self.exp_comp.setValue(c["exposure_comp"])
        self.aelock.setChecked(c["aelock"])
        self.tnr_mode.setCurrentIndex(max(0, self.tnr_mode.findData(c["tnr_mode"])))
        self.tnr_strength.setValue(c["tnr_strength"])
        self.ee_mode.setCurrentIndex(max(0, self.ee_mode.findData(c["ee_mode"])))
        self.ee_strength.setValue(c["ee_strength"])

    def values(self) -> Dict[str, Any]:
        return merge_isp({
            "enabled": self.enabled.isChecked(),
            "wbmode": self.wbmode.currentData(),
            "awblock": self.awblock.isChecked(),
            "saturation": self.saturation.value(),
            "exposure_auto": self.exp_auto.isChecked(),
            "exposure_min_us": min(self.exp_min.value(), self.exp_max.value()),
            "exposure_max_us": max(self.exp_min.value(), self.exp_max.value()),
            "gain_max": self.gain_max.value(),
            "exposure_comp": self.exp_comp.value(),
            "aelock": self.aelock.isChecked(),
            "tnr_mode": self.tnr_mode.currentData(),
            "tnr_strength": self.tnr_strength.value(),
            "ee_mode": self.ee_mode.currentData(),
            "ee_strength": self.ee_strength.value(),
        })


def edit_isp_settings(parent=None) -> Optional[Dict[str, Any]]:
    """Dialog; returns the saved settings on OK, None on Cancel."""
    dlg = CameraIspDialog(parent)
    if qt_exec(dlg) != QDialog.Accepted:
        return None
    cfg = dlg.values()
    save_isp_settings(cfg)
    return cfg


# ---------------------------------------------------------------- live image adjust


class ImageAdjustDialog(QDialog):
    """Non-modal: every change is applied to the running detection immediately."""

    def __init__(self, cfg: Dict[str, Any], on_change: Callable[[Dict[str, Any]], None], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image adjust / تنظیم رنگ تصویر")
        self.setMinimumWidth(420)
        self._on_change = on_change
        c = merge_adjust(cfg)
        root = QVBoxLayout(self)
        tip = QLabel(
            "Applied to every frame before detection — changes take effect while running.\n"
            "روی هر فریم قبل از تشخیص اعمال می‌شود؛ تغییرات در حین اجرا اثر می‌کنند."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#94a3b8;")
        root.addWidget(tip)

        g = QGroupBox("Colour normalisation / یکسان‌سازی رنگ")
        f = QFormLayout(g)
        self.match = QCheckBox("Match training imagery colours (package)")
        self.match.setChecked(c["match_reference"])
        self.match.setToolTip("Uses reference/color_stats.json of newer packages; older ones fall back to grey-world")
        f.addRow(self.match)
        self.wb = QCheckBox("Grey-world white balance (fallback)")
        self.wb.setChecked(c["white_balance"])
        f.addRow(self.wb)
        self.strength = _dspin(c["strength"], 0.0, 1.0, 0.1, 2)
        f.addRow("Strength", self.strength)
        root.addWidget(g)

        g2 = QGroupBox("Manual / دستی")
        f2 = QFormLayout(g2)
        self.brightness = _dspin(c["brightness"], -100.0, 100.0, 5.0, 0)
        self.contrast = _dspin(c["contrast"], 0.5, 2.0, 0.05)
        self.gamma = _dspin(c["gamma"], 0.4, 2.5, 0.05)
        self.gamma.setToolTip("< 1 brightens shadows")
        self.saturation = _dspin(c["saturation"], 0.0, 2.0, 0.1)
        f2.addRow("Brightness", self.brightness)
        f2.addRow("Contrast", self.contrast)
        f2.addRow("Gamma", self.gamma)
        f2.addRow("Saturation", self.saturation)
        self.clahe = QCheckBox("Local contrast (CLAHE) — slower")
        self.clahe.setChecked(c["clahe"])
        self.clahe_clip = _dspin(c["clahe_clip"], 1.0, 6.0, 0.5, 1)
        f2.addRow(self.clahe)
        f2.addRow("CLAHE clip", self.clahe_clip)
        self.show_adj = QCheckBox("Preview adjusted image (what the model sees)")
        self.show_adj.setChecked(c["show_adjusted"])
        f2.addRow(self.show_adj)
        root.addWidget(g2)

        row = QHBoxLayout()
        b_reset = QPushButton("Reset / بازنشانی")
        b_reset.clicked.connect(self._reset)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.close)
        row.addWidget(b_reset)
        row.addStretch(1)
        row.addWidget(b_close)
        root.addLayout(row)

        for w in (self.match, self.wb, self.clahe, self.show_adj):
            w.toggled.connect(self._changed)
        for w in (self.strength, self.brightness, self.contrast, self.gamma, self.saturation, self.clahe_clip):
            w.valueChanged.connect(self._changed)

    def _reset(self) -> None:
        d = merge_adjust({})
        self.match.setChecked(d["match_reference"])
        self.wb.setChecked(d["white_balance"])
        self.strength.setValue(d["strength"])
        self.brightness.setValue(d["brightness"])
        self.contrast.setValue(d["contrast"])
        self.gamma.setValue(d["gamma"])
        self.saturation.setValue(d["saturation"])
        self.clahe.setChecked(d["clahe"])
        self.clahe_clip.setValue(d["clahe_clip"])
        self.show_adj.setChecked(d["show_adjusted"])

    def values(self) -> Dict[str, Any]:
        return {
            "match_reference": self.match.isChecked(),
            "white_balance": self.wb.isChecked(),
            "strength": self.strength.value(),
            "brightness": self.brightness.value(),
            "contrast": self.contrast.value(),
            "gamma": self.gamma.value(),
            "saturation": self.saturation.value(),
            "clahe": self.clahe.isChecked(),
            "clahe_clip": self.clahe_clip.value(),
            "show_adjusted": self.show_adj.isChecked(),
        }

    def _changed(self, *_a) -> None:
        self._on_change(self.values())


class ImageControlBar(QWidget):
    """Main-window row: Image adjust on/off + settings, and the CSI camera ISP settings."""

    def __init__(self, parent=None, on_isp_changed: Optional[Callable[[Dict[str, Any]], None]] = None):
        super().__init__(parent)
        self._cfg = load_adjust_settings()
        self._dlg: Optional[ImageAdjustDialog] = None
        self._on_isp_changed = on_isp_changed
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.chk = QCheckBox("Image adjust / تنظیم تصویر")
        self.chk.setToolTip("Colour-normalise frames before detection (towards the training imagery)")
        self.chk.setChecked(self._cfg["enabled"])
        self.chk.toggled.connect(self._toggled)
        row.addWidget(self.chk)
        b_adj = QPushButton("Adjust… / تنظیم رنگ")
        b_adj.clicked.connect(self._open_adjust)
        row.addWidget(b_adj)
        b_isp = QPushButton("Camera ISP… / تنظیمات دوربین")
        b_isp.setToolTip("CSI (IMX477) white balance, exposure, gain, noise reduction, sharpening")
        b_isp.clicked.connect(self._open_isp)
        row.addWidget(b_isp)
        self.label = QLabel()
        self.label.setStyleSheet("color:#94a3b8;")
        row.addWidget(self.label, stretch=1)
        self._refresh_label()

    def config(self) -> Dict[str, Any]:
        """Polled by the runtime every frame (cheap)."""
        return dict(self._cfg)

    def _refresh_label(self) -> None:
        self.label.setText(f"{describe_adjust(self._cfg)}  ·  {describe_isp(load_isp_settings())}")

    def _set(self, cfg: Dict[str, Any]) -> None:
        self._cfg = merge_adjust(cfg)
        save_adjust_settings(self._cfg)
        self._refresh_label()

    def _toggled(self, on: bool) -> None:
        self._set({**self._cfg, "enabled": bool(on)})

    def _open_adjust(self) -> None:
        if self._dlg is None:
            self._dlg = ImageAdjustDialog(self._cfg, lambda v: self._set({**v, "enabled": self.chk.isChecked()}), self)
        if not self.chk.isChecked():
            self.chk.setChecked(True)
        self._dlg.show()
        self._dlg.raise_()

    def _open_isp(self) -> None:
        cfg = edit_isp_settings(self)
        if cfg is not None:
            self._refresh_label()
            if self._on_isp_changed is not None:
                self._on_isp_changed(cfg)
