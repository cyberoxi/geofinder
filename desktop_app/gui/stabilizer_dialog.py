"""Detection filter settings (false-target suppression) — live test + Jetson package."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from desktop_app.gui import stabilizer_help as H
from shared.qt_compat import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSettings,
    QSpinBox,
    Qt,
    QVBoxLayout,
    QWidget,
    qt_exec,
)
from shared.target_stabilizer import merge_stabilizer_config

ORG = "GeoFinder"
APP = "Desktop"
_KEY = "detection/stabilizer"

# Unicode BiDi: force RTL even when a line starts with Latin / digits / symbols.
_RLI, _PDI, _RLM = "\u2067", "\u2069", "\u200f"


def _force_rtl_text(text: str) -> str:
    if not text:
        return text
    if text.startswith(_RLI) or text.startswith(_RLM):
        return text
    return f"{_RLI}{text}{_PDI}"


def load_stabilizer_settings() -> Dict[str, Any]:
    raw = QSettings(ORG, APP).value(_KEY, "")
    try:
        cfg = json.loads(str(raw)) if raw else {}
    except (TypeError, ValueError):
        cfg = {}
    return merge_stabilizer_config(cfg if isinstance(cfg, dict) else {})


def save_stabilizer_settings(cfg: Dict[str, Any]) -> None:
    QSettings(ORG, APP).setValue(_KEY, json.dumps(merge_stabilizer_config(cfg)))


def describe_stabilizer(cfg: Dict[str, Any]) -> str:
    on = [
        name
        for key, name in (
            ("geometry_check", "geometry"),
            ("temporal", f"temporal {cfg['confirm_hits']}/{cfg['confirm_window']}"),
            ("landmark_consistency", "aux-consistency"),
        )
        if cfg.get(key)
    ]
    return "filters: " + (", ".join(on) if on else "OFF")


def _dspin(value: float, lo: float, hi: float, step: float, suffix: str = "") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(2)
    w.setValue(float(value))
    if suffix:
        w.setSuffix(suffix)
    return w


def _ispin(value: int, lo: int, hi: int, suffix: str = "") -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(int(value))
    if suffix:
        w.setSuffix(suffix)
    return w


def _text(text: str, size: int, color: str) -> QLabel:
    lab = QLabel(_force_rtl_text(text))
    lab.setWordWrap(True)
    lab.setAlignment(Qt.AlignRight | Qt.AlignTop)
    lab.setLayoutDirection(Qt.RightToLeft)
    lab.setStyleSheet(f"color:{color}; font-size:{size}px;")
    return lab


def _group(title: str, checked: bool, desc: str) -> Tuple[QGroupBox, QFormLayout]:
    """Checkable group: 'how it works' paragraph on top, then the parameter form."""
    g = QGroupBox(_force_rtl_text(title))
    g.setCheckable(True)
    g.setChecked(checked)
    g.setLayoutDirection(Qt.RightToLeft)
    g.setAlignment(Qt.AlignRight)
    g.setStyleSheet("QGroupBox{font-weight:bold; font-size:14px; padding-top:16px;}")
    box = QVBoxLayout(g)
    box.setSpacing(10)
    box.addWidget(_text(desc, 13, "#cbd5e1"))
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    form.setFormAlignment(Qt.AlignRight | Qt.AlignTop)
    form.setVerticalSpacing(4)
    form.setHorizontalSpacing(16)
    box.addLayout(form)
    return g, form


def _param(form: QFormLayout, title: str, widget, help_text: str) -> None:
    """Parameter row followed by a full-width explanation of what it does."""
    widget.setMinimumWidth(170)
    widget.setMaximumWidth(240)
    widget.setToolTip(help_text)
    form.addRow(_force_rtl_text(title), widget)
    help_lab = _text(help_text, 12, "#9ca3af")
    help_lab.setContentsMargins(0, 0, 0, 10)
    form.addRow(help_lab)


class StabilizerSettingsDialog(QDialog):
    def __init__(self, cfg: Dict[str, Any] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Filters / فیلتر تشخیص اشتباه")
        self.setLayoutDirection(Qt.RightToLeft)
        self.setMinimumSize(760, 560)
        height = 960
        try:
            height = min(height, self.screen().availableGeometry().height() - 80)
        except Exception:  # noqa: BLE001
            pass
        self.resize(920, height)
        self._build(merge_stabilizer_config(cfg) if cfg is not None else load_stabilizer_settings())

    def _build(self, cfg: Dict[str, Any]) -> None:
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setLayoutDirection(Qt.RightToLeft)
        body = QWidget()
        body.setLayoutDirection(Qt.RightToLeft)
        layout = QVBoxLayout(body)
        layout.setSpacing(16)
        layout.addWidget(_text(H.TOP_NOTE, 13, "#93c5fd"))

        g1, f1 = _group("1) Geometry check / بررسی هندسی با اشیای کمکی", bool(cfg["geometry_check"]), H.GEO_DESC)
        self.geo_err = _dspin(cfg["geometry_max_center_err"], 0.1, 5.0, 0.1, " × size")
        _param(f1, "Max centre error / حداکثر خطای مرکز", self.geo_err, H.GEO_ERR_HELP)
        self.geo_size = _dspin(cfg["geometry_max_size_ratio"], 1.1, 10.0, 0.1, " ×")
        _param(f1, "Max size ratio / حداکثر نسبت اندازه", self.geo_size, H.GEO_SIZE_HELP)
        layout.addWidget(g1)
        self.g_geo = g1

        g2, f2 = _group("2) Temporal confirm + gate / تأیید زمانی و دروازه حرکت", bool(cfg["temporal"]), H.TMP_DESC)
        self.hits = _ispin(cfg["confirm_hits"], 1, 30, " hits")
        _param(f2, "Confirm N / تعداد تأیید", self.hits, H.HITS_HELP)
        self.window = _ispin(cfg["confirm_window"], 1, 60, " frames")
        _param(f2, "Within M frames / در M فریم", self.window, H.WINDOW_HELP)
        self.hold = _dspin(cfg["hold_conf_ratio"], 0.1, 1.0, 0.05, " × min conf")
        _param(f2, "Hold confidence / دقت نگه‌داشتن", self.hold, H.HOLD_HELP)
        self.gate = _dspin(cfg["gate_ratio"], 0.2, 10.0, 0.1, " × size")
        _param(f2, "Gate radius / شعاع دروازه", self.gate, H.GATE_HELP)
        self.gate_min = _dspin(cfg["gate_min_px"], 0.0, 2000.0, 5.0, " px")
        _param(f2, "Min gate / حداقل شعاع", self.gate_min, H.GATE_MIN_HELP)
        self.coast = _ispin(cfg["coast_frames"], 0, 300, " frames")
        _param(f2, "Coast / ادامه بدون تشخیص", self.coast, H.COAST_HELP)
        layout.addWidget(g2)
        self.g_tmp = g2

        g3, f3 = _group(
            "3) Aux-object consistency / سازگاری اشیای کمکی", bool(cfg["landmark_consistency"]), H.LM_DESC
        )
        self.min_lm = _ispin(cfg["min_landmarks_acquire"], 1, 10)
        _param(f3, "Min aux to acquire / حداقل اشیای کمکی", self.min_lm, H.MIN_LM_HELP)
        layout.addWidget(g3)
        self.g_lm = g3
        layout.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.setLayoutDirection(Qt.RightToLeft)
        reset = QPushButton("Defaults / پیش‌فرض")
        reset.setLayoutDirection(Qt.RightToLeft)
        reset.clicked.connect(self._reset)
        buttons.addButton(reset, QDialogButtonBox.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _reset(self) -> None:
        d = merge_stabilizer_config({})
        self.g_geo.setChecked(d["geometry_check"])
        self.geo_err.setValue(d["geometry_max_center_err"])
        self.geo_size.setValue(d["geometry_max_size_ratio"])
        self.g_tmp.setChecked(d["temporal"])
        self.hits.setValue(d["confirm_hits"])
        self.window.setValue(d["confirm_window"])
        self.hold.setValue(d["hold_conf_ratio"])
        self.gate.setValue(d["gate_ratio"])
        self.gate_min.setValue(d["gate_min_px"])
        self.coast.setValue(d["coast_frames"])
        self.g_lm.setChecked(d["landmark_consistency"])
        self.min_lm.setValue(d["min_landmarks_acquire"])

    def values(self) -> Dict[str, Any]:
        return merge_stabilizer_config(
            {
                "geometry_check": self.g_geo.isChecked(),
                "geometry_max_center_err": self.geo_err.value(),
                "geometry_max_size_ratio": self.geo_size.value(),
                "temporal": self.g_tmp.isChecked(),
                "confirm_hits": self.hits.value(),
                "confirm_window": self.window.value(),
                "hold_conf_ratio": self.hold.value(),
                "gate_ratio": self.gate.value(),
                "gate_min_px": self.gate_min.value(),
                "coast_frames": self.coast.value(),
                "landmark_consistency": self.g_lm.isChecked(),
                "min_landmarks_acquire": self.min_lm.value(),
            }
        )


def edit_stabilizer_settings(parent=None) -> Dict[str, Any]:
    """Open the dialog; on OK save and return the new config, else the stored one."""
    dlg = StabilizerSettingsDialog(parent=parent)
    if qt_exec(dlg) == QDialog.Accepted:
        cfg = dlg.values()
        save_stabilizer_settings(cfg)
        return cfg
    return load_stabilizer_settings()
