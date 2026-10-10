"""False-target filter toggles for Jetson (same three as desktop Live Test)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from shared.config import deep_update, default_runtime_config, load_yaml
from shared.qt_compat import (
    QCheckBox,
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
from shared.target_stabilizer import merge_stabilizer_config

ORG = "GeoFinder"
APP = "JetsonRuntime"
_KEY = "detection/stabilizer"


def load_stabilizer_settings(base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Package/default *base*, overlaid with any Jetson QSettings overrides."""
    out = merge_stabilizer_config(base)
    raw = QSettings(ORG, APP).value(_KEY, "")
    try:
        saved = json.loads(str(raw)) if raw else {}
    except (TypeError, ValueError):
        saved = {}
    if isinstance(saved, dict) and saved:
        out = merge_stabilizer_config(deep_update(out, saved))
    return out


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


def stabilizer_from_package(package: Optional[str]) -> Dict[str, Any]:
    """Read ``config/runtime.yaml`` stabilizer from a package folder/ZIP root."""
    base = default_runtime_config().get("stabilizer", {})
    if not package:
        return merge_stabilizer_config(base)
    root = Path(package)
    cfg_path = root / "config" / "runtime.yaml"
    if not cfg_path.is_file():
        # Unpacked validate dirs sometimes nest one level.
        for cand in root.glob("**/config/runtime.yaml"):
            cfg_path = cand
            break
    if cfg_path.is_file():
        try:
            data = load_yaml(cfg_path) or {}
            return merge_stabilizer_config(data.get("stabilizer"))
        except Exception:  # noqa: BLE001
            pass
    return merge_stabilizer_config(base)


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


class StabilizerSettingsDialog(QDialog):
    """Compact filter settings (same keys as desktop Detection Filters)."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Filters / فیلتر تشخیص اشتباه")
        self.setMinimumWidth(420)
        self._build(merge_stabilizer_config(cfg))

    def _build(self, cfg: Dict[str, Any]) -> None:
        root = QVBoxLayout(self)
        tip = QLabel(
            "Same three filters as the desktop Live Test.\n"
            "همان سه فیلتر تست زنده دسکتاپ — برای دوربین می‌توانید خاموششان کنید."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#94a3b8;")
        root.addWidget(tip)

        self.g_geo = QGroupBox("1) Geometry / هندسی")
        self.g_geo.setCheckable(True)
        self.g_geo.setChecked(bool(cfg["geometry_check"]))
        f1 = QFormLayout(self.g_geo)
        self.geo_err = _dspin(cfg["geometry_max_center_err"], 0.1, 5.0, 0.1, " × size")
        self.geo_size = _dspin(cfg["geometry_max_size_ratio"], 1.1, 10.0, 0.1, " ×")
        f1.addRow("Max centre error:", self.geo_err)
        f1.addRow("Max size ratio:", self.geo_size)
        root.addWidget(self.g_geo)

        self.g_tmp = QGroupBox("2) Temporal / زمانی")
        self.g_tmp.setCheckable(True)
        self.g_tmp.setChecked(bool(cfg["temporal"]))
        f2 = QFormLayout(self.g_tmp)
        self.hits = _ispin(cfg["confirm_hits"], 1, 30, " hits")
        self.window = _ispin(cfg["confirm_window"], 1, 60, " frames")
        self.hold = _dspin(cfg["hold_conf_ratio"], 0.1, 1.0, 0.05, " × min")
        self.gate = _dspin(cfg["gate_ratio"], 0.2, 10.0, 0.1, " × size")
        self.gate_min = _dspin(cfg["gate_min_px"], 0.0, 2000.0, 5.0, " px")
        self.coast = _ispin(cfg["coast_frames"], 0, 300, " frames")
        f2.addRow("Confirm N:", self.hits)
        f2.addRow("Within M frames:", self.window)
        f2.addRow("Hold conf:", self.hold)
        f2.addRow("Gate radius:", self.gate)
        f2.addRow("Min gate:", self.gate_min)
        f2.addRow("Coast frames:", self.coast)
        root.addWidget(self.g_tmp)

        self.g_lm = QGroupBox("3) Aux consistency / سازگاری اشیای کمکی")
        self.g_lm.setCheckable(True)
        self.g_lm.setChecked(bool(cfg["landmark_consistency"]))
        f3 = QFormLayout(self.g_lm)
        self.min_lm = _ispin(cfg["min_landmarks_acquire"], 1, 10)
        f3.addRow("Min aux to acquire:", self.min_lm)
        root.addWidget(self.g_lm)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        reset = QPushButton("Defaults")
        reset.clicked.connect(self._reset)
        buttons.addButton(reset, QDialogButtonBox.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

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


class StabilizerFilterBar(QWidget):
    """Three on/off checkboxes + Settings — mirrors desktop Live Test."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cfg = load_stabilizer_settings()
        self._on_change = None  # Optional[Callable[[Dict], None]]
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        lab = QLabel("False-target filters / فیلتر تشخیص اشتباه:")
        lab.setToolTip("Same filters as desktop Live Test; disable if camera lock never happens")
        row.addWidget(lab)
        self.chk_geometry = QCheckBox("Geometry / هندسی")
        self.chk_geometry.setToolTip("Reject targets that disagree with aux-object geometry")
        self.chk_temporal = QCheckBox("Temporal / زمانی")
        self.chk_temporal.setToolTip("Lock only after N-of-M confirmations; gate far jumps")
        self.chk_landmarks = QCheckBox("Aux consistency / سازگاری اشیای کمکی")
        self.chk_landmarks.setToolTip("Aux objects must fit the scene layout")
        for chk in (self.chk_geometry, self.chk_temporal, self.chk_landmarks):
            chk.toggled.connect(self._on_toggled)
            row.addWidget(chk)
        self.btn_settings = QPushButton("Settings… / تنظیمات")
        self.btn_settings.clicked.connect(self._edit)
        row.addWidget(self.btn_settings)
        row.addStretch(1)
        self._sync_checks()

    def set_change_callback(self, cb) -> None:
        self._on_change = cb

    def config(self) -> Dict[str, Any]:
        return dict(self._cfg)

    def set_config(self, cfg: Dict[str, Any]) -> None:
        self._cfg = merge_stabilizer_config(cfg)
        self._sync_checks()

    def reload_from_package(self, package: Optional[str]) -> None:
        """Seed from package, then re-apply any saved Jetson overrides."""
        self._cfg = load_stabilizer_settings(stabilizer_from_package(package))
        self._sync_checks()

    def _sync_checks(self) -> None:
        for chk, key in (
            (self.chk_geometry, "geometry_check"),
            (self.chk_temporal, "temporal"),
            (self.chk_landmarks, "landmark_consistency"),
        ):
            chk.blockSignals(True)
            chk.setChecked(bool(self._cfg.get(key)))
            chk.blockSignals(False)

    def _emit(self) -> None:
        save_stabilizer_settings(self._cfg)
        if self._on_change is not None:
            self._on_change(dict(self._cfg))

    def _on_toggled(self, *_args) -> None:
        self._cfg = dict(self._cfg)
        self._cfg["geometry_check"] = self.chk_geometry.isChecked()
        self._cfg["temporal"] = self.chk_temporal.isChecked()
        self._cfg["landmark_consistency"] = self.chk_landmarks.isChecked()
        self._cfg = merge_stabilizer_config(self._cfg)
        self._emit()

    def _edit(self) -> None:
        dlg = StabilizerSettingsDialog(self._cfg, self)
        if qt_exec(dlg) == QDialog.Accepted:
            self._cfg = dlg.values()
            self._sync_checks()
            self._emit()
