"""Min-confidence slider for the Jetson main window (same control as the desktop Live Test).

The value starts at the package's ``min_confidence`` (what was set in the desktop Live Test),
can be dragged while processing (applies on the next frame) and is remembered per package.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Optional

from shared.qt_compat import QHBoxLayout, QLabel, QPushButton, QSettings, QSlider, Qt, QWidget

ORG = "GeoFinder"
APP = "JetsonRuntime"
_KEY = "detect/min_confidence_pct"
_PKG_KEY = "detect/min_confidence_package"
_DEFAULT_PCT = 45


def package_min_confidence_pct(package: str) -> Optional[int]:
    """``manifest.json`` → min_confidence in percent, from a package folder or ZIP (None if unreadable)."""
    try:
        p = Path(package)
        if p.is_dir():
            text = (p / "manifest.json").read_text(encoding="utf-8")
        else:
            with zipfile.ZipFile(p) as zf:
                names = sorted((n for n in zf.namelist() if n.endswith("manifest.json")), key=len)
                if not names:
                    return None
                text = zf.read(names[0]).decode("utf-8")
        return max(1, min(99, round(float(json.loads(text)["min_confidence"]) * 100)))
    except Exception:  # noqa: BLE001 — missing / old package: keep the slider as it is
        return None


class ConfidenceControl(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._package: Optional[str] = None
        self._package_pct: Optional[int] = None
        self._fraction = _DEFAULT_PCT / 100.0
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        lab = QLabel("Min confidence / حداقل دقت:")
        lab.setToolTip("Live threshold — drag while processing; remembered for this package.")
        row.addWidget(lab)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setMinimumWidth(180)
        self.slider.setToolTip("Detection confidence must be ≥ this percent (applies immediately)")
        row.addWidget(self.slider, stretch=1)
        self.value = QLabel()
        self.value.setMinimumWidth(40)
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.value)
        self.reset_btn = QPushButton("Package")
        self.reset_btn.setToolTip("Back to the value saved in the package (desktop Live Test)")
        self.reset_btn.clicked.connect(self._reset_to_package)
        row.addWidget(self.reset_btn)
        s = QSettings(ORG, APP)
        self.slider.setValue(self._int(s.value(_KEY, _DEFAULT_PCT)))
        self.slider.valueChanged.connect(self._on_changed)
        self._on_changed(self.slider.value())

    @staticmethod
    def _int(raw) -> int:
        try:
            return max(1, min(99, int(raw)))
        except (TypeError, ValueError):
            return _DEFAULT_PCT

    def set_package(self, package: str) -> None:
        """New package: use its value, unless this same package was tuned here before."""
        self._package = package
        self._package_pct = package_min_confidence_pct(package)
        self.reset_btn.setEnabled(self._package_pct is not None)
        s = QSettings(ORG, APP)
        if s.value(_PKG_KEY, "") != package and self._package_pct is not None:
            self.slider.setValue(self._package_pct)
        self._on_changed(self.slider.value())

    def _reset_to_package(self) -> None:
        if self._package_pct is not None:
            self.slider.setValue(self._package_pct)

    def _on_changed(self, pct: int) -> None:
        self._fraction = pct / 100.0
        self.value.setText(f"{pct} %")
        self.value.setToolTip(f"Package value: {self._package_pct} %" if self._package_pct is not None else "")
        s = QSettings(ORG, APP)
        s.setValue(_KEY, int(pct))
        if self._package:
            s.setValue(_PKG_KEY, self._package)

    def fraction(self) -> float:
        """Current threshold 0–1 — a plain float, safe to read from the worker thread."""
        return self._fraction
