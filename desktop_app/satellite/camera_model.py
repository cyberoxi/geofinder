"""Real-camera simulation settings: camera + lens presets, field of view, altitude → zoom range.

``CameraSim`` bundles three independent options used when rendering training videos:

* ``oblique``  — tilted camera (pitch / roll) instead of a straight-down view (see oblique.py)
* ``sensor``   — the chosen camera + lens: distortion, vignetting, noise, white balance, blur… (sensor_sim.py)
* ``altitude`` — zoom range taken from the flight altitude range and the lens field of view

All three are off by default; with everything off the renderer output is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

# Sensor area used by the video mode (mm). IMX477 1080p/4K modes read a 3840×2160 window
# (1.55 µm pixels; 1080p is 2×2 binned), the full mode reads 4032×3040.
CAMERAS: Dict[str, Dict[str, Any]] = {
    "imx477_1080p": {"title": "IMX477 (RPi HQ) — 1920×1080", "sensor_mm": (5.952, 3.348), "noise": 1.0,
                     "interchangeable": True},
    "imx477_full": {"title": "IMX477 (RPi HQ) — 4032×3040", "sensor_mm": (6.250, 4.712), "noise": 0.8,
                    "interchangeable": True},
    "usb_webcam": {"title": "USB webcam (fixed ~70° lens)", "sensor_mm": (3.6, 2.03), "noise": 1.6,
                   "interchangeable": False, "focal_mm": 2.6, "k1": -0.18, "k2": 0.04, "vignette": 0.35},
}

# focal (mm), radial distortion k1/k2, vignetting strength, lateral chromatic aberration (fraction)
LENSES: Dict[str, Dict[str, Any]] = {
    "m12_2.8": {"title": "M12 2.8 mm (very wide)", "focal_mm": 2.8, "k1": -0.30, "k2": 0.08, "vignette": 0.45, "ca": 0.003},
    "m12_4": {"title": "M12 4 mm (wide)", "focal_mm": 4.0, "k1": -0.20, "k2": 0.05, "vignette": 0.35, "ca": 0.0025},
    "cs_6": {"title": "6 mm CS (official wide)", "focal_mm": 6.0, "k1": -0.12, "k2": 0.02, "vignette": 0.28, "ca": 0.0015},
    "m12_8": {"title": "M12 8 mm", "focal_mm": 8.0, "k1": -0.06, "k2": 0.01, "vignette": 0.2, "ca": 0.001},
    "m12_12": {"title": "M12 12 mm", "focal_mm": 12.0, "k1": -0.03, "k2": 0.0, "vignette": 0.15, "ca": 0.001},
    "c_16": {"title": "16 mm C (official telephoto)", "focal_mm": 16.0, "k1": -0.015, "k2": 0.0, "vignette": 0.12, "ca": 0.0008},
    "c_25": {"title": "25 mm C", "focal_mm": 25.0, "k1": 0.0, "k2": 0.0, "vignette": 0.1, "ca": 0.0005},
    "c_35": {"title": "35 mm C", "focal_mm": 35.0, "k1": 0.005, "k2": 0.0, "vignette": 0.08, "ca": 0.0005},
    "custom": {"title": "Custom focal length", "focal_mm": 0.0, "k1": -0.05, "k2": 0.0, "vignette": 0.2, "ca": 0.001},
}


@dataclass
class CameraSim:
    camera: str = "imx477_1080p"
    lens: str = "cs_6"
    custom_focal_mm: float = 8.0
    # 1. oblique view
    oblique: bool = False
    max_pitch: float = 35.0  # degrees from straight down
    max_roll: float = 6.0
    oblique_fraction: float = 0.6  # share of videos (the canonical first video stays straight down)
    # 2. real sensor + lens
    sensor: bool = False
    sensor_fraction: float = 0.8
    sensor_strength: float = 1.0  # 0.5 = mild … 1.5 = harsh
    # 3. altitude profile
    altitude: bool = False
    alt_min_m: float = 80.0
    alt_max_m: float = 400.0

    # -------------------------------------------------------------- io
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "CameraSim":
        known = cls.__dataclass_fields__.keys()
        sim = cls(**{k: v for k, v in (d or {}).items() if k in known})
        if sim.camera not in CAMERAS:
            sim.camera = "imx477_1080p"
        if sim.lens not in LENSES:
            sim.lens = "cs_6"
        return sim

    @property
    def any_enabled(self) -> bool:
        return bool(self.oblique or self.sensor or self.altitude)

    # -------------------------------------------------------------- optics
    def lens_params(self) -> Dict[str, float]:
        cam = CAMERAS[self.camera]
        if not cam.get("interchangeable", True):
            return {k: float(cam[k]) for k in ("focal_mm", "k1", "k2", "vignette")} | {"ca": 0.002}
        lens = dict(LENSES[self.lens])
        if self.lens == "custom":
            lens["focal_mm"] = max(1.0, float(self.custom_focal_mm))
        return {k: float(lens[k]) for k in ("focal_mm", "k1", "k2", "vignette", "ca")}

    def fov_deg(self) -> Tuple[float, float]:
        """(horizontal, vertical) field of view of the camera + lens."""
        sw, sh = CAMERAS[self.camera]["sensor_mm"]
        f = self.lens_params()["focal_mm"]
        return math.degrees(2 * math.atan(sw / (2 * f))), math.degrees(2 * math.atan(sh / (2 * f)))

    def fov_v(self) -> float:
        return self.fov_deg()[1]

    def view_h_range(self, meters_per_px: float) -> Optional[Tuple[float, float]]:
        """Mosaic px covered by the frame height at the min / max altitude (None if unknown scale)."""
        if not meters_per_px or meters_per_px <= 0:
            return None
        t = math.tan(math.radians(self.fov_v()) / 2.0)
        lo, hi = sorted((max(1.0, self.alt_min_m), max(1.0, self.alt_max_m)))
        return 2 * lo * t / meters_per_px, 2 * hi * t / meters_per_px

    def describe(self) -> str:
        if not self.any_enabled:
            return "Camera simulation: OFF (straight-down satellite views)"
        fh, fv = self.fov_deg()
        lens = CAMERAS[self.camera]["title"]
        if CAMERAS[self.camera].get("interchangeable", True):
            lens += f" + {self.lens_params()['focal_mm']:g} mm"
        on = []
        if self.oblique:
            on.append(f"tilt ≤{self.max_pitch:g}°")
        if self.sensor:
            on.append("sensor/lens")
        if self.altitude:
            on.append(f"altitude {self.alt_min_m:g}–{self.alt_max_m:g} m")
        return f"Camera simulation: {', '.join(on)} — {lens} (FOV {fh:.0f}°×{fv:.0f}°)"
