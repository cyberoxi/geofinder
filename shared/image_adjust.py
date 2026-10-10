"""Live colour normalisation of camera frames before detection (Jetson runtime).

The model was trained on satellite imagery; a real camera has another white balance, exposure
and contrast. ``ImageAdjuster`` pulls each frame's per-channel statistics towards the training
imagery (``reference/color_stats.json`` in the package: mean / std of the scene mosaic), or
towards a neutral grey-world balance for older packages, plus optional manual tweaks and CLAHE.

Cost: per-channel LUT (fast, ~1–2 ms at 1080p); statistics are measured on a small copy every
few frames and smoothed, so the image does not flicker. CLAHE / saturation cost more.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

DEFAULT_ADJUST: Dict[str, Any] = {
    "enabled": False,
    "match_reference": True,  # towards the training imagery colours (package color_stats.json)
    "white_balance": True,  # grey-world balance when no reference is available
    "strength": 0.7,  # 0 … 1 blend towards the target statistics
    "brightness": 0.0,  # −100 … 100
    "contrast": 1.0,  # 0.5 … 2
    "gamma": 1.0,  # 0.4 … 2.5 (<1 brightens shadows)
    "saturation": 1.0,  # 0 … 2
    "clahe": False,  # local contrast (costs ~10 ms at 1080p on Orin)
    "clahe_clip": 2.0,
    "show_adjusted": True,  # preview the adjusted frame (what the model sees)
}

STATS_EVERY = 5  # frames between statistics updates
STATS_EMA = 0.25


def merge_adjust(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(DEFAULT_ADJUST)
    for k, v in (cfg or {}).items():
        if k in out:
            out[k] = type(out[k])(v)
    return out


def image_color_stats(img: np.ndarray, max_side: int = 1024) -> Dict[str, list]:
    """Per-channel (B, G, R) mean / std — written into packages from the scene mosaic."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    px = small.reshape(-1, 3).astype(np.float32)
    return {"mean": px.mean(axis=0).round(3).tolist(), "std": px.std(axis=0).round(3).tolist()}


def load_color_stats(path: str | Path) -> Optional[Dict[str, list]]:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if len(d.get("mean", [])) == 3 and len(d.get("std", [])) == 3:
            return d
    except (OSError, ValueError):
        pass
    return None


class ImageAdjuster:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, reference: Optional[Dict[str, list]] = None):
        self.reference = reference
        self.cfg = merge_adjust(cfg)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._n = 0
        self._lut: Optional[np.ndarray] = None
        self._clahe = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg["enabled"])

    def configure(self, cfg: Optional[Dict[str, Any]]) -> bool:
        """Apply new settings; True if they changed."""
        new = merge_adjust(cfg)
        if new == self.cfg:
            return False
        self.cfg = new
        self._lut = None
        self._clahe = None
        return True

    # ------------------------------------------------------------ stats + LUT
    def _update_stats(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
        px = small.reshape(-1, 3).astype(np.float32)
        m, s = px.mean(axis=0), px.std(axis=0)
        if self._mean is None:
            self._mean, self._std = m, s
        else:
            self._mean += STATS_EMA * (m - self._mean)
            self._std += STATS_EMA * (s - self._std)
        self._lut = None

    def _target(self) -> tuple:
        m, s = self._mean, np.maximum(self._std, 1.0)
        c = self.cfg
        if c["match_reference"] and self.reference is not None:
            tm = np.asarray(self.reference["mean"], np.float32)
            ts = np.asarray(self.reference["std"], np.float32)
        elif c["white_balance"]:
            tm = np.full(3, float(m.mean()), np.float32)  # grey world: equal channel means
            ts = np.full(3, float(s.mean()), np.float32)
        else:
            return m, s
        # Do not amplify noise / crush detail too much
        ts = np.clip(ts, s * 0.5, s * 2.5)
        return tm, ts

    def _build_lut(self) -> np.ndarray:
        c = self.cfg
        x = np.arange(256, dtype=np.float32)
        lut = np.empty((256, 1, 3), np.uint8)
        tm, ts = self._target()
        a = float(np.clip(c["strength"], 0.0, 1.0))
        for ch in range(3):
            m, s = float(self._mean[ch]), float(max(self._std[ch], 1.0))
            y = (x - m) / s * float(ts[ch]) + float(tm[ch])
            y = x + a * (y - x)
            y = (y - 128.0) * float(c["contrast"]) + 128.0 + float(c["brightness"])
            y = np.clip(y, 0, 255)
            if abs(float(c["gamma"]) - 1.0) > 1e-3:
                y = 255.0 * np.power(y / 255.0, float(c["gamma"]))
            lut[:, 0, ch] = np.clip(y + 0.5, 0, 255).astype(np.uint8)
        return lut

    # ------------------------------------------------------------ apply
    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or frame is None or frame.ndim != 3:
            return frame
        if self._n % STATS_EVERY == 0 or self._mean is None:
            self._update_stats(frame)
        self._n += 1
        if self._lut is None:
            self._lut = self._build_lut()
        out = cv2.LUT(frame, self._lut)
        c = self.cfg
        if abs(float(c["saturation"]) - 1.0) > 1e-3:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=float(c["saturation"]))
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if c["clahe"]:
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(clipLimit=float(c["clahe_clip"]), tileGridSize=(8, 8))
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[..., 0] = self._clahe.apply(lab[..., 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return out


def describe_adjust(cfg: Optional[Dict[str, Any]], has_reference: Optional[bool] = None) -> str:
    c = merge_adjust(cfg)
    if not c["enabled"]:
        return "Image adjust: OFF"
    parts = []
    if c["match_reference"]:
        parts.append("match training colours" + ("" if has_reference in (None, True) else " (n/a → grey-world)"))
    elif c["white_balance"]:
        parts.append("grey-world WB")
    if c["clahe"]:
        parts.append("CLAHE")
    for k, d in (("brightness", 0.0), ("contrast", 1.0), ("gamma", 1.0), ("saturation", 1.0)):
        if abs(float(c[k]) - d) > 1e-3:
            parts.append(f"{k} {c[k]:g}")
    return "Image adjust: " + (", ".join(parts) if parts else "on")
