"""Real camera + lens look for rendered frames (applied after the pinhole render).

Geometry: radial lens distortion (Brown k1/k2). The frame is rendered as a slightly wider pinhole
view (``LensModel.pre``) and remapped, so the distorted frame has no empty corners; labels go
through ``LensModel.distort_points`` — image and labels stay exactly aligned.

Look: lateral chromatic aberration, vignetting, motion blur from the camera's own movement,
defocus, white-balance / exposure / tone / saturation of the ISP, shot + read noise, sharpening
and compression. Random per video, scaled by the camera's noise level and ``strength``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

from desktop_app.satellite.camera_model import CAMERAS, CameraSim


# ---------------------------------------------------------------- lens geometry


class LensModel:
    def __init__(self, out_w: int, out_h: int, fov_v: float, k1: float, k2: float):
        self.w, self.h = int(out_w), int(out_h)
        self.f = (out_h / 2.0) / math.tan(math.radians(fov_v) / 2.0)
        self.cx, self.cy = out_w / 2.0, out_h / 2.0
        self.k1, self.k2 = float(k1), float(k2)
        self.active = abs(self.k1) > 1e-6 or abs(self.k2) > 1e-6
        self.pre: Optional[np.ndarray] = None
        self.pre_scale = 1.0
        self.r_lim = math.hypot(self.cx, self.cy) / self.f
        self.mapx = self.mapy = None
        if not self.active:
            return
        K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]], np.float64)
        dist = np.array([self.k1, self.k2, 0.0, 0.0], np.float64)
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        grid = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(grid, K, dist, P=K).reshape(self.h, self.w, 2)
        ux, uy = und[..., 0] - self.cx, und[..., 1] - self.cy
        # Wider pinhole render so every distorted pixel has content (barrel lenses pull corners in)
        m = max(1.0, float(np.abs(ux).max()) / self.cx, float(np.abs(uy).max()) / self.cy)
        self.pre_scale = 1.0 / m
        self.pre = np.array([[1 / m, 0, self.cx * (1 - 1 / m)], [0, 1 / m, self.cy * (1 - 1 / m)], [0, 0, 1.0]])
        self.mapx = (self.cx + ux / m).astype(np.float32)
        self.mapy = (self.cy + uy / m).astype(np.float32)
        self.r_lim = float(np.sqrt(ux**2 + uy**2).max()) / self.f

    def distort_points(self, pts: np.ndarray) -> np.ndarray:
        """Pinhole frame px → distorted frame px (beyond the frame corner: constant factor)."""
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        if not self.active:
            return p.astype(np.float32)
        xn, yn = (p[:, 0] - self.cx) / self.f, (p[:, 1] - self.cy) / self.f
        r2 = np.minimum(xn * xn + yn * yn, self.r_lim**2)
        fac = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return np.stack([self.cx + xn * fac * self.f, self.cy + yn * fac * self.f], axis=1).astype(np.float32)

    def remap(self, img: np.ndarray) -> np.ndarray:
        if not self.active:
            return img
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@lru_cache(maxsize=8)
def lens_model(out_w: int, out_h: int, fov_v: float, k1: float, k2: float) -> LensModel:
    return LensModel(out_w, out_h, round(fov_v, 3), k1, k2)


@lru_cache(maxsize=8)
def _radius2(w: int, h: int) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return (((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2) / 2.0  # 0 centre … 1 corner


# ---------------------------------------------------------------- per-video look


@dataclass
class SensorSim:
    lens: LensModel
    light: str = "day"
    wb: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # B, G, R gains
    exposure: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    shot: float = 0.0  # Poisson-like noise: σ² = shot · I
    read: float = 0.0  # Gaussian read noise σ
    defocus: float = 0.0
    sharpen: float = 0.0
    shutter: float = 0.5  # exposure time as a fraction of the frame interval (motion blur)
    jpeg: int = 90
    vignette: float = 0.0
    ca: float = 0.0

    @classmethod
    def random(cls, sim: CameraSim, out_size: Tuple[int, int], rng: random.Random, light: str = "day") -> "SensorSim":
        lp = sim.lens_params()
        lens = lens_model(int(out_size[0]), int(out_size[1]), float(sim.fov_v()), lp["k1"], lp["k2"])
        k = max(0.2, float(sim.sensor_strength))
        nz = float(CAMERAS[sim.camera].get("noise", 1.0)) * k
        day = light == "day"
        u = rng.uniform
        return cls(
            lens=lens,
            light=light,
            wb=(u(1 - 0.15 * k, 1 + 0.15 * k), u(1 - 0.04 * k, 1 + 0.04 * k), u(1 - 0.15 * k, 1 + 0.15 * k)) if day else (1, 1, 1),
            exposure=u(1 - 0.2 * k, 1 + 0.2 * k) if day else 1.0,
            contrast=u(0.9, 1.0 + 0.2 * k) if day else 1.0,
            saturation=u(1 - 0.25 * k, 1 + 0.15 * k) if day else 1.0,
            shot=u(0.2, 1.2) * nz * (1.0 if day else 0.5),
            read=u(0.8, 3.0) * nz * (1.0 if day else 0.5),
            defocus=rng.choice([0.0, 0.0, 0.0, 0.7, 1.2]) * k,
            sharpen=u(0.0, 0.7) if day else 0.0,
            shutter=min(1.0, u(0.1, 0.7) * k),
            jpeg=int(max(35, min(95, u(92 - 35 * k, 95)))),
            vignette=lp["vignette"] * u(0.7, 1.2),
            ca=lp["ca"] * u(0.5, 1.2),
        )

    def apply(self, img: np.ndarray, motion: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
        out = self.lens.remap(img)
        h, w = out.shape[:2]
        if self.ca > 0:  # lateral CA: red slightly larger, blue slightly smaller around the centre
            b, g, r = cv2.split(out)

            def _scale(ch: np.ndarray, s: float) -> np.ndarray:
                M = np.array([[s, 0, w / 2 * (1 - s)], [0, s, h / 2 * (1 - s)]], np.float32)
                return cv2.warpAffine(ch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            out = cv2.merge([_scale(b, 1 - self.ca), g, _scale(r, 1 + self.ca)])
        L = math.hypot(*motion) * self.shutter
        if L > 1.5:
            n = int(min(31, math.ceil(L))) | 1
            L = min(L, n - 1.0)
            ker = np.zeros((n, n), np.float32)
            c = n // 2
            dx, dy = motion[0] / max(1e-6, math.hypot(*motion)), motion[1] / max(1e-6, math.hypot(*motion))
            for t in np.linspace(-L / 2, L / 2, n * 2):
                ker[int(np.clip(round(c + dy * t), 0, n - 1)), int(np.clip(round(c + dx * t), 0, n - 1))] = 1.0
            out = cv2.filter2D(out, -1, ker / ker.sum())
        if self.defocus > 0:
            out = cv2.GaussianBlur(out, (0, 0), self.defocus)
        f = out.astype(np.float32)
        if self.light == "day":
            f *= np.array(self.wb, np.float32) * self.exposure
            if self.contrast != 1.0:  # ISP tone curve: contrast around mid-grey
                f = (f - 128.0) * self.contrast + 128.0
            if self.saturation != 1.0:
                grey = f.mean(axis=2, keepdims=True)
                f = grey + (f - grey) * self.saturation
        if self.vignette > 0:
            f *= (1.0 - self.vignette * _radius2(w, h))[..., None]
        if self.shot > 0 or self.read > 0:
            sigma = np.sqrt(np.clip(f, 0, 255) * self.shot + self.read**2)
            f += rng.standard_normal(f.shape).astype(np.float32) * sigma
        out = np.clip(f, 0, 255).astype(np.uint8)
        if self.sharpen > 0:
            blur = cv2.GaussianBlur(out, (0, 0), 1.2)
            out = cv2.addWeighted(out, 1 + self.sharpen, blur, -self.sharpen, 0)
        if self.jpeg < 95:
            ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg)])
            if ok:
                out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return out
