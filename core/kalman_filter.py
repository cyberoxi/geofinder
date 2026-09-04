"""EMA and Kalman smoothing for ROI geometry."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from core.types import ROI


class EMAFilter:
    def __init__(self, alpha: float = 0.35):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self._points: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._points = None

    def update(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if self._points is None or self._points.shape != pts.shape:
            self._points = pts.copy()
            return pts.copy()
        self._points = self.alpha * pts + (1.0 - self.alpha) * self._points
        return self._points.copy()


class KalmanBoxFilter:
    """Simple constant-velocity Kalman filter on (cx, cy, w, h)."""

    def __init__(self, process_var: float = 1e-2, meas_var: float = 1e-1):
        self.process_var = process_var
        self.meas_var = meas_var
        self.x: Optional[np.ndarray] = None  # [cx, cy, w, h, vcx, vcy, vw, vh]
        self.P: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.x = None
        self.P = None

    def _init(self, z: np.ndarray) -> None:
        self.x = np.array([z[0], z[1], z[2], z[3], 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(8) * 10.0

    def update(self, z: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        z_arr = np.asarray(z, dtype=np.float64)
        if self.x is None:
            self._init(z_arr)
            return float(z_arr[0]), float(z_arr[1]), float(z_arr[2]), float(z_arr[3])

        # Predict
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = 1.0
        Q = np.eye(8) * self.process_var
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Update
        H = np.zeros((4, 8))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        R = np.eye(4) * self.meas_var
        y = z_arr - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P
        return float(self.x[0]), float(self.x[1]), float(abs(self.x[2])), float(abs(self.x[3]))


class GeometrySmoother:
    def __init__(
        self,
        method: str = "ema",
        ema_alpha: float = 0.35,
        max_center_jump_px: float = 80.0,
        max_scale_jump: float = 0.35,
    ):
        self.method = method.lower()
        self.max_center_jump_px = max_center_jump_px
        self.max_scale_jump = max_scale_jump
        self.ema = EMAFilter(ema_alpha)
        self.kalman = KalmanBoxFilter()
        self._last_roi: Optional[ROI] = None

    def reset(self) -> None:
        self.ema.reset()
        self.kalman.reset()
        self._last_roi = None

    def filter(self, roi: ROI) -> Optional[ROI]:
        if self._last_roi is not None:
            lc = np.array(self._last_roi.center)
            cc = np.array(roi.center)
            jump = float(np.linalg.norm(cc - lc))
            if jump > self.max_center_jump_px:
                return None  # reject outlier
            la = max(self._last_roi.area, 1.0)
            scale_jump = abs(np.sqrt(roi.area / la) - 1.0)
            if scale_jump > self.max_scale_jump:
                return None

        if self.method == "kalman":
            x, y, w, h = roi.bbox
            cx, cy = x + w / 2, y + h / 2
            scx, scy, sw, sh = self.kalman.update((cx, cy, w, h))
            # Keep polygon shape; translate/scale about center
            pts = roi.points.copy()
            old_c = np.array(roi.center)
            new_c = np.array([scx, scy])
            sx = sw / max(w, 1e-6)
            sy = sh / max(h, 1e-6)
            s = 0.5 * (sx + sy)
            pts = new_c + (pts - old_c) * s
            out = ROI(pts.astype(np.float32), roi.roi_type, roi.frame_index)
        else:
            pts = self.ema.update(roi.points)
            out = ROI(pts, roi.roi_type, roi.frame_index)

        self._last_roi = out.copy()
        return out
