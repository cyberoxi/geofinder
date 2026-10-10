"""Tilted (oblique) camera over a flat mosaic: pinhole homography + per-video pitch / roll profiles.

The mosaic is the ground plane. A pose keeps its meaning — the frame centre looks at (cx, cy) and
``view_h`` sets the scale at that point — while ``pitch`` tilts the camera forward (the top of the
frame sees farther ground) and ``roll`` banks it. With pitch = roll = 0 this equals the affine view.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any, List, Optional, Sequence

import numpy as np

# The top frame ray must still hit the ground well before the horizon
HORIZON_MARGIN_DEG = 12.0


def ground_homography(
    cx: float, cy: float, view_h: float, angle: float, pitch: float, roll: float,
    out_w: int, out_h: int, fov_v: float,
) -> np.ndarray:
    """3×3 homography: mosaic px (ground plane) → frame px."""
    f = (out_h / 2.0) / math.tan(math.radians(max(1.0, fov_v)) / 2.0)
    dist = f * view_h / out_h  # camera → look-at point: scale f/dist = out_h/view_h at the centre
    th, ph = math.radians(pitch), math.radians(roll)
    s, c = math.sin(th), math.cos(th)
    # Local ground axes (u, v) are the nadir frame axes; w points into the ground.
    C = np.array([0.0, dist * s, -dist * c])  # behind (towards frame bottom) and above
    z = np.array([0.0, -s, c])
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, c, s])
    xr = math.cos(ph) * x + math.sin(ph) * y
    yr = -math.sin(ph) * x + math.cos(ph) * y
    R = np.vstack([xr, yr, z])
    t = -R @ C
    K = np.array([[f, 0, out_w / 2.0], [0, f, out_h / 2.0], [0, 0, 1.0]])
    H_local = K @ np.column_stack([R[:, 0], R[:, 1], t])
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    Rot = np.array([[ca, sa], [-sa, ca]])  # same heading convention as MosaicCamera.matrix
    A = np.eye(3)
    A[:2, :2] = Rot
    A[:2, 2] = -Rot @ np.array([cx, cy])
    return H_local @ A


def project(points: np.ndarray, H: np.ndarray) -> Optional[np.ndarray]:
    """Project N×2 points with a 3×3 homography; None if any point lies behind the camera."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = hom[:, 2]
    if np.any(w <= 1e-6):
        return None
    return (hom[:, :2] / w[:, None]).astype(np.float32)


def frame_footprint(H: np.ndarray, out_w: int, out_h: int) -> Optional[np.ndarray]:
    """Mosaic px seen by the 4 frame corners (None if a corner ray misses the ground)."""
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    corners = np.array([[0, 0, 1], [out_w, 0, 1], [out_w, out_h, 1], [0, out_h, 1]], np.float64)
    g = corners @ Hi.T
    if np.any(g[:, 2] <= 1e-9):
        return None
    return g[:, :2] / g[:, 2:3]


def max_pitch_for_fov(fov_v: float) -> float:
    return max(0.0, 90.0 - fov_v / 2.0 - HORIZON_MARGIN_DEG)


def fit_tilt(cam: Any, pose: Any) -> Any:
    """Reduce pitch (then roll) until the whole tilted footprint lies inside the mosaic."""
    if not (pose.pitch or pose.roll):
        return pose
    pitch = min(abs(pose.pitch), max_pitch_for_fov(cam.fov_v)) * (1 if pose.pitch >= 0 else -1)
    roll = pose.roll
    for _ in range(14):
        H = ground_homography(pose.cx, pose.cy, pose.view_h, pose.angle, pitch, roll, cam.out_w, cam.out_h, cam.fov_v)
        fp = frame_footprint(H, cam.out_w, cam.out_h)
        if fp is not None and fp[:, 0].min() >= 0 and fp[:, 1].min() >= 0 and fp[:, 0].max() <= cam.W and fp[:, 1].max() <= cam.H:
            return replace(pose, pitch=pitch, roll=roll)
        pitch *= 0.8
        roll *= 0.8
        if abs(pitch) < 1.0 and abs(roll) < 0.5:
            break
    return replace(pose, pitch=0.0, roll=0.0)


def tilt_profile(poses: Sequence[Any], max_pitch: float, max_roll: float, rng: random.Random) -> List[Any]:
    """Smoothly varying pitch / roll over one video (a different flight attitude per video)."""
    n = len(poses)
    if n == 0:
        return []
    peak = max_pitch * rng.uniform(0.5, 1.0)
    low = peak * rng.uniform(0.0, 0.5)
    cyc, ph = rng.uniform(0.3, 1.0), rng.uniform(0, 2 * math.pi)
    r_amp, r_cyc, r_ph = max_roll * rng.uniform(0.3, 1.0), rng.uniform(1.0, 3.0), rng.uniform(0, 2 * math.pi)
    out = []
    for i, p in enumerate(poses):
        t = i / max(1, n - 1)
        pitch = low + (peak - low) * (0.5 + 0.5 * math.sin(2 * math.pi * cyc * t + ph))
        roll = r_amp * math.sin(2 * math.pi * r_cyc * t + r_ph)
        out.append(replace(p, pitch=float(pitch), roll=float(roll)))
    return out
