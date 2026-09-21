"""Synthetic camera paths over a satellite mosaic with exact per-frame labels.

Each frame is an affine view of the mosaic (centre, height in mosaic px,
rotation).  The same affine maps every scene object into the frame, so the
target and landmark labels are exact — no manual annotation needed.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop_app.satellite.video_gen import _ease_in_out, _overlay_attribution, _write_video
from shared.landmarks import SceneLayout, apply_affine, clip_polygon_to_frame, polygon_area_abs
from shared.logging import get_logger

logger = get_logger("desktop.satellite.synth")

ProgressCb = Callable[[int, int, str], None]
FrameCb = Callable[[np.ndarray], None]


@dataclass
class Pose:
    cx: float
    cy: float
    view_h: float  # mosaic px covered by the frame height
    angle: float  # degrees (camera heading)


@dataclass
class Photometric:
    alpha: float = 1.0  # contrast
    beta: float = 0.0  # brightness
    gamma: float = 1.0
    sat: float = 1.0
    hue: float = 0.0
    haze: float = 0.0  # 0..1 blend with light grey
    blur: int = 0  # gaussian kernel (odd) or 0
    noise: float = 0.0  # stddev

    @classmethod
    def random(cls, rng: random.Random) -> "Photometric":
        return cls(
            alpha=rng.uniform(0.75, 1.25),
            beta=rng.uniform(-25, 25),
            gamma=rng.uniform(0.8, 1.25),
            sat=rng.uniform(0.6, 1.3),
            hue=rng.uniform(-6, 6),
            haze=rng.choice([0.0, 0.0, rng.uniform(0.08, 0.3)]),
            blur=rng.choice([0, 0, 3, 5]),
            noise=rng.choice([0.0, rng.uniform(2, 7)]),
        )

    def apply(self, img: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        out = img
        if self.sat != 1.0 or self.hue != 0.0:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + self.hue) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * self.sat, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        f = out.astype(np.float32)
        if self.gamma != 1.0:
            f = 255.0 * np.power(f / 255.0, self.gamma)
        f = f * self.alpha + self.beta
        if self.haze > 0:
            f = f * (1 - self.haze) + 210.0 * self.haze
        if self.noise > 0 and rng is not None:
            f = f + rng.normal(0, self.noise, f.shape)
        out = np.clip(f, 0, 255).astype(np.uint8)
        if self.blur:
            out = cv2.GaussianBlur(out, (self.blur, self.blur), 0)
        return out


class MosaicCamera:
    """Renders affine views with a mip pyramid (no aliasing when zoomed out)."""

    def __init__(self, mosaic: np.ndarray, out_size: Tuple[int, int]):
        self.out_w, self.out_h = out_size
        self.pyr = [mosaic]
        while min(self.pyr[-1].shape[:2]) > 2 * max(self.out_w, self.out_h) // 2 and len(self.pyr) < 8:
            self.pyr.append(cv2.pyrDown(self.pyr[-1]))
        self.H, self.W = mosaic.shape[:2]

    @property
    def aspect(self) -> float:
        return self.out_w / self.out_h

    def matrix(self, pose: Pose) -> np.ndarray:
        """2x3 affine: mosaic px → frame px."""
        s = self.out_h / pose.view_h
        a = math.radians(pose.angle)
        c, si = math.cos(a) * s, math.sin(a) * s
        R = np.array([[c, si], [-si, c]])
        t = np.array([self.out_w / 2.0, self.out_h / 2.0]) - R @ np.array([pose.cx, pose.cy])
        return np.hstack([R, t.reshape(2, 1)])

    def max_view_h(self, cx: float, cy: float, angle: float) -> float:
        """Largest view height at this centre/angle that keeps the frame inside the mosaic."""
        a = math.radians(angle)
        ca, sa = abs(math.cos(a)), abs(math.sin(a))
        ex = (ca * self.aspect + sa) / 2.0  # half extent per unit view_h
        ey = (sa * self.aspect + ca) / 2.0
        lim = min(cx / ex, (self.W - cx) / ex, cy / ey, (self.H - cy) / ey)
        return max(8.0, lim)

    def fit(self, pose: Pose) -> Pose:
        vh = min(pose.view_h, self.max_view_h(pose.cx, pose.cy, pose.angle))
        return Pose(pose.cx, pose.cy, vh, pose.angle)

    def render(self, pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
        M = self.matrix(pose)
        s = self.out_h / pose.view_h
        level = 0
        while level + 1 < len(self.pyr) and s * (2 ** (level + 1)) <= 1.0:
            level += 1
        Ml = M.copy()
        Ml[:, :2] *= 2**level
        img = cv2.warpAffine(
            self.pyr[level],
            Ml,
            (self.out_w, self.out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return img, M


def frame_labels(
    layout: SceneLayout,
    M: np.ndarray,
    width: int,
    height: int,
    min_side_px: float = 10.0,
    min_visible: Dict[int, float] | None = None,
) -> List[Tuple[int, List[List[float]]]]:
    """Project every scene object; keep those sufficiently visible."""
    min_visible = min_visible or {}
    out = []
    for cid in range(len(layout.classes)):
        poly = apply_affine(layout.polygon(cid), M)
        full = polygon_area_abs(poly)
        if full <= 0 or math.sqrt(full) < min_side_px:
            continue
        clipped = clip_polygon_to_frame(poly, width, height)
        if len(clipped) < 3:
            continue
        vis = polygon_area_abs(clipped) / full
        if vis < min_visible.get(cid, 0.5 if cid else 0.3):
            continue
        out.append((cid, [[round(float(x), 2), round(float(y), 2)] for x, y in clipped]))
    return out


# ---------------------------------------------------------------- camera paths


def zoom_path(
    cam: MosaicCamera,
    target_center: Tuple[float, float],
    target_size: float,
    n: int,
    rng: random.Random,
    *,
    angle: float = 0.0,
    angle_drift: float = 0.0,
    end_view_factor: float = 1.6,
    start_offset: float = 0.0,
    end_jitter: float = 0.0,
) -> List[Pose]:
    """Zoom from the widest possible view down to ``end_view_factor`` × target size."""
    tcx, tcy = target_center
    end_h = max(16.0, target_size * end_view_factor)
    end_c = (
        tcx + rng.uniform(-1, 1) * end_jitter * end_h,
        tcy + rng.uniform(-1, 1) * end_jitter * end_h,
    )
    start_h = cam.max_view_h(tcx, tcy, angle)
    start_c = (
        tcx + rng.uniform(-1, 1) * start_offset * start_h,
        tcy + rng.uniform(-1, 1) * start_offset * start_h,
    )
    poses = []
    for i in range(n):
        t = _ease_in_out(i / max(1, n - 1))
        # log-space zoom: every zoom level gets equal screen time
        vh = math.exp(math.log(start_h) + (math.log(end_h) - math.log(start_h)) * t)
        cx = start_c[0] + (end_c[0] - start_c[0]) * t
        cy = start_c[1] + (end_c[1] - start_c[1]) * t
        ang = angle + angle_drift * t
        poses.append(cam.fit(Pose(cx, cy, vh, ang)))
    return poses


def pan_path(
    cam: MosaicCamera,
    target_center: Tuple[float, float],
    target_size: float,
    n: int,
    rng: random.Random,
    *,
    angle: float = 0.0,
    view_factor: float = 4.0,
    length_factor: float = 3.0,
) -> List[Pose]:
    """Straight pass at fixed altitude crossing near the target (target enters and leaves view)."""
    tcx, tcy = target_center
    vh = target_size * view_factor
    heading = rng.uniform(0, 2 * math.pi)
    miss = rng.uniform(-0.4, 0.4) * vh  # lateral miss distance
    half_len = vh * length_factor / 2.0
    dx, dy = math.cos(heading), math.sin(heading)
    px, py = -dy, dx
    a = (tcx - dx * half_len + px * miss, tcy - dy * half_len + py * miss)
    b = (tcx + dx * half_len + px * miss, tcy + dy * half_len + py * miss)
    # Keep the whole view inside the mosaic by clamping the centre (not by zooming in)
    rad = math.radians(angle)
    ca, sa = abs(math.cos(rad)), abs(math.sin(rad))
    ex = (ca * cam.aspect + sa) / 2.0
    ey = (sa * cam.aspect + ca) / 2.0
    vh = min(vh, 0.95 * cam.W / (2 * ex), 0.95 * cam.H / (2 * ey))
    lo_x, hi_x = ex * vh, cam.W - ex * vh
    lo_y, hi_y = ey * vh, cam.H - ey * vh
    poses = []
    for i in range(n):
        t = i / max(1, n - 1)
        cx = min(max(a[0] + (b[0] - a[0]) * t, lo_x), hi_x)
        cy = min(max(a[1] + (b[1] - a[1]) * t, lo_y), hi_y)
        poses.append(cam.fit(Pose(cx, cy, vh, angle)))
    return poses


# ---------------------------------------------------------------- rendering


def render_labeled_video(
    cam: MosaicCamera,
    poses: Sequence[Pose],
    layout: SceneLayout,
    output: str | Path,
    *,
    fps: float,
    photometric: Optional[Photometric] = None,
    seed: int = 0,
    kind: str = "zoom",
    attribution: bool = True,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
    progress_label: str = "Rendering",
) -> Tuple[Path, Path]:
    """Write the video plus ``<video>.labels.json`` with exact per-frame objects."""
    np_rng = np.random.default_rng(seed)
    labels: List[List[Tuple[int, List[List[float]]]]] = []
    scales: List[float] = []

    def frames():
        for i, pose in enumerate(poses):
            img, M = cam.render(pose)
            if photometric is not None:
                img = photometric.apply(img, np_rng)
            if attribution and (i == 0 or i == len(poses) - 1):
                img = _overlay_attribution(img)
            labels.append(frame_labels(layout, M, cam.out_w, cam.out_h))
            scales.append(cam.out_h / pose.view_h)
            yield img

    video_path = _write_video(
        frames(),
        Path(output),
        fps,
        (cam.out_w, cam.out_h),
        total_frames=len(poses),
        progress_cb=progress_cb,
        frame_cb=frame_cb,
        progress_label=progress_label,
    )
    labels_path = labels_path_for(video_path)
    labels_path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": kind,
                "width": cam.out_w,
                "height": cam.out_h,
                "fps": fps,
                "classes": list(layout.classes),
                "scale": [round(s, 6) for s in scales],
                "frames": labels,
            }
        ),
        encoding="utf-8",
    )
    return video_path, labels_path


def labels_path_for(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.with_name(p.stem + ".labels.json")


def load_labels(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan_variants(n_videos: int, seed: int = 0) -> List[Dict]:
    """
    Variant recipe list.  Video 0 is the canonical north-up zoom; the rest mix
    rotated / off-centre zooms, deep zooms and low-altitude passes (where the
    target leaves the view and only landmarks remain) with photometric changes.
    """
    rng = random.Random(seed)
    plans: List[Dict] = [
        {"kind": "zoom", "angle": 0.0, "drift": 0.0, "end": 1.6, "start_off": 0.0, "jitter": 0.0, "photo": False}
    ]
    i = 1
    while len(plans) < n_videos:
        slot = i % 3
        if slot == 2:
            plans.append(
                {
                    "kind": "pan",
                    "angle": rng.uniform(0, 360),
                    "view": rng.uniform(2.5, 6.0),
                    "length": rng.uniform(2.5, 4.0),
                    "photo": True,
                }
            )
        else:
            plans.append(
                {
                    "kind": "zoom",
                    "angle": rng.uniform(0, 360),
                    "drift": rng.uniform(-25, 25),
                    "end": rng.choice([0.9, 1.3, 1.8, 2.5]),
                    "start_off": rng.uniform(0.0, 0.2),
                    "jitter": rng.uniform(0.0, 0.25),
                    "photo": True,
                }
            )
        i += 1
    return plans[:n_videos]
