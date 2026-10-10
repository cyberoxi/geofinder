"""Synthetic camera paths over a satellite mosaic with exact per-frame labels.

Each frame is an affine view of the mosaic (centre, height in mosaic px,
rotation) — or, with pitch / roll, a tilted pinhole view (homography, see oblique.py).
The same transform maps every scene object into the frame, so the target and landmark
labels are exact — no manual annotation needed.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop_app.satellite.video_gen import _ease_in_out, _overlay_attribution, _write_video
from shared.landmarks import SceneLayout, apply_affine, clip_polygon_to_frame, json_default, polygon_area_abs
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
    pitch: float = 0.0  # degrees of forward tilt from straight down (oblique view)
    roll: float = 0.0  # degrees of bank


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


def make_night_mosaic(mosaic: np.ndarray, meters_per_px: float, rng: random.Random, mode: str = "night") -> np.ndarray:
    """
    Simulated night view of a daytime mosaic (no free high-resolution night satellite
    imagery exists — VIIRS/Black Marble is ~500 m/px).

    ``night``: moonlit dark scene + streetlights along detected roads + lit windows /
    floodlit fields.  ``ir``: thermal-like white-hot rendering (asphalt warm,
    vegetation / water cool).  Lights live in mosaic coordinates, so they stay fixed
    on the ground in every frame of a video.
    """
    from desktop_app.satellite.objects import _SceneStats

    H, W = mosaic.shape[:2]
    ds = min(1.0, 1400.0 / max(W, H))
    small = cv2.resize(mosaic, (max(1, int(W * ds)), max(1, int(H * ds))), interpolation=cv2.INTER_AREA)
    mpp = max(0.05, (meters_per_px or 0.6) / ds)
    st = _SceneStats(small, mpp)
    road = (st.dark & (st.texture < max(14.0, st.median_tex))).astype(np.uint8)
    L = st.L / 255.0
    green = (st.lab[..., 1] < st.median_lab[1] - 5).astype(np.float32)
    nprng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    if mode == "ir":
        # White-hot thermal look: dark asphalt retains heat, vegetation / water are cool.
        temp = 0.55 * (1.0 - L) + 0.35 * (1.0 - green) * L + 0.25 * road
        temp = cv2.GaussianBlur(temp.astype(np.float32), (0, 0), max(0.8, 1.5 / mpp))  # thermal sensors are soft
        lo, hi = np.percentile(temp, 1), np.percentile(temp, 99.5)
        temp = np.clip((temp - lo) / max(1e-6, hi - lo), 0, 1)
        # Thermal AGC: spread the histogram (avoid a washed-out image)
        eq = cv2.equalizeHist((temp * 255).astype(np.uint8)).astype(np.float32) / 255.0
        temp = 0.5 * temp + 0.5 * eq
        if rng.random() < 0.5:
            temp = 1.0 - temp  # black-hot polarity
        g = cv2.resize((temp * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # --- night (visible camera)
    base = mosaic.astype(np.float32) / 255.0
    dark = np.power(base, 2.0) * rng.uniform(0.08, 0.2)
    dark *= np.array([1.2, 0.95, 0.75], np.float32)  # moonlight tint (BGR)

    light = np.zeros(small.shape[:2], np.float32)
    ys, xs = np.nonzero(road)
    if len(xs):
        spacing_px = rng.uniform(25.0, 45.0) / mpp
        n_lamps = int(min(len(xs), road.sum() / max(1.0, spacing_px * spacing_px) * 4))
        idx = nprng.choice(len(xs), size=max(1, n_lamps), replace=False)
        light[ys[idx], xs[idx]] = nprng.uniform(0.6, 1.0, len(idx))
    # Sparse lit windows / courtyards off the road network
    off = np.nonzero((road == 0) & (st.texture > st.median_tex * 0.6))
    if len(off[0]):
        k = int(len(off[0]) * rng.uniform(0.0015, 0.004))
        idx = nprng.choice(len(off[0]), size=max(1, k), replace=False)
        light[off[0][idx], off[1][idx]] = nprng.uniform(0.2, 0.6, len(idx))
    core = cv2.GaussianBlur(light, (0, 0), max(0.6, 3.0 / mpp)) * 20.0
    halo = cv2.GaussianBlur(light, (0, 0), max(1.5, 12.0 / mpp)) * 60.0
    glow = np.clip(core + halo, 0, 1.5)
    # Floodlit sports fields / large plazas (some of them)
    flat = (st.texture < 0.4 * st.median_tex).astype(np.uint8)
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(flat)
    min_area = (40.0 / mpp) ** 2
    flood = np.zeros_like(glow)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area and rng.random() < 0.3:
            flood[lab_img == i] = rng.uniform(0.2, 0.4)
    # Soft falloff at the edges of floodlit areas
    glow = np.maximum(glow, cv2.GaussianBlur(flood, (0, 0), max(1.0, 6.0 / mpp)))
    glow = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
    lamp = np.array([0.35, 0.7, 1.0], np.float32) if rng.random() < 0.6 else np.array([1.0, 0.95, 0.9], np.float32)
    lit = dark + glow[..., None] * lamp * (0.35 + 0.65 * base)
    return (np.clip(lit, 0, 1) * 255).astype(np.uint8)


class MosaicCamera:
    """Renders affine views with a mip pyramid (no aliasing when zoomed out)."""

    def __init__(self, mosaic: np.ndarray, out_size: Tuple[int, int], fov_v: float = 45.0):
        self.out_w, self.out_h = out_size
        self.fov_v = float(fov_v)  # vertical field of view — only used by tilted poses
        self.pyr = [mosaic]
        while min(self.pyr[-1].shape[:2]) > 2 * max(self.out_w, self.out_h) // 2 and len(self.pyr) < 8:
            self.pyr.append(cv2.pyrDown(self.pyr[-1]))
        self.H, self.W = mosaic.shape[:2]

    @property
    def aspect(self) -> float:
        return self.out_w / self.out_h

    def matrix(self, pose: Pose) -> np.ndarray:
        """2x3 affine (straight down) or 3x3 homography (tilted pose): mosaic px → frame px."""
        if pose.pitch or pose.roll:
            from desktop_app.satellite.oblique import ground_homography

            return ground_homography(pose.cx, pose.cy, pose.view_h, pose.angle, pose.pitch, pose.roll,
                                     self.out_w, self.out_h, self.fov_v)
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
        fitted = replace(pose, view_h=vh)
        if fitted.pitch or fitted.roll:
            from desktop_app.satellite.oblique import fit_tilt

            fitted = fit_tilt(self, fitted)
        return fitted

    def render(self, pose: Pose, pre: Optional[np.ndarray] = None, pre_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Frame + the mosaic→frame transform used for labels.

        ``pre`` (3x3, scale ``pre_scale``) is applied to the image only — the lens simulation
        renders a slightly wider pinhole view and then distorts it back to the frame.
        """
        M = self.matrix(pose)
        s = self.out_h / pose.view_h * pre_scale
        level = 0
        while level + 1 < len(self.pyr) and s * (2 ** (level + 1)) <= 1.0:
            level += 1
        if M.shape == (2, 3) and pre is None:
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
        M3 = M if M.shape == (3, 3) else np.vstack([M, [0.0, 0.0, 1.0]])
        Mr = (pre @ M3 if pre is not None else M3) @ np.diag([2.0**level, 2.0**level, 1.0])
        img = cv2.warpPerspective(
            self.pyr[level], Mr, (self.out_w, self.out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        return img, M


def frame_labels(
    layout: SceneLayout,
    M: np.ndarray,
    width: int,
    height: int,
    min_side_px: float = 10.0,
    min_visible: Dict[int, float] | None = None,
    keep: Optional[set] = None,
    point_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> List[Tuple[int, List[List[float]]]]:
    """Project every scene object; keep those sufficiently visible (and covered by the imagery).

    ``M`` is a 2x3 affine or a 3x3 homography (tilted camera). ``point_fn`` maps projected frame
    points once more (lens distortion); polygon edges are densified first so they can bend.
    """
    min_visible = min_visible or {}
    out = []
    for cid in range(len(layout.classes)):
        if keep is not None and cid not in keep:
            continue
        if M.shape == (3, 3):
            from desktop_app.satellite.oblique import project

            poly = project(layout.polygon(cid), M)
            if poly is None:
                continue  # partly behind the camera
        else:
            poly = apply_affine(layout.polygon(cid), M)
        if point_fn is not None:
            poly = point_fn(_densify(poly))
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


def _densify(poly: np.ndarray, per_edge: int = 6) -> np.ndarray:
    pts = np.asarray(poly, np.float32).reshape(-1, 2)
    nxt = np.roll(pts, -1, axis=0)
    t = (np.arange(per_edge, dtype=np.float32) / per_edge)[None, :, None]
    return (pts[:, None, :] + (nxt - pts)[:, None, :] * t).reshape(-1, 2)


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
    start_view_factor: Optional[float] = None,
    start_offset: float = 0.0,
    end_jitter: float = 0.0,
) -> List[Pose]:
    """
    Zoom between view heights expressed as multiples of ``target_size``.

    If ``start_view_factor`` is None, start at the widest in-mosaic view.
    Larger factor = more zoomed out.  start/end are swapped if needed so the
    path still covers the requested range (zoom-in or zoom-out).
    """
    tcx, tcy = target_center
    max_h = cam.max_view_h(tcx, tcy, angle)
    end_h = max(16.0, target_size * float(end_view_factor))
    if start_view_factor is None:
        start_h = max_h
    else:
        start_h = max(16.0, target_size * float(start_view_factor))
    start_h = min(start_h, max_h)
    end_h = min(end_h, max_h)
    # Keep a tiny gap so log-space interpolation is stable
    if abs(math.log(max(start_h, 1e-3)) - math.log(max(end_h, 1e-3))) < 1e-6:
        end_h = max(16.0, start_h * 0.85)
    end_c = (
        tcx + rng.uniform(-1, 1) * end_jitter * end_h,
        tcy + rng.uniform(-1, 1) * end_jitter * end_h,
    )
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
    extra_meta: Optional[Dict] = None,
    keep_classes: Optional[set] = None,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
    progress_label: str = "Rendering",
    sensor: Optional[Any] = None,
) -> Tuple[Path, Path]:
    """Write the video plus ``<video>.labels.json`` with exact per-frame objects.

    ``sensor``: a ``sensor_sim.SensorSim`` (real camera + lens look; labels follow the lens distortion).
    """
    np_rng = np.random.default_rng(seed)
    labels: List[List[Tuple[int, List[List[float]]]]] = []
    scales: List[float] = []
    point_fn = sensor.lens.distort_points if sensor is not None and sensor.lens.active else None

    def frames():
        for i, pose in enumerate(poses):
            if sensor is not None:
                img, M = cam.render(pose, pre=sensor.lens.pre, pre_scale=sensor.lens.pre_scale)
            else:
                img, M = cam.render(pose)
            if photometric is not None:
                img = photometric.apply(img, np_rng)
            if sensor is not None:
                img = sensor.apply(img, _frame_motion(cam, poses, i), np_rng)
            if attribution and (i == 0 or i == len(poses) - 1):
                img = _overlay_attribution(img)
            labels.append(frame_labels(layout, M, cam.out_w, cam.out_h, keep=keep_classes, point_fn=point_fn))
            scales.append(float(cam.out_h / pose.view_h))
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
                **(extra_meta or {}),
                "width": cam.out_w,
                "height": cam.out_h,
                "fps": fps,
                "classes": list(layout.classes),
                "scale": [round(s, 6) for s in scales],
                "frames": labels,
            },
            default=json_default,
        ),
        encoding="utf-8",
    )
    return video_path, labels_path


def _frame_motion(cam: MosaicCamera, poses: Sequence[Pose], i: int) -> Tuple[float, float]:
    """Image-centre motion (frame px per frame) — drives the sensor's motion blur."""
    if len(poses) < 2:
        return 0.0, 0.0
    a, b = poses[max(0, i - 1)], poses[min(len(poses) - 1, max(1, i))]
    M = cam.matrix(a)
    p = np.array([b.cx, b.cy, 1.0])
    q = (M @ p) if M.shape == (2, 3) else (M @ p)[:2] / max(1e-9, (M @ p)[2])
    return float(q[0] - cam.out_w / 2.0), float(q[1] - cam.out_h / 2.0)


def labels_path_for(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.with_name(p.stem + ".labels.json")


def load_labels(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan_variants(n_videos: int, seed: int = 0, night_fraction: float = 0.0, ir_fraction: float = 0.0) -> List[Dict]:
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
    plans = plans[:n_videos]
    for p in plans:
        p["light"] = "day"
    # Lighting: video 0 stays the canonical daylight zoom; spread night / IR over the rest
    others = list(range(1, len(plans)))
    rng.shuffle(others)
    n_night = int(round(night_fraction * len(plans)))
    n_ir = int(round(ir_fraction * len(plans)))
    if night_fraction > 0 and n_night == 0 and others:
        n_night = 1
    if ir_fraction > 0 and n_ir == 0 and len(others) > n_night:
        n_ir = 1
    for j in others[:n_night]:
        plans[j]["light"] = "night"
    for j in others[n_night : n_night + n_ir]:
        plans[j]["light"] = "ir"
    return plans


# ---------------------------------------------------------------- user-designed camera paths


@dataclass
class CameraKey:
    """Keyframe of a designed camera path (mosaic px / degrees / seconds)."""

    cx: float
    cy: float
    view_h: float  # mosaic px covered by the frame height (smaller = more zoomed in)
    angle: float = 0.0
    duration: float = 2.0  # seconds to travel from the previous keyframe (ignored for the first)

    def to_dict(self) -> Dict:
        return {k: round(float(getattr(self, k)), 3) for k in ("cx", "cy", "view_h", "angle", "duration")}

    @classmethod
    def from_dict(cls, d: Dict) -> "CameraKey":
        return cls(**{k: float(d[k]) for k in ("cx", "cy", "view_h", "angle", "duration") if k in d})


def _catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def path_from_keyframes(keys: Sequence[CameraKey], fps: float, smooth: bool = True, ease: bool = True) -> List[Pose]:
    """Poses for every frame: smooth (Catmull-Rom) position, log-space zoom, unwrapped rotation."""
    if not keys:
        return []
    if len(keys) == 1:
        k = keys[0]
        return [Pose(k.cx, k.cy, k.view_h, k.angle)] * max(1, int(round(max(k.duration, 1.0) * fps)))
    # unwrap angles so a turn from 170° to -170° goes the short way
    angles = [keys[0].angle]
    for k in keys[1:]:
        d = (k.angle - angles[-1] + 180.0) % 360.0 - 180.0
        angles.append(angles[-1] + d)
    pts = np.array([[k.cx, k.cy] for k in keys], np.float64)
    logs = np.log([max(4.0, k.view_h) for k in keys])
    poses: List[Pose] = []
    for i in range(1, len(keys)):
        n = max(1, int(round(max(0.05, keys[i].duration) * fps)))
        p0 = pts[max(0, i - 2)]
        p1, p2 = pts[i - 1], pts[i]
        p3 = pts[min(len(keys) - 1, i + 1)]
        for j in range(n):
            u = j / n
            e = _ease_in_out(u) if ease else u
            if smooth:
                c = _catmull(p0, p1, p2, p3, e)
            else:
                c = p1 + (p2 - p1) * e
            vh = float(np.exp(logs[i - 1] + (logs[i] - logs[i - 1]) * e))
            ang = angles[i - 1] + (angles[i] - angles[i - 1]) * e
            poses.append(Pose(float(c[0]), float(c[1]), vh, float(ang)))
    last = keys[-1]
    poses.append(Pose(last.cx, last.cy, last.view_h, angles[-1]))
    return poses


def preset_keys(
    name: str,
    target_center: Tuple[float, float],
    target_size: float,
    max_view: float,
    angle: float = 0.0,
) -> List[CameraKey]:
    """Ready-made editable paths: zoom_in, zoom_out, orbit, flyover, spiral."""
    tx, ty = target_center
    far = max_view * 0.95
    near = max(16.0, target_size * 1.6)
    mid = math.sqrt(far * near)
    if name == "zoom_in":
        return [CameraKey(tx, ty, far, angle, 0), CameraKey(tx, ty, mid, angle, 3), CameraKey(tx, ty, near, angle, 3)]
    if name == "zoom_out":
        return [CameraKey(tx, ty, near, angle, 0), CameraKey(tx, ty, mid, angle, 3), CameraKey(tx, ty, far, angle, 3)]
    if name == "orbit":
        # frame must fit inside the map at any heading: diagonal ≈ 2.05 × view_h for 16:9
        vh = min(max(near * 2.0, target_size * 4.0), max_view * 0.4)
        r = min(target_size * 2.0, max(0.0, (max_view * 0.95 - vh * 2.05) / 2.0))
        return [
            CameraKey(tx + r * math.cos(a), ty + r * math.sin(a), vh, angle + math.degrees(a) + 90, 0 if k == 0 else 1.5)
            for k, a in enumerate(np.linspace(0, 2 * math.pi, 9))
        ]
    if name == "flyover":
        vh = min(max(near * 2.5, target_size * 3.5), max_view * 0.4)
        d = min(target_size * 8, max(0.0, (max_view * 0.95 - vh * 2.05) / 2.0))
        a = math.radians(angle)
        dx, dy = math.cos(a) * d, math.sin(a) * d
        return [CameraKey(tx - dx, ty - dy, vh, angle, 0), CameraKey(tx, ty, vh, angle, 3), CameraKey(tx + dx, ty + dy, vh, angle, 3)]
    if name == "spiral":
        keys = []
        n = 8
        for k in range(n + 1):
            f = k / n
            vh = math.exp(math.log(far) + (math.log(near) - math.log(far)) * f)
            r = vh * 0.3 * (1 - f)
            a = 2 * math.pi * f * 1.5
            keys.append(CameraKey(tx + r * math.cos(a), ty + r * math.sin(a), vh, angle + 45 * f * 4, 0 if k == 0 else 1.0))
        return keys
    raise ValueError(f"Unknown preset {name!r}")


def frame_corners(pose: Pose, aspect: float) -> np.ndarray:
    """Corners (4,2) of the camera frame footprint on the mosaic (for drawing)."""
    hw, hh = pose.view_h * aspect / 2.0, pose.view_h / 2.0
    a = math.radians(pose.angle)
    # frame x axis in mosaic coords (inverse of MosaicCamera.matrix rotation)
    # MosaicCamera.matrix uses R = [[c, s], [-s, c]]; frame axes in mosaic = columns of R^T
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -math.sin(a), math.cos(a)
    return np.array(
        [[pose.cx + sx * hw * ux + sy * hh * vx, pose.cy + sx * hw * uy + sy * hh * vy] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))],
        np.float32,
    )
