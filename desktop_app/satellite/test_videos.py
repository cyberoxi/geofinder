"""Generate varied test videos over a prepared scene (with / without target in view)."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from desktop_app.satellite.synth import (
    MosaicCamera,
    Photometric,
    Pose,
    make_night_mosaic,
    pan_path,
    render_labeled_video,
    zoom_path,
)
from shared.landmarks import SceneLayout

ProgressCb = Callable[[int, int, str], None]
FrameCb = Callable[[np.ndarray], None]


def _target_box(layout: SceneLayout) -> Tuple[Tuple[float, float], float]:
    poly = layout.polygon(0)
    (x0, y0), (x1, y1) = poly.min(axis=0).tolist(), poly.max(axis=0).tolist()
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0), float(max(x1 - x0, y1 - y0, 16.0))


def _exclusion_radius(t_size: float, view_h: float, aspect: float = 16 / 9) -> float:
    """Keep target outside the frame: half-diagonal of view + half target + margin."""
    half_diag = 0.5 * view_h * math.hypot(aspect, 1.0)
    return half_diag + 0.5 * t_size + 0.15 * view_h


def _sample_empty_center(
    cam: MosaicCamera,
    t_center: Tuple[float, float],
    t_size: float,
    view_h: float,
    rng: random.Random,
    *,
    attempts: int = 80,
) -> Optional[Tuple[float, float]]:
    """Random mosaic centre whose mid-altitude view never covers the target."""
    excl = _exclusion_radius(t_size, view_h, cam.aspect)
    margin = view_h * 0.55
    lo_x, hi_x = margin, cam.W - margin
    lo_y, hi_y = margin, cam.H - margin
    if hi_x <= lo_x or hi_y <= lo_y:
        return None
    tx, ty = t_center
    for _ in range(attempts):
        cx = rng.uniform(lo_x, hi_x)
        cy = rng.uniform(lo_y, hi_y)
        if math.hypot(cx - tx, cy - ty) >= excl:
            return (cx, cy)
    return None


def plan_test_variants(
    n_with: int,
    n_without: int,
    seed: int = 0,
    *,
    modes: Sequence[str] = ("zoom", "pan"),
    night_fraction: float = 0.0,
    ir_fraction: float = 0.0,
) -> List[Dict[str, Any]]:
    """Recipes for positive (target in path) and negative (empty region) test clips."""
    rng = random.Random(seed)
    mode_set = {m for m in modes if m in ("zoom", "pan")} or {"zoom", "pan"}
    plans: List[Dict[str, Any]] = []

    def _kind(i: int) -> str:
        if mode_set == {"zoom"}:
            return "zoom"
        if mode_set == {"pan"}:
            return "pan"
        return "pan" if i % 3 == 2 else "zoom"

    for i in range(max(0, n_with)):
        kind = _kind(i)
        if kind == "pan":
            plans.append(
                {
                    "has_target": True,
                    "kind": "pan",
                    "angle": rng.uniform(0, 360),
                    "view": rng.uniform(2.5, 6.0),
                    "length": rng.uniform(2.5, 4.5),
                    "photo": True,
                }
            )
        else:
            plans.append(
                {
                    "has_target": True,
                    "kind": "zoom",
                    "angle": rng.uniform(0, 360),
                    "drift": rng.uniform(-30, 30),
                    "end": rng.choice([0.9, 1.3, 1.8, 2.5]),
                    "start_off": rng.uniform(0.0, 0.25),
                    "jitter": rng.uniform(0.0, 0.3),
                    "photo": True,
                }
            )

    for i in range(max(0, n_without)):
        kind = _kind(i)
        if kind == "pan":
            plans.append(
                {
                    "has_target": False,
                    "kind": "pan",
                    "angle": rng.uniform(0, 360),
                    "view": rng.uniform(2.0, 5.0),
                    "length": rng.uniform(2.0, 3.5),
                    "photo": True,
                }
            )
        else:
            plans.append(
                {
                    "has_target": False,
                    "kind": "zoom",
                    "angle": rng.uniform(0, 360),
                    "drift": rng.uniform(-20, 20),
                    "end": rng.choice([1.2, 1.8, 2.5, 3.5]),
                    "start_off": rng.uniform(0.0, 0.15),
                    "jitter": rng.uniform(0.0, 0.2),
                    "photo": True,
                }
            )

    for p in plans:
        p["light"] = "day"
    idxs = list(range(len(plans)))
    rng.shuffle(idxs)
    n_night = int(round(night_fraction * len(plans))) if plans else 0
    n_ir = int(round(ir_fraction * len(plans))) if plans else 0
    if night_fraction > 0 and n_night == 0 and idxs:
        n_night = 1
    if ir_fraction > 0 and n_ir == 0 and len(idxs) > n_night:
        n_ir = 1
    for j in idxs[:n_night]:
        plans[j]["light"] = "night"
    for j in idxs[n_night : n_night + n_ir]:
        plans[j]["light"] = "ir"
    return plans


def _region_center_size(region: Sequence[float] | Dict[str, Any]) -> Tuple[Tuple[float, float], float]:
    """Accept bbox ``(x0,y0,x1,y1)`` or ``{"center": (cx,cy), "size": s}``."""
    if isinstance(region, dict):
        c = region["center"]
        return (float(c[0]), float(c[1])), float(region["size"])
    x0, y0, x1, y1 = (float(v) for v in region[:4])
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0), float(max(abs(x1 - x0), abs(y1 - y0), 16.0))


def render_test_videos(
    mosaic: np.ndarray,
    layout: SceneLayout,
    out_dir: str | Path,
    *,
    n_with_target: int = 4,
    n_without_target: int = 4,
    render_modes: Sequence[str] = ("zoom", "pan"),
    duration_s: float = 8.0,
    fps: float = 24.0,
    out_size: Tuple[int, int] = (1280, 720),
    night_fraction: float = 0.0,
    ir_fraction: float = 0.0,
    empty_region_mode: str = "random",
    empty_regions: Optional[Sequence[Any]] = None,
    zoom_start: float = 12.0,
    zoom_end: float = 1.6,
    zoom_speed: float = 1.0,
    pan_speed: float = 1.0,
    pan_view: float = 4.0,
    seed: int = 0,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
) -> Dict[str, Any]:
    """
    Render evaluation clips over the existing satellite scene.

    *with-target* paths fly over the selected target; *without-target* paths use
    either random hard-negative centres or user-drawn map regions.

    ``empty_region_mode``: ``"random"`` samples centres that keep the target out of
    frame; ``"manual"`` cycles through ``empty_regions`` (any user-drawn bboxes on
    the mosaic — not rejected if the target is also visible).

    ``zoom_start`` / ``zoom_end``: view height as multiples of the region size
    (larger = more zoomed out).  ``zoom_speed`` / ``pan_speed`` scale motion time
    (2× ≈ half the frames for the same duration setting).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_center, t_size = _target_box(layout)
    manual = [_region_center_size(r) for r in (empty_regions or [])]
    if empty_region_mode == "manual" and n_without_target > 0 and not manual:
        raise ValueError("Draw at least one region on the map, or switch to Random")
    plans = plan_test_variants(
        n_with_target,
        n_without_target,
        seed=seed,
        modes=render_modes,
        night_fraction=night_fraction,
        ir_fraction=ir_fraction,
    )
    if not plans:
        raise ValueError("Set at least one video with or without target")

    z_start = max(1.0, float(zoom_start))
    z_end = max(0.5, float(zoom_end))
    z_speed = max(0.1, float(zoom_speed))
    p_speed = max(0.1, float(pan_speed))
    p_view = max(1.0, float(pan_view))
    n_zoom = max(2, int(round(duration_s / z_speed * fps)))
    n_pan = max(2, int(round(duration_s / p_speed * fps)))
    cams: Dict[str, MosaicCamera] = {}
    videos: List[Dict[str, Any]] = []
    skipped = 0
    empty_i = 0

    for i, plan in enumerate(plans):
        rng = random.Random(seed * 1000 + i + 17)
        light = plan.get("light", "day")
        if light not in cams:
            img = mosaic
            if light in ("night", "ir"):
                img = make_night_mosaic(mosaic, layout.meters_per_px, random.Random(seed + i), mode=light)
            cams[light] = MosaicCamera(img, out_size)
        cam = cams[light]

        manual_pick = False
        if plan["has_target"]:
            center, size = t_center, t_size
        elif empty_region_mode == "manual" and manual:
            center, size = manual[empty_i % len(manual)]
            empty_i += 1
            manual_pick = True
        else:
            probe_vh = max(t_size * 3.5, 80.0)
            empty = _sample_empty_center(cam, t_center, t_size, probe_vh, rng)
            if empty is None:
                skipped += 1
                continue
            center = empty
            size = max(t_size * 0.8, probe_vh / 4.0)

        if plan["kind"] == "zoom":
            poses = zoom_path(
                cam,
                center,
                size,
                n_zoom,
                rng,
                angle=plan["angle"],
                angle_drift=plan.get("drift", 0.0),
                start_view_factor=z_start,
                end_view_factor=z_end,
                start_offset=plan.get("start_off", 0.0),
                end_jitter=plan.get("jitter", 0.0),
            )
        else:
            poses = pan_path(
                cam,
                center,
                size,
                n_pan,
                rng,
                angle=plan["angle"],
                view_factor=p_view,
                length_factor=plan.get("length", 3.0),
            )

        # Random hard-negatives must not show the target. Manual map picks keep the
        # user-chosen place even if the (wide) view also catches the green target.
        if plan["has_target"]:
            has_target = True
        elif manual_pick:
            has_target = _target_visible_in_poses(layout, cam, poses)
        elif _target_visible_in_poses(layout, cam, poses):
            skipped += 1
            continue
        else:
            has_target = False
        tag_side = "with" if has_target else "without"

        if light == "day":
            photo = Photometric.random(rng) if plan.get("photo") else None
        else:
            photo = Photometric(noise=rng.uniform(4, 12), blur=rng.choice([0, 3, 3, 5]))

        light_tag = "" if light == "day" else f"_{light}"
        name = f"test_{tag_side}_{plan['kind']}_{i:02d}{light_tag}.mp4"

        def _cb(done: int, total: int, msg: str, i: int = i) -> None:
            if progress_cb and total > 0:
                pct = int(100 * (i + done / total) / len(plans))
                progress_cb(min(99, pct), 100, f"Test video {i + 1}/{len(plans)}: {msg}")

        vpath, lpath = render_labeled_video(
            cam,
            poses,
            layout,
            out / name,
            fps=fps,
            photometric=photo,
            seed=seed * 1000 + i,
            kind=f"test_{tag_side}_{plan['kind']}",
            progress_cb=_cb,
            frame_cb=frame_cb,
            progress_label=f"Rendering test {tag_side} {plan['kind']} ({light})",
            extra_meta={
                "light": light,
                "has_target": has_target,
                "role": "positive" if has_target else "negative",
                "region": "manual" if manual_pick else ("target" if has_target else "random"),
            },
        )
        videos.append(
            {
                "path": vpath,
                "labels": lpath,
                "kind": plan["kind"],
                "has_target": has_target,
                "light": light,
            }
        )

    if progress_cb:
        progress_cb(100, 100, "Done")
    if not videos:
        raise ValueError(
            "Could not place empty-region cameras (mosaic too small vs target). "
            "Try fewer without-target videos or a wider scene."
        )
    return {"videos": videos, "skipped": skipped, "out_dir": out}


def _target_visible_in_poses(layout: SceneLayout, cam: MosaicCamera, poses: Sequence[Pose]) -> bool:
    """True if the target polygon is meaningfully visible in any frame."""
    from desktop_app.satellite.synth import frame_labels

    for pose in poses[:: max(1, len(poses) // 8)]:
        M = cam.matrix(pose)
        if frame_labels(layout, M, cam.out_w, cam.out_h, keep={0}):
            return True
    return False
