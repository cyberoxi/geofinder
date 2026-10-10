"""Apply a :class:`CameraSim` to one rendered video: altitude zoom range, tilt profile, sensor look."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from desktop_app.satellite.camera_model import CameraSim
from desktop_app.satellite.oblique import tilt_profile
from desktop_app.satellite.sensor_sim import SensorSim
from shared.logging import get_logger

logger = get_logger("desktop.satellite.camera_sim")

MIN_VIEW_H = 48.0  # mosaic px — below this the imagery is upscaled too much to be useful


def plan_videos(sim: Optional[CameraSim], n_videos: int, seed: int) -> List[Dict[str, bool]]:
    """Which videos get the tilt / sensor simulation (video 0 stays the canonical straight-down view)."""
    plans = [{"oblique": False, "sensor": False} for _ in range(max(0, n_videos))]
    if sim is None or not sim.any_enabled:
        return plans
    rng = random.Random(seed * 7919 + 13)
    for i, p in enumerate(plans):
        p["oblique"] = bool(sim.oblique and i > 0 and rng.random() < sim.oblique_fraction)
        p["sensor"] = bool(sim.sensor and rng.random() < sim.sensor_fraction)
    if sim.oblique and n_videos > 1 and not any(p["oblique"] for p in plans):
        plans[-1]["oblique"] = True
    return plans


def altitude_remap(poses: Sequence[Any], sim: CameraSim, meters_per_px: float) -> List[Any]:
    """Map the path's zoom range (log view height) onto the camera's altitude range."""
    rng_vh = sim.view_h_range(meters_per_px)
    if not poses or rng_vh is None:
        return list(poses)
    lo, hi = max(MIN_VIEW_H, rng_vh[0]), max(MIN_VIEW_H, rng_vh[1])
    if rng_vh[0] < MIN_VIEW_H:
        logger.warning(
            "Altitude %.0f m with this lens needs finer imagery (%.2f m/px now) — clamped; "
            "use a higher download zoom", sim.alt_min_m, meters_per_px,
        )
    logs = [math.log(max(1.0, p.view_h)) for p in poses]
    a, b = min(logs), max(logs)
    la, lb = math.log(lo), math.log(max(hi, lo * 1.01))
    out = []
    for p, lv in zip(poses, logs):
        t = 0.5 if b - a < 1e-6 else (lv - a) / (b - a)
        out.append(replace(p, view_h=float(math.exp(la + (lb - la) * t))))
    return out


def adjust_poses(
    poses: Sequence[Any], cam: Any, sim: Optional[CameraSim], plan: Dict[str, bool],
    meters_per_px: float, rng: random.Random, *, use_altitude: bool = True,
) -> List[Any]:
    """Altitude range → tilt profile → fit to the mosaic (tilt is reduced where it would leave it)."""
    if sim is None or not sim.any_enabled:
        return list(poses)
    out = list(poses)
    if sim.altitude and use_altitude:
        out = altitude_remap(out, sim, meters_per_px)
    if plan.get("oblique"):
        cam.fov_v = sim.fov_v()
        out = tilt_profile(out, sim.max_pitch, sim.max_roll, rng)
    return [cam.fit(p) for p in out]


def make_sensor(sim: Optional[CameraSim], cam: Any, plan: Dict[str, bool], rng: random.Random, light: str) -> Optional[SensorSim]:
    if sim is None or not plan.get("sensor"):
        return None
    return SensorSim.random(sim, (cam.out_w, cam.out_h), rng, light=light)


def tag(plan: Dict[str, bool]) -> str:
    return ("_tilt" if plan.get("oblique") else "") + ("_cam" if plan.get("sensor") else "")
