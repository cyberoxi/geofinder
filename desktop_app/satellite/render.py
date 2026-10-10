"""Render labeled satellite training videos from a prepared scene."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop_app.satellite.prepare import _safe
from desktop_app.satellite.synth import (
    MosaicCamera,
    Photometric,
    make_night_mosaic,
    pan_path,
    plan_variants,
    render_labeled_video,
    zoom_path,
)
from desktop_app.satellite import camera_sim as csim
from desktop_app.satellite.camera_model import CameraSim
from shared.landmarks import SceneLayout

ProgressCb = Callable[[int, int, str], None]
FrameCb = Callable[[np.ndarray], None]


def render_scene_videos(
    mosaic: "np.ndarray | Sequence[Tuple[str, np.ndarray]]",
    layout: SceneLayout,
    out_dir: str | Path,
    *,
    n_videos: int = 6,
    night_fraction: float = 0.0,
    ir_fraction: float = 0.0,
    render_modes: Sequence[str] = ("zoom", "pan"),
    duration_s: float = 8.0,
    fps: float = 24.0,
    out_size: Tuple[int, int] = (1280, 720),
    seed: int = 0,
    keep_classes: Optional[set] = None,
    camera_sim: Optional[CameraSim] = None,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
) -> Dict[str, Any]:
    """
    Stage 2: render varied videos with exact per-frame labels for every scene object.

    ``mosaic`` may be one image or a list of (source_name, image) that are pixel-aligned
    (same area / zoom from different sources or dates); videos cycle through them.
    ``night_fraction`` / ``ir_fraction`` of the videos use simulated night / thermal views.
    ``keep_classes``: only these class ids are labeled (None = all).
    ``camera_sim``: optional real-camera simulation (tilt / sensor + lens / altitude range).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = [("primary", mosaic, None)] if isinstance(mosaic, np.ndarray) else list(mosaic)
    base_shape = raw[0][1].shape[:2]
    sources: List[Tuple[str, np.ndarray]] = []
    keeps: List[Optional[set]] = []
    for item in raw:
        n, m = str(item[0]), item[1]
        mask = item[2] if len(item) > 2 else None
        if m.shape[:2] != base_shape:
            m = cv2.resize(m, base_shape[::-1])
        sources.append((n, m))
        if mask is not None:
            from desktop_app.satellite.align import covered_classes

            keeps.append(covered_classes(layout, mask))
        else:
            keeps.append(None)
    if keep_classes is not None:
        keeps = [set(keep_classes) if k is None else (k & set(keep_classes)) for k in keeps]
    tpoly = layout.polygon(0)
    (tx0, ty0), (tx1, ty1) = tpoly.min(axis=0).tolist(), tpoly.max(axis=0).tolist()
    t_center = ((tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0)
    t_size = float(max(tx1 - tx0, ty1 - ty0))
    cams: Dict[Tuple[int, str], MosaicCamera] = {}

    def camera(src_idx: int, light: str, vid_seed: int) -> MosaicCamera:
        key = (src_idx, light)
        if key not in cams:
            img = sources[src_idx][1]
            if light in ("night", "ir"):
                img = make_night_mosaic(img, layout.meters_per_px, random.Random(vid_seed), mode=light)
            cams.clear() if len(cams) > 3 else None  # bound memory (pyramids of large mosaics)
            cams[key] = MosaicCamera(img, out_size)
        return cams[key]

    plans = plan_variants(max(1, n_videos), seed=seed, night_fraction=night_fraction, ir_fraction=ir_fraction)
    if "pan" not in render_modes:
        plans = [p for p in plans if p["kind"] != "pan"] or plans[:1]
    if "zoom" not in render_modes:
        pans = [p for p in plans if p["kind"] == "pan"]
        plans = pans or [{"kind": "pan", "angle": 0.0, "view": 4.0, "length": 3.0, "photo": False}]

    n_frames = max(2, int(round(duration_s * fps)))
    sim_plans = csim.plan_videos(camera_sim, len(plans), seed)
    results: Dict[str, Any] = {}
    videos: List[Dict[str, Any]] = []
    for i, plan in enumerate(plans):
        rng = random.Random(seed * 1000 + i)
        src_idx = i % len(sources)
        light = plan.get("light", "day")
        if sources[src_idx][0].startswith("night:"):
            light = "night_real"  # real night imagery: no simulation on top
        cam = camera(src_idx, "day" if light == "night_real" else light, seed * 1000 + i)
        if plan["kind"] == "zoom":
            poses = zoom_path(
                cam,
                t_center,
                t_size,
                n_frames,
                rng,
                angle=plan["angle"],
                angle_drift=plan["drift"],
                end_view_factor=plan["end"],
                start_offset=plan["start_off"],
                end_jitter=plan["jitter"],
            )
        else:
            poses = pan_path(
                cam,
                t_center,
                t_size,
                n_frames,
                rng,
                angle=plan["angle"],
                view_factor=plan["view"],
                length_factor=plan["length"],
            )
        sim_rng = random.Random(seed * 1000 + i + 555)
        poses = csim.adjust_poses(poses, cam, camera_sim, sim_plans[i], layout.meters_per_px, sim_rng)
        sensor = csim.make_sensor(camera_sim, cam, sim_plans[i], sim_rng, light)
        if light == "day":
            photo = Photometric.random(rng) if plan.get("photo") and sensor is None else None
        else:  # low-light sensor: noise + softness, no daylight colour jitter
            photo = Photometric(noise=rng.uniform(4, 12), blur=rng.choice([0, 3, 3, 5]))
        tag = ("" if light == "day" else f"_{light}") + csim.tag(sim_plans[i])
        name = f"satellite_{plan['kind']}_{i:02d}{tag}.mp4"

        def _cb(done: int, total: int, msg: str, i: int = i) -> None:
            if progress_cb and total > 0:
                pct = int(100 * (i + done / total) / len(plans))
                progress_cb(min(99, pct), 100, f"Video {i + 1}/{len(plans)}: {msg}")

        vpath, lpath = render_labeled_video(
            cam,
            poses,
            layout,
            out / name,
            fps=fps,
            photometric=photo,
            seed=seed * 1000 + i,
            kind=plan["kind"],
            progress_cb=_cb,
            frame_cb=frame_cb,
            progress_label=f"Rendering {plan['kind']} ({light}, {sources[src_idx][0]})",
            extra_meta={"light": light, "source": sources[src_idx][0], "camera_sim": _sim_meta(camera_sim, sim_plans[i])},
            keep_classes=keeps[src_idx],
            sensor=sensor,
        )
        videos.append({"path": vpath, "labels": lpath, "kind": plan["kind"], "light": light, "source": sources[src_idx][0]})
        results.setdefault(plan["kind"], vpath)
    results["videos"] = videos
    if progress_cb:
        progress_cb(100, 100, "Done")
    return results


def render_path_videos(
    mosaics: "np.ndarray | Sequence[Tuple]",
    layout: SceneLayout,
    keys: Sequence[Any],
    out_dir: str | Path,
    *,
    lights: Sequence[str] = ("day",),
    all_sources: bool = False,
    fps: float = 24.0,
    out_size: Tuple[int, int] = (1280, 720),
    name: str = "custom",
    seed: int = 0,
    keep_classes: Optional[set] = None,
    camera_sim: Optional[CameraSim] = None,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
) -> Dict[str, Any]:
    """Render a user-designed camera path (CameraKey list) as labeled videos: one per light × source.

    ``keep_classes``: only these class ids are labeled (None = all).
    ``camera_sim``: tilt + sensor/lens simulation (the designed altitudes are kept as they are).
    """
    from desktop_app.satellite.synth import path_from_keyframes

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = [("primary", mosaics, None)] if isinstance(mosaics, np.ndarray) else list(mosaics)
    if not all_sources:
        raw = raw[:1]
    poses = path_from_keyframes(keys, fps)
    jobs = [(item, light) for item in raw for light in lights]
    sim_plans = csim.plan_videos(camera_sim, len(jobs) + 1, seed)[1:]  # paths have no canonical video
    videos: List[Dict[str, Any]] = []
    for j, (item, light) in enumerate(jobs):
        src_name, img = str(item[0]), item[1]
        mask = item[2] if len(item) > 2 else None
        keep = None
        if mask is not None:
            from desktop_app.satellite.align import covered_classes

            keep = covered_classes(layout, mask)
        if keep_classes is not None:
            keep = set(keep_classes) if keep is None else (keep & set(keep_classes))
        real_night = src_name.startswith("night:")
        mode = "day" if (real_night or light == "day") else light
        view = make_night_mosaic(img, layout.meters_per_px, random.Random(seed + j), mode=mode) if mode != "day" else img
        cam = MosaicCamera(view, out_size)
        fitted = [cam.fit(p) for p in poses]
        tag = "night_real" if real_night else light
        sim_rng = random.Random(seed * 1000 + j + 777)
        fitted = csim.adjust_poses(fitted, cam, camera_sim, sim_plans[j], layout.meters_per_px, sim_rng,
                                   use_altitude=False)
        sensor = csim.make_sensor(camera_sim, cam, sim_plans[j], sim_rng, "day" if mode == "day" else mode)
        photo = None if mode == "day" else Photometric(noise=6.0, blur=3)

        def _cb(done: int, total: int, msg: str, j: int = j) -> None:
            if progress_cb and total > 0:
                progress_cb(min(99, int(100 * (j + done / total) / len(jobs))), 100, f"Video {j + 1}/{len(jobs)}: {msg}")

        vpath, lpath = render_labeled_video(
            cam,
            fitted,
            layout,
            out / f"path_{_safe(name)}_{j:02d}_{tag}{csim.tag(sim_plans[j])}.mp4",
            fps=fps,
            photometric=photo,
            seed=seed + j,
            kind="custom",
            progress_cb=_cb,
            frame_cb=frame_cb,
            progress_label=f"Rendering {name} ({tag}, {src_name})",
            extra_meta={"light": tag, "source": src_name, "path": [k.to_dict() for k in keys],
                        "camera_sim": _sim_meta(camera_sim, sim_plans[j])},
            keep_classes=keep,
            sensor=sensor,
        )
        videos.append({"path": vpath, "labels": lpath, "kind": "custom", "light": tag, "source": src_name})
    if progress_cb:
        progress_cb(100, 100, "Done")
    return {"videos": videos}


def _sim_meta(sim: Optional[CameraSim], plan: Dict[str, bool]) -> Optional[Dict[str, Any]]:
    if sim is None or not sim.any_enabled:
        return None
    return {"camera": sim.camera, "lens": sim.lens, "fov_v": round(sim.fov_v(), 2), **plan,
            "altitude": [sim.alt_min_m, sim.alt_max_m] if sim.altitude else None}
