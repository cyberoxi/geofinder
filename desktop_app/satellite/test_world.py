"""Hard-negative test clips from user-picked places anywhere on Earth."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from desktop_app.satellite.generate import DEFAULT_ZOOM_OUT_FACTOR
from desktop_app.satellite.scene import meters_per_pixel, selection_rect_px
from desktop_app.satellite.synth import (
    MosaicCamera,
    Photometric,
    make_night_mosaic,
    pan_path,
    render_labeled_video,
    zoom_path,
)
from desktop_app.satellite.test_videos import plan_test_variants
from desktop_app.satellite.tiles import expand_bbox, fetch_mosaic, fit_zoom_for_bbox
from shared.landmarks import SceneLayout

ProgressCb = Callable[[int, int, str], None]
FrameCb = Callable[[np.ndarray], None]
Swne = Tuple[float, float, float, float]


def _empty_layout(mosaic: np.ndarray, mpp: float) -> SceneLayout:
    h, w = mosaic.shape[:2]
    return SceneLayout(classes=[], polygons=[], mosaic_size=(w, h), meters_per_px=float(mpp))


def _fetch_focus(
    swne: Swne,
    *,
    zoom: int,
    zoom_out_factor: float,
    layer: str,
    cache_dir: Path,
    max_tiles: int,
    progress_cb: Optional[ProgressCb],
) -> Tuple[np.ndarray, Tuple[float, float], float, float]:
    """Download mosaiced imagery around ``swne``; return mosaic, center, size, mpp."""
    s, w, n, e = (float(v) for v in swne)
    factor = max(1.0, float(zoom_out_factor))
    fetch = expand_bbox(s, w, n, e, factor) if factor > 1.0 else (s, w, n, e)
    z = fit_zoom_for_bbox(*fetch, int(zoom), max_tiles=max_tiles)
    mosaic = fetch_mosaic(
        *fetch,
        zoom=z,
        cache_dir=cache_dir,
        max_tiles=max_tiles,
        layer=layer,
        progress_cb=progress_cb,
    )
    x0, y0, x1, y1 = selection_rect_px((s, w, n, e), fetch, z, mosaic.shape)
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    size = float(max(x1 - x0, y1 - y0, 16.0))
    mpp = meters_per_pixel((s + n) / 2.0, z)
    return mosaic, center, size, mpp


def render_world_test_videos(
    world_regions: Sequence[Swne],
    out_dir: str | Path,
    *,
    n_videos: int,
    render_modes: Sequence[str] = ("zoom", "pan"),
    duration_s: float = 8.0,
    fps: float = 24.0,
    out_size: Tuple[int, int] = (1280, 720),
    night_fraction: float = 0.0,
    ir_fraction: float = 0.0,
    zoom_start: float = 12.0,
    zoom_end: float = 1.6,
    zoom_speed: float = 1.0,
    pan_speed: float = 1.0,
    pan_view: float = 4.0,
    tile_zoom: int = 16,
    zoom_out_factor: float = DEFAULT_ZOOM_OUT_FACTOR,
    layer: str = "satellite",
    cache_dir: str | Path | None = None,
    max_tiles: int = 400,
    seed: int = 0,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
) -> Dict[str, Any]:
    """
    Fetch Esri imagery for each geo bbox and render without-target evaluation clips
    centred on that place (true hard negatives from anywhere on Earth).
    """
    regions = [tuple(map(float, r[:4])) for r in world_regions]  # type: ignore[misc]
    if n_videos < 1:
        raise ValueError("Set at least one without-target video")
    if not regions:
        raise ValueError("Pick at least one place on the world map")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else Path("data") / "satellite_cache"

    plans = plan_test_variants(
        0,
        n_videos,
        seed=seed,
        modes=render_modes,
        night_fraction=night_fraction,
        ir_fraction=ir_fraction,
    )
    z_start = max(1.0, float(zoom_start))
    z_end = max(0.5, float(zoom_end))
    z_speed = max(0.1, float(zoom_speed))
    p_speed = max(0.1, float(pan_speed))
    p_view = max(1.0, float(pan_view))
    n_zoom = max(2, int(round(duration_s / z_speed * fps)))
    n_pan = max(2, int(round(duration_s / p_speed * fps)))

    videos: List[Dict[str, Any]] = []
    for i, plan in enumerate(plans):
        swne = regions[i % len(regions)]
        rng = random.Random(seed * 1000 + i + 17)

        def tile_progress(done: int, total: int, msg: str, i: int = i) -> None:
            if progress_cb and total > 0:
                base = int(100 * i / max(1, len(plans)))
                span = max(1, int(100 / max(1, len(plans))) - 1)
                progress_cb(min(99, base + int(span * 0.4 * done / total)), 100, f"Tiles {i + 1}/{len(plans)}: {msg}")

        mosaic, center, size, mpp = _fetch_focus(
            swne,  # type: ignore[arg-type]
            zoom=tile_zoom,
            zoom_out_factor=zoom_out_factor,
            layer=layer,
            cache_dir=cache,
            max_tiles=max_tiles,
            progress_cb=tile_progress,
        )
        layout = _empty_layout(mosaic, mpp)
        light = plan.get("light", "day")
        img = mosaic
        if light in ("night", "ir"):
            img = make_night_mosaic(mosaic, mpp, random.Random(seed + i), mode=light)
        cam = MosaicCamera(img, out_size)

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

        if light == "day":
            photo = Photometric.random(rng) if plan.get("photo") else None
        else:
            photo = Photometric(noise=rng.uniform(4, 12), blur=rng.choice([0, 3, 3, 5]))

        light_tag = "" if light == "day" else f"_{light}"
        name = f"test_without_{plan['kind']}_{i:02d}{light_tag}.mp4"

        def _cb(done: int, total: int, msg: str, i: int = i) -> None:
            if progress_cb and total > 0:
                base = int(100 * i / max(1, len(plans)))
                span = max(1, int(100 / max(1, len(plans))) - 1)
                progress_cb(
                    min(99, base + int(span * (0.4 + 0.6 * done / total))),
                    100,
                    f"World test {i + 1}/{len(plans)}: {msg}",
                )

        vpath, lpath = render_labeled_video(
            cam,
            poses,
            layout,
            out / name,
            fps=fps,
            photometric=photo,
            seed=seed * 1000 + i,
            kind=f"test_without_{plan['kind']}",
            progress_cb=_cb,
            frame_cb=frame_cb,
            progress_label=f"Rendering world without {plan['kind']} ({light})",
            extra_meta={
                "light": light,
                "has_target": False,
                "role": "negative",
                "region": "world",
                "swne": list(swne),
            },
        )
        videos.append(
            {
                "path": vpath,
                "labels": lpath,
                "kind": plan["kind"],
                "has_target": False,
                "light": light,
            }
        )

    if progress_cb:
        progress_cb(100, 100, "Done")
    if not videos:
        raise ValueError("No world-map test videos were rendered")
    return {"videos": videos, "skipped": 0, "out_dir": out}
