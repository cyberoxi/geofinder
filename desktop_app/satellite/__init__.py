"""Satellite imagery → synthetic video helpers."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop_app.satellite.tiles import (
    bbox_from_center,
    expand_bbox,
    fetch_mosaic,
    fit_zoom_for_bbox,
)
from desktop_app.satellite.scene import (
    build_scene_layout,
    draw_scene_preview,
    meters_per_pixel,
    selection_rect_px,
)
from desktop_app.satellite.synth import (
    MosaicCamera,
    Photometric,
    pan_path,
    plan_variants,
    render_labeled_video,
    zoom_path,
)
from desktop_app.satellite.video_gen import render_pan_video, render_zoom_video
from shared.logging import get_logger

logger = get_logger("desktop.satellite")

ProgressCb = Callable[[int, int, str], None]
FrameCb = Callable[[np.ndarray], None]

# How many times wider the zoom-video start view is vs the selected region.
DEFAULT_ZOOM_OUT_FACTOR = 12.0


def _center_fraction_crop(mosaic: np.ndarray, fraction: float) -> np.ndarray:
    """Crop the center ``fraction`` of mosaic width/height (for pan on selection)."""
    fraction = max(0.02, min(1.0, float(fraction)))
    h, w = mosaic.shape[:2]
    cw = max(8, int(round(w * fraction)))
    ch = max(8, int(round(h * fraction)))
    x0 = max(0, (w - cw) // 2)
    y0 = max(0, (h - ch) // 2)
    return mosaic[y0 : y0 + ch, x0 : x0 + cw].copy()


def generate_satellite_videos(
    *,
    lat: float,
    lon: float,
    span: float = 0.02,
    south: Optional[float] = None,
    west: Optional[float] = None,
    north: Optional[float] = None,
    east: Optional[float] = None,
    zoom: int = 17,
    modes: Sequence[str] = ("zoom", "pan"),
    out_dir: str | Path = "data/output",
    duration_s: float = 8.0,
    fps: float = 24.0,
    out_size: Tuple[int, int] = (1280, 720),
    cache_dir: str | Path | None = None,
    save_mosaic: bool = True,
    zoom_out_factor: float = DEFAULT_ZOOM_OUT_FACTOR,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
    auto_label: bool = True,
    num_landmarks: int = 4,
    n_videos: int = 6,
    target_class: str = "target_region",
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Download Esri World Imagery for a region and render zoom and/or pan MP4s.

    For zoom mode the selected bbox is the *end* of the flyover; imagery is
    fetched for a wider area (``zoom_out_factor``×) so the video starts far out
    and zooms into the selection.

    With ``auto_label`` (default) the exact target polygon is computed from the
    selection, ``num_landmarks`` distinctive landmarks are discovered around it,
    and ``n_videos`` varied clips (rotated / off-centre zooms, low passes,
    lighting changes) are rendered, each with a ``.labels.json`` sidecar.

    Returns dict with keys among: mosaic, zoom, pan, scene, scene_preview, videos
    (``videos`` = list of {"path", "labels", "kind"}).
    """
    if all(v is not None for v in (south, west, north, east)):
        s, w, n, e = float(south), float(west), float(north), float(east)  # type: ignore[arg-type]
    else:
        s, w, n, e = bbox_from_center(lat, lon, span)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mode_set = {m.strip().lower() for m in modes}
    if "both" in mode_set:
        mode_set = {"zoom", "pan"}
    render_modes = [m for m in ("zoom", "pan") if m in mode_set]
    if not render_modes:
        raise ValueError("modes must include zoom, pan, or both")

    want_zoom = "zoom" in render_modes
    factor = max(1.0, float(zoom_out_factor)) if (want_zoom or auto_label) else 1.0
    fetch_s, fetch_w, fetch_n, fetch_e = (
        expand_bbox(s, w, n, e, factor) if factor > 1.0 else (s, w, n, e)
    )
    fetch_zoom = fit_zoom_for_bbox(fetch_s, fetch_w, fetch_n, fetch_e, zoom)
    if fetch_zoom != zoom:
        logger.info(
            "Lowered download zoom %d → %d to fit expanded region (factor=%.1f)",
            zoom,
            fetch_zoom,
            factor,
        )

    # Overall progress: tiles ~0-60%, each render mode shares the rest
    tile_weight = 60
    render_weight = 40
    per_mode = render_weight // len(render_modes)

    def tile_progress(done: int, total: int, msg: str) -> None:
        if not progress_cb or total <= 0:
            return
        pct = int(tile_weight * done / total)
        progress_cb(pct, 100, msg)

    mosaic = fetch_mosaic(
        fetch_s,
        fetch_w,
        fetch_n,
        fetch_e,
        zoom=fetch_zoom,
        cache_dir=cache_dir,
        progress_cb=tile_progress,
        preview_cb=frame_cb,
    )
    results: Dict[str, Any] = {}

    if save_mosaic:
        mosaic_path = out / "satellite_mosaic.jpg"
        cv2.imwrite(str(mosaic_path), mosaic)
        results["mosaic"] = mosaic_path

    if progress_cb:
        progress_cb(tile_weight, 100, "Mosaic ready — rendering video…")

    if auto_label:
        _render_auto_labeled(
            results,
            mosaic,
            selection=(s, w, n, e),
            fetch_bbox=(fetch_s, fetch_w, fetch_n, fetch_e),
            fetch_zoom=fetch_zoom,
            out=out,
            render_modes=render_modes,
            duration_s=duration_s,
            fps=fps,
            out_size=out_size,
            num_landmarks=num_landmarks,
            n_videos=n_videos,
            target_class=target_class,
            seed=seed,
            base_pct=tile_weight,
            progress_cb=progress_cb,
            frame_cb=frame_cb,
        )
        if progress_cb:
            progress_cb(100, 100, "Done")
        logger.info("Generated %d labeled videos", len(results.get("videos", [])))
        return results

    # Zoom ends on the original selection ≈ center 1/factor of the mosaic.
    end_scale = 1.0 / factor if factor > 1.0 else 0.22
    pan_source = _center_fraction_crop(mosaic, 1.0 / factor) if factor > 1.0 else mosaic

    base = tile_weight
    for i, mode in enumerate(render_modes):
        mode_base = base + i * per_mode
        mode_span = per_mode if i < len(render_modes) - 1 else (100 - mode_base)

        def make_progress(mb: int = mode_base, ms: int = mode_span):
            def _cb(done: int, total: int, msg: str) -> None:
                if not progress_cb or total <= 0:
                    return
                pct = mb + int(ms * done / total)
                progress_cb(min(99, pct), 100, msg)

            return _cb

        if mode == "zoom":
            results["zoom"] = render_zoom_video(
                mosaic,
                out / "satellite_zoom.mp4",
                duration_s=duration_s,
                fps=fps,
                out_size=out_size,
                end_scale=end_scale,
                progress_cb=make_progress(),
                frame_cb=frame_cb,
            )
        else:
            results["pan"] = render_pan_video(
                pan_source,
                out / "satellite_pan.mp4",
                duration_s=duration_s,
                fps=fps,
                out_size=out_size,
                progress_cb=make_progress(),
                frame_cb=frame_cb,
            )

    if progress_cb:
        progress_cb(100, 100, "Done")

    logger.info("Generated: %s", {k: str(v) for k, v in results.items()})
    return results


def _render_auto_labeled(
    results: Dict[str, Any],
    mosaic: np.ndarray,
    *,
    selection: Tuple[float, float, float, float],
    fetch_bbox: Tuple[float, float, float, float],
    fetch_zoom: int,
    out: Path,
    render_modes: List[str],
    duration_s: float,
    fps: float,
    out_size: Tuple[int, int],
    num_landmarks: int,
    n_videos: int,
    target_class: str,
    seed: int,
    base_pct: int,
    progress_cb: Optional[ProgressCb],
    frame_cb: Optional[FrameCb],
) -> None:
    target = selection_rect_px(selection, fetch_bbox, fetch_zoom, mosaic.shape)
    lat_c = (selection[0] + selection[2]) / 2.0
    if progress_cb:
        progress_cb(base_pct, 100, "Discovering landmarks around the target…")
    layout = build_scene_layout(
        mosaic,
        target,
        num_landmarks=num_landmarks,
        target_class=target_class,
        meters_per_px=meters_per_pixel(lat_c, fetch_zoom),
        meta={
            "selection_swne": list(selection),
            "fetch_swne": list(fetch_bbox),
            "tile_zoom": fetch_zoom,
            "attribution": "Esri World Imagery",
        },
    )
    results["scene"] = layout.save(out / "scene.json")
    preview_path = out / "scene_preview.jpg"
    preview = draw_scene_preview(mosaic, layout)
    cv2.imwrite(str(preview_path), preview)
    results["scene_preview"] = preview_path
    if frame_cb:
        frame_cb(preview)

    tx0, ty0, tx1, ty1 = target
    t_center = ((tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0)
    t_size = max(tx1 - tx0, ty1 - ty0)
    cam = MosaicCamera(mosaic, out_size)
    plans = plan_variants(max(1, n_videos), seed=seed)
    if "pan" not in render_modes:
        plans = [p for p in plans if p["kind"] != "pan"] or plans[:1]
    if "zoom" not in render_modes:
        pans = [p for p in plans if p["kind"] == "pan"]
        plans = pans or [{"kind": "pan", "angle": 0.0, "view": 4.0, "length": 3.0, "photo": False}]

    n_frames = max(2, int(round(duration_s * fps)))
    span = 100 - base_pct - 2
    videos: List[Dict[str, Any]] = []
    for i, plan in enumerate(plans):
        rng = random.Random(seed * 1000 + i)
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
        photo = Photometric.random(rng) if plan.get("photo") else None
        name = f"satellite_{plan['kind']}_{i:02d}.mp4"
        lo = base_pct + 2 + span * i // len(plans)
        hi = base_pct + 2 + span * (i + 1) // len(plans)

        def _cb(done: int, total: int, msg: str, lo: int = lo, hi: int = hi, i: int = i) -> None:
            if progress_cb and total > 0:
                progress_cb(min(99, lo + (hi - lo) * done // total), 100, f"Video {i + 1}/{len(plans)}: {msg}")

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
            progress_label=f"Rendering {plan['kind']}",
        )
        videos.append({"path": vpath, "labels": lpath, "kind": plan["kind"]})
        results.setdefault(plan["kind"], vpath)
    results["videos"] = videos
