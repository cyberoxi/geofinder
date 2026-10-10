"""Satellite imagery → synthetic video helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop_app.satellite.prepare import PreparedScene, prepare_scene
from desktop_app.satellite.render import render_scene_videos
from desktop_app.satellite.tiles import bbox_from_center, expand_bbox, fetch_mosaic, fit_zoom_for_bbox
from desktop_app.satellite.video_gen import render_pan_video, render_zoom_video
from shared.landmarks import SceneLayout
from shared.logging import get_logger

logger = get_logger("desktop.satellite")

ProgressCb = Callable[[int, int, str], None]
FrameCb = Callable[[np.ndarray], None]

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
    num_landmarks: int = 8,
    n_videos: int = 6,
    target_class: str = "target_region",
    seed: int = 0,
    use_sam: bool = True,
    max_tiles: int = 900,
    review_cb: Optional[Callable[["PreparedScene"], Optional[SceneLayout]]] = None,
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

    if auto_label:
        prepared = prepare_scene(
            south=s,
            west=w,
            north=n,
            east=e,
            zoom=zoom,
            zoom_out_factor=zoom_out_factor,
            out_dir=out,
            cache_dir=cache_dir,
            num_landmarks=num_landmarks,
            target_class=target_class,
            use_sam=use_sam,
            max_tiles=max_tiles,
            progress_cb=(lambda d, t, m: progress_cb(int(60 * d / max(1, t)), 100, m)) if progress_cb else None,
            frame_cb=frame_cb,
        )
        layout = prepared.layout
        if review_cb is not None:
            edited = review_cb(prepared)
            if edited is not None:
                layout = edited
                prepared.save_layout(layout)
        results = render_scene_videos(
            prepared.mosaic,
            layout,
            out,
            n_videos=n_videos,
            render_modes=render_modes,
            duration_s=duration_s,
            fps=fps,
            out_size=out_size,
            seed=seed,
            progress_cb=(lambda d, t, m: progress_cb(60 + int(40 * d / max(1, t)), 100, m)) if progress_cb else None,
            frame_cb=frame_cb,
        )
        results.update(prepared.paths())
        logger.info("Generated %d labeled videos", len(results.get("videos", [])))
        return results

    want_zoom = "zoom" in render_modes
    factor = max(1.0, float(zoom_out_factor)) if want_zoom else 1.0
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
