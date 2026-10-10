"""Fast sub-region training: a target inside a region atlas, fine-tuned from the atlas base model.

The new scene keeps the atlas class list (target + every atlas landmark, projected from
lon/lat), so the base model's whole network — head included — is reused and only the
target class has to be learned. Imagery comes from the tile cache filled by the atlas build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop_app.atlas.model import RegionAtlas, bbox_of, global_px, lonlat_to_mosaic
from desktop_app.satellite.scene import draw_scene_preview, meters_per_pixel, save_candidates, selection_polygon_px
from desktop_app.satellite.tiles import count_tiles, expand_bbox, fetch_mosaic
from shared.landmarks import SceneLayout
from shared.logging import get_logger

logger = get_logger("desktop.atlas")

ProgressCb = Callable[[int, int, str], None]
StopCb = Callable[[], bool]
DEFAULT_CACHE = Path("data") / "satellite_cache"


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _landmarks_in_view(atlas: RegionAtlas, fetch_swne, target_ring) -> int:
    """Atlas landmarks fully inside ``fetch_swne`` and not overlapping the target (geometry only)."""
    s, w, n, e = fetch_swne
    gx0, gy0 = global_px(w, n, atlas.zoom)
    gx1, gy1 = global_px(e, s, atlas.zoom)
    tb = bbox_of([global_px(lon, lat, atlas.zoom) for lon, lat in target_ring])
    count = 0
    for lm in atlas.landmarks:
        b = bbox_of([global_px(lon, lat, atlas.zoom) for lon, lat in lm["polygon_lonlat"]])
        if b[0] >= gx0 + 2 and b[1] >= gy0 + 2 and b[2] <= gx1 - 2 and b[3] <= gy1 - 2 and not _overlap(b, tb):
            count += 1
    return count


def validate_target(atlas: RegionAtlas, target_lonlat: Sequence[Sequence[float]]) -> None:
    if not target_lonlat or len(target_lonlat) < 3:
        raise ValueError("Draw the target region (polygon or box) first")
    outside = [p for p in target_lonlat if not atlas.contains(float(p[0]), float(p[1]))]
    if outside:
        raise ValueError(
            f"{len(outside)} point(s) of the target are outside the atlas area.\n"
            "Draw the target inside the dashed orange boundary."
        )
    if not atlas.landmarks:
        raise ValueError("This atlas has no landmarks yet — build it first")


def prepare_subregion_scene(
    atlas: RegionAtlas,
    target_lonlat: Sequence[Sequence[float]],
    out_dir: str | Path,
    *,
    zoom_out_factor: float = 8.0,
    min_landmarks: int = 3,
    max_tiles: int = 900,
    cache_dir: str | Path | None = None,
    progress_cb: Optional[ProgressCb] = None,
) -> Dict[str, Any]:
    """Mosaic around the target (at the atlas zoom, from cache) + layout with the atlas classes."""
    from desktop_app.satellite.objects import ObjectCandidate

    validate_target(atlas, target_lonlat)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ring = [[float(lon), float(lat)] for lon, lat in target_lonlat]
    w, s, e, n = bbox_of(ring)
    selection = (s, w, n, e)
    factor = max(1.0, float(zoom_out_factor))
    while factor > 1.5 and count_tiles(*expand_bbox(*selection, factor), atlas.zoom) > max_tiles:
        factor *= 0.85
    # Widen the view until enough atlas landmarks are fully in it (the localizer needs ≥2)
    while _landmarks_in_view(atlas, expand_bbox(*selection, factor), ring) < min_landmarks:
        bigger = factor * 1.3
        if bigger > 60 or count_tiles(*expand_bbox(*selection, bigger), atlas.zoom) > max_tiles:
            break
        factor = bigger
    fetch = expand_bbox(*selection, factor)
    logger.info("Sub-region zoom-out factor %.1f", factor)
    if count_tiles(*fetch, atlas.zoom) > max_tiles:
        raise ValueError(
            f"The target is too large for the atlas zoom {atlas.zoom}. Draw a smaller target region."
        )
    mosaic = fetch_mosaic(
        *fetch, zoom=atlas.zoom, cache_dir=Path(cache_dir) if cache_dir else DEFAULT_CACHE,
        max_tiles=max_tiles, layer=atlas.layer, progress_cb=progress_cb,
    )
    H, W = mosaic.shape[:2]
    target_px = selection_polygon_px(ring, fetch, atlas.zoom, mosaic.shape)
    tb = bbox_of(target_px)
    polys = [target_px]
    keep = {0}
    candidates = []
    for i, lm in enumerate(atlas.landmarks, start=1):
        poly = lonlat_to_mosaic(lm["polygon_lonlat"], fetch, atlas.zoom)
        polys.append(poly)
        x0, y0, x1, y1 = bbox_of(poly)
        inside = x0 >= 2 and y0 >= 2 and x1 <= W - 2 and y1 <= H - 2
        if inside and not _overlap((x0, y0, x1, y1), tb):
            keep.add(i)
        candidates.append(ObjectCandidate(polygon=np.asarray(poly, np.float32), kind=str(lm.get("kind", "object")),
                                          score=float(lm.get("score", 0.0))))
    layout = SceneLayout(
        classes=atlas.classes,
        polygons=polys,
        mosaic_size=(W, H),
        meters_per_px=meters_per_pixel((s + n) / 2.0, atlas.zoom),
        meta={
            "selection_swne": list(selection),
            "selection_lonlat": ring,
            "fetch_swne": list(fetch),
            "tile_zoom": atlas.zoom,
            "source": atlas.layer,
            "atlas": str(atlas.root),
            "visible_landmarks": sorted(keep - {0}),
            "landmarks": [c.to_dict() for c in candidates],
        },
    )
    paths = {
        "mosaic": out / "satellite_mosaic.jpg",
        "scene": out / "scene.json",
        "candidates": out / "candidates.json",
        "scene_preview": out / "scene_preview.jpg",
    }
    cv2.imwrite(str(paths["mosaic"]), mosaic)
    layout.save(paths["scene"])
    save_candidates(paths["candidates"], [candidates[i - 1] for i in sorted(keep - {0})])
    visible = sorted(keep)
    preview_layout = SceneLayout(
        classes=[layout.classes[i] for i in visible], polygons=[layout.polygons[i] for i in visible],
        mosaic_size=layout.mosaic_size,
    )
    cv2.imwrite(str(paths["scene_preview"]), draw_scene_preview(mosaic, preview_layout))
    logger.info("Sub-region scene: %d of %d atlas landmarks in view", len(keep) - 1, len(atlas.landmarks))
    return {"mosaic": mosaic, "layout": layout, "keep": keep, "paths": paths}


def prepare_subregion_project(
    atlas: RegionAtlas,
    target_lonlat: Sequence[Sequence[float]],
    project_dir: str | Path,
    *,
    project_name: str = "",
    n_videos: int = 3,
    zoom_out_factor: float = 10.0,
    night_fraction: float = 0.2,
    ir_fraction: float = 0.1,
    freeze: int = 10,
    cache_dir: str | Path | None = None,
    progress_cb: Optional[ProgressCb] = None,
    should_stop: Optional[StopCb] = None,
) -> Path:
    """New project with labeled videos of the target, linked to the atlas base model. Returns its folder."""
    from desktop_app.project_manager.manager import ProjectManager
    from desktop_app.satellite.render import render_scene_videos

    progress = progress_cb or (lambda d, t, m: None)
    stop = should_stop or (lambda: False)
    weights = atlas.base_weights_path()
    if weights is None:
        raise RuntimeError("The atlas base model is not trained yet — build the atlas first")
    validate_target(atlas, target_lonlat)
    root = Path(project_dir)
    if (root / "project.json").exists():
        raise FileExistsError(f"A project already exists in {root}")
    pm = ProjectManager.create(root, project_name or root.name)
    try:
        def _fetch(d: int, t: int, msg: str) -> None:
            if t > 0:
                progress(int(15 * d / t), 100, msg)

        prep = prepare_subregion_scene(
            atlas, target_lonlat, pm.exports_dir / "atlas_scene",
            zoom_out_factor=zoom_out_factor, cache_dir=cache_dir, progress_cb=_fetch,
        )
        if stop():
            raise RuntimeError("Stopped by user")

        def _render(d: int, t: int, msg: str) -> None:
            if t > 0:
                progress(15 + int(80 * d / t), 100, msg)

        res = render_scene_videos(
            prep["mosaic"], prep["layout"], pm.exports_dir / "satellite",
            n_videos=max(1, int(n_videos)), night_fraction=night_fraction, ir_fraction=ir_fraction,
            keep_classes=prep["keep"], progress_cb=_render,
        )
        if stop():
            raise RuntimeError("Stopped by user")
        progress(96, 100, "Importing videos into the new project…")
        for v in res.get("videos") or []:
            asset = pm.add_video(v["path"])
            pm.attach_auto_labels(asset.video_id, v["labels"])
        p = prep["paths"]
        pm.set_scene(p["scene"], p["scene_preview"], p["mosaic"], p["candidates"])
        assert pm.meta is not None
        pm.meta.model_type = atlas.model_type
        pm.meta.settings["atlas"] = {
            "root": str(atlas.root),
            "role": "subregion",
            "base_weights": str(weights),
            "classes": atlas.classes,
            "freeze": int(freeze),
            "imgsz": atlas.imgsz,
        }
        pm.save_meta()
    finally:
        pm.close()
    progress(100, 100, f"Sub-region ready: {len(prep['keep']) - 1} atlas landmarks in view")
    return root


def finetune_source(pm: Any) -> Optional[Tuple[str, int]]:
    """(base_weights, freeze) when ``pm`` is an atlas sub-region whose classes still match the atlas."""
    meta = getattr(pm, "meta", None)
    info = (getattr(meta, "settings", None) or {}).get("atlas") if meta else None
    if not info or info.get("role") != "subregion":
        return None
    weights = Path(str(info.get("base_weights") or ""))
    if not weights.is_file():
        return None
    try:
        if list(pm.class_names()) != list(info.get("classes") or []):
            return None  # landmarks were edited → the base head no longer matches
    except Exception:  # noqa: BLE001
        return None
    return str(weights), int(info.get("freeze", 10))
