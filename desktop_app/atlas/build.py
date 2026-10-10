"""Build a region atlas: chunk the large area, find a global landmark set, render training videos.

Stages are resumable — progress is stored per chunk in ``atlas.json``:
``planned`` → ``scanned`` (landmarks found) → ``rendered`` → ``imported`` (in the base project).
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop_app.atlas.model import (
    RegionAtlas,
    bbox_of,
    global_px,
    global_px_to_lonlat,
    lonlat_to_mosaic,
    mosaic_to_lonlat,
    point_in_ring,
)
from desktop_app.atlas.tour import tour_keys, tour_lights
from desktop_app.satellite.scene import draw_scene_preview, meters_per_pixel
from desktop_app.satellite.tiles import TILE_SIZE, count_tiles, fetch_mosaic
from shared.landmarks import SceneLayout
from shared.logging import get_logger

logger = get_logger("desktop.atlas")

ProgressCb = Callable[[int, int, str], None]
StopCb = Callable[[], bool]
DEFAULT_CACHE = Path("data") / "satellite_cache"


@dataclass
class AtlasBuildOptions:
    landmarks_per_chunk: int = 10
    max_landmarks: int = 120
    videos_per_chunk: int = 4
    focus_per_chunk: int = 2
    tour_videos: int = 2  # flights over EVERY landmark of the chunk (see tour.py)
    max_tiles_per_chunk: int = 300
    overlap: float = 0.15
    use_sam: bool = True
    night_fraction: float = 0.2
    ir_fraction: float = 0.1
    duration_s: float = 8.0
    fps: float = 24.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AtlasBuildOptions":
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ------------------------------------------------------------------ planning


def _starts(a: float, b: float, side: float, step: float) -> List[Tuple[float, float]]:
    if b - a <= side:
        return [(a, b)]
    out, x = [], a
    while x + side < b:
        out.append((x, x + side))
        x += step
    out.append((b - side, b))
    return out


def _chunk_touches_polygon(swne: Sequence[float], ring: Sequence[Sequence[float]]) -> bool:
    if len(ring) < 3:
        return True
    s, w, n, e = swne
    for lon, lat in ring:
        if s <= lat <= n and w <= lon <= e:
            return True
    for i in range(5):
        for j in range(5):
            if point_in_ring(w + (e - w) * i / 4.0, s + (n - s) * j / 4.0, ring):
                return True
    return False


def plan_chunks(
    swne: Sequence[float],
    polygon_lonlat: Sequence[Sequence[float]],
    zoom: int,
    max_tiles_per_chunk: int = 300,
    overlap: float = 0.15,
) -> List[Dict[str, Any]]:
    """Overlapping grid of chunk bboxes (each ≤ ``max_tiles_per_chunk`` tiles) covering the area."""
    s, w, n, e = swne
    gx0, gy0 = global_px(w, n, zoom)
    gx1, gy1 = global_px(e, s, zoom)
    side = max(2, int(math.sqrt(max_tiles_per_chunk)) - 1) * TILE_SIZE
    step = side * (1.0 - max(0.0, min(0.5, overlap)))
    chunks: List[Dict[str, Any]] = []
    for y0, y1 in _starts(gy0, gy1, side, step):
        for x0, x1 in _starts(gx0, gx1, side, step):
            lw, ln = global_px_to_lonlat(x0, y0, zoom)
            le, ls = global_px_to_lonlat(x1, y1, zoom)
            c_swne = [ls, lw, ln, le]
            if _chunk_touches_polygon(c_swne, polygon_lonlat):
                chunks.append({"id": len(chunks), "swne": c_swne, "status": "planned", "videos": []})
    return chunks


def estimate(
    swne: Sequence[float],
    polygon_lonlat: Sequence[Sequence[float]],
    zoom: int,
    opts: Optional[AtlasBuildOptions] = None,
) -> Dict[str, Any]:
    """Rough size of an atlas build (for the dialog)."""
    opts = opts or AtlasBuildOptions()
    chunks = plan_chunks(swne, polygon_lonlat, zoom, opts.max_tiles_per_chunk, opts.overlap)
    tiles = count_tiles(*swne, zoom)
    s, w, n, e = swne
    km_ns = (n - s) * 111.32
    km_ew = (e - w) * 111.32 * math.cos(math.radians((s + n) / 2.0))
    return {
        "chunks": len(chunks),
        "tiles": tiles,
        "download_mb": round(tiles * 0.025, 1),
        "area_km2": round(km_ns * km_ew, 2),
        "videos": len(chunks) * (opts.videos_per_chunk + opts.tour_videos),
        "meters_per_px": round(meters_per_pixel((s + n) / 2.0, zoom), 2),
    }


def new_atlas(
    root: str | Path,
    name: str,
    polygon_lonlat: Sequence[Sequence[float]],
    *,
    zoom: int,
    layer: str = "satellite",
    model_type: str = "segmentation",
    imgsz: int = 640,
    opts: Optional[AtlasBuildOptions] = None,
) -> RegionAtlas:
    opts = opts or AtlasBuildOptions()
    ring = [[float(lon), float(lat)] for lon, lat in polygon_lonlat]
    w, s, e, n = bbox_of(ring)
    atlas = RegionAtlas(
        name=name,
        root=Path(root).resolve(),
        swne=(s, w, n, e),
        polygon_lonlat=ring,
        zoom=int(zoom),
        layer=layer,
        model_type=model_type,
        imgsz=int(imgsz),
        settings={"build": asdict(opts)},
    )
    atlas.chunks = plan_chunks(atlas.swne, ring, atlas.zoom, opts.max_tiles_per_chunk, opts.overlap)
    if not atlas.chunks:
        raise ValueError("The selected area produced no chunks")
    atlas.save()
    return atlas


# ------------------------------------------------------------------ helpers


def _chunk_mosaic(atlas: RegionAtlas, chunk: Dict[str, Any], cache_dir: Path, progress_cb=None) -> np.ndarray:
    path = atlas.chunk_dir(chunk["id"]) / "mosaic.jpg"
    if path.exists():
        img = cv2.imread(str(path))
        if img is not None:
            return img
    opts = AtlasBuildOptions.from_dict(atlas.settings.get("build", {}))
    img = fetch_mosaic(
        *chunk["swne"], zoom=atlas.zoom, cache_dir=cache_dir,
        max_tiles=opts.max_tiles_per_chunk * 2, layer=atlas.layer, progress_cb=progress_cb,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return img


def _global_bbox(lm: Dict[str, Any], zoom: int) -> Tuple[float, float, float, float]:
    pts = [global_px(lon, lat, zoom) for lon, lat in lm["polygon_lonlat"]]
    return bbox_of(pts)


def _overlaps(a, b, pad: float = 0.0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def chunk_layout(
    atlas: RegionAtlas, chunk: Dict[str, Any], mosaic: np.ndarray, focus: Tuple[float, float], focus_size: float
) -> Tuple[SceneLayout, set]:
    """Layout with ALL atlas classes projected into this chunk; class 0 = camera focus (never labeled)."""
    H, W = mosaic.shape[:2]
    fx = float(np.clip(focus[0], focus_size / 2, W - focus_size / 2))
    fy = float(np.clip(focus[1], focus_size / 2, H - focus_size / 2))
    h = focus_size / 2.0
    polys = [[[fx - h, fy - h], [fx + h, fy - h], [fx + h, fy + h], [fx - h, fy + h]]]
    keep: set = set()
    for i, lm in enumerate(atlas.landmarks, start=1):
        poly = lonlat_to_mosaic(lm["polygon_lonlat"], chunk["swne"], atlas.zoom)
        polys.append(poly)
        x0, y0, x1, y1 = bbox_of(poly)
        if x0 >= 2 and y0 >= 2 and x1 <= W - 2 and y1 <= H - 2:
            keep.add(i)
    s, w, n, e = chunk["swne"]
    layout = SceneLayout(
        classes=atlas.classes,
        polygons=polys,
        mosaic_size=(W, H),
        meters_per_px=meters_per_pixel((s + n) / 2.0, atlas.zoom),
        meta={"atlas": str(atlas.root), "chunk": chunk["id"], "fetch_swne": list(chunk["swne"]), "tile_zoom": atlas.zoom,
              "source": atlas.layer, "focus_only_target": True},
    )
    return layout, keep


def _scan_chunk(
    mosaic: np.ndarray, mpp: float, k: int, use_sam: bool
) -> List[Tuple[List[List[float]], str, float]]:
    """``k`` distinctive objects spread over the whole chunk (grid cells → best per cell)."""
    from desktop_app.satellite.objects import propose_objects
    from desktop_app.satellite.scene import _rect_poly, discover_landmarks

    H, W = mosaic.shape[:2]
    centre = (W / 2 - 1, H / 2 - 1, W / 2 + 1, H / 2 + 1)
    try:
        cands = propose_objects(mosaic, centre, mpp, use_sam=use_sam, distance_visibility=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Object proposals failed (%s) — texture patches only", exc)
        cands = []
    g = max(1, math.ceil(math.sqrt(k)))
    cells: Dict[Tuple[int, int], Any] = {}
    for c in cands:  # best first
        cx, cy = c.center
        key = (min(g - 1, int(cx / W * g)), min(g - 1, int(cy / H * g)))
        cells.setdefault(key, c)
    chosen = list(cells.values())
    for c in cands:
        if len(chosen) >= k:
            break
        if c not in chosen and not any(_overlaps(c.bbox(), o.bbox(), pad=8) for o in chosen):
            chosen.append(c)
    chosen = sorted(chosen, key=lambda c: -c.score)[:k]
    out = [([[float(x), float(y)] for x, y in c.polygon], c.kind, float(c.score)) for c in chosen]
    if len(out) < k:  # sparse terrain (desert, farmland): distinctive texture patches
        T = min(W, H) / 6.0
        taken = [bbox_of(p) for p, _, _ in out]
        for gx in range(2):
            for gy in range(2):
                fx, fy = W * (0.25 + 0.5 * gx), H * (0.25 + 0.5 * gy)
                for r in discover_landmarks(mosaic, (fx - T / 4, fy - T / 4, fx + T / 4, fy + T / 4), k=2):
                    if len(out) >= k:
                        break
                    if any(_overlaps(r, b, pad=4) for b in taken):
                        continue
                    out.append((_rect_poly(r), "patch", 0.3))
                    taken.append(r)
    return out


# ------------------------------------------------------------------ stages


def scan_landmarks(atlas: RegionAtlas, cache_dir: Path, progress_cb: ProgressCb, should_stop: StopCb) -> None:
    """Stage 1: download every chunk and collect a de-duplicated global landmark set."""
    if atlas.settings.get("landmarks_frozen"):
        return
    opts = AtlasBuildOptions.from_dict(atlas.settings.get("build", {}))
    n = len(atlas.chunks)
    k = max(4, min(opts.landmarks_per_chunk, math.ceil(opts.max_landmarks / max(1, n)) + 2))
    for idx, chunk in enumerate(atlas.chunks):
        if chunk.get("status") != "planned":
            continue
        if should_stop():
            raise RuntimeError("Stopped by user")

        def _p(d: int, t: int, msg: str, idx: int = idx) -> None:
            if t > 0:
                progress_cb(int(40 * (idx + 0.8 * d / t) / n), 100, f"Chunk {idx + 1}/{n}: {msg}")

        mosaic = _chunk_mosaic(atlas, chunk, cache_dir, _p)
        progress_cb(int(40 * (idx + 0.85) / n), 100, f"Chunk {idx + 1}/{n}: detecting landmarks…")
        s, w, nn, e = chunk["swne"]
        picked = _scan_chunk(mosaic, meters_per_pixel((s + nn) / 2.0, atlas.zoom), k, opts.use_sam)
        existing = [_global_bbox(lm, atlas.zoom) for lm in atlas.landmarks]
        added = 0
        for poly, kind, score in picked:
            if len(atlas.landmarks) >= opts.max_landmarks:
                break
            ring = mosaic_to_lonlat(poly, chunk["swne"], atlas.zoom)
            lm = {
                "kind": kind or "object",
                "score": float(score),
                "polygon_lonlat": [[round(a, 8), round(b, 8)] for a, b in ring],
                "chunk": chunk["id"],
            }
            gb = _global_bbox(lm, atlas.zoom)
            if any(_overlaps(gb, b, pad=4.0) for b in existing):
                continue
            atlas.landmarks.append(lm)
            existing.append(gb)
            added += 1
        chunk["status"] = "scanned"
        chunk["landmarks_added"] = added
        atlas.save()
        logger.info("Atlas chunk %d: +%d landmarks (total %d)", chunk["id"], added, len(atlas.landmarks))
    if not atlas.landmarks:
        raise RuntimeError("No landmarks were found in the selected area")
    atlas.settings["landmarks_frozen"] = True
    atlas.save()


def render_chunks(atlas: RegionAtlas, cache_dir: Path, progress_cb: ProgressCb, should_stop: StopCb) -> None:
    """Stage 2: labeled videos per chunk (all atlas landmarks; no target labels)."""
    from desktop_app.satellite.render import render_path_videos, render_scene_videos

    opts = AtlasBuildOptions.from_dict(atlas.settings.get("build", {}))
    n = len(atlas.chunks)
    for idx, chunk in enumerate(atlas.chunks):
        if chunk.get("status") not in ("scanned",):
            continue
        if should_stop():
            raise RuntimeError("Stopped by user")
        mosaic = _chunk_mosaic(atlas, chunk, cache_dir)
        H, W = mosaic.shape[:2]
        focus_size = min(W, H) / 5.0
        _, keep = chunk_layout(atlas, chunk, mosaic, (W / 2, H / 2), focus_size)
        if not keep:
            chunk["status"] = "empty"
            atlas.save()
            continue
        rng = random.Random(1000 + int(chunk["id"]))
        focuses = [(W / 2.0, H / 2.0)]
        extra = rng.sample(sorted(keep), min(len(keep), max(0, opts.focus_per_chunk - 1)))
        for cid in extra:
            pts = np.asarray(lonlat_to_mosaic(atlas.landmarks[cid - 1]["polygon_lonlat"], chunk["swne"], atlas.zoom))
            focuses.append(tuple(pts.mean(axis=0).tolist()))
        per = [opts.videos_per_chunk // len(focuses)] * len(focuses)
        for j in range(opts.videos_per_chunk - sum(per)):
            per[j % len(per)] += 1
        # Night / IR counts per chunk (rounding per focus would make half the videos night)
        n_night = int(round(opts.night_fraction * opts.videos_per_chunk))
        n_ir = int(round(opts.ir_fraction * opts.videos_per_chunk))
        lights = [[0, 0] for _ in focuses]
        for k in range(n_night + n_ir):
            fi = (len(focuses) - 1 - k) % len(focuses)
            lights[fi][0 if k < n_night else 1] += 1
        videos: List[Dict[str, str]] = []
        tours = tour_lights(opts.tour_videos, opts.night_fraction, opts.ir_fraction)
        n_steps = len(focuses) + len(tours)

        def _add(res: Dict[str, Any]) -> None:
            for v in res.get("videos") or []:
                videos.append({
                    "path": str(Path(v["path"]).relative_to(atlas.root)),
                    "labels": str(Path(v["labels"]).relative_to(atlas.root)),
                })

        for fi, (focus, nv) in enumerate(zip(focuses, per)):
            if nv <= 0:
                continue
            night_f, ir_f = (min(1.0, c / nv) for c in lights[fi])
            layout, keep = chunk_layout(atlas, chunk, mosaic, focus, focus_size)

            def _p(d: int, t: int, msg: str, idx: int = idx, fi: int = fi) -> None:
                if t > 0:
                    frac = (fi + d / t) / n_steps
                    progress_cb(40 + int(55 * (idx + frac) / n), 100, f"Chunk {idx + 1}/{n} videos: {msg}")

            res = render_scene_videos(
                mosaic, layout, atlas.chunk_dir(chunk["id"]) / f"videos_{fi}",
                n_videos=nv, night_fraction=night_f, ir_fraction=ir_f,
                duration_s=opts.duration_s, fps=opts.fps, seed=int(chunk["id"]) * 17 + fi,
                keep_classes=keep, progress_cb=_p,
            )
            _add(res)
            if fi == 0:
                layout.save(atlas.chunk_dir(chunk["id"]) / "scene.json")
                cv2.imwrite(str(atlas.chunk_dir(chunk["id"]) / "scene_preview.jpg"), draw_scene_preview(mosaic, layout))
        # Landmark tours: every landmark of the chunk seen up close (day + night / IR)
        layout, keep = chunk_layout(atlas, chunk, mosaic, (W / 2.0, H / 2.0), focus_size)
        for ti, light in enumerate(tours):
            if should_stop():
                raise RuntimeError("Stopped by user")
            keys = tour_keys(atlas, chunk, keep, (W, H), ti, random.Random(5000 + int(chunk["id"]) * 31 + ti))
            if not keys:
                break

            def _pt(d: int, t: int, msg: str, idx: int = idx, ti: int = ti) -> None:
                if t > 0:
                    frac = (len(focuses) + ti + d / t) / n_steps
                    progress_cb(40 + int(55 * (idx + frac) / n), 100, f"Chunk {idx + 1}/{n} landmark tour: {msg}")

            _add(render_path_videos(
                mosaic, layout, keys, atlas.chunk_dir(chunk["id"]) / "tours", lights=(light,), fps=opts.fps,
                name=f"tour{ti}", seed=int(chunk["id"]) * 31 + ti, keep_classes=keep, progress_cb=_pt,
            ))
        chunk["videos"] = videos
        chunk["status"] = "rendered"
        atlas.save()


def import_base_project(atlas: RegionAtlas, progress_cb: ProgressCb) -> Path:
    """Stage 3: put every rendered chunk video into ``<atlas>/base_project`` (a normal project)."""
    from desktop_app.project_manager.manager import ProjectManager

    root = atlas.base_project_dir
    pm = ProjectManager.open(root) if (root / "project.json").exists() else ProjectManager.create(root, f"{atlas.name} (atlas base)")
    try:
        todo = [c for c in atlas.chunks if c.get("status") == "rendered"]
        for k, chunk in enumerate(todo):
            progress_cb(95 + int(4 * k / max(1, len(todo))), 100, f"Importing chunk {chunk['id']} videos…")
            for v in chunk.get("videos") or []:
                src = atlas.root / v["path"]
                if not src.exists():
                    continue
                asset = pm.add_video(src)
                pm.attach_auto_labels(asset.video_id, atlas.root / v["labels"])
                try:
                    src.unlink()  # the project keeps its own copy
                except OSError:
                    pass
            chunk["status"] = "imported"
            atlas.save()
        if not pm.scene_path.exists() or pm.class_names() != atlas.classes:
            first = next((c for c in atlas.chunks if (atlas.chunk_dir(c["id"]) / "scene.json").exists()), None)
            if first is None:
                raise RuntimeError("No chunk produced training videos")
            d = atlas.chunk_dir(first["id"])
            pm.set_scene(d / "scene.json", d / "scene_preview.jpg", d / "mosaic.jpg")
        assert pm.meta is not None
        pm.meta.model_type = atlas.model_type
        pm.meta.settings["atlas"] = {"root": str(atlas.root), "role": "base"}
        pm.save_meta()
        if not pm.meta.videos:
            raise RuntimeError("The base project has no videos")
    finally:
        pm.close()
    return root


def prepare_atlas_data(
    atlas: RegionAtlas,
    *,
    cache_dir: str | Path | None = None,
    progress_cb: Optional[ProgressCb] = None,
    should_stop: Optional[StopCb] = None,
) -> Path:
    """Run (or resume) stages 1–3. Returns the base project folder, ready for training."""
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    progress = progress_cb or (lambda d, t, m: None)
    stop = should_stop or (lambda: False)
    scan_landmarks(atlas, cache, progress, stop)
    render_chunks(atlas, cache, progress, stop)
    root = import_base_project(atlas, progress)
    progress(100, 100, f"Atlas data ready: {len(atlas.landmarks)} landmarks, {len(atlas.chunks)} chunks")
    return root


def finalize_base_model(atlas: RegionAtlas, best_pt: str | Path, summary: Optional[Dict[str, Any]] = None) -> Path:
    """Copy the trained weights into ``<atlas>/base_model/best.pt`` and record metrics."""
    import shutil

    dest = atlas.root / "base_model" / "best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, dest)
    atlas.base_weights = str(dest.relative_to(atlas.root))
    train = (summary or {}).get("train") or {}
    atlas.metrics = {k: v for k, v in train.items() if isinstance(v, (int, float, str)) and k != "best_weights"}
    atlas.save()
    return dest
