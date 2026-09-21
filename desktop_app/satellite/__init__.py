"""Satellite imagery → synthetic video helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop_app.satellite.tiles import (
    LAYERS,
    bbox_from_center,
    expand_bbox,
    fetch_mosaic,
    fit_zoom_for_bbox,
    layer_max_zoom,
)
from desktop_app.satellite.scene import (
    build_scene,
    draw_scene_preview,
    load_candidates,
    meters_per_pixel,
    save_candidates,
    selection_rect_px,
)
from shared.landmarks import SceneLayout
from desktop_app.satellite.synth import (
    MosaicCamera,
    Photometric,
    make_night_mosaic,
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


def _safe(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:60]


@dataclass
class PreparedScene:
    """Downloaded mosaic + detected landmarks, ready for review and rendering."""

    mosaic: np.ndarray
    layout: SceneLayout
    candidates: List[Any]
    out_dir: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    source_name: str = "primary"
    extra_mosaics: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    extra_masks: Dict[str, np.ndarray] = field(default_factory=dict)  # coverage of partial images

    def all_mosaics(self) -> List[Tuple[str, np.ndarray, Optional[np.ndarray]]]:
        """Primary + pixel-aligned extra sources/dates (+ coverage mask), for render_scene_videos."""
        return [(self.source_name, self.mosaic, None)] + [
            (n, m, self.extra_masks.get(n)) for n, m in self.extra_mosaics
        ]

    def save_extras(self) -> None:
        for f in self.out_dir.glob("mosaic_extra_*"):
            f.unlink()
        for name, img in self.extra_mosaics:
            cv2.imwrite(str(self.out_dir / f"mosaic_extra_{_safe(name)}.jpg"), img)
            if name in self.extra_masks:
                cv2.imwrite(str(self.out_dir / f"mosaic_extra_{_safe(name)}.mask.png"), self.extra_masks[name])

    @property
    def mosaic_path(self) -> Path:
        return self.out_dir / "satellite_mosaic.jpg"

    @property
    def scene_path(self) -> Path:
        return self.out_dir / "scene.json"

    @property
    def candidates_path(self) -> Path:
        return self.out_dir / "candidates.json"

    @property
    def preview_path(self) -> Path:
        return self.out_dir / "scene_preview.jpg"

    def save_layout(self, layout: SceneLayout) -> None:
        self.layout = layout
        layout.save(self.scene_path)
        cv2.imwrite(str(self.preview_path), draw_scene_preview(self.mosaic, layout))

    def paths(self) -> Dict[str, Any]:
        return {
            "extra_mosaics": [self.out_dir / f"mosaic_extra_{_safe(n)}.jpg" for n, _ in self.extra_mosaics],
            "mosaic": self.mosaic_path,
            "scene": self.scene_path,
            "scene_preview": self.preview_path,
            "candidates": self.candidates_path,
        }


def prepare_scene(
    *,
    south: float,
    west: float,
    north: float,
    east: float,
    zoom: int = 17,
    zoom_out_factor: float = DEFAULT_ZOOM_OUT_FACTOR,
    out_dir: str | Path = "data/output",
    cache_dir: str | Path | None = None,
    num_landmarks: int = 8,
    target_class: str = "target_region",
    use_sam: bool = True,
    max_tiles: int = 900,
    layer: str = "satellite",
    extra_layers: Sequence[str] = (),
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
) -> PreparedScene:
    """
    Stage 1: fetch imagery around the selection and detect landmark objects.

    ``layer``: imagery source (see tiles.LAYERS / mosaic_layers()).  ``extra_layers``:
    more sources or Wayback dates fetched for the *same* pixels — rendered as extra
    training videos (other seasons / sensors) with the same labels.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection = (south, west, north, east)
    factor = max(1.0, float(zoom_out_factor))
    fetch = expand_bbox(south, west, north, east, factor)
    zoom = min(int(zoom), layer_max_zoom(layer), *[layer_max_zoom(x) for x in extra_layers])
    fetch_zoom = fit_zoom_for_bbox(*fetch, zoom, max_tiles=max_tiles)
    if fetch_zoom != zoom:
        logger.info("Lowered download zoom %d → %d to fit expanded region (factor=%.1f)", zoom, fetch_zoom, factor)

    def tile_progress(done: int, total: int, msg: str) -> None:
        if progress_cb and total > 0:
            progress_cb(int(70 * done / total), 100, msg)

    n_src = 1 + len(extra_layers)

    def src_progress(k: int):
        def _cb(done: int, total: int, msg: str) -> None:
            if progress_cb and total > 0:
                progress_cb(int(70 * (k + done / total) / n_src), 100, f"[{k + 1}/{n_src}] {msg}")

        return _cb

    mosaic = fetch_mosaic(
        *fetch, zoom=fetch_zoom, cache_dir=cache_dir, max_tiles=max_tiles, layer=layer,
        progress_cb=src_progress(0), preview_cb=frame_cb,
    )
    extras: List[Tuple[str, np.ndarray]] = []
    for k, xl in enumerate(extra_layers, start=1):
        try:
            img = fetch_mosaic(
                *fetch, zoom=fetch_zoom, cache_dir=cache_dir, max_tiles=max_tiles, layer=xl, progress_cb=src_progress(k)
            )
            if img.shape[:2] != mosaic.shape[:2]:
                img = cv2.resize(img, (mosaic.shape[1], mosaic.shape[0]))
            extras.append((LAYERS[xl].get("title", xl), img))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extra imagery source %s skipped: %s", xl, exc)
    target = selection_rect_px(selection, fetch, fetch_zoom, mosaic.shape)
    if progress_cb:
        progress_cb(72, 100, "Detecting landmark objects (fields, buildings, roundabouts…)")
    layout, candidates = build_scene(
        mosaic,
        target,
        num_landmarks=num_landmarks,
        target_class=target_class,
        meters_per_px=meters_per_pixel((south + north) / 2.0, fetch_zoom),
        meta={
            "selection_swne": list(selection),
            "fetch_swne": list(fetch),
            "tile_zoom": fetch_zoom,
            "source": layer,
            "attribution": "; ".join(
                [LAYERS[layer].get("attribution", layer)] + [LAYERS[x].get("attribution", x) for x in extra_layers]
            ),
        },
        use_sam=use_sam,
    )
    prepared = PreparedScene(
        mosaic=mosaic,
        layout=layout,
        candidates=candidates,
        out_dir=out,
        source_name=LAYERS[layer].get("title", layer),
        extra_mosaics=extras,
    )
    cv2.imwrite(str(prepared.mosaic_path), mosaic)
    prepared.save_extras()
    save_candidates(prepared.candidates_path, candidates)
    prepared.save_layout(layout)
    if frame_cb:
        frame_cb(cv2.imread(str(prepared.preview_path)))
    if progress_cb:
        progress_cb(100, 100, f"Found {len(candidates)} candidate objects, selected {layout.num_landmarks} landmarks")
    return prepared


def read_world_file_mpp(image_path: str | Path) -> Optional[float]:
    """Metres per pixel from an ESRI world file (.tfw/.jgw/.pgw/.wld) next to the image, if any."""
    import math as _m

    ip = Path(image_path)
    ext = ip.suffix.lower()
    cands = [ip.with_suffix(ext[:2] + ext[-1] + "w"), ip.with_suffix(ext + "w"), ip.with_suffix(".wld")]
    for wf in cands:
        if wf.exists():
            try:
                vals = [float(v) for v in wf.read_text().split()[:6]]
            except ValueError:
                continue
            px = abs(vals[0])
            if px < 0.01:  # degrees (EPSG:4326) → metres at the image latitude
                return px * 111320.0 * _m.cos(_m.radians(abs(vals[5])))
            return px
    return None


def prepare_scene_from_image(
    image: "str | Path | np.ndarray",
    target_rect: Tuple[float, float, float, float],
    *,
    meters_per_px: float,
    out_dir: str | Path,
    num_landmarks: int = 8,
    target_class: str = "target_region",
    use_sam: bool = True,
    source_name: str = "own image",
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
) -> PreparedScene:
    """Stage 1 for your own satellite / aerial image (any sensor): target given in image pixels."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(image, (str, Path)):
        mosaic = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if mosaic is None:
            raise FileNotFoundError(f"Cannot read image {image}")
        source_name = Path(image).name
    else:
        mosaic = image
    if progress_cb:
        progress_cb(10, 100, "Detecting landmark objects (fields, buildings, roundabouts…)")
    layout, candidates = build_scene(
        mosaic,
        tuple(map(float, target_rect)),
        num_landmarks=num_landmarks,
        target_class=target_class,
        meters_per_px=float(meters_per_px),
        meta={"source": source_name, "attribution": source_name},
        use_sam=use_sam,
    )
    prepared = PreparedScene(
        mosaic=mosaic, layout=layout, candidates=candidates, out_dir=out, source_name=source_name
    )
    cv2.imwrite(str(prepared.mosaic_path), mosaic)
    save_candidates(prepared.candidates_path, candidates)
    prepared.save_layout(layout)
    if frame_cb:
        frame_cb(cv2.imread(str(prepared.preview_path)))
    if progress_cb:
        progress_cb(100, 100, f"Found {len(candidates)} candidate objects, selected {layout.num_landmarks} landmarks")
    return prepared


def load_prepared_scene(scene_dir: str | Path, mosaic_name: str = "satellite_mosaic.jpg") -> PreparedScene:
    """Re-open a saved scene (mosaic + scene.json + candidates.json) for editing."""
    d = Path(scene_dir)
    mosaic_path = d / mosaic_name
    if not mosaic_path.exists():
        mosaic_path = d / "mosaic.jpg"
    mosaic = cv2.imread(str(mosaic_path))
    if mosaic is None:
        raise FileNotFoundError(f"No mosaic image in {d}")
    layout = SceneLayout.load(d / "scene.json")
    extras = []
    masks: Dict[str, np.ndarray] = {}
    for f in sorted(d.glob("mosaic_extra_*.jpg")):
        img = cv2.imread(str(f))
        if img is None:
            continue
        name = f.stem.replace("mosaic_extra_", "")
        if name.startswith("night_"):
            name = "night:" + name[len("night_"):]
        extras.append((name, img))
        mf = f.with_name(f.stem + ".mask.png")
        if mf.exists():
            masks[name] = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
    return PreparedScene(
        mosaic=mosaic,
        layout=layout,
        candidates=load_candidates(d / "candidates.json"),
        out_dir=d,
        source_name=str(layout.meta.get("source", "primary")),
        extra_mosaics=extras,
        extra_masks=masks,
    )


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
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
) -> Dict[str, Any]:
    """
    Stage 2: render varied videos with exact per-frame labels for every scene object.

    ``mosaic`` may be one image or a list of (source_name, image) that are pixel-aligned
    (same area / zoom from different sources or dates); videos cycle through them.
    ``night_fraction`` / ``ir_fraction`` of the videos use simulated night / thermal views.
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
        if light == "day":
            photo = Photometric.random(rng) if plan.get("photo") else None
        else:  # low-light sensor: noise + softness, no daylight colour jitter
            photo = Photometric(noise=rng.uniform(4, 12), blur=rng.choice([0, 3, 3, 5]))
        tag = "" if light == "day" else f"_{light}"
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
            extra_meta={"light": light, "source": sources[src_idx][0]},
            keep_classes=keeps[src_idx],
        )
        videos.append({"path": vpath, "labels": lpath, "kind": plan["kind"], "light": light, "source": sources[src_idx][0]})
        results.setdefault(plan["kind"], vpath)
    results["videos"] = videos
    if progress_cb:
        progress_cb(100, 100, "Done")
    return results
