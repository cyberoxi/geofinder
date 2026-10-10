"""Prepare satellite scenes (imagery + landmarks) for labeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop_app.satellite.tiles import (
    LAYERS,
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
    polygon_bbox_px,
    save_candidates,
    selection_polygon_px,
    selection_rect_px,
)
from shared.landmarks import SceneLayout
from shared.logging import get_logger

logger = get_logger("desktop.satellite")

ProgressCb = Callable[[int, int, str], None]
FrameCb = Callable[[np.ndarray], None]
DEFAULT_ZOOM_OUT_FACTOR = 12.0


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
    target_lonlat: Optional[Sequence[Sequence[float]]] = None,
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
    target_poly_px: Optional[List[List[float]]] = None
    if target_lonlat and len(target_lonlat) >= 3:
        target_poly_px = selection_polygon_px(target_lonlat, fetch, fetch_zoom, mosaic.shape)
        target = polygon_bbox_px(target_poly_px)
    else:
        target = selection_rect_px(selection, fetch, fetch_zoom, mosaic.shape)
    if progress_cb:
        progress_cb(72, 100, "Detecting landmark objects (fields, buildings, roundabouts…)")
    layout, candidates = build_scene(
        mosaic,
        target,
        target_polygon=target_poly_px,
        num_landmarks=num_landmarks,
        target_class=target_class,
        meters_per_px=meters_per_pixel((south + north) / 2.0, fetch_zoom),
        meta={
            "selection_swne": list(selection),
            "selection_lonlat": [list(map(float, p)) for p in (target_lonlat or [])],
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
        for name in ("mosaic.jpg", "satellite_mosaic.jpg"):
            cand = d / name
            if cand.exists():
                mosaic_path = cand
                break
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
