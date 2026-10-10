"""Fast region / browse previews for the satellite compare gallery (no full granules)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np

from shared.logging import get_logger

logger = get_logger("desktop.satellite.preview_fetch")

USER_AGENT = "geofinder-satellite-preview/1.0"

# Basemap layers shown side-by-side for the same bbox (XYZ, no login)
COMPARE_BASEMAPS: List[Tuple[str, str]] = [
    ("satellite", "Esri World Imagery"),
    ("esri_clarity", "Esri Clarity"),
]


def fetch_browse_image(url: str, *, timeout: float = 45.0, max_side: int = 512) -> Optional[np.ndarray]:
    """Download a browse/quicklook JPG/PNG → BGR, optionally downscale."""
    if not url or not url.startswith("http"):
        return None
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return _fit(img, max_side)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Browse fetch failed (%s): %s", url[:80], exc)
        return None


def fetch_basemap_preview(
    south: float,
    west: float,
    north: float,
    east: float,
    layer: str,
    *,
    cache_dir: str | Path | None = None,
    max_side: int = 512,
    max_tiles: int = 36,
) -> Optional[np.ndarray]:
    """Low-res XYZ mosaic of the region for gallery compare."""
    from desktop_app.satellite.tiles import fetch_mosaic, fit_zoom_for_bbox, layer_max_zoom

    cache = Path(cache_dir) if cache_dir else Path("data") / "satellite_cache"
    try:
        zoom = min(14, layer_max_zoom(layer))
        zoom = fit_zoom_for_bbox(south, west, north, east, zoom, max_tiles=max_tiles)
        mosaic = fetch_mosaic(
            south, west, north, east,
            zoom=zoom,
            cache_dir=cache,
            max_tiles=max_tiles,
            layer=layer,
        )
        return _fit(mosaic, max_side)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Basemap preview %s failed: %s", layer, exc)
        return None


def load_gallery_previews(
    south: float,
    west: float,
    north: float,
    east: float,
    scenes: Sequence,
    *,
    basemap_layers: Optional[Sequence[Tuple[str, str]]] = None,
    include_basemaps: bool = True,
    cache_dir: str | Path | None = None,
    progress_cb=None,
) -> Dict[str, np.ndarray]:
    """
    Build id → BGR thumbnail map.
    Basemap keys are ``basemap:<layer>``; scene keys are ``scene:<id>``.
    ``basemap_layers`` overrides the default COMPARE_BASEMAPS list when provided.
    """
    out: Dict[str, np.ndarray] = {}
    items: List[Tuple[str, str, Optional[object]]] = []
    if include_basemaps:
        layers = list(basemap_layers) if basemap_layers is not None else list(COMPARE_BASEMAPS)
        for layer, _title in layers:
            items.append((f"basemap:{layer}", "basemap", layer))
    for sc in scenes:
        items.append((f"scene:{sc.id}", "scene", sc))

    n = max(1, len(items))
    for i, (key, kind, payload) in enumerate(items):
        if progress_cb:
            progress_cb(i, n, f"Loading preview {i + 1}/{n}…")
        img = None
        if kind == "basemap":
            img = fetch_basemap_preview(south, west, north, east, str(payload), cache_dir=cache_dir)
        else:
            browse = getattr(payload, "browse_url", "") or ""
            img = fetch_browse_image(browse) if browse else None
        if img is not None:
            out[key] = img
    if progress_cb:
        progress_cb(n, n, f"Loaded {len(out)} preview(s)")
    return out


def _fit(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
