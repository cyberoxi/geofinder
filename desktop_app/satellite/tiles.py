"""Esri / OSM tile download and mosaic stitching."""

from __future__ import annotations

import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from shared.logging import get_logger

logger = get_logger("desktop.satellite.tiles")

ProgressCb = Callable[[int, int, str], None]
TILE_SIZE = 256
USER_AGENT = "geofinder-satellite-video/1.0 (research; contact: local-dev)"

# Basemap layers for preview / mosaic. Video generation always uses satellite.
LAYERS: Dict[str, Dict[str, str]] = {
    "satellite": {
        "url": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "attribution": "Esri World Imagery",
        "cache": "esri_imagery",
    },
    "map": {
        "url": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        "attribution": "Esri Street Map",
        "cache": "esri_streets",
    },
}

# Back-compat alias
ESRI_URL = LAYERS["satellite"]["url"]


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[float, float]:
    """Web Mercator fractional tile coordinates."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x: float, y: float, zoom: int) -> Tuple[float, float]:
    """Convert fractional Web Mercator tile coords to (lon, lat)."""
    n = 2.0**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def bbox_from_center(lat: float, lon: float, span: float) -> Tuple[float, float, float, float]:
    """Return (south, west, north, east) from center + span in degrees."""
    half = span / 2.0
    return lat - half, lon - half, lat + half, lon + half


def expand_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    factor: float,
) -> Tuple[float, float, float, float]:
    """Grow bbox about its center by ``factor`` (≥1). Clamps latitude to Web Mercator range."""
    factor = max(1.0, float(factor))
    lat_c = (south + north) / 2.0
    lon_c = (west + east) / 2.0
    half_lat = (north - south) / 2.0 * factor
    half_lon = (east - west) / 2.0 * factor
    s = max(-85.05112878, lat_c - half_lat)
    n = min(85.05112878, lat_c + half_lat)
    w = max(-180.0, lon_c - half_lon)
    e = min(180.0, lon_c + half_lon)
    if n <= s:
        s, n = max(-85.0, lat_c - 0.01), min(85.0, lat_c + 0.01)
    if e <= w:
        w, e = max(-180.0, lon_c - 0.01), min(180.0, lon_c + 0.01)
    return s, w, n, e


def count_tiles(south: float, west: float, north: float, east: float, zoom: int) -> int:
    """Approximate tile count needed to cover bbox at zoom."""
    x0, y0 = lonlat_to_tile(west, north, zoom)
    x1, y1 = lonlat_to_tile(east, south, zoom)
    n_x = int(math.floor(x1)) - int(math.floor(x0)) + 1
    n_y = int(math.floor(y1)) - int(math.floor(y0)) + 1
    return max(0, n_x) * max(0, n_y)


def fit_zoom_for_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    zoom: int,
    max_tiles: int = 400,
) -> int:
    """Lower tile zoom until bbox fits within ``max_tiles`` (minimum zoom 3)."""
    z = int(max(1, min(20, zoom)))
    while z > 3 and count_tiles(south, west, north, east, z) > max_tiles:
        z -= 1
    if count_tiles(south, west, north, east, z) > max_tiles:
        raise ValueError(
            f"Region needs more than {max_tiles} tiles even at zoom {z}. "
            "Select a smaller area or lower Download zoom / Zoom depth."
        )
    return z


def _tile_path(cache_dir: Path, z: int, x: int, y: int) -> Path:
    return cache_dir / str(z) / str(x) / f"{y}.jpg"


def _layer_cache_dir(cache_dir: Path, layer: str) -> Path:
    if layer not in LAYERS:
        raise ValueError(f"Unknown layer {layer!r}; expected one of {list(LAYERS)}")
    return Path(cache_dir) / LAYERS[layer]["cache"]


def try_load_cached_tile(
    z: int,
    x: int,
    y: int,
    cache_dir: Path,
    layer: str = "satellite",
) -> Optional[np.ndarray]:
    """Return BGR tile from disk cache only; None if missing/unreadable (no network)."""
    path = _tile_path(_layer_cache_dir(cache_dir, layer), z, x, y)
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def download_tile(
    z: int,
    x: int,
    y: int,
    cache_dir: Path,
    retries: int = 3,
    layer: str = "satellite",
    timeout: float = 15.0,
) -> np.ndarray:
    """Fetch one tile as BGR image; uses disk cache under cache_dir/layer."""
    cached = try_load_cached_tile(z, x, y, cache_dir, layer=layer)
    if cached is not None:
        return cached

    cfg = LAYERS[layer]
    layer_cache = _layer_cache_dir(cache_dir, layer)
    path = _tile_path(layer_cache, z, x, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["url"].format(z=z, y=y, x=x)
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to decode tile z={z} x={x} y={y}")
            cv2.imwrite(str(path), img)
            return img
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            last_err = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Tile download failed {url}: {last_err}") from last_err


def fetch_mosaic(
    south: float,
    west: float,
    north: float,
    east: float,
    zoom: int = 17,
    cache_dir: str | Path | None = None,
    max_tiles: int = 400,
    layer: str = "satellite",
    progress_cb: Optional[ProgressCb] = None,
    preview_cb: Optional[Callable[[np.ndarray], None]] = None,
) -> np.ndarray:
    """
    Download and stitch tiles covering the bbox (default: Esri World Imagery).
    Returns BGR mosaic (uint8).
    """
    if north <= south or east <= west:
        raise ValueError("Invalid bbox: need north>south and east>west")
    if zoom < 1 or zoom > 20:
        raise ValueError("zoom must be in 1..20")

    root = Path(cache_dir) if cache_dir else Path("data") / "satellite_cache"
    root.mkdir(parents=True, exist_ok=True)

    x0, y0 = lonlat_to_tile(west, north, zoom)  # NW
    x1, y1 = lonlat_to_tile(east, south, zoom)  # SE
    tx0, ty0 = int(math.floor(x0)), int(math.floor(y0))
    tx1, ty1 = int(math.floor(x1)), int(math.floor(y1))
    n_x = tx1 - tx0 + 1
    n_y = ty1 - ty0 + 1
    total = n_x * n_y
    if total > max_tiles:
        raise ValueError(
            f"Region needs {total} tiles (max {max_tiles}). "
            "Reduce span or lower zoom."
        )
    if total < 1:
        raise ValueError("Empty tile range")

    logger.info("Fetching %d tiles at z=%d (%dx%d) layer=%s", total, zoom, n_x, n_y, layer)
    if progress_cb:
        progress_cb(0, total, f"Downloading tiles 0/{total}")
    mosaic = np.zeros((n_y * TILE_SIZE, n_x * TILE_SIZE, 3), dtype=np.uint8)
    done = 0
    preview_every = max(1, total // 20)

    def _emit_preview() -> None:
        if not preview_cb:
            return
        # Downscale partial mosaic for UI preview
        h, w = mosaic.shape[:2]
        scale = min(1.0, 640.0 / max(w, h))
        if scale < 1.0:
            thumb = cv2.resize(
                mosaic,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            thumb = mosaic.copy()
        preview_cb(thumb)

    for iy, ty in enumerate(range(ty0, ty1 + 1)):
        for ix, tx in enumerate(range(tx0, tx1 + 1)):
            tile = download_tile(zoom, tx, ty, root, layer=layer)
            if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                tile = cv2.resize(tile, (TILE_SIZE, TILE_SIZE))
            y_off = iy * TILE_SIZE
            x_off = ix * TILE_SIZE
            mosaic[y_off : y_off + TILE_SIZE, x_off : x_off + TILE_SIZE] = tile
            done += 1
            if done % 20 == 0 or done == total:
                logger.info("Tiles %d/%d", done, total)
            if progress_cb and (done == 1 or done % 5 == 0 or done == total):
                progress_cb(done, total, f"Downloading tiles {done}/{total}")
            if preview_cb and (done == 1 or done % preview_every == 0 or done == total):
                _emit_preview()

    px0 = int((x0 - tx0) * TILE_SIZE)
    py0 = int((y0 - ty0) * TILE_SIZE)
    px1 = int((x1 - tx0) * TILE_SIZE)
    py1 = int((y1 - ty0) * TILE_SIZE)
    px0, px1 = max(0, px0), min(mosaic.shape[1], max(px0 + 1, px1))
    py0, py1 = max(0, py0), min(mosaic.shape[0], max(py0 + 1, py1))
    cropped = mosaic[py0:py1, px0:px1]
    if cropped.size == 0:
        raise RuntimeError("Mosaic crop empty")
    if preview_cb:
        h, w = cropped.shape[:2]
        scale = min(1.0, 640.0 / max(w, h))
        if scale < 1.0:
            preview_cb(
                cv2.resize(
                    cropped,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            )
        else:
            preview_cb(cropped.copy())
    return cropped
