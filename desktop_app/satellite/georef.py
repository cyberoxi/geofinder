"""Georeferenced raster load + warp/crop into a mosaic pixel grid (EPSG:4326 SWNE)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from shared.logging import get_logger

logger = get_logger("desktop.satellite.georef")

Swne = Tuple[float, float, float, float]  # south, west, north, east


def warp_scene_to_mosaic(
    scene,
    fetch_swne: Swne,
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load scene local files and warp into BGR uint8 matching out_shape (H, W).
    Returns (image, coverage_mask_or_None). Mask is uint8 255 where data exists.
    """
    paths = list(scene.local_paths or [])
    if not paths:
        raise FileNotFoundError(f"No local files for scene {scene.id}")
    if scene.kind == "sar":
        return _warp_sar(paths, fetch_swne, out_shape)
    return _warp_hls_rgb(paths, fetch_swne, out_shape)


def _warp_hls_rgb(
    paths: Sequence[Path],
    fetch_swne: Swne,
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Stack B04/B03/B02 (or any 3 GeoTIFFs) into BGR and warp to SWNE."""
    # Order paths: prefer names containing B04, B03, B02
    ordered = _order_rgb_paths(paths)
    bands: List[np.ndarray] = []
    mask: Optional[np.ndarray] = None
    for p in ordered[:3]:
        arr, m = _reproject_band(p, fetch_swne, out_shape)
        bands.append(arr)
        if mask is None:
            mask = m
        else:
            mask = np.minimum(mask, m)
    while len(bands) < 3:
        bands.append(bands[-1] if bands else np.zeros(out_shape, dtype=np.float32))
    # HLS reflectance is often uint16 scaled; stretch each band
    stacked = np.stack(bands[:3], axis=-1)  # R,G,B float
    rgb = _stretch_rgb(stacked)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, mask


def _warp_sar(
    paths: Sequence[Path],
    fetch_swne: Swne,
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    tif = _resolve_sar_geotiff(paths[0])
    arr, mask = _reproject_band(tif, fetch_swne, out_shape)
    gray = _sar_to_uint8(arr)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr, mask


def _resolve_sar_geotiff(path: Path) -> Path:
    """If ASF gave a ZIP, extract first measurement .tif; else return path."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        return path
    if path.suffix.lower() == ".zip" or path.is_dir():
        extract_dir = path.with_suffix("") if path.suffix.lower() == ".zip" else path
        if path.suffix.lower() == ".zip":
            extract_dir.mkdir(parents=True, exist_ok=True)
            marker = extract_dir / ".extracted"
            if not marker.exists():
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(extract_dir)
                marker.write_text("ok", encoding="utf-8")
        # Prefer measurement/*.tif inside SAFE
        cands = list(extract_dir.rglob("*.tif")) + list(extract_dir.rglob("*.tiff"))
        # Prefer VV over VH, measurement folder
        def score(p: Path) -> tuple:
            n = p.name.lower()
            return (
                0 if "measurement" in str(p).lower() else 1,
                0 if "-vv-" in n or "_vv_" in n else 1,
                0 if "grd" in n else 1,
                len(n),
            )

        if not cands:
            raise FileNotFoundError(f"No GeoTIFF inside {path}")
        return sorted(cands, key=score)[0]
    raise FileNotFoundError(f"Unsupported SAR product path: {path}")


def _order_rgb_paths(paths: Sequence[Path]) -> List[Path]:
    by_band = {}
    rest = []
    for p in paths:
        name = p.name.upper()
        for b in ("B04", "B03", "B02"):
            if b in name:
                by_band[b] = p
                break
        else:
            rest.append(p)
    ordered = [by_band[b] for b in ("B04", "B03", "B02") if b in by_band]
    for p in rest:
        if p not in ordered:
            ordered.append(p)
    return ordered


def _reproject_band(
    path: Path,
    fetch_swne: Swne,
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproject one band into float32 array of out_shape over SWNE (EPSG:4326)."""
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject, transform_bounds
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is required to align archive GeoTIFFs.\n"
            "Install with: pip install rasterio"
        ) from exc

    south, west, north, east = fetch_swne
    h, w = int(out_shape[0]), int(out_shape[1])
    dst_transform = from_bounds(west, south, east, north, w, h)
    dst = np.zeros((h, w), dtype=np.float32)

    with rasterio.open(path) as src:
        src_crs = src.crs
        if src_crs is None:
            # Assume already geographic
            src_crs = "EPSG:4326"
        # Read first band
        source = src.read(1).astype(np.float32)
        nodata = src.nodata
        reproject(
            source=source,
            destination=dst,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            dst_nodata=0.0,
        )
        # Coverage: non-zero after warp (approx)
        mask = (np.abs(dst) > 0).astype(np.uint8) * 255
        # Also mark nodata zeros in empty areas — refine with bounds overlap
        try:
            sb = transform_bounds(src_crs, "EPSG:4326", *src.bounds, densify_pts=21)
            # sb: left, bottom, right, top
            if sb[2] < west or sb[0] > east or sb[3] < south or sb[1] > north:
                logger.warning("Scene %s does not overlap fetch bbox", path.name)
        except Exception:  # noqa: BLE001
            pass
    return dst, mask


def _stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    """Percentile stretch float RGB → uint8."""
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for c in range(3):
        band = rgb[:, :, c]
        valid = band[np.isfinite(band) & (band > 0)]
        if valid.size < 10:
            continue
        lo, hi = np.percentile(valid, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        scaled = (np.clip(band, lo, hi) - lo) / (hi - lo) * 255.0
        out[:, :, c] = scaled.astype(np.uint8)
    return out


def _sar_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Log-scale SAR amplitude/power to uint8."""
    x = arr.astype(np.float64)
    x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    # If values look like linear amplitude, use 10*log10
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if np.nanmedian(finite) > 1.5:
        db = 10.0 * np.log10(np.maximum(x, 1e-6))
    else:
        db = x
    valid = db[np.isfinite(db)]
    lo, hi = np.percentile(valid, (2, 98))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (np.clip(db, lo, hi) - lo) / (hi - lo) * 255.0
    scaled = np.nan_to_num(scaled, nan=0.0)
    return scaled.astype(np.uint8)
