"""Unified archive search / download for Earthdata + ASF scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from shared.logging import get_logger

logger = get_logger("desktop.satellite.archives")

ProgressCb = Callable[[int, int, str], None]

# Sensor keys used by UI checkboxes
SENSOR_HLS_S30 = "hls_s30"
SENSOR_HLS_L30 = "hls_l30"
SENSOR_S1_GRD = "s1_grd"

SENSOR_TITLES = {
    SENSOR_HLS_S30: "HLS Sentinel-2",
    SENSOR_HLS_L30: "HLS Landsat",
    SENSOR_S1_GRD: "Sentinel-1 SAR",
}


@dataclass
class ArchiveCredentials:
    """Earthdata Login — used by both CMR and ASF downloads."""

    username: str = ""
    password: str = ""
    token: str = ""

    def has_auth(self) -> bool:
        return bool(self.token.strip()) or bool(self.username.strip() and self.password)


@dataclass
class ArchiveScene:
    """One searchable / downloadable archive granule."""

    id: str
    sensor: str
    title: str
    date: str
    cloud_cover: Optional[float] = None
    download_urls: List[str] = field(default_factory=list)
    # For HLS: band name → URL (B02/B03/B04). Empty for SAR single product.
    band_urls: Dict[str, str] = field(default_factory=dict)
    kind: str = "optical"  # optical | sar
    platform: str = ""
    size_mb: Optional[float] = None
    local_paths: List[Path] = field(default_factory=list)
    browse_url: str = ""  # quick-look JPG/PNG (no full granule download)

    @property
    def label(self) -> str:
        cloud = f", cloud {self.cloud_cover:.0f}%" if self.cloud_cover is not None else ""
        return f"{self.title} {self.date[:10]}{cloud}"


def search_archives(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    sensors: Sequence[str],
    start_date: str,
    end_date: str,
    max_per_sensor: int = 5,
) -> List[ArchiveScene]:
    """Search selected archives for scenes overlapping the bbox (SWNE)."""
    from desktop_app.satellite.asf_client import search_sentinel1
    from desktop_app.satellite.earthdata_cmr import search_hls

    results: List[ArchiveScene] = []
    for sensor in sensors:
        try:
            if sensor in (SENSOR_HLS_S30, SENSOR_HLS_L30):
                results.extend(
                    search_hls(
                        south, west, north, east,
                        sensor=sensor,
                        start_date=start_date,
                        end_date=end_date,
                        max_results=max_per_sensor,
                    )
                )
            elif sensor == SENSOR_S1_GRD:
                results.extend(
                    search_sentinel1(
                        south, west, north, east,
                        start_date=start_date,
                        end_date=end_date,
                        max_results=max_per_sensor,
                    )
                )
            else:
                logger.warning("Unknown archive sensor: %s", sensor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Archive search failed for %s: %s", sensor, exc)
            raise
    results.sort(key=lambda s: (s.cloud_cover if s.cloud_cover is not None else 999.0, s.date), reverse=False)
    return results


def download_scene(
    scene: ArchiveScene,
    out_dir: str | Path,
    credentials: ArchiveCredentials,
    *,
    progress_cb: Optional[ProgressCb] = None,
) -> ArchiveScene:
    """Download granule files into out_dir; fills scene.local_paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if scene.sensor == SENSOR_S1_GRD:
        from desktop_app.satellite.asf_client import download_sentinel1

        paths = download_sentinel1(scene, out, credentials, progress_cb=progress_cb)
    else:
        from desktop_app.satellite.earthdata_cmr import download_hls

        paths = download_hls(scene, out, credentials, progress_cb=progress_cb)
    scene.local_paths = list(paths)
    return scene


def attach_archive_extras(
    prepared,
    scenes: Sequence[ArchiveScene],
    credentials: ArchiveCredentials,
    *,
    cache_dir: str | Path | None = None,
    progress_cb: Optional[ProgressCb] = None,
) -> int:
    """
    Download + warp selected archive scenes into prepared.extra_mosaics
    (same pixel grid as the primary mosaic). Returns number attached.
    """
    from desktop_app.satellite.georef import warp_scene_to_mosaic

    meta = prepared.layout.meta or {}
    fetch = meta.get("fetch_swne")
    if not fetch or len(fetch) != 4:
        raise RuntimeError("Prepared scene missing fetch_swne — cannot align archives")
    h, w = prepared.mosaic.shape[:2]
    root = Path(cache_dir) if cache_dir else Path("data") / "satellite_cache" / "archives"
    root.mkdir(parents=True, exist_ok=True)
    attached = 0
    n = len(scenes)
    for i, scene in enumerate(scenes):
        if progress_cb:
            progress_cb(i, n, f"Archive [{i + 1}/{n}] {scene.label}")
        try:
            scene_dir = root / scene.sensor / _safe_id(scene.id)
            if not scene.local_paths:
                download_scene(scene, scene_dir, credentials, progress_cb=None)
            img, mask = warp_scene_to_mosaic(
                scene, (float(fetch[0]), float(fetch[1]), float(fetch[2]), float(fetch[3])), (h, w)
            )
            name = scene.label
            # Avoid duplicate keys if same date appears twice
            existing = {n for n, _ in prepared.extra_mosaics}
            if name in existing:
                name = f"{name} ({scene.id[-8:]})"
            prepared.extra_mosaics.append((name, img))
            if mask is not None:
                prepared.extra_masks[name] = mask
            attached += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping archive scene %s: %s", scene.id, exc)
            if progress_cb:
                progress_cb(i + 1, n, f"Skipped {scene.title}: {exc}")
    if progress_cb:
        progress_cb(n, n, f"Attached {attached}/{n} archive scenes")
    return attached


def _safe_id(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]
