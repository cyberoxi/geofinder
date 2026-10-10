"""ASF Search API — Sentinel-1 GRD scenes overlapping a bbox."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

from shared.logging import get_logger

logger = get_logger("desktop.satellite.asf_client")

ProgressCb = Callable[[int, int, str], None]


def search_sentinel1(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    start_date: str,
    end_date: str,
    max_results: int = 5,
) -> List:
    """Search Sentinel-1 IW GRD products intersecting SWNE. Dates: YYYY-MM-DD."""
    from desktop_app.satellite.archives import ArchiveScene, SENSOR_S1_GRD

    try:
        import asf_search as asf
    except ImportError as exc:
        raise RuntimeError(
            "asf_search is required for Sentinel-1 archive search.\n"
            "Install with: pip install asf_search"
        ) from exc

    wkt = (
        f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    )
    start = _date(start_date)
    end = _date(end_date)
    results = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=["GRD_HD", "GRD_MD", "GRD_HS", "GRD_MS"],
        intersectsWith=wkt,
        start=start,
        end=end,
        maxResults=max(1, min(int(max_results) * 2, 50)),
        beamMode=["IW"],
    )
    scenes: List[ArchiveScene] = []
    for prod in results:
        props = getattr(prod, "properties", None) or {}
        if isinstance(props, dict):
            pid = str(props.get("fileID") or props.get("sceneName") or props.get("url") or id(prod))
            date = str(props.get("startTime") or props.get("stopTime") or start_date)[:19]
            url = str(props.get("url") or "")
            size = props.get("bytes")
            size_mb = float(size) / (1024 * 1024) if size else None
            pol = str(props.get("polarization") or "")
            browse = _asf_browse(props.get("browse"))
        else:
            pid = str(getattr(prod, "fileID", None) or getattr(prod, "sceneName", id(prod)))
            date = str(getattr(prod, "startTime", start_date))[:19]
            url = str(getattr(prod, "url", "") or "")
            size_mb = None
            pol = ""
            browse = _asf_browse(getattr(prod, "browse", None))
        if not url:
            # asf_search Product has .download_url or properties
            url = str(getattr(prod, "download_url", "") or "")
        if not url:
            continue
        title = "Sentinel-1 SAR"
        if pol:
            title = f"Sentinel-1 SAR ({pol})"
        scenes.append(
            ArchiveScene(
                id=pid,
                sensor=SENSOR_S1_GRD,
                title=title,
                date=date,
                cloud_cover=None,
                download_urls=[url],
                kind="sar",
                platform="SENTINEL-1",
                size_mb=size_mb,
                browse_url=browse,
            )
        )
        if len(scenes) >= max_results:
            break
    scenes.sort(key=lambda s: s.date, reverse=True)
    return scenes[:max_results]


def download_sentinel1(
    scene,
    out_dir: Path,
    credentials,
    *,
    progress_cb: Optional[ProgressCb] = None,
) -> List[Path]:
    """Download one Sentinel-1 GRD product with Earthdata credentials."""
    if not credentials.has_auth():
        raise RuntimeError(
            "Earthdata Login required for ASF / Sentinel-1 download.\n"
            "Create an account at https://urs.earthdata.nasa.gov/ and authorize ASF at "
            "https://search.asf.alaska.edu/"
        )
    try:
        import asf_search as asf
    except ImportError as exc:
        raise RuntimeError("asf_search is required for Sentinel-1 download") from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0, 1, f"Downloading {scene.title}…")

    session = asf.ASFSession()
    if credentials.token.strip():
        session.auth_with_token(credentials.token.strip())
    else:
        session.auth_with_creds(credentials.username.strip(), credentials.password)

    urls = list(scene.download_urls)
    if not urls:
        raise RuntimeError(f"No download URL for {scene.id}")

    before = {p.name for p in out_dir.iterdir()} if out_dir.exists() else set()
    asf.download_urls(urls=urls, path=str(out_dir), session=session, processes=1)
    after = [p for p in out_dir.iterdir() if p.name not in before or p.suffix.lower() in (".zip", ".tif", ".tiff", ".SAFE")]
    # Prefer newly written; else any zip/tif in dir
    paths = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        paths = sorted(
            [p for p in out_dir.iterdir() if p.suffix.lower() in (".zip", ".tif", ".tiff")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not paths:
        raise RuntimeError(f"ASF download produced no files in {out_dir}")
    if progress_cb:
        progress_cb(1, 1, "Sentinel-1 product ready")
    return paths[:1]


def _date(d: str) -> str:
    d = d.strip()
    if "T" in d:
        return d
    return f"{d}T00:00:00Z"


def _asf_browse(raw) -> str:
    """Normalize ASF browse field (str | list | dict) to one HTTP URL."""
    if not raw:
        return ""
    if isinstance(raw, str) and raw.startswith("http"):
        return raw
    if isinstance(raw, (list, tuple)):
        for item in raw:
            got = _asf_browse(item)
            if got:
                return got
        return ""
    if isinstance(raw, dict):
        for key in ("url", "href", "browse"):
            v = raw.get(key)
            if isinstance(v, str) and v.startswith("http"):
                return v
    return ""
