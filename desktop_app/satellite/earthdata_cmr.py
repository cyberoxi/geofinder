"""NASA CMR search + download for HLS optical granules (S30 / L30)."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from shared.logging import get_logger

logger = get_logger("desktop.satellite.earthdata_cmr")

ProgressCb = Callable[[int, int, str], None]

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
USER_AGENT = "geofinder-satellite-archives/1.0"

# HLS V2.0 LPCLOUD concept IDs
HLS_COLLECTIONS = {
    "hls_s30": {
        "concept_id": "C2021957295-LPCLOUD",
        "title": "HLS Sentinel-2",
        "short_name": "HLSS30",
    },
    "hls_l30": {
        "concept_id": "C2021957657-LPCLOUD",
        "title": "HLS Landsat",
        "short_name": "HLSL30",
    },
}

# True-color bands (HLS naming)
RGB_BANDS = ("B04", "B03", "B02")  # R, G, B


def search_hls(
    south: float,
    west: float,
    north: float,
    east: float,
    *,
    sensor: str,
    start_date: str,
    end_date: str,
    max_results: int = 5,
) -> List:
    """Search HLS granules overlapping SWNE bbox. Dates: YYYY-MM-DD."""
    from desktop_app.satellite.archives import ArchiveScene

    meta = HLS_COLLECTIONS.get(sensor)
    if not meta:
        raise ValueError(f"Unknown HLS sensor: {sensor}")
    start = _iso_start(start_date)
    end = _iso_end(end_date)
    # CMR bounding_box: west,south,east,north
    params = {
        "concept_id": meta["concept_id"],
        "bounding_box": f"{west},{south},{east},{north}",
        "temporal": f"{start},{end}",
        "page_size": str(max(1, min(int(max_results), 50))),
        "sort_key": "-start_date",
    }
    url = CMR_GRANULES + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    entries = data.get("feed", {}).get("entry", []) or []
    scenes: List[ArchiveScene] = []
    for entry in entries:
        band_urls = _band_urls_from_links(entry.get("links") or [])
        if not all(b in band_urls for b in RGB_BANDS):
            # Fallback: collect any GET DATA hrefs
            hrefs = [
                ln.get("href", "")
                for ln in (entry.get("links") or [])
                if ln.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"
                or "data" in str(ln.get("rel", "")).lower()
            ]
            if not hrefs:
                continue
            band_urls = _guess_rgb_from_hrefs(hrefs)
            if not all(b in band_urls for b in RGB_BANDS):
                continue
        cloud = _cloud_cover(entry)
        granule_id = str(entry.get("id") or entry.get("producer_granule_id") or entry.get("title") or "")
        date = str(entry.get("time_start") or start_date)[:19]
        scenes.append(
            ArchiveScene(
                id=granule_id,
                sensor=sensor,
                title=meta["title"],
                date=date,
                cloud_cover=cloud,
                band_urls=band_urls,
                download_urls=[band_urls[b] for b in RGB_BANDS if b in band_urls],
                kind="optical",
                platform=meta["short_name"],
                size_mb=_granule_size_mb(entry),
                browse_url=_browse_url(entry.get("links") or []),
            )
        )
        if len(scenes) >= max_results:
            break
    # Prefer low cloud cover
    scenes.sort(key=lambda s: (s.cloud_cover if s.cloud_cover is not None else 999.0, s.date))
    return scenes[:max_results]


def download_hls(
    scene,
    out_dir: Path,
    credentials,
    *,
    progress_cb: Optional[ProgressCb] = None,
) -> List[Path]:
    """Download B04/B03/B02 COGs for an HLS scene. Requires Earthdata auth."""
    if not credentials.has_auth():
        raise RuntimeError(
            "Earthdata Login required for HLS download.\n"
            "Create an account at https://urs.earthdata.nasa.gov/ and enter credentials."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    bands = list(RGB_BANDS)
    urls = scene.band_urls or {}
    for i, band in enumerate(bands):
        url = urls.get(band)
        if not url:
            raise RuntimeError(f"Missing {band} for {scene.id}")
        dest = out_dir / f"{_safe(scene.id)}_{band}.tif"
        if progress_cb:
            progress_cb(i, len(bands), f"Downloading {band}…")
        if dest.exists() and dest.stat().st_size > 1000:
            paths.append(dest)
            continue
        _download_authenticated(url, dest, credentials)
        paths.append(dest)
    if progress_cb:
        progress_cb(len(bands), len(bands), "HLS bands ready")
    return paths


def _download_authenticated(url: str, dest: Path, credentials) -> None:
    """Download with Earthdata Bearer token or basic auth (redirect-aware)."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests package required for Earthdata downloads") from exc

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    if credentials.token.strip():
        session.headers["Authorization"] = f"Bearer {credentials.token.strip()}"
    else:
        session.auth = (credentials.username.strip(), credentials.password)

    # Follow redirects; URS may ask for auth
    resp = session.get(url, stream=True, timeout=120, allow_redirects=True)
    if resp.status_code in (401, 403) and not credentials.token.strip():
        # Retry via URS-style basic on final URL
        resp.close()
        resp = session.get(url, stream=True, timeout=120, allow_redirects=True)
    if resp.status_code >= 400:
        resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    tmp.replace(dest)
    resp.close()


def _band_urls_from_links(links: Sequence[dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ln in links:
        href = ln.get("href") or ""
        if not href:
            continue
        rel = str(ln.get("rel") or "")
        title = str(ln.get("title") or "")
        if "data" not in rel.lower() and "GET DATA" not in title.upper() and not href.lower().endswith((".tif", ".tiff")):
            # Still accept COG hrefs that name a band
            if not re.search(r"B\d{2}", href, re.I):
                continue
        m = re.search(r"\.(B\d{2}[A-Za-z]?)\.", href) or re.search(r"_(B\d{2}[A-Za-z]?)(?:_|\.|$)", href)
        if not m:
            m = re.search(r"(B\d{2})", title)
        if m:
            band = m.group(1).upper()
            if band.startswith("B") and len(band) >= 3:
                band = band[:3] if band[1:3].isdigit() else band
            out[band] = href
    return out


def _guess_rgb_from_hrefs(hrefs: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for href in hrefs:
        m = re.search(r"\.(B\d{2})\.", href) or re.search(r"_(B\d{2})(?:_|\.|$)", href)
        if m:
            out[m.group(1).upper()] = href
    return out


def _browse_url(links: Sequence[dict]) -> str:
    """First HTTP(S) browse / quicklook URL from CMR links."""
    for ln in links:
        rel = str(ln.get("rel") or "").lower()
        href = str(ln.get("href") or "")
        if not href.startswith("http"):
            continue
        if "browse" in rel or href.lower().endswith((".jpg", ".jpeg", ".png")):
            if "browse" in rel or "/browse" in href.lower() or href.lower().endswith((".jpg", ".jpeg", ".png")):
                return href
    for ln in links:
        href = str(ln.get("href") or "")
        if href.startswith("http") and href.lower().endswith((".jpg", ".jpeg", ".png")):
            return href
    return ""


def _cloud_cover(entry: dict) -> Optional[float]:
    for key in ("cloud_cover", "cloudCover"):
        if key in entry:
            try:
                return float(entry[key])
            except (TypeError, ValueError):
                pass
    # UMM-style nested
    for attr in entry.get("umm", {}).get("AdditionalAttributes", []) if isinstance(entry.get("umm"), dict) else []:
        if str(attr.get("Name", "")).lower() in ("cloudcover", "cloud_cover", "cloud cover"):
            vals = attr.get("Values") or []
            if vals:
                try:
                    return float(vals[0])
                except (TypeError, ValueError):
                    pass
    return None


def _granule_size_mb(entry: dict) -> Optional[float]:
    raw = entry.get("granule_size") or entry.get("granuleSizeMB")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _iso_start(d: str) -> str:
    d = d.strip()
    if "T" in d:
        return d if d.endswith("Z") else d + "Z"
    return f"{d}T00:00:00Z"


def _iso_end(d: str) -> str:
    d = d.strip()
    if "T" in d:
        return d if d.endswith("Z") else d + "Z"
    return f"{d}T23:59:59Z"


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:60]
