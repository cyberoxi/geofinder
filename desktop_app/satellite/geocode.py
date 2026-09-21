"""Place search via OpenStreetMap Nominatim (no API key)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from desktop_app.satellite.tiles import USER_AGENT
from shared.logging import get_logger

logger = get_logger("desktop.satellite.geocode")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


@dataclass
class GeocodeHit:
    display_name: str
    lat: float
    lon: float
    # Optional bounding box from Nominatim: [south, north, west, east] as strings
    south: Optional[float] = None
    north: Optional[float] = None
    west: Optional[float] = None
    east: Optional[float] = None

    @property
    def has_bbox(self) -> bool:
        return None not in (self.south, self.north, self.west, self.east)


def search_places(query: str, limit: int = 8, language: str = "fa,en") -> List[GeocodeHit]:
    """Search places by name. Returns empty list on failure / no results."""
    q = (query or "").strip()
    if len(q) < 2:
        return []

    params = urllib.parse.urlencode(
        {
            "q": q,
            "format": "json",
            "limit": str(limit),
            "addressdetails": "0",
            "accept-language": language,
        }
    )
    url = f"{NOMINATIM_URL}?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Geocode failed: %s", exc)
        raise RuntimeError(f"Place search failed: {exc}") from exc

    hits: List[GeocodeHit] = []
    for item in data if isinstance(data, list) else []:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
            name = str(item.get("display_name") or query)
            south = north = west = east = None
            bb = item.get("boundingbox")
            if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                # Nominatim: [south_lat, north_lat, west_lon, east_lon]
                south, north, west, east = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
            hits.append(
                GeocodeHit(
                    display_name=name,
                    lat=lat,
                    lon=lon,
                    south=south,
                    north=north,
                    west=west,
                    east=east,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return hits
