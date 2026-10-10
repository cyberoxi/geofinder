"""Region atlas: one large area + a fixed global landmark set + a base model trained on all of it."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from desktop_app.satellite.tiles import TILE_SIZE, lonlat_to_tile, tile_to_lonlat

ATLAS_FILE = "atlas.json"
TARGET_CLASS = "target_region"

SWNE = Tuple[float, float, float, float]
LonLat = Tuple[float, float]


# ------------------------------------------------------------------ geo helpers


def global_px(lon: float, lat: float, zoom: int) -> Tuple[float, float]:
    """Web Mercator pixel coordinates of (lon, lat) for the whole world at ``zoom``."""
    x, y = lonlat_to_tile(lon, lat, zoom)
    return x * TILE_SIZE, y * TILE_SIZE


def global_px_to_lonlat(x: float, y: float, zoom: int) -> LonLat:
    return tile_to_lonlat(x / TILE_SIZE, y / TILE_SIZE, zoom)


def mosaic_origin_px(fetch_swne: SWNE, zoom: int) -> Tuple[float, float]:
    """Global pixel of mosaic pixel (0, 0) for a mosaic returned by ``fetch_mosaic(fetch_swne, zoom)``."""
    s, w, n, e = fetch_swne
    ox, oy = lonlat_to_tile(w, n, zoom)
    tx0, ty0 = math.floor(ox), math.floor(oy)
    return tx0 * TILE_SIZE + int((ox - tx0) * TILE_SIZE), ty0 * TILE_SIZE + int((oy - ty0) * TILE_SIZE)


def lonlat_to_mosaic(points: Sequence[Sequence[float]], fetch_swne: SWNE, zoom: int) -> List[List[float]]:
    """Unclamped lon/lat ring → mosaic pixels (points may fall outside the mosaic)."""
    ox, oy = mosaic_origin_px(fetch_swne, zoom)
    out = []
    for lon, lat in points:
        gx, gy = global_px(float(lon), float(lat), zoom)
        out.append([gx - ox, gy - oy])
    return out


def mosaic_to_lonlat(points: Sequence[Sequence[float]], fetch_swne: SWNE, zoom: int) -> List[List[float]]:
    ox, oy = mosaic_origin_px(fetch_swne, zoom)
    return [list(global_px_to_lonlat(float(x) + ox, float(y) + oy, zoom)) for x, y in points]


def point_in_ring(x: float, y: float, ring: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def bbox_of(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# ------------------------------------------------------------------ atlas record


@dataclass
class RegionAtlas:
    name: str
    root: Path
    swne: SWNE
    polygon_lonlat: List[List[float]]
    zoom: int = 16
    layer: str = "satellite"
    model_type: str = "segmentation"
    imgsz: int = 640
    landmarks: List[Dict[str, Any]] = field(default_factory=list)  # {kind, score, polygon_lonlat}
    chunks: List[Dict[str, Any]] = field(default_factory=list)  # {id, swne, status, videos}
    base_weights: str = ""  # relative to root
    metrics: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)  # build options (for resume)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""

    # -------------------------------------------------------------- paths
    @property
    def atlas_path(self) -> Path:
        return self.root / ATLAS_FILE

    @property
    def chunks_dir(self) -> Path:
        return self.root / "chunks"

    @property
    def base_project_dir(self) -> Path:
        return self.root / "base_project"

    @property
    def subregions_dir(self) -> Path:
        return self.root / "subregions"

    def chunk_dir(self, chunk_id: int) -> Path:
        return self.chunks_dir / f"chunk_{int(chunk_id):03d}"

    def base_weights_path(self) -> Optional[Path]:
        if not self.base_weights:
            return None
        p = self.root / self.base_weights
        return p if p.is_file() else None

    # -------------------------------------------------------------- classes
    @property
    def classes(self) -> List[str]:
        return [TARGET_CLASS] + [f"landmark_{i + 1}_{lm.get('kind', 'object')}" for i, lm in enumerate(self.landmarks)]

    def contains(self, lon: float, lat: float) -> bool:
        ring = self.polygon_lonlat
        if len(ring) >= 3:
            return point_in_ring(lon, lat, ring)
        s, w, n, e = self.swne
        return s <= lat <= n and w <= lon <= e

    def summary(self) -> str:
        done = sum(1 for c in self.chunks if c.get("status") in ("rendered", "imported"))
        model = "trained" if self.base_weights_path() else "not trained yet"
        return (
            f"{self.name} — zoom {self.zoom}, {len(self.landmarks)} landmarks, "
            f"{done}/{len(self.chunks)} chunks rendered, base model {model}"
        )

    # -------------------------------------------------------------- io
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("root", None)
        d["version"] = 1
        d["classes"] = self.classes
        return d

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        tmp = self.atlas_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.atlas_path)
        return self.atlas_path

    @classmethod
    def load(cls, path: str | Path) -> "RegionAtlas":
        p = Path(path)
        if p.is_dir():
            p = p / ATLAS_FILE
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            name=str(d.get("name") or p.parent.name),
            root=p.parent.resolve(),
            swne=tuple(float(v) for v in d["swne"]),  # type: ignore[arg-type]
            polygon_lonlat=[list(map(float, pt)) for pt in d.get("polygon_lonlat") or []],
            zoom=int(d.get("zoom", 16)),
            layer=str(d.get("layer", "satellite")),
            model_type=str(d.get("model_type", "segmentation")),
            imgsz=int(d.get("imgsz", 640)),
            landmarks=list(d.get("landmarks") or []),
            chunks=list(d.get("chunks") or []),
            base_weights=str(d.get("base_weights") or ""),
            metrics=dict(d.get("metrics") or {}),
            settings=dict(d.get("settings") or {}),
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
        )
