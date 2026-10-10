"""Offline map packages (MBTiles / SQLite) + global offline mode.

A package is one ``.mbtiles`` file holding the tiles of one or more regions at a
zoom range for one imagery layer. When offline mode is on, ``tiles.download_tile``
and ``tiles.try_load_cached_tile`` read from the imported packages (then the local
disk cache) and never touch the network.
"""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from shared.logging import get_logger

logger = get_logger("desktop.satellite.offline")

OFFLINE_DIR = Path("data") / "offline_maps"
LIBRARY_FILE = "library.json"
PACKAGE_EXT = ".mbtiles"
OVERZOOM_LEVELS = 6  # enlarge a parent tile up to this many levels when a zoom is missing
UNDERZOOM_LEVELS = 2  # stitch child tiles up to this many levels for a lower zoom
AVG_TILE_KB = 25.0
# Video / detail mosaics need roughly this zoom; packages below are "overview only".
HIGH_ZOOM_MIN = 16

LonLat = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # south, west, north, east
ProgressCb = Callable[[int, int, str], None]


def _is_imagery(layer: str) -> bool:
    from desktop_app.satellite.tiles import LAYERS

    return layer != "map" and LAYERS.get(layer, {}).get("mosaic", "1") != "0"


class OfflinePackage:
    """One MBTiles file. Thread-safe reads (one connection guarded by a lock)."""

    def __init__(self, path: str | Path, create: bool = False):
        self.path = Path(path).resolve()
        if not create and not self.path.exists():
            raise FileNotFoundError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT)")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS tiles (zoom_level INTEGER, tile_column INTEGER, "
            "tile_row INTEGER, tile_data BLOB)"
        )
        self._db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS tile_index ON tiles (zoom_level, tile_column, tile_row)"
        )
        self._db.commit()
        self.meta: Dict[str, str] = dict(self._db.execute("SELECT name, value FROM metadata").fetchall())
        self._zooms: Optional[List[int]] = None

    # ------------------------------------------------------------ metadata
    @property
    def name(self) -> str:
        return self.meta.get("name") or self.path.stem

    @property
    def layer(self) -> str:
        return self.meta.get("geofinder_layer", "satellite")

    @property
    def regions(self) -> List[List[LonLat]]:
        try:
            return [[(float(p[0]), float(p[1])) for p in r] for r in json.loads(self.meta.get("geofinder_regions", "[]"))]
        except (TypeError, ValueError):
            return []

    def zooms(self) -> List[int]:
        if self._zooms is None:
            with self._lock:
                rows = self._db.execute("SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level").fetchall()
            self._zooms = [int(r[0]) for r in rows]
        return self._zooms

    def tile_count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0])

    def set_meta(self, **values: str) -> None:
        with self._lock:
            for k, v in values.items():
                self._db.execute("INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)", (k, str(v)))
                self.meta[k] = str(v)
            self._db.commit()

    def add_region(self, polygon: Sequence[LonLat], layer: str, zmin: int, zmax: int, attribution: str) -> None:
        regions = self.regions + [[(float(lon), float(lat)) for lon, lat in polygon]]
        lons = [p[0] for r in regions for p in r]
        lats = [p[1] for r in regions for p in r]
        zs = [int(z) for z in (self.meta.get("minzoom"), self.meta.get("maxzoom")) if z not in (None, "")]
        self.set_meta(
            name=self.name,
            format=self.meta.get("format", "jpg"),
            type="baselayer",
            version="1.1",
            attribution=attribution,
            bounds=f"{min(lons)},{min(lats)},{max(lons)},{max(lats)}",
            minzoom=min(zs + [zmin]),
            maxzoom=max(zs + [zmax]),
            geofinder_layer=layer,
            geofinder_regions=json.dumps(regions),
            geofinder_updated=time.strftime("%Y-%m-%d %H:%M"),
        )

    # ------------------------------------------------------------ tiles (XYZ y; stored as TMS row)
    def has(self, z: int, x: int, y: int) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, (1 << z) - 1 - y),
            ).fetchone()
        return row is not None

    def get(self, z: int, x: int, y: int) -> Optional[bytes]:
        with self._lock:
            row = self._db.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (z, x, (1 << z) - 1 - y),
            ).fetchone()
        return bytes(row[0]) if row else None

    def put_many(self, items: Sequence[Tuple[int, int, int, bytes]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)",
                [(z, x, (1 << z) - 1 - y, sqlite3.Binary(data)) for z, x, y, data in items],
            )
            self._db.commit()
        self._zooms = None

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def summary(self) -> str:
        zs = self.zooms()
        zr = f"z{zs[0]}–{zs[-1]}" if zs else "empty"
        mb = self.path.stat().st_size / 1e6 if self.path.exists() else 0.0
        return f"{self.name} · {self.layer} · {zr} · {len(self.regions)} region(s) · {mb:.1f} MB"


# ---------------------------------------------------------------- library + offline flag
class _Library:
    def __init__(self) -> None:
        self.enabled = False
        self.packages: List[OfflinePackage] = []
        self._loaded = False
        self._lock = threading.RLock()

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            f = OFFLINE_DIR / LIBRARY_FILE
            try:
                data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            except (OSError, ValueError) as exc:
                logger.warning("Offline library unreadable: %s", exc)
                data = {}
            self.enabled = bool(data.get("enabled", False))
            for p in data.get("packages", []):
                try:
                    self.packages.append(OfflinePackage(p))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Offline package %s skipped: %s", p, exc)

    def save(self) -> None:
        OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
        data = {"enabled": self.enabled, "packages": [str(p.path) for p in self.packages]}
        (OFFLINE_DIR / LIBRARY_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")


_LIB = _Library()


def offline_enabled() -> bool:
    _LIB.ensure_loaded()
    return _LIB.enabled


def set_offline_enabled(enabled: bool) -> None:
    _LIB.ensure_loaded()
    _LIB.enabled = bool(enabled)
    _LIB.save()
    logger.info("Offline mode %s", "ON" if enabled else "OFF")


def packages() -> List[OfflinePackage]:
    _LIB.ensure_loaded()
    return list(_LIB.packages)


def _pkg_max_zoom(pkg: OfflinePackage) -> int:
    zs = pkg.zooms()
    if zs:
        return int(zs[-1])
    try:
        return int(pkg.meta.get("maxzoom") or 0)
    except (TypeError, ValueError):
        return 0


def region_boxes(*, imagery_only: bool = True, min_zoom: int = 0) -> List[BBox]:
    """Axis-aligned boxes for offline package regions.

    Each stored region (or package ``bounds`` fallback) whose package reaches
    ``min_zoom`` is returned as ``(south, west, north, east)``.
    """
    out: List[BBox] = []
    for pkg in packages():
        if imagery_only and not _is_imagery(pkg.layer):
            continue
        if _pkg_max_zoom(pkg) < int(min_zoom):
            continue
        regions = pkg.regions
        if regions:
            for region in regions:
                if len(region) < 2:
                    continue
                lons = [float(p[0]) for p in region]
                lats = [float(p[1]) for p in region]
                out.append((min(lats), min(lons), max(lats), max(lons)))
            continue
        bounds = pkg.meta.get("bounds") or ""
        try:
            west, south, east, north = (float(x) for x in bounds.split(","))
        except (TypeError, ValueError):
            continue
        if east > west and north > south:
            out.append((south, west, north, east))
    return out


def high_zoom_region_boxes(min_zoom: int = HIGH_ZOOM_MIN) -> List[BBox]:
    """Boxes for imagery packages that include high-zoom tiles (see ``HIGH_ZOOM_MIN``)."""
    return region_boxes(imagery_only=True, min_zoom=min_zoom)


def find_package(path: str | Path) -> Optional[OfflinePackage]:
    target = Path(path).resolve()
    return next((p for p in packages() if p.path == target), None)


def register_package(path: str | Path | OfflinePackage) -> OfflinePackage:
    """Add an existing ``.mbtiles`` file (or open package) to the library (no copy)."""
    existing = find_package(path.path if isinstance(path, OfflinePackage) else path)
    if existing is not None:
        return existing
    pkg = path if isinstance(path, OfflinePackage) else OfflinePackage(path)
    with _LIB._lock:
        _LIB.packages.append(pkg)
        _LIB.save()
    return pkg


def import_package(path: str | Path, copy_into_library: bool = True) -> OfflinePackage:
    """Import a package file; optionally copy it under ``data/offline_maps`` first."""
    src = Path(path).resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    sqlite3.connect(f"file:{src}?mode=ro", uri=True).execute("SELECT COUNT(*) FROM tiles").fetchone()
    if copy_into_library and OFFLINE_DIR.resolve() not in src.parents:
        OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
        dest = OFFLINE_DIR / src.name
        k = 1
        while dest.exists():
            dest = OFFLINE_DIR / f"{src.stem}_{k}{src.suffix}"
            k += 1
        shutil.copy2(src, dest)
        src = dest
    return register_package(src)


def export_package(pkg: OfflinePackage, dest: str | Path) -> Path:
    dest = Path(dest)
    with pkg._lock:
        pkg._db.commit()
        out = sqlite3.connect(str(dest))
        try:
            pkg._db.backup(out)
        finally:
            out.close()
    return dest


def remove_package(pkg: OfflinePackage, delete_file: bool = False) -> None:
    with _LIB._lock:
        _LIB.packages = [p for p in _LIB.packages if p.path != pkg.path]
        _LIB.save()
    pkg.close()
    if delete_file:
        pkg.path.unlink(missing_ok=True)


# ---------------------------------------------------------------- tile lookup (used by tiles.py)
def _ordered(layer: str) -> List[OfflinePackage]:
    pk = packages()
    same = [p for p in pk if p.layer == layer]
    other = [p for p in pk if p.layer != layer and _is_imagery(p.layer) and _is_imagery(layer)]
    return same + other


def _decode(data: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def _tile_any(pkgs: List[OfflinePackage], z: int, x: int, y: int) -> Optional[np.ndarray]:
    if z < 0 or x < 0 or y < 0 or x >= (1 << z) or y >= (1 << z):
        return None
    for p in pkgs:
        if z not in p.zooms():
            continue
        data = p.get(z, x, y)
        if data:
            img = _decode(data)
            if img is not None:
                return img
    return None


def lookup_tile(z: int, x: int, y: int, layer: str = "satellite") -> Optional[np.ndarray]:
    """BGR 256×256 tile from offline packages (exact, enlarged parent, or stitched children)."""
    from desktop_app.satellite.tiles import TILE_SIZE

    pkgs = _ordered(layer)
    if not pkgs:
        return None
    img = _tile_any(pkgs, z, x, y)
    if img is not None:
        return cv2.resize(img, (TILE_SIZE, TILE_SIZE)) if img.shape[:2] != (TILE_SIZE, TILE_SIZE) else img
    # Missing zoom → enlarge the matching quarter of a parent tile
    for dz in range(1, OVERZOOM_LEVELS + 1):
        parent = _tile_any(pkgs, z - dz, x >> dz, y >> dz)
        if parent is None:
            continue
        parent = cv2.resize(parent, (TILE_SIZE, TILE_SIZE)) if parent.shape[:2] != (TILE_SIZE, TILE_SIZE) else parent
        n = 1 << dz
        sub = max(1, TILE_SIZE // n)
        ox, oy = (x % n) * sub, (y % n) * sub
        return cv2.resize(parent[oy : oy + sub, ox : ox + sub], (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_CUBIC)
    # Lower zoom than stored → stitch children
    for dz in range(1, UNDERZOOM_LEVELS + 1):
        n = 1 << dz
        canvas = np.zeros((TILE_SIZE * n, TILE_SIZE * n, 3), np.uint8)
        found = 0
        for j in range(n):
            for i in range(n):
                child = _tile_any(pkgs, z + dz, x * n + i, y * n + j)
                if child is not None:
                    found += 1
                    canvas[j * TILE_SIZE : (j + 1) * TILE_SIZE, i * TILE_SIZE : (i + 1) * TILE_SIZE] = (
                        cv2.resize(child, (TILE_SIZE, TILE_SIZE))
                    )
        if found >= (n * n + 1) // 2:
            return cv2.resize(canvas, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
    return None


# ---------------------------------------------------------------- region download
def _point_in_poly(px: float, py: float, ring: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    return (orient(a, b, c) > 0) != (orient(a, b, d) > 0) and (orient(c, d, a) > 0) != (orient(c, d, b) > 0)


def iter_region_tiles(polygon: Sequence[LonLat], zoom: int) -> Iterator[Tuple[int, int]]:
    """(x, y) of every tile at ``zoom`` that touches the polygon."""
    from desktop_app.satellite.tiles import lonlat_to_tile

    ring = [lonlat_to_tile(lon, lat, zoom) for lon, lat in polygon]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    n = 1 << zoom
    x0, x1 = max(0, int(math.floor(min(xs)))), min(n - 1, int(math.floor(max(xs))))
    y0, y1 = max(0, int(math.floor(min(ys)))), min(n - 1, int(math.floor(max(ys))))
    edges = list(zip(ring, ring[1:] + ring[:1]))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            corners = [(tx, ty), (tx + 1, ty), (tx + 1, ty + 1), (tx, ty + 1)]
            if (
                _point_in_poly(tx + 0.5, ty + 0.5, ring)
                or any(_point_in_poly(cx, cy, ring) for cx, cy in corners)
                or any(tx <= px <= tx + 1 and ty <= py <= ty + 1 for px, py in ring)
                or any(_segments_cross(a, b, c, d) for a, b in edges for c, d in zip(corners, corners[1:] + corners[:1]))
            ):
                yield tx, ty


def estimate_region(polygon: Sequence[LonLat], zmin: int, zmax: int) -> Dict[str, float]:
    per_zoom = {z: sum(1 for _ in iter_region_tiles(polygon, z)) for z in range(zmin, zmax + 1)}
    total = sum(per_zoom.values())
    return {"tiles": total, "mb": round(total * AVG_TILE_KB / 1024.0, 1), "per_zoom": per_zoom}


def download_region(
    package_path: str | Path,
    polygon: Sequence[LonLat],
    zmin: int,
    zmax: int,
    layer: str = "satellite",
    cache_dir: str | Path | None = None,
    progress_cb: Optional[ProgressCb] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    workers: int = 8,
) -> OfflinePackage:
    """Download every tile of ``polygon`` for zooms ``zmin..zmax`` into a package (resumable / appendable)."""
    from desktop_app.satellite.tiles import LAYERS, fetch_tile_bytes, layer_max_zoom

    if len(polygon) < 3:
        raise ValueError("Region needs at least 3 points")
    zmax = min(int(zmax), layer_max_zoom(layer))
    zmin = max(0, min(int(zmin), zmax))
    pkg = find_package(package_path) or OfflinePackage(package_path, create=True)
    if pkg.meta.get("geofinder_layer") and pkg.layer != layer:
        raise ValueError(f"Package '{pkg.name}' holds layer '{pkg.layer}', not '{layer}'. Use a new package.")
    pkg.add_region(polygon, layer, zmin, zmax, LAYERS[layer].get("attribution", layer))

    jobs = [(z, x, y) for z in range(zmin, zmax + 1) for x, y in iter_region_tiles(polygon, z)]
    total = len(jobs)
    cache = Path(cache_dir) if cache_dir else Path("data") / "satellite_cache"
    done = failed = 0
    batch: List[Tuple[int, int, int, bytes]] = []

    def fetch(job: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], Optional[bytes]]:
        z, x, y = job
        if (should_stop and should_stop()) or pkg.has(z, x, y):
            return job, None
        try:
            return job, fetch_tile_bytes(z, x, y, layer, cache_dir=cache)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Offline tile %s failed: %s", job, exc)
            return job, b""

    if progress_cb:
        progress_cb(0, total, f"Downloading {total} tiles (z{zmin}–{zmax})…")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for chunk_start in range(0, total, 256):
            if should_stop and should_stop():
                break
            for (z, x, y), data in pool.map(fetch, jobs[chunk_start : chunk_start + 256]):
                done += 1
                if data == b"":
                    failed += 1
                elif data:
                    batch.append((z, x, y, data))
            if batch:
                pkg.put_many(batch)
                batch.clear()
            if progress_cb:
                progress_cb(done, total, f"Tiles {done}/{total}" + (f" · {failed} failed" if failed else ""))
    pkg = register_package(pkg)
    stopped = bool(should_stop and should_stop())
    logger.info("Offline package %s: %d/%d tiles processed, %d failed%s", pkg.path, done, total, failed,
                " (stopped)" if stopped else "")
    if failed and failed == done:
        raise RuntimeError(f"All {failed} tiles failed — check the internet connection or imagery source.")
    return pkg
