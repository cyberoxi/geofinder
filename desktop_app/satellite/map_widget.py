"""Interactive map with satellite/street basemap, search, and region selection."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from desktop_app.satellite.geocode import GeocodeHit, search_places
from desktop_app.satellite.scene import meters_per_pixel
from desktop_app.satellite.tiles import (
    LAYERS,
    USER_AGENT,
    layer_max_zoom,
    TILE_SIZE,
    download_tile,
    lonlat_to_tile,
    tile_to_lonlat,
    try_load_cached_tile,
)
from shared.logging import get_logger
from shared.qt_compat import (
    QColor,
    QComboBox,
    QFont,
    QImage,
    QLabel,
    QLineEdit,
    QPainter,
    QPen,
    QPixmap,
    QPointF,
    QPolygonF,
    Qt,
    QThread,
    QTimer,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    Signal,
    mouse_xy,
)

LonLat = Tuple[float, float]  # (lon, lat)
HIT_PX = 12.0
_EARTH_R_M = 6_371_000.0
# Google Maps–compatible constant: eye altitude ≈ C / 2^zoom / cos(lat)
_MAP_EYE_ALT_C = 591_657_550.5
logger = get_logger("desktop.satellite.map")


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * _EARTH_R_M * math.asin(min(1.0, math.sqrt(a)))


def polygon_edge_lengths_m(poly: Sequence[LonLat], *, closed: bool = True) -> List[float]:
    """Length of each edge (and closing edge when ``closed``)."""
    if len(poly) < 2:
        return []
    edges = [
        haversine_m(poly[i][0], poly[i][1], poly[i + 1][0], poly[i + 1][1])
        for i in range(len(poly) - 1)
    ]
    if closed and len(poly) >= 3:
        edges.append(haversine_m(poly[-1][0], poly[-1][1], poly[0][0], poly[0][1]))
    return edges


def polygon_area_m2(poly: Sequence[LonLat]) -> float:
    """Approximate geodesic area (m²) via equirectangular shoelace — fine for map-scale regions."""
    if len(poly) < 3:
        return 0.0
    lat0 = sum(p[1] for p in poly) / len(poly)
    lon0 = sum(p[0] for p in poly) / len(poly)
    m_lat = 111_320.0
    m_lon = 111_320.0 * math.cos(math.radians(lat0))
    pts = [((p[0] - lon0) * m_lon, (p[1] - lat0) * m_lat) for p in poly]
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def format_length_m(meters: float) -> str:
    m = abs(float(meters))
    if m >= 1000:
        km = m / 1000.0
        return f"{km:.2f} km" if km < 10 else f"{km:.1f} km"
    if m >= 100:
        return f"{m:.0f} m"
    if m >= 10:
        return f"{m:.1f} m"
    return f"{m:.2f} m"


def format_area_m2(area: float) -> str:
    a = abs(float(area))
    if a >= 1_000_000:
        return f"{a / 1_000_000:.3g} km²"
    if a >= 10_000:
        return f"{a / 10_000:.2f} ha ({a:.0f} m²)"
    if a >= 100:
        return f"{a:.0f} m²"
    return f"{a:.1f} m²"


def format_altitude_m(meters: float, *, prefer_meters: bool = False) -> str:
    m = abs(float(meters))
    if prefer_meters or m < 1000:
        if m >= 100:
            return f"{m:.0f} m"
        return f"{m:.1f} m"
    km = m / 1000.0
    return f"{km:.2f} km" if km < 10 else f"{km:.1f} km"


def eye_altitude_m(lat: float, zoom: int, view_height_px: float = 0.0, fov_deg: float = 50.0) -> float:
    """
    Approximate viewing altitude above ground (m) for the current Web Mercator zoom.

    Uses the Google Maps–style zoom→altitude mapping (matches what users expect
    from map apps). ``view_height_px`` / ``fov_deg`` kept for API compatibility.
    """
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    return _MAP_EYE_ALT_C / (2.0 ** int(zoom)) / cos_lat

TileKey = Tuple[int, int, int, str]  # z, x, y, layer


class GeocodeWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        try:
            hits = search_places(self.query)
            self.finished_ok.emit(hits)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ElevationWorker(QThread):
    """Fetch ground elevation (m ASL) at a lat/lon. Tries Open-Meteo, then OpenTopoData."""

    finished_ok = Signal(float, float, float)  # elev_m, lat, lon
    failed = Signal(str)

    def __init__(self, lat: float, lon: float, parent=None):
        super().__init__(parent)
        self.lat = float(lat)
        self.lon = float(lon)

    @staticmethod
    def _http_json(url: str, timeout: float = 10.0) -> dict:
        import json
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    def _fetch_open_meteo(self) -> float:
        import urllib.parse

        qs = urllib.parse.urlencode({"latitude": f"{self.lat:.5f}", "longitude": f"{self.lon:.5f}"})
        data = self._http_json(f"https://api.open-meteo.com/v1/elevation?{qs}")
        elevs = data.get("elevation") or []
        if not elevs or elevs[0] is None:
            raise ValueError("Open-Meteo: empty elevation")
        return float(elevs[0])

    def _fetch_opentopodata(self) -> float:
        # Public DEM mosaic; no API key
        url = f"https://api.opentopodata.org/v1/aster30m?locations={self.lat:.5f},{self.lon:.5f}"
        data = self._http_json(url)
        results = data.get("results") or []
        if not results or results[0].get("elevation") is None:
            raise ValueError("OpenTopoData: empty elevation")
        return float(results[0]["elevation"])

    def run(self):
        errors: List[str] = []
        for name, fn in (("open-meteo", self._fetch_open_meteo), ("opentopodata", self._fetch_opentopodata)):
            try:
                elev = fn()
                self.finished_ok.emit(float(elev), self.lat, self.lon)
                return
            except Exception as exc:  # noqa: BLE001 — network / parse / empty payload
                errors.append(f"{name}: {exc}")
                logger.warning("Elevation fetch via %s failed: %s", name, exc)
        self.failed.emit("; ".join(errors) or "elevation unavailable")


class TileFetchWorker(QThread):
    """Download missing map tiles off the UI thread (parallel, reports each tile as it lands)."""

    finished_ok = Signal(int)  # request_id
    failed = Signal(int, str)
    tile_done = Signal(int, object, bool)  # request_id, key, ok

    def __init__(
        self,
        request_id: int,
        keys: List[TileKey],
        cache_dir: Path,
        parent=None,
    ):
        super().__init__(parent)
        self.request_id = request_id
        self.keys = keys
        self.cache_dir = cache_dir
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch(key):
            if self._cancel:
                return key, False
            z, x, y, layer = key
            try:
                download_tile(z, x, y, self.cache_dir, retries=2, layer=layer, timeout=12.0)
                return key, True
            except Exception:  # noqa: BLE001
                return key, False

        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for fut in as_completed([pool.submit(fetch, k) for k in self.keys]):
                    key, ok = fut.result()
                    if not self._cancel:
                        self.tile_done.emit(self.request_id, key, ok)
            self.finished_ok.emit(self.request_id)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.request_id, str(exc))


class SatelliteMapWidget(QLabel):
    """
    Pan (left-drag) / zoom (wheel).
    Basemap: satellite or street map.
    Region select: click polygon vertices (numbered) or drag a box;
    closed shapes can be reshaped by dragging nodes, clicking an edge to
    insert a node, double-clicking a node to delete it, or moving the fill.
    """

    view_changed = Signal()
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Keep modest so dialog + right sidebar fit without horizontal clipping
        self.setMinimumSize(520, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background:#111;border:1px solid #333;")

        self.center_lat = 35.6892
        self.center_lon = 51.3890
        self.zoom = 15
        self.select_mode = False
        self.select_tool = "polygon"  # "polygon" | "box"
        self.basemap = "satellite"
        self._tiles_enabled = True  # False → blank (no imagery service selected)
        self.cache_dir = Path("data") / "satellite_cache"

        self._drag_origin: Optional[Tuple[float, float]] = None
        self._pan_origin_center: Optional[Tuple[float, float]] = None

        # Closed region as lon/lat ring (first≠last). Empty = none.
        self._poly: List[LonLat] = []
        self._draft: List[LonLat] = []
        self._ref_poly: List[LonLat] = []  # read-only outline (e.g. region atlas boundary)
        self._draft_cursor: Optional[Tuple[float, float]] = None  # widget px
        self._box_start_px: Optional[Tuple[float, float]] = None
        self._box_end_px: Optional[Tuple[float, float]] = None
        self._drag_node: Optional[int] = None
        self._move_origin_px: Optional[Tuple[float, float]] = None
        self._move_origin_poly: Optional[List[LonLat]] = None

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._render)
        self._base_pixmap: Optional[QPixmap] = None

        self._tile_worker: Optional[TileFetchWorker] = None
        self._tile_req_id = 0
        self._pending_keys: Set[TileKey] = set()
        self._failed: Dict[TileKey, float] = {}  # key → time of failure (retry later, no endless loop)
        self._loading = False

        # Ground elevation (m ASL) at map center — fetched lazily
        self._ground_elev_m: Optional[float] = None
        self._elev_at: Optional[Tuple[float, float]] = None  # lat, lon of last successful fetch
        self._elev_pending: Optional[Tuple[float, float]] = None
        self._elev_worker: Optional[ElevationWorker] = None
        self._elev_timer = QTimer(self)
        self._elev_timer.setSingleShot(True)
        self._elev_timer.timeout.connect(self._request_elevation)
        self.view_changed.connect(self._schedule_elevation)

        QTimer.singleShot(50, self._render)
        QTimer.singleShot(300, self._request_elevation)

    def set_basemap(self, layer: str) -> None:
        key = layer.strip().lower()
        if key not in LAYERS:
            return
        self._tiles_enabled = True
        if key == self.basemap:
            self._schedule_render()
            return
        self.basemap = key
        self._schedule_render()

    def set_tiles_enabled(self, enabled: bool) -> None:
        """When False, paint a blank map (no satellite imagery fetched)."""
        enabled = bool(enabled)
        if self._tiles_enabled == enabled:
            return
        self._tiles_enabled = enabled
        self._schedule_render()

    def set_select_mode(self, enabled: bool) -> None:
        self.select_mode = bool(enabled)
        if not enabled:
            self._cancel_draft()
        self._update_cursor(None)
        self.update()

    def set_select_tool(self, tool: str) -> None:
        tool = "box" if tool == "box" else "polygon"
        if tool == self.select_tool:
            return
        self.select_tool = tool
        self._cancel_draft()
        self._update_cursor(None)
        self.update()

    def clear_selection(self) -> None:
        self._poly = []
        self._cancel_draft()
        self.selection_changed.emit()
        self.update()

    def undo_last_point(self) -> None:
        if self._draft:
            self._draft.pop()
            self.selection_changed.emit()
            self.update()

    def finish_polygon(self) -> bool:
        """Close the draft polygon (≥3 points). Returns True if closed."""
        if len(self._draft) < 3:
            return False
        self._poly = list(self._draft)
        self._cancel_draft(keep_emit=False)
        self.selection_changed.emit()
        self.update()
        return True

    def selection_polygon(self) -> Optional[List[LonLat]]:
        """Closed region as [(lon, lat), ...] or None."""
        return list(self._poly) if len(self._poly) >= 3 else None

    def selection_metrics(self) -> Optional[Dict[str, object]]:
        """Area / edge lengths for the closed selection (or None)."""
        poly = self.selection_polygon()
        if not poly:
            return None
        edges = polygon_edge_lengths_m(poly, closed=True)
        return {
            "area_m2": polygon_area_m2(poly),
            "perimeter_m": float(sum(edges)),
            "edges_m": edges,
            "n_nodes": len(poly),
        }

    def selection_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        poly = self.selection_polygon()
        if not poly:
            return None
        lons = [p[0] for p in poly]
        lats = [p[1] for p in poly]
        return (min(lats), min(lons), max(lats), max(lons))

    def visible_bbox(self) -> Tuple[float, float, float, float]:
        """SWNE of the currently visible map viewport (for synced mini previews)."""
        w = max(1, int(self.width()))
        h = max(1, int(self.height()))
        lon0, lat0 = self._pixel_to_lonlat(0.0, 0.0)
        lon1, lat1 = self._pixel_to_lonlat(float(w), float(h))
        south, north = min(lat0, lat1), max(lat0, lat1)
        west, east = min(lon0, lon1), max(lon0, lon1)
        if north <= south:
            north = south + 1e-5
        if east <= west:
            east = west + 1e-5
        return south, west, north, east

    def set_selection_bbox(self, south: float, west: float, north: float, east: float) -> None:
        if north <= south or east <= west:
            return
        self._poly = [(west, north), (east, north), (east, south), (west, south)]
        self._cancel_draft(keep_emit=False)
        self.selection_changed.emit()
        self.update()

    def set_selection_polygon(self, points: List[LonLat]) -> None:
        if len(points) < 3:
            return
        self._poly = [(float(lon), float(lat)) for lon, lat in points]
        self._cancel_draft(keep_emit=False)
        self.selection_changed.emit()
        self.update()

    def set_reference_polygon(self, points: Optional[List[LonLat]]) -> None:
        """Show a dashed, non-editable outline (e.g. the boundary of a region atlas)."""
        self._ref_poly = [(float(lon), float(lat)) for lon, lat in (points or [])]
        self.update()

    def reference_contains(self, lon: float, lat: float) -> bool:
        """True when (lon, lat) is inside the reference outline (or no outline is set)."""
        if len(self._ref_poly) < 3:
            return True
        return self._point_in_poly(lon, lat, self._ref_poly)

    def _cancel_draft(self, keep_emit: bool = True) -> None:
        had = bool(self._draft or self._box_start_px)
        self._draft = []
        self._draft_cursor = None
        self._box_start_px = None
        self._box_end_px = None
        self._drag_node = None
        self._move_origin_px = None
        self._move_origin_poly = None
        if had and keep_emit:
            self.selection_changed.emit()

    @staticmethod
    def _point_in_poly(px: float, py: float, ring: List[Tuple[float, float]]) -> bool:
        # Ray casting in widget pixel space
        inside = False
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def _poly_to_px(self, poly: List[LonLat]) -> List[Tuple[float, float]]:
        return [self._lonlat_to_pixel(lon, lat) for lon, lat in poly]

    def _hit_node(self, x: float, y: float, poly: List[LonLat]) -> Optional[int]:
        for i, (lon, lat) in enumerate(poly):
            px, py = self._lonlat_to_pixel(lon, lat)
            if math.hypot(px - x, py - y) <= HIT_PX:
                return i
        return None

    def _hit_edge(self, x: float, y: float, poly: List[LonLat]) -> Optional[Tuple[int, LonLat]]:
        """If ``(x,y)`` is near a closed-ring edge, return ``(insert_index, lonlat)``."""
        n = len(poly)
        if n < 2:
            return None
        pts = self._poly_to_px(poly)
        best_i: Optional[int] = None
        best_d = HIT_PX
        best_ll: Optional[LonLat] = None
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            dx, dy = x1 - x0, y1 - y0
            len2 = dx * dx + dy * dy
            if len2 < 1e-6:
                continue
            t = ((x - x0) * dx + (y - y0) * dy) / len2
            # Skip endpoints so node-hit / duplicate vertices stay clean
            if t < 0.05 or t > 0.95:
                continue
            cx = x0 + t * dx
            cy = y0 + t * dy
            d = math.hypot(x - cx, y - cy)
            if d <= best_d:
                best_d = d
                best_i = i + 1
                best_ll = self._pixel_to_lonlat(cx, cy)
        if best_i is None or best_ll is None:
            return None
        return best_i, best_ll

    def _update_cursor(self, pos: Optional[Tuple[float, float]]) -> None:
        if self.select_mode and not self._poly and not self._drag_node and self._move_origin_poly is None:
            self.setCursor(Qt.CrossCursor)
            return
        if pos is not None and self._poly:
            node = self._hit_node(pos[0], pos[1], self._poly)
            if node is not None:
                self.setCursor(Qt.SizeAllCursor)
                return
            if self._hit_edge(pos[0], pos[1], self._poly) is not None:
                self.setCursor(Qt.PointingHandCursor)
                return
            ring = self._poly_to_px(self._poly)
            if self._point_in_poly(pos[0], pos[1], ring):
                self.setCursor(Qt.SizeAllCursor)
                return
        if self.select_mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)

    def go_to(self, lat: float, lon: float, zoom: Optional[int] = None) -> None:
        self.center_lat = float(lat)
        self.center_lon = float(lon)
        if zoom is not None:
            self.zoom = int(max(3, min(19, zoom)))
        self._schedule_render()

    def fit_bbox(self, south: float, west: float, north: float, east: float, pad: float = 1.15) -> None:
        """Center on bbox and pick a zoom that roughly fits the widget."""
        lat = (south + north) / 2.0
        lon = (west + east) / 2.0
        self.center_lat = lat
        self.center_lon = lon
        span_lat = max((north - south) * pad, 1e-6)
        span_lon = max((east - west) * pad, 1e-6)
        h = max(self.height(), 480)
        w = max(self.width(), 720)
        best = 12
        for z in range(3, 19):
            _, y0 = lonlat_to_tile(lon, lat + span_lat / 2, z)
            _, y1 = lonlat_to_tile(lon, lat - span_lat / 2, z)
            x0, _ = lonlat_to_tile(lon - span_lon / 2, lat, z)
            x1, _ = lonlat_to_tile(lon + span_lon / 2, lat, z)
            px_h = abs(y1 - y0) * TILE_SIZE
            px_w = abs(x1 - x0) * TILE_SIZE
            if px_h <= h and px_w <= w:
                best = z
            else:
                break
        self.zoom = best
        self._schedule_render()

    def _schedule_render(self) -> None:
        self._refresh_timer.start(80)

    def _pixel_to_lonlat(self, px: float, py: float) -> Tuple[float, float]:
        w, h = max(self.width(), 1), max(self.height(), 1)
        cx, cy = lonlat_to_tile(self.center_lon, self.center_lat, self.zoom)
        tx = cx + (px - w / 2.0) / TILE_SIZE
        ty = cy + (py - h / 2.0) / TILE_SIZE
        return tile_to_lonlat(tx, ty, self.zoom)

    def _lonlat_to_pixel(self, lon: float, lat: float) -> Tuple[float, float]:
        w, h = max(self.width(), 1), max(self.height(), 1)
        cx, cy = lonlat_to_tile(self.center_lon, self.center_lat, self.zoom)
        tx, ty = lonlat_to_tile(lon, lat, self.zoom)
        px = (tx - cx) * TILE_SIZE + w / 2.0
        py = (ty - cy) * TILE_SIZE + h / 2.0
        return px, py

    def _visible_tile_range(self) -> Tuple[int, int, int, int, float, float]:
        w, h = max(self.width(), 1), max(self.height(), 1)
        cx, cy = lonlat_to_tile(self.center_lon, self.center_lat, self.zoom)
        left = cx - (w / 2.0) / TILE_SIZE
        top = cy - (h / 2.0) / TILE_SIZE
        right = cx + (w / 2.0) / TILE_SIZE
        bottom = cy + (h / 2.0) / TILE_SIZE
        tx0, ty0 = int(math.floor(left)), int(math.floor(top))
        tx1, ty1 = int(math.floor(right)), int(math.floor(bottom))
        return tx0, ty0, tx1, ty1, left, top

    def _render(self) -> None:
        """Paint from disk cache only; queue network fetches in a background thread."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        if not self._tiles_enabled:
            self._loading = False
            self._pending_keys.clear()
            msg = "Tick a satellite service to show imagery"
            cv2.putText(
                canvas, msg, (max(10, w // 2 - 200), h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA,
            )
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self._base_pixmap = QPixmap.fromImage(qimg.copy())
            self.setPixmap(self._base_pixmap)
            self.view_changed.emit()
            self.update()
            return

        tx0, ty0, tx1, ty1, left, top = self._visible_tile_range()
        n_world = 2**self.zoom
        attr = LAYERS.get(self.basemap, LAYERS["satellite"])["attribution"]
        missing: List[TileKey] = []

        max_z = layer_max_zoom(self.basemap)
        dz = max(0, self.zoom - max_z)
        now = time.time()
        n_failed = 0
        for ty in range(ty0, ty1 + 1):
            if ty < 0 or ty >= n_world:
                continue
            for tx in range(tx0, tx1 + 1):
                wrapped = tx % n_world
                # Beyond the source's max zoom: enlarge the matching part of the parent tile
                pz, px_, py_ = self.zoom - dz, wrapped >> dz, ty >> dz
                key = (pz, px_, py_, self.basemap)
                tile = try_load_cached_tile(pz, px_, py_, self.cache_dir, layer=self.basemap)
                if tile is None:
                    if now - self._failed.get(key, -1e9) < 30.0:
                        n_failed += 1
                        tile = np.full((TILE_SIZE, TILE_SIZE, 3), 45, np.uint8)
                        cv2.putText(tile, "no data", (80, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
                    else:
                        if key not in missing:
                            missing.append(key)
                        continue
                elif dz:
                    n = 2**dz
                    sub = TILE_SIZE // n
                    ox, oy = (wrapped % n) * sub, (ty % n) * sub
                    tile = cv2.resize(tile[oy : oy + sub, ox : ox + sub], (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_CUBIC)
                if tile.shape[0] != TILE_SIZE or tile.shape[1] != TILE_SIZE:
                    tile = cv2.resize(tile, (TILE_SIZE, TILE_SIZE))
                sx = int(round((tx - left) * TILE_SIZE))
                sy = int(round((ty - top) * TILE_SIZE))
                x0, y0 = max(0, sx), max(0, sy)
                x1, y1 = min(w, sx + TILE_SIZE), min(h, sy + TILE_SIZE)
                if x1 <= x0 or y1 <= y0:
                    continue
                src_x0, src_y0 = x0 - sx, y0 - sy
                src_x1, src_y1 = src_x0 + (x1 - x0), src_y0 + (y1 - y0)
                canvas[y0:y1, x0:x1] = tile[src_y0:src_y1, src_x0:src_x1]

        status = attr
        if dz:
            status += f"  ·  max zoom {max_z} (enlarged)"
        if missing:
            status = f"{status}  ·  loading {len(missing)} tile(s)…"
        elif n_failed:
            status = f"{status}  ·  {n_failed} tile(s) unavailable"
        cv2.putText(
            canvas,
            status,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._base_pixmap = QPixmap.fromImage(qimg.copy())
        self.setPixmap(self._base_pixmap)
        self.view_changed.emit()
        self.update()

        if missing:
            self._queue_tile_fetch(missing)
        else:
            self._loading = False
            self._pending_keys.clear()

    def _queue_tile_fetch(self, keys: List[TileKey]) -> None:
        needed = set(keys)
        # Already downloading all of these — wait for that worker.
        if self._tile_worker is not None and self._tile_worker.isRunning():
            if needed <= self._pending_keys:
                return
            self._tile_worker.cancel()
        self._pending_keys = needed
        self._loading = True
        self._tile_req_id += 1
        req_id = self._tile_req_id
        worker = TileFetchWorker(req_id, list(needed), self.cache_dir, parent=self)
        worker.tile_done.connect(self._on_tile_done)
        worker.finished_ok.connect(self._on_tiles_ready)
        worker.failed.connect(self._on_tiles_failed)
        self._tile_worker = worker
        worker.start()

    def _on_tile_done(self, request_id: int, key: object, ok: bool) -> None:
        if not ok:
            self._failed[tuple(key)] = time.time()
        else:
            self._failed.pop(tuple(key), None)
        # Repaint progressively (coalesced by the single-shot timer)
        if not self._refresh_timer.isActive():
            self._refresh_timer.start(120)

    def _on_tiles_ready(self, request_id: int) -> None:
        if request_id != self._tile_req_id:
            # A superseded worker finished: its tiles are cached now — repaint anyway
            self._schedule_render()
            return
        self._loading = False
        self._pending_keys.clear()
        self._schedule_render()

    def _on_tiles_failed(self, request_id: int, _err: str) -> None:
        if request_id != self._tile_req_id:
            return
        self._loading = False
        self._pending_keys.clear()
        self._schedule_render()

    def _draw_numbered_poly(
        self,
        painter: QPainter,
        poly: List[LonLat],
        *,
        closed: bool,
        fill: QColor,
        stroke: QColor,
        draft_cursor: Optional[Tuple[float, float]] = None,
    ) -> None:
        if not poly and draft_cursor is None:
            return
        pts_px = self._poly_to_px(poly)
        path = QPolygonF([QPointF(px, py) for px, py in pts_px])
        if draft_cursor is not None and pts_px:
            path.append(QPointF(draft_cursor[0], draft_cursor[1]))
        painter.setPen(QPen(stroke, 2, Qt.SolidLine if closed else Qt.DashLine))
        painter.setBrush(fill if closed or len(pts_px) >= 3 else QColor(0, 0, 0, 0))
        if closed and len(pts_px) >= 3:
            painter.drawPolygon(path)
        elif len(pts_px) >= 1:
            painter.drawPolyline(path)
            if draft_cursor is not None and pts_px:
                # Soft close hint toward first vertex when ≥3 points
                if len(pts_px) >= 3:
                    fx, fy = pts_px[0]
                    if math.hypot(draft_cursor[0] - fx, draft_cursor[1] - fy) <= HIT_PX * 1.6:
                        painter.setPen(QPen(QColor(255, 220, 80), 2, Qt.DotLine))
                        painter.drawEllipse(QPointF(fx, fy), HIT_PX, HIT_PX)

        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        for i, (px, py) in enumerate(pts_px):
            r = 9
            painter.setBrush(QColor(20, 20, 20, 220))
            painter.setPen(QPen(stroke, 2))
            painter.drawEllipse(QPointF(px, py), r, r)
            painter.setPen(QColor(255, 255, 255))
            label = str(i + 1)
            br = painter.fontMetrics().boundingRect(label)
            painter.drawText(int(px - br.width() / 2), int(py + br.height() / 2 - 2), label)

        # Edge length labels (mid-segment); include rubber-band while drafting
        measure: List[LonLat] = list(poly)
        extra_px: Optional[Tuple[float, float]] = None
        if draft_cursor is not None and measure:
            lon_c, lat_c = self._pixel_to_lonlat(draft_cursor[0], draft_cursor[1])
            measure = measure + [(lon_c, lat_c)]
            extra_px = draft_cursor
        if len(measure) >= 2:
            edges = polygon_edge_lengths_m(measure, closed=closed and draft_cursor is None)
            px_ring = list(pts_px)
            if extra_px is not None:
                px_ring.append(extra_px)
            small = QFont()
            small.setPointSize(8)
            small.setBold(True)
            painter.setFont(small)
            for i, dist in enumerate(edges):
                if closed and draft_cursor is None:
                    a = px_ring[i]
                    b = px_ring[(i + 1) % len(px_ring)]
                else:
                    if i + 1 >= len(px_ring):
                        break
                    a = px_ring[i]
                    b = px_ring[i + 1]
                mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
                text = format_length_m(dist)
                br = painter.fontMetrics().boundingRect(text)
                pad = 3
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, 170))
                painter.drawRoundedRect(
                    int(mx - br.width() / 2 - pad),
                    int(my - br.height() / 2 - pad - 1),
                    br.width() + pad * 2,
                    br.height() + pad * 2,
                    3,
                    3,
                )
                painter.setPen(QColor(255, 240, 180))
                painter.drawText(int(mx - br.width() / 2), int(my + br.height() / 2 - 3), text)

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if len(self._ref_poly) >= 3:
            ref = QPolygonF([QPointF(px, py) for px, py in self._poly_to_px(self._ref_poly)])
            painter.setPen(QPen(QColor(255, 170, 40), 2, Qt.DashLine))
            painter.setBrush(QColor(255, 170, 40, 18))
            painter.drawPolygon(ref)

        if self._box_start_px and self._box_end_px:
            x0, y0 = self._box_start_px
            x1, y1 = self._box_end_px
            lon_a, lat_a = self._pixel_to_lonlat(x0, y0)
            lon_b, lat_b = self._pixel_to_lonlat(x1, y1)
            box_poly: List[LonLat] = [
                (min(lon_a, lon_b), max(lat_a, lat_b)),
                (max(lon_a, lon_b), max(lat_a, lat_b)),
                (max(lon_a, lon_b), min(lat_a, lat_b)),
                (min(lon_a, lon_b), min(lat_a, lat_b)),
            ]
            self._draw_numbered_poly(
                painter,
                box_poly,
                closed=True,
                fill=QColor(0, 180, 255, 50),
                stroke=QColor(0, 220, 255),
            )
        elif self._draft:
            self._draw_numbered_poly(
                painter,
                self._draft,
                closed=False,
                fill=QColor(0, 180, 255, 45),
                stroke=QColor(0, 220, 255),
                draft_cursor=self._draft_cursor,
            )
        elif self._poly:
            self._draw_numbered_poly(
                painter,
                self._poly,
                closed=True,
                fill=QColor(0, 255, 120, 40),
                stroke=QColor(0, 255, 120),
            )

        painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
        cx, cy = self.width() // 2, self.height() // 2
        painter.drawLine(cx - 8, cy, cx + 8, cy)
        painter.drawLine(cx, cy - 8, cx, cy + 8)

        self._draw_scale_bar(painter)
        painter.end()

    def _nice_scale_meters(self, target_m: float) -> float:
        """Round to a clean 1 / 2 / 5 × 10^n distance for the scale bar."""
        if target_m <= 0:
            return 1.0
        exp = math.floor(math.log10(target_m))
        base = 10.0**exp
        for mult in (1.0, 2.0, 5.0, 10.0):
            cand = mult * base
            if cand >= target_m * 0.6:
                return cand
        return 10.0 * base

    def _scale_bar_spec(self) -> Tuple[float, float, str]:
        """Return (bar_width_px, meters, label) for the current view."""
        mpp = meters_per_pixel(self.center_lat, self.zoom)
        target_px = 110.0
        nice_m = self._nice_scale_meters(mpp * target_px)
        bar_px = nice_m / max(mpp, 1e-9)
        if nice_m >= 1000:
            val = nice_m / 1000.0
            label = f"{val:g} km"
        else:
            label = f"{nice_m:g} m"
        return bar_px, nice_m, label

    def _schedule_elevation(self) -> None:
        # Debounce while panning/zooming; only refetch when center moved enough
        self._elev_timer.start(450)

    def _elev_needs_refresh(self, lat: float, lon: float) -> bool:
        if self._ground_elev_m is None or self._elev_at is None:
            return True
        # ~150 m at mid-latitudes
        return abs(lat - self._elev_at[0]) >= 0.0015 or abs(lon - self._elev_at[1]) >= 0.0015

    def _request_elevation(self) -> None:
        lat, lon = self.center_lat, self.center_lon
        if not self._elev_needs_refresh(lat, lon):
            return
        if self._elev_worker is not None and self._elev_worker.isRunning():
            self._elev_pending = (lat, lon)
            return
        self._elev_pending = None
        worker = ElevationWorker(lat, lon, parent=self)
        worker.finished_ok.connect(self._on_elevation_ok)
        worker.failed.connect(self._on_elevation_failed)
        self._elev_worker = worker
        worker.start()

    def _on_elevation_ok(self, elev_m: float, lat: float, lon: float) -> None:
        self._ground_elev_m = float(elev_m)
        self._elev_at = (float(lat), float(lon))
        self.update()
        pending = self._elev_pending
        if pending is not None and self._elev_needs_refresh(pending[0], pending[1]):
            self._elev_pending = None
            self._elev_timer.start(50)
        elif self._elev_needs_refresh(self.center_lat, self.center_lon):
            self._elev_timer.start(50)

    def _on_elevation_failed(self, err: str) -> None:
        logger.warning("Elevation unavailable: %s", err)
        if self._ground_elev_m is None:
            self.update()
        if self._elev_pending is not None or self._elev_needs_refresh(self.center_lat, self.center_lon):
            self._elev_pending = None
            # brief backoff then retry once center settles
            self._elev_timer.start(2000)

    def _draw_scale_bar(self, painter: QPainter) -> None:
        bar_px, _meters, label = self._scale_bar_spec()
        bar_px = max(40.0, min(bar_px, self.width() * 0.35))
        margin = 14
        h = self.height()
        x0 = margin
        y_bar = h - margin - 8

        mpp = meters_per_pixel(self.center_lat, self.zoom)
        alt = eye_altitude_m(self.center_lat, self.zoom)
        alt_line = f"ارتفاع از زمین ≈ {format_altitude_m(alt)}"
        if self._ground_elev_m is not None:
            ground_line = f"ارتفاع سطح زمین: {format_altitude_m(self._ground_elev_m, prefer_meters=True)}"
        else:
            ground_line = "ارتفاع سطح زمین: در حال دریافت…"
        gsd_line = f"{mpp:.2f} m/px" if mpp >= 0.1 else f"{mpp:.3f} m/px"

        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        lines = [alt_line, ground_line, gsd_line]
        text_w = max(fm.boundingRect(t).width() for t in lines)
        scale_w = bar_px + 8 + fm.boundingRect(label).width()
        pad = 6
        box_w = int(max(text_w, scale_w) + pad * 2)
        line_h = fm.height() + 2
        box_h = int(16 + line_h * 3 + pad * 2)
        box_top = int(y_bar - box_h + 10)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(x0 - pad, box_top, box_w, box_h, 5, 5)

        # Altitude first — viewing height of the imagery at this zoom
        painter.setPen(QColor(120, 220, 255))
        painter.drawText(int(x0), box_top + pad + line_h, alt_line)
        painter.setPen(QColor(180, 255, 180))
        painter.drawText(int(x0), box_top + pad + line_h * 2, ground_line)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(int(x0), box_top + pad + line_h * 3, gsd_line)

        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(int(x0), int(y_bar), int(x0 + bar_px), int(y_bar))
        painter.drawLine(int(x0), int(y_bar - 6), int(x0), int(y_bar + 2))
        painter.drawLine(int(x0 + bar_px), int(y_bar - 6), int(x0 + bar_px), int(y_bar + 2))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(int(x0 + bar_px + 8), int(y_bar + 4), label)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._schedule_render()

    def wheelEvent(self, event):  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            return
        mx, my = mouse_xy(event)
        lon_before, lat_before = self._pixel_to_lonlat(mx, my)
        new_zoom = self.zoom + (1 if delta > 0 else -1)
        new_zoom = max(3, min(19, new_zoom))
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom
        lon_after, lat_after = self._pixel_to_lonlat(mx, my)
        self.center_lon += lon_before - lon_after
        self.center_lat += lat_before - lat_after
        self._schedule_render()

    def keyPressEvent(self, event):  # noqa: N802
        key = event.key()
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            if self._draft:
                self.undo_last_point()
                return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self.finish_polygon():
                return
        if key == Qt.Key_Escape:
            if self._draft or self._box_start_px:
                self._cancel_draft()
                self.update()
                return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            x, y = mouse_xy(event)
            # Closed polygon: double-click a node to delete it (keep ≥3)
            if self._poly and not self._draft:
                node = self._hit_node(x, y, self._poly)
                if node is not None:
                    if len(self._poly) > 3:
                        del self._poly[node]
                        self._drag_node = None
                        self.selection_changed.emit()
                        self.update()
                    return
            if self.select_mode and self.select_tool == "polygon":
                if self.finish_polygon():
                    return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        self.setFocus(Qt.MouseFocusReason)
        x, y = mouse_xy(event)

        # Right-click finishes polygon (≥3 pts) or cancels empty draft
        if event.button() == Qt.RightButton:
            if self.select_mode and self.select_tool == "polygon":
                if len(self._draft) >= 3:
                    self.finish_polygon()
                elif self._draft:
                    self._cancel_draft()
                    self.update()
            return

        if event.button() != Qt.LeftButton:
            return

        # Edit closed polygon: drag node, insert on edge, or move whole shape
        if self._poly and not self._draft:
            node = self._hit_node(x, y, self._poly)
            if node is not None:
                self._drag_node = node
                self.setCursor(Qt.SizeAllCursor)
                return
            edge = self._hit_edge(x, y, self._poly)
            if edge is not None:
                insert_at, lonlat = edge
                self._poly.insert(insert_at, lonlat)
                self._drag_node = insert_at
                self.setCursor(Qt.SizeAllCursor)
                self.selection_changed.emit()
                self.update()
                return
            ring = self._poly_to_px(self._poly)
            if self._point_in_poly(x, y, ring):
                self._move_origin_px = (x, y)
                self._move_origin_poly = list(self._poly)
                self.setCursor(Qt.SizeAllCursor)
                return

        if self.select_mode:
            if self.select_tool == "box":
                self._poly = []
                self._box_start_px = (x, y)
                self._box_end_px = (x, y)
                self.update()
                return
            # Polygon: click near first vertex closes; otherwise append
            if len(self._draft) >= 3:
                fx, fy = self._lonlat_to_pixel(*self._draft[0])
                if math.hypot(x - fx, y - fy) <= HIT_PX * 1.6:
                    self.finish_polygon()
                    return
            if not self._draft:
                self._poly = []  # start fresh over previous selection
            lon, lat = self._pixel_to_lonlat(x, y)
            self._draft.append((lon, lat))
            self._draft_cursor = (x, y)
            self.selection_changed.emit()
            self.update()
            return

        self._drag_origin = (x, y)
        self._pan_origin_center = (self.center_lon, self.center_lat)
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):  # noqa: N802
        x, y = mouse_xy(event)

        if self._drag_node is not None and self._poly:
            lon, lat = self._pixel_to_lonlat(x, y)
            self._poly[self._drag_node] = (lon, lat)
            self.selection_changed.emit()
            self.update()
            return

        if self._move_origin_px is not None and self._move_origin_poly is not None:
            lon0, lat0 = self._pixel_to_lonlat(*self._move_origin_px)
            lon1, lat1 = self._pixel_to_lonlat(x, y)
            dlon, dlat = lon1 - lon0, lat1 - lat0
            self._poly = [(lon + dlon, lat + dlat) for lon, lat in self._move_origin_poly]
            self.selection_changed.emit()
            self.update()
            return

        if self._box_start_px is not None:
            self._box_end_px = (x, y)
            self.update()
            return

        if self.select_mode and self.select_tool == "polygon" and self._draft:
            self._draft_cursor = (x, y)
            self.update()
            return

        if self._drag_origin and self._pan_origin_center:
            dx = x - self._drag_origin[0]
            dy = y - self._drag_origin[1]
            dtx = -dx / TILE_SIZE
            dty = -dy / TILE_SIZE
            lon0, lat0 = self._pan_origin_center
            cx, cy = lonlat_to_tile(lon0, lat0, self.zoom)
            self.center_lon, self.center_lat = tile_to_lonlat(cx + dtx, cy + dty, self.zoom)
            self._schedule_render()
            return

        self._update_cursor((x, y))

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton:
            return

        if self._drag_node is not None:
            self._drag_node = None
            self._update_cursor(mouse_xy(event))
            return

        if self._move_origin_px is not None:
            self._move_origin_px = None
            self._move_origin_poly = None
            self._update_cursor(mouse_xy(event))
            return

        if self._box_start_px and self._box_end_px:
            x0, y0 = self._box_start_px
            x1, y1 = self._box_end_px
            self._box_start_px = None
            self._box_end_px = None
            if abs(x1 - x0) > 8 and abs(y1 - y0) > 8:
                lon_a, lat_a = self._pixel_to_lonlat(x0, y0)
                lon_b, lat_b = self._pixel_to_lonlat(x1, y1)
                south, north = min(lat_a, lat_b), max(lat_a, lat_b)
                west, east = min(lon_a, lon_b), max(lon_a, lon_b)
                self.set_selection_bbox(south, west, north, east)
            else:
                self.update()
            self._update_cursor(mouse_xy(event))
            return

        self._drag_origin = None
        self._pan_origin_center = None
        self._update_cursor(mouse_xy(event))


class SatelliteMapPanel(QWidget):
    """Map + search + basemap toggle + pan/select tools."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        search_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search place (e.g. Tehran, میدان آزادی, Shiraz)…")
        self.btn_search = QPushButton("Search")
        self.results = QComboBox()
        self.results.setMinimumWidth(220)
        self.results.setEnabled(False)
        self.results.addItem("Search results…")
        search_row.addWidget(self.search_edit, stretch=2)
        search_row.addWidget(self.btn_search)
        search_row.addWidget(self.results, stretch=2)
        layout.addLayout(search_row)

        bar = QHBoxLayout()
        self.btn_satellite = QPushButton("Satellite")
        self.btn_map = QPushButton("Map")
        self.btn_satellite.setCheckable(True)
        self.btn_map.setCheckable(True)
        self.btn_satellite.setChecked(True)

        self.btn_pan = QPushButton("Pan")
        self.btn_select = QPushButton("Select Region")
        self.btn_poly = QPushButton("Polygon")
        self.btn_box = QPushButton("Box")
        self.btn_undo = QPushButton("Undo point")
        self.btn_done = QPushButton("Finish")
        self.btn_clear = QPushButton("Clear")
        self.btn_pan.setCheckable(True)
        self.btn_select.setCheckable(True)
        self.btn_poly.setCheckable(True)
        self.btn_box.setCheckable(True)
        self.btn_pan.setChecked(True)
        self.btn_poly.setChecked(True)
        self.btn_poly.setToolTip(
            "Click vertices · double-click / Enter / right-click to close · "
            "drag nodes · click edge to add · double-click node to delete"
        )
        self.btn_box.setToolTip("Drag a rectangle (saved as a 4-point polygon you can still edit)")
        self.btn_undo.setToolTip("Remove the last polygon vertex (Backspace)")
        self.btn_done.setToolTip("Close the polygon (≥3 points)")
        self.btn_undo.setEnabled(False)
        self.btn_done.setEnabled(False)

        bar.addWidget(self.btn_satellite)
        bar.addWidget(self.btn_map)
        bar.addSpacing(12)
        bar.addWidget(self.btn_pan)
        bar.addWidget(self.btn_select)
        bar.addSpacing(8)
        bar.addWidget(self.btn_poly)
        bar.addWidget(self.btn_box)
        bar.addWidget(self.btn_undo)
        bar.addWidget(self.btn_done)
        bar.addWidget(self.btn_clear)
        bar.addStretch(1)
        layout.addLayout(bar)

        self.status = QLabel(
            "Search a place · Select Region · polygon nodes numbered · "
            "drag nodes · click edge to add · double-click node to delete · drag fill to move"
        )
        self.status.setStyleSheet("color:#aaa;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # Map + locator overview inset (bottom-right)
        self._map_host = _MapHost(self)
        host_lay = QVBoxLayout(self._map_host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        self.map = SatelliteMapWidget(self._map_host)
        host_lay.addWidget(self.map, stretch=1)
        from desktop_app.gui.map_overview import MapOverviewWidget

        self.overview = MapOverviewWidget(self._map_host)
        self.overview.raise_()
        self.overview.jump_requested.connect(self._overview_jump)
        layout.addWidget(self._map_host, stretch=1)

        self._hits: List[GeocodeHit] = []
        self._geo_worker: Optional[GeocodeWorker] = None

        self.btn_pan.clicked.connect(lambda: self._set_mode(False))
        self.btn_select.clicked.connect(lambda: self._set_mode(True))
        self.btn_poly.clicked.connect(lambda: self._set_tool("polygon"))
        self.btn_box.clicked.connect(lambda: self._set_tool("box"))
        self.btn_undo.clicked.connect(self.map.undo_last_point)
        self.btn_done.clicked.connect(self._finish)
        self.btn_clear.clicked.connect(self.map.clear_selection)
        self.btn_satellite.clicked.connect(lambda: self._set_basemap("satellite"))
        self.btn_map.clicked.connect(lambda: self._set_basemap("map"))
        self.btn_search.clicked.connect(self._run_search)
        self.search_edit.returnPressed.connect(self._run_search)
        self.results.activated.connect(self._on_result_chosen)
        self.map.selection_changed.connect(self._on_sel)
        self.map.view_changed.connect(self._on_view)
        self.map.selection_changed.connect(self._sync_overview)
        self.map.view_changed.connect(self._sync_overview)
        self.map.set_select_mode(False)
        self.map.set_select_tool("polygon")
        QTimer.singleShot(0, self._place_overview)
        QTimer.singleShot(80, self._sync_overview)

    def _set_basemap(self, layer: str) -> None:
        self.btn_satellite.setChecked(layer == "satellite")
        self.btn_map.setChecked(layer == "map")
        self.map.set_basemap(layer)
        self.status.setText(f"Basemap: {layer}")

    def _set_mode(self, select: bool) -> None:
        self.btn_pan.setChecked(not select)
        self.btn_select.setChecked(select)
        self.map.set_select_mode(select)
        if select:
            tool = "box" if self.btn_box.isChecked() else "polygon"
            hint = (
                "Drag a box on the map — corners become numbered nodes you can move"
                if tool == "box"
                else (
                    "Click vertices · Finish / double-click / right-click to close · "
                    "then drag nodes · click edge to add · double-click node to delete"
                )
            )
            self.status.setText(hint)

    def _set_tool(self, tool: str) -> None:
        self.btn_poly.setChecked(tool == "polygon")
        self.btn_box.setChecked(tool == "box")
        self.map.set_select_tool(tool)
        if not self.btn_select.isChecked():
            self._set_mode(True)
        else:
            self._set_mode(True)

    def _finish(self) -> None:
        if not self.map.finish_polygon():
            self.status.setText("Need at least 3 points to finish the polygon")

    def _run_search(self) -> None:
        q = self.search_edit.text().strip()
        if len(q) < 2:
            self.status.setText("Enter at least 2 characters to search")
            return
        if self._geo_worker is not None and self._geo_worker.isRunning():
            return
        self.btn_search.setEnabled(False)
        self.status.setText(f"Searching “{q}”…")
        self._geo_worker = GeocodeWorker(q)
        self._geo_worker.finished_ok.connect(self._on_search_ok)
        self._geo_worker.failed.connect(self._on_search_failed)
        self._geo_worker.start()

    def _on_search_ok(self, hits_obj: object) -> None:
        self.btn_search.setEnabled(True)
        hits: List[GeocodeHit] = list(hits_obj) if isinstance(hits_obj, list) else []
        self._hits = hits
        self.results.blockSignals(True)
        self.results.clear()
        if not hits:
            self.results.addItem("No results")
            self.results.setEnabled(False)
            self.status.setText("No places found")
            self.results.blockSignals(False)
            return
        self.results.addItem(f"{len(hits)} result(s) — pick one")
        for h in hits:
            label = h.display_name
            if len(label) > 90:
                label = label[:87] + "…"
            self.results.addItem(label)
        self.results.setEnabled(True)
        self.results.blockSignals(False)
        self.status.setText(f"Found {len(hits)} place(s)")
        self._apply_hit(hits[0])
        self.results.setCurrentIndex(1)

    def _on_search_failed(self, err: str) -> None:
        self.btn_search.setEnabled(True)
        self.status.setText(f"Search failed: {err}")

    def _on_result_chosen(self, index: int) -> None:
        if index <= 0 or index - 1 >= len(self._hits):
            return
        self._apply_hit(self._hits[index - 1])

    def _apply_hit(self, hit: GeocodeHit) -> None:
        if hit.has_bbox and hit.south is not None:
            self.map.fit_bbox(hit.south, hit.west, hit.north, hit.east)  # type: ignore[arg-type]
            self.map.set_selection_bbox(hit.south, hit.west, hit.north, hit.east)  # type: ignore[arg-type]
        else:
            self.map.go_to(hit.lat, hit.lon, zoom=max(self.map.zoom, 14))
        self.status.setText(hit.display_name[:120])

    def _on_sel(self) -> None:
        draft_n = len(self.map._draft)
        self.btn_undo.setEnabled(draft_n > 0)
        self.btn_done.setEnabled(draft_n >= 3)
        poly = self.map.selection_polygon()
        if draft_n:
            edges = polygon_edge_lengths_m(self.map._draft, closed=False)
            edge_txt = " · ".join(
                f"{i + 1}→{i + 2}: {format_length_m(d)}" for i, d in enumerate(edges)
            )
            more = " · Finish / double-click / right-click to close" if draft_n >= 3 else " · keep clicking"
            area_bit = ""
            if draft_n >= 3:
                area_bit = f" · area≈{format_area_m2(polygon_area_m2(self.map._draft))}"
            base = f"Drawing: {draft_n} pts{area_bit}{more}"
            self.status.setText(f"{base}" + (f"  |  {edge_txt}" if edge_txt else ""))
            return
        if not poly:
            self.status.setText("No region selected — Select Region, then click vertices or drag a Box")
            return
        m = self.map.selection_metrics()
        assert m is not None
        edges = m["edges_m"]  # type: ignore[assignment]
        edge_txt = " · ".join(
            f"{i + 1}→{(i + 1) % len(poly) + 1}: {format_length_m(d)}"  # type: ignore[arg-type]
            for i, d in enumerate(edges)  # type: ignore[arg-type]
        )
        self.status.setText(
            f"Area {format_area_m2(float(m['area_m2']))}  ·  "
            f"perimeter {format_length_m(float(m['perimeter_m']))}  ·  "
            f"{len(poly)} nodes  |  {edge_txt}"
        )

    def _on_view(self) -> None:
        if self.map.selection_polygon() or self.map._draft:
            self._on_sel()
            return
        loading = "  · loading…" if self.map._loading else ""
        self.status.setText(
            f"Center {self.map.center_lat:.5f}, {self.map.center_lon:.5f}  "
            f"zoom={self.map.zoom}  [{self.map.basemap}]{loading}"
        )

    def _place_overview(self) -> None:
        host = getattr(self, "_map_host", None)
        ov = getattr(self, "overview", None)
        if host is None or ov is None:
            return
        margin = 10
        ov.move(max(0, host.width() - ov.width() - margin), max(0, host.height() - ov.height() - margin))
        ov.raise_()
        ov.show()

    def _sync_overview(self) -> None:
        ov = getattr(self, "overview", None)
        if ov is None:
            return
        view = self.map.visible_bbox()
        poly = self.map.selection_polygon()
        ov.update_from_main(view, poly)

    def refresh_overview_offline(self) -> None:
        """Reload green offline high-zoom boxes on the locator inset."""
        ov = getattr(self, "overview", None)
        if ov is not None:
            ov.refresh_offline_boxes()
            self._sync_overview()

    def _overview_jump(self, lat: float, lon: float) -> None:
        self.map.go_to(lat, lon)
        self.status.setText(f"Jumped to {lat:.5f}, {lon:.5f}")


class _MapHost(QWidget):
    """Holds the main map and repositions the overview inset on resize."""

    def __init__(self, panel: "SatelliteMapPanel"):
        super().__init__()
        self._panel = panel

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._panel._place_overview()
