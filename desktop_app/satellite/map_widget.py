"""Interactive map with satellite/street basemap, search, and region selection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from desktop_app.satellite.geocode import GeocodeHit, search_places
from desktop_app.satellite.tiles import (
    LAYERS,
    TILE_SIZE,
    download_tile,
    lonlat_to_tile,
    tile_to_lonlat,
    try_load_cached_tile,
)
from shared.qt_compat import (
    QColor,
    QComboBox,
    QImage,
    QLabel,
    QLineEdit,
    QPainter,
    QPen,
    QPixmap,
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


class TileFetchWorker(QThread):
    """Download missing map tiles off the UI thread."""

    finished_ok = Signal(int)  # request_id
    failed = Signal(int, str)

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
        try:
            for z, x, y, layer in self.keys:
                if self._cancel:
                    break
                try:
                    download_tile(z, x, y, self.cache_dir, retries=2, layer=layer, timeout=12.0)
                except Exception:  # noqa: BLE001
                    continue
            self.finished_ok.emit(self.request_id)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.request_id, str(exc))


class SatelliteMapWidget(QLabel):
    """
    Pan (left-drag) / zoom (wheel).
    Basemap: satellite or street map.
    In select mode, left-drag draws a geographic bbox.
    """

    view_changed = Signal()
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(720, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background:#111;border:1px solid #333;")

        self.center_lat = 35.6892
        self.center_lon = 51.3890
        self.zoom = 15
        self.select_mode = False
        self.basemap = "satellite"
        self.cache_dir = Path("data") / "satellite_cache"

        self._drag_origin: Optional[Tuple[float, float]] = None
        self._pan_origin_center: Optional[Tuple[float, float]] = None
        self._sel_start_px: Optional[Tuple[float, float]] = None
        self._sel_end_px: Optional[Tuple[float, float]] = None
        self._selection: Optional[Tuple[float, float, float, float]] = None

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._render)
        self._base_pixmap: Optional[QPixmap] = None

        self._tile_worker: Optional[TileFetchWorker] = None
        self._tile_req_id = 0
        self._pending_keys: Set[TileKey] = set()
        self._loading = False

        QTimer.singleShot(50, self._render)

    def set_basemap(self, layer: str) -> None:
        key = layer.strip().lower()
        if key not in LAYERS:
            return
        if key == self.basemap:
            return
        self.basemap = key
        self._schedule_render()

    def set_select_mode(self, enabled: bool) -> None:
        self.select_mode = bool(enabled)
        self.setCursor(Qt.CrossCursor if self.select_mode else Qt.OpenHandCursor)

    def clear_selection(self) -> None:
        self._selection = None
        self._sel_start_px = None
        self._sel_end_px = None
        self.selection_changed.emit()
        self.update()

    def selection_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        return self._selection

    def set_selection_bbox(self, south: float, west: float, north: float, east: float) -> None:
        if north <= south or east <= west:
            return
        self._selection = (south, west, north, east)
        self.selection_changed.emit()
        self.update()

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

        tx0, ty0, tx1, ty1, left, top = self._visible_tile_range()
        n_world = 2**self.zoom
        attr = LAYERS.get(self.basemap, LAYERS["satellite"])["attribution"]
        missing: List[TileKey] = []

        for ty in range(ty0, ty1 + 1):
            if ty < 0 or ty >= n_world:
                continue
            for tx in range(tx0, tx1 + 1):
                wrapped = tx % n_world
                tile = try_load_cached_tile(
                    self.zoom, wrapped, ty, self.cache_dir, layer=self.basemap
                )
                if tile is None:
                    missing.append((self.zoom, wrapped, ty, self.basemap))
                    continue
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
        if missing:
            status = f"{attr}  ·  loading {len(missing)} tile(s)…"
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
        # Already downloading a superset / same set — wait for that worker.
        if self._tile_worker is not None and self._tile_worker.isRunning():
            if needed <= self._pending_keys:
                return
            self._tile_worker.cancel()
        self._pending_keys = needed
        self._loading = True
        self._tile_req_id += 1
        req_id = self._tile_req_id
        worker = TileFetchWorker(req_id, list(needed), self.cache_dir, parent=self)
        worker.finished_ok.connect(self._on_tiles_ready)
        worker.failed.connect(self._on_tiles_failed)
        self._tile_worker = worker
        worker.start()

    def _on_tiles_ready(self, request_id: int) -> None:
        if request_id != self._tile_req_id:
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

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._sel_start_px and self._sel_end_px:
            x0, y0 = self._sel_start_px
            x1, y1 = self._sel_end_px
            pen = QPen(QColor(0, 220, 255), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 180, 255, 50))
            painter.drawRect(int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))
        elif self._selection:
            s, west, n, e = self._selection
            px0, py0 = self._lonlat_to_pixel(west, n)
            px1, py1 = self._lonlat_to_pixel(e, s)
            pen = QPen(QColor(0, 255, 120), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 120, 40))
            painter.drawRect(int(min(px0, px1)), int(min(py0, py1)), int(abs(px1 - px0)), int(abs(py1 - py0)))

        painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
        cx, cy = self.width() // 2, self.height() // 2
        painter.drawLine(cx - 8, cy, cx + 8, cy)
        painter.drawLine(cx, cy - 8, cx, cy + 8)
        painter.end()

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

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        x, y = mouse_xy(event)
        if self.select_mode:
            self._sel_start_px = (x, y)
            self._sel_end_px = (x, y)
            self.update()
        else:
            self._drag_origin = (x, y)
            self._pan_origin_center = (self.center_lon, self.center_lat)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):  # noqa: N802
        x, y = mouse_xy(event)
        if self.select_mode and self._sel_start_px is not None:
            self._sel_end_px = (x, y)
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

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        if self.select_mode and self._sel_start_px and self._sel_end_px:
            x0, y0 = self._sel_start_px
            x1, y1 = self._sel_end_px
            if abs(x1 - x0) > 8 and abs(y1 - y0) > 8:
                lon_a, lat_a = self._pixel_to_lonlat(x0, y0)
                lon_b, lat_b = self._pixel_to_lonlat(x1, y1)
                south, north = min(lat_a, lat_b), max(lat_a, lat_b)
                west, east = min(lon_a, lon_b), max(lon_a, lon_b)
                self._selection = (south, west, north, east)
                self.selection_changed.emit()
            self._sel_start_px = None
            self._sel_end_px = None
            self.update()
        self._drag_origin = None
        self._pan_origin_center = None
        self.setCursor(Qt.CrossCursor if self.select_mode else Qt.OpenHandCursor)


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
        self.btn_clear = QPushButton("Clear Selection")
        self.btn_pan.setCheckable(True)
        self.btn_select.setCheckable(True)
        self.btn_pan.setChecked(True)

        bar.addWidget(self.btn_satellite)
        bar.addWidget(self.btn_map)
        bar.addSpacing(12)
        bar.addWidget(self.btn_pan)
        bar.addWidget(self.btn_select)
        bar.addWidget(self.btn_clear)
        bar.addStretch(1)
        self.status = QLabel("Search a place · switch Satellite/Map · Select Region")
        self.status.setStyleSheet("color:#aaa;")
        bar.addWidget(self.status)
        layout.addLayout(bar)

        self.map = SatelliteMapWidget()
        layout.addWidget(self.map, stretch=1)

        self._hits: List[GeocodeHit] = []
        self._geo_worker: Optional[GeocodeWorker] = None

        self.btn_pan.clicked.connect(lambda: self._set_mode(False))
        self.btn_select.clicked.connect(lambda: self._set_mode(True))
        self.btn_clear.clicked.connect(self.map.clear_selection)
        self.btn_satellite.clicked.connect(lambda: self._set_basemap("satellite"))
        self.btn_map.clicked.connect(lambda: self._set_basemap("map"))
        self.btn_search.clicked.connect(self._run_search)
        self.search_edit.returnPressed.connect(self._run_search)
        self.results.activated.connect(self._on_result_chosen)
        self.map.selection_changed.connect(self._on_sel)
        self.map.view_changed.connect(self._on_view)
        self.map.set_select_mode(False)

    def _set_basemap(self, layer: str) -> None:
        self.btn_satellite.setChecked(layer == "satellite")
        self.btn_map.setChecked(layer == "map")
        self.map.set_basemap(layer)
        self.status.setText(f"Basemap: {layer}")

    def _set_mode(self, select: bool) -> None:
        self.btn_pan.setChecked(not select)
        self.btn_select.setChecked(select)
        self.map.set_select_mode(select)

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
        bbox = self.map.selection_bbox()
        if not bbox:
            self.status.setText("No region selected")
            return
        s, w, n, e = bbox
        self.status.setText(
            f"Selected  S={s:.5f}  W={w:.5f}  N={n:.5f}  E={e:.5f}  "
            f"(z={self.map.zoom}, {self.map.basemap})"
        )

    def _on_view(self) -> None:
        if self.map.selection_bbox():
            self._on_sel()
            return
        loading = "  · loading…" if self.map._loading else ""
        self.status.setText(
            f"Center {self.map.center_lat:.5f}, {self.map.center_lon:.5f}  "
            f"zoom={self.map.zoom}  [{self.map.basemap}]{loading}"
        )
