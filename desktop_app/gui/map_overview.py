"""Locator inset: wider earth map with the main viewport highlighted."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from shared.qt_compat import (
    QLabel,
    QPainter,
    QPen,
    QColor,
    QPixmap,
    QTimer,
    Qt,
    Signal,
)
from desktop_app.gui.archive_cards import numpy_to_pixmap
from desktop_app.gui.workers import StageWorker
from desktop_app.satellite.preview_fetch import fetch_basemap_preview
from desktop_app.satellite.tiles import lonlat_to_tile, tile_to_lonlat

_W, _H = 200, 150
_EXPAND = 14.0  # overview covers ~14× the main viewport
# Arbitrary zoom — only relative Web Mercator tile coords matter for mapping.
_MAP_Z = 10
BBox = Tuple[float, float, float, float]  # south, west, north, east


def _expand_bbox(
    south: float, west: float, north: float, east: float, factor: float = _EXPAND
) -> Tuple[float, float, float, float]:
    lat_c = 0.5 * (south + north)
    lon_c = 0.5 * (west + east)
    half_lat = max(0.02, 0.5 * (north - south) * factor)
    half_lon = max(0.02, 0.5 * (east - west) * factor)
    # Widen longitude near poles so the inset still looks square-ish
    cos_lat = max(0.2, abs(np.cos(np.radians(lat_c))))
    half_lon = max(half_lon, half_lat / cos_lat)
    s = max(-85.0, lat_c - half_lat)
    n = min(85.0, lat_c + half_lat)
    w = max(-180.0, lon_c - half_lon)
    e = min(180.0, lon_c + half_lon)
    if n <= s:
        n = s + 0.05
    if e <= w:
        e = w + 0.05
    return s, w, n, e


def _lonlat_to_px(
    lon: float,
    lat: float,
    south: float,
    west: float,
    north: float,
    east: float,
    w: int,
    h: int,
) -> Tuple[float, float]:
    """Map lon/lat into mosaic pixels (Web Mercator — matches fetch_mosaic crop)."""
    x0, y0 = lonlat_to_tile(west, north, _MAP_Z)
    x1, y1 = lonlat_to_tile(east, south, _MAP_Z)
    tx, ty = lonlat_to_tile(lon, lat, _MAP_Z)
    x = (tx - x0) / max(1e-9, x1 - x0) * w
    y = (ty - y0) / max(1e-9, y1 - y0) * h
    return x, y


def _px_to_lonlat(
    px: float,
    py: float,
    south: float,
    west: float,
    north: float,
    east: float,
    w: int,
    h: int,
) -> Tuple[float, float]:
    """Inverse of ``_lonlat_to_px`` → (lon, lat)."""
    x0, y0 = lonlat_to_tile(west, north, _MAP_Z)
    x1, y1 = lonlat_to_tile(east, south, _MAP_Z)
    tx = x0 + (px / max(1, w)) * (x1 - x0)
    ty = y0 + (py / max(1, h)) * (y1 - y0)
    return tile_to_lonlat(tx, ty, _MAP_Z)


class MapOverviewWidget(QLabel):
    """
    Small street-map inset. Red box = what the main map shows;
    green boxes = offline high-zoom coverage; cyan = selected region.
    Click to re-center the main map.
    """

    jump_requested = Signal(float, float)  # lat, lon

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(_W, _H)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { background:#1a1a1a; color:#888; border:1px solid #555; border-radius:3px; }"
        )
        self.setToolTip(
            "Overview · نمای کلی\n"
            "Red = map view · Green = offline high-zoom · Cyan = selection\n"
            "قرمز = نمای نقشه · سبز = آفلاین زوم بالا · فیروزه‌ای = انتخاب"
        )
        self.setText("Overview…")
        self.setCursor(Qt.PointingHandCursor)

        self._view: Optional[BBox] = None
        self._overview: Optional[BBox] = None
        self._poly: Optional[List[Tuple[float, float]]] = None  # lon,lat
        self._offline_boxes: List[BBox] = []
        self._base: Optional[QPixmap] = None
        self._alive = True
        self._pending = False
        self._worker: Optional[StageWorker] = None
        self._fetched_for: Optional[BBox] = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self._fetch_now)
        self.refresh_offline_boxes()

    def refresh_offline_boxes(self) -> None:
        """Reload high-zoom offline coverage boxes from the package library."""
        try:
            from desktop_app.satellite.offline import high_zoom_region_boxes

            self._offline_boxes = high_zoom_region_boxes()
        except Exception:  # noqa: BLE001 — overview must never break the map
            self._offline_boxes = []
        self.update()

    def update_from_main(
        self,
        view_bbox: BBox,
        selection: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """view_bbox = (south, west, north, east); selection = [(lon, lat), ...]"""
        self._view = view_bbox
        self._poly = list(selection) if selection else None
        s, w, n, e = view_bbox
        ov = _expand_bbox(s, w, n, e)
        self._overview = ov
        # Only re-fetch when the overview window moved a lot
        if self._fetched_for is None or self._needs_refetch(ov):
            self._debounce.start()
        self.update()

    def shutdown(self) -> None:
        self._alive = False
        self._debounce.stop()
        self._pending = False
        w = self._worker
        self._worker = None
        if w is None:
            return
        try:
            w.finished_ok.disconnect(self._on_ready)
        except (TypeError, RuntimeError):
            pass
        if w.isRunning():
            w.finished.connect(w.deleteLater)
            if not w.wait(3000):
                w.terminate()
                w.wait(500)
        else:
            w.deleteLater()

    def _needs_refetch(self, ov: Tuple[float, float, float, float]) -> bool:
        prev = self._fetched_for
        if prev is None:
            return True
        ps, pw, pn, pe = prev
        s, w, n, e = ov
        lat_span = max(1e-6, pn - ps)
        lon_span = max(1e-6, pe - pw)
        # Refetch if center drifted >25% of previous overview span
        pc_lat, pc_lon = 0.5 * (ps + pn), 0.5 * (pw + pe)
        c_lat, c_lon = 0.5 * (s + n), 0.5 * (w + e)
        if abs(c_lat - pc_lat) > 0.25 * lat_span or abs(c_lon - pc_lon) > 0.25 * lon_span:
            return True
        # Or zoom changed a lot
        if abs((n - s) - lat_span) > 0.4 * lat_span:
            return True
        return False

    def _fetch_now(self) -> None:
        if not self._alive or self._overview is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._pending = True
            return
        south, west, north, east = self._overview
        target = (south, west, north, east)

        def job(progress, _frame):
            if progress:
                progress(0, 1, "Overview map")
            img = fetch_basemap_preview(
                south, west, north, east, "map", max_side=max(_W, _H) * 2, max_tiles=24
            )
            if progress:
                progress(1, 1, "Overview ready")
            return (target, img)

        self._worker = StageWorker(job)
        self._worker.finished_ok.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, _err: str) -> None:
        if not self._alive:
            return
        if self._pending:
            self._pending = False
            self._fetch_now()

    def _on_ready(self, payload: object) -> None:
        if not self._alive:
            return
        if isinstance(payload, tuple) and len(payload) == 2:
            target, img = payload
            if img is not None:
                # Only bind geo extent when the pixmap actually matches that bbox
                self._fetched_for = target
                dim = cv2.convertScaleAbs(img, alpha=0.85, beta=-8)
                self._base = numpy_to_pixmap(dim, _W - 4, _H - 4)
                self.setText("")
            elif self._base is None:
                self.setText("No overview")
        self.update()
        if self._pending:
            self._pending = False
            self._fetch_now()

    def _image_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Geographic extent of the currently displayed base image (not the live overview target)."""
        return self._fetched_for

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        geo = self._image_bbox()
        if self._base is None or geo is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        bw, bh = self._base.width(), self._base.height()
        ox = (self.width() - bw) // 2
        oy = (self.height() - bh) // 2
        p.drawPixmap(ox, oy, self._base)

        s, w, n, e = geo

        # Offline high-zoom coverage (green) — under selection / viewport
        if self._offline_boxes:
            fill = QColor(74, 222, 128, 45)
            pen = QPen(QColor(74, 222, 128, 230), 1.5, Qt.DashLine)
            p.setPen(pen)
            for box in self._offline_boxes:
                rect = self._box_to_rect(box, s, w, n, e, bw, bh)
                if rect is None:
                    continue
                left, top, rw, rh = rect
                p.fillRect(int(ox + left), int(oy + top), int(rw), int(rh), fill)
                p.drawRect(int(ox + left), int(oy + top), int(rw), int(rh))

        # Selection polygon
        if self._poly and len(self._poly) >= 3:
            pen = QPen(QColor(80, 220, 255, 220), 1.5)
            p.setPen(pen)
            pts = [
                _lonlat_to_px(lon, lat, s, w, n, e, bw, bh) for lon, lat in self._poly
            ]
            for i in range(len(pts)):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % len(pts)]
                p.drawLine(int(ox + x0), int(oy + y0), int(ox + x1), int(oy + y1))

        # Main viewport rectangle — mapped into the *fetched* image extent
        if self._view is not None:
            rect = self._box_to_rect(self._view, s, w, n, e, bw, bh, min_px=4)
            if rect is not None:
                left, top, rw, rh = rect
                pen = QPen(QColor(255, 60, 60, 240), 2)
                p.setPen(pen)
                p.drawRect(int(ox + left), int(oy + top), int(rw), int(rh))

        # Caption strip
        p.fillRect(0, self.height() - 16, self.width(), 16, QColor(0, 0, 0, 160))
        p.setPen(QColor(220, 220, 220))
        n_off = len(self._offline_boxes)
        cap = "Overview · نمای کلی"
        if n_off:
            cap = f"Overview · {n_off} offline"
        p.drawText(4, self.height() - 4, cap)
        p.end()

    @staticmethod
    def _box_to_rect(
        box: BBox,
        south: float,
        west: float,
        north: float,
        east: float,
        bw: int,
        bh: int,
        min_px: float = 3.0,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Map a geo bbox to (left, top, width, height) in mosaic pixels, or None if off-screen."""
        vs, vw, vn, ve = box
        # Skip boxes that do not intersect the overview image
        if vn <= south or vs >= north or ve <= west or vw >= east:
            return None
        x0, y0 = _lonlat_to_px(vw, vn, south, west, north, east, bw, bh)
        x1, y1 = _lonlat_to_px(ve, vs, south, west, north, east, bw, bh)
        left, right = min(x0, x1), max(x0, x1)
        top, bottom = min(y0, y1), max(y0, y1)
        left = max(0.0, min(float(bw - 1), left))
        right = max(0.0, min(float(bw - 1), right))
        top = max(0.0, min(float(bh - 1), top))
        bottom = max(0.0, min(float(bh - 1), bottom))
        if right - left < min_px:
            mid = 0.5 * (left + right)
            left, right = mid - 0.5 * min_px, mid + 0.5 * min_px
        if bottom - top < min_px:
            mid = 0.5 * (top + bottom)
            top, bottom = mid - 0.5 * min_px, mid + 0.5 * min_px
        return left, top, right - left, bottom - top

    def mousePressEvent(self, event):  # noqa: N802
        super().mousePressEvent(event)
        geo = self._image_bbox()
        if geo is None or self._base is None:
            return
        s, w, n, e = geo
        bw, bh = self._base.width(), self._base.height()
        ox = (self.width() - bw) // 2
        oy = (self.height() - bh) // 2
        px = event.position().x() if hasattr(event, "position") else event.x()
        py = event.position().y() if hasattr(event, "position") else event.y()
        x = float(px) - ox
        y = float(py) - oy
        if x < 0 or y < 0 or x > bw or y > bh:
            return
        lon, lat = _px_to_lonlat(x, y, s, w, n, e, bw, bh)
        self.jump_requested.emit(float(lat), float(lon))
