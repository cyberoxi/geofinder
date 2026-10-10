"""Annotation frame canvas for the desktop GUI."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from shared.geometry import ROISelector
from shared.qt_compat import (
    Qt,
    QImage,
    QLabel,
    QPixmap,
    Signal,
    mouse_xy,
)


class FrameCanvas(QLabel):
    roi_done = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background:#0b1220;color:#64748b;border:1px solid #243041;border-radius:8px;"
        )
        self.setText("Open a project and select a video")
        self._frame = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._mode = None
        self._points: List = []
        self._rect_start = None
        self._cursor = None
        self._overlay_ann = None
        self._overlay_objects: List = []
        self.setMouseTracking(True)

    def set_mode(self, mode: Optional[str]) -> None:
        self._mode = mode
        self._points = []
        self._rect_start = None
        self.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)

    def clear(self) -> None:
        self._frame = None
        self._overlay_ann = None
        self._overlay_objects = []
        self._points = []
        self._rect_start = None
        self._cursor = None
        self._mode = None
        self.setCursor(Qt.ArrowCursor)
        super().clear()
        self.setText("Open a project and select a video")

    def show_frame(self, frame, annotation=None, objects=None) -> None:
        """``objects``: auto labels [(class_id, polygon, name), ...]."""
        self._frame = frame
        self._overlay_ann = annotation
        self._overlay_objects = objects or []
        self._paint()

    def _paint(self) -> None:
        if self._frame is None:
            return
        img = self._frame.copy()
        if self._overlay_ann is not None:
            pts = np.array(self._overlay_ann.polygon_points, dtype=np.int32)
            color = (0, 255, 255) if self._overlay_ann.annotation_source == "interpolated" else (0, 220, 0)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
        for cid, poly, name in self._overlay_objects:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            color = (0, 220, 0) if cid == 0 else (0, 160, 255)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
            x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
            cv2.putText(img, name, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if self._mode == "polygon" and self._points:
            pts = np.array(self._points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)
        if self._mode == "rectangle" and self._rect_start and self._cursor:
            cv2.rectangle(img, (int(self._rect_start[0]), int(self._rect_start[1])), (int(self._cursor[0]), int(self._cursor[1])), (0, 255, 255), 2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy()).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scale = pix.width() / w
        self._offset = ((self.width() - pix.width()) // 2, (self.height() - pix.height()) // 2)
        self.setPixmap(pix)

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._paint()

    def _to_image(self, x, y):
        if self._frame is None or self._scale <= 0:
            return None
        ix = (x - self._offset[0]) / self._scale
        iy = (y - self._offset[1]) / self._scale
        h, w = self._frame.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            return float(ix), float(iy)
        return None

    def mousePressEvent(self, e):  # noqa: N802
        pt = self._to_image(*mouse_xy(e))
        if not pt or not self._mode:
            return
        if self._mode == "rectangle" and e.button() == Qt.LeftButton:
            self._rect_start = pt
            self._cursor = pt
        elif self._mode == "polygon":
            if e.button() == Qt.LeftButton:
                self._points.append(pt)
            elif e.button() == Qt.RightButton and len(self._points) >= 3:
                self._finish_poly()
        self._paint()

    def mouseMoveEvent(self, e):  # noqa: N802
        pt = self._to_image(*mouse_xy(e))
        if pt:
            self._cursor = pt
            self._paint()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self._mode == "rectangle" and self._rect_start and e.button() == Qt.LeftButton:
            pt = self._to_image(*mouse_xy(e))
            if not pt:
                return
            try:
                roi = ROISelector.from_rectangle(self._rect_start[0], self._rect_start[1], pt[0], pt[1])
                self.roi_done.emit(roi)
                self._mode = None
                self.setCursor(Qt.ArrowCursor)
            except ValueError:
                pass
            self._rect_start = None
            self._paint()

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if self._mode == "polygon" and len(self._points) >= 3:
            self._finish_poly()

    def _finish_poly(self):
        try:
            roi = ROISelector.from_polygon([[p[0], p[1]] for p in self._points])
            self.roi_done.emit(roi)
            self._mode = None
            self.setCursor(Qt.ArrowCursor)
        except ValueError:
            pass
        self._paint()
