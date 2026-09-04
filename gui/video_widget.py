"""Video display widget with ROI selection overlays."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from gui.qt_compat import Qt, Signal, QImage, QPixmap, QMouseEvent, QLabel, QSizePolicy, mouse_xy

from core.types import ROI, ROIType, STATUS_COLORS, TrackingStatus


class VideoWidget(QLabel):
    roi_completed = Signal(object)  # ROI
    roi_updated = Signal(list)  # points so far

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #1a1a1a; color: #aaa;")
        self.setText("Open a video to begin")
        self.setMouseTracking(True)

        self._frame: Optional[np.ndarray] = None
        self._display: Optional[np.ndarray] = None
        self._scale = 1.0
        self._offset = (0, 0)

        self._mode: Optional[str] = None  # rectangle | polygon | None
        self._points: List[Tuple[float, float]] = []
        self._rect_start: Optional[Tuple[float, float]] = None
        self._cursor: Optional[Tuple[float, float]] = None
        self._confirmed_roi: Optional[ROI] = None
        self._overlay_info: dict = {}
        self._trail: List[Tuple[float, float]] = []

    def set_selection_mode(self, mode: Optional[str]) -> None:
        self._mode = mode
        self._points = []
        self._rect_start = None
        self._cursor = None
        if mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self._repaint_frame()

    def clear_roi(self) -> None:
        self._confirmed_roi = None
        self._points = []
        self._trail = []
        self._repaint_frame()

    def set_roi(self, roi: ROI) -> None:
        self._confirmed_roi = roi
        self._repaint_frame()

    def set_overlay(self, info: dict, trail: Optional[list] = None) -> None:
        self._overlay_info = info or {}
        if trail is not None:
            self._trail = list(trail)
        self._repaint_frame()

    def show_frame(self, frame: np.ndarray, overlay_info: Optional[dict] = None) -> None:
        self._frame = frame
        if overlay_info is not None:
            self._overlay_info = overlay_info
        self._repaint_frame()

    def _image_to_widget(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._scale + self._offset[0], y * self._scale + self._offset[1]

    def _widget_to_image(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if self._frame is None or self._scale <= 0:
            return None
        ix = (x - self._offset[0]) / self._scale
        iy = (y - self._offset[1]) / self._scale
        h, w = self._frame.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return float(ix), float(iy)

    def _compose(self) -> Optional[np.ndarray]:
        if self._frame is None:
            return None
        img = self._frame.copy()

        # Confirmed / live ROI
        roi = self._confirmed_roi
        color = (0, 220, 0)
        status = self._overlay_info.get("status")
        if status:
            try:
                color = STATUS_COLORS.get(TrackingStatus(status), color)
            except ValueError:
                pass

        if roi is not None and len(roi.points) >= 2:
            pts = roi.as_int_points().reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, color, 2)
            cx, cy = map(int, roi.center)
            cv2.circle(img, (cx, cy), 4, color, -1)

        if self._trail and len(self._trail) > 1:
            for i in range(1, len(self._trail)):
                p1 = (int(self._trail[i - 1][0]), int(self._trail[i - 1][1]))
                p2 = (int(self._trail[i][0]), int(self._trail[i][1]))
                cv2.line(img, p1, p2, (0, 255, 255), 1)

        # In-progress selection
        if self._mode == "polygon" and self._points:
            pts = np.array(self._points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)
            for p in self._points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
            if self._cursor:
                cv2.line(
                    img,
                    (int(self._points[-1][0]), int(self._points[-1][1])),
                    (int(self._cursor[0]), int(self._cursor[1])),
                    (0, 200, 255),
                    1,
                )

        if self._mode == "rectangle" and self._rect_start and self._cursor:
            x1, y1 = self._rect_start
            x2, y2 = self._cursor
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # HUD text
        if self._overlay_info:
            y0 = 24
            keys = [
                ("status", "Status"),
                ("tracking_method", "Method"),
                ("confidence", "Conf"),
                ("inlier_count", "Inliers"),
                ("process_fps", "FPS"),
                ("inference_ms", "Infer ms"),
            ]
            for i, (k, label) in enumerate(keys):
                if k not in self._overlay_info:
                    continue
                val = self._overlay_info[k]
                if isinstance(val, float):
                    text = f"{label}: {val:.2f}"
                else:
                    text = f"{label}: {val}"
                cv2.putText(img, text, (10, y0 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(img, text, (10, y0 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        return img

    def _repaint_frame(self) -> None:
        composed = self._compose()
        if composed is None:
            return
        self._display = composed
        rgb = cv2.cvtColor(composed, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        scaled = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scale = scaled.width() / w
        ox = (self.width() - scaled.width()) // 2
        oy = (self.height() - scaled.height()) // 2
        self._offset = (ox, oy)
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._repaint_frame()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mode is None or self._frame is None:
            return
        pt = self._widget_to_image(*mouse_xy(event))
        if pt is None:
            return
        if self._mode == "rectangle":
            if event.button() == Qt.LeftButton:
                self._rect_start = pt
                self._cursor = pt
        elif self._mode == "polygon":
            if event.button() == Qt.LeftButton:
                self._points.append(pt)
                self.roi_updated.emit(self._points)
            elif event.button() == Qt.RightButton and len(self._points) >= 3:
                self._finish_polygon()
        self._repaint_frame()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mode is None:
            return
        pt = self._widget_to_image(*mouse_xy(event))
        if pt is None:
            return
        self._cursor = pt
        self._repaint_frame()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mode == "rectangle" and self._rect_start and event.button() == Qt.LeftButton:
            pt = self._widget_to_image(*mouse_xy(event))
            if pt is None:
                return
            x1, y1 = self._rect_start
            x2, y2 = pt
            from core.roi_selector import ROISelector

            try:
                roi = ROISelector.from_rectangle(x1, y1, x2, y2)
                self._confirmed_roi = roi
                self._mode = None
                self.setCursor(Qt.ArrowCursor)
                self.roi_completed.emit(roi)
            except ValueError:
                pass
            self._rect_start = None
            self._repaint_frame()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._mode == "polygon" and len(self._points) >= 3:
            self._finish_polygon()

    def _finish_polygon(self) -> None:
        from core.roi_selector import ROISelector

        try:
            roi = ROISelector.from_polygon([[p[0], p[1]] for p in self._points])
            self._confirmed_roi = roi
            self._mode = None
            self.setCursor(Qt.ArrowCursor)
            self.roi_completed.emit(roi)
        except ValueError:
            pass
        self._repaint_frame()
