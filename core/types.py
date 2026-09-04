"""Shared types, enums, and data structures for Ground Region Tracker."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TrackingMode(str, Enum):
    MANUAL = "manual"
    YOLO = "yolo"
    HYBRID = "hybrid"


class TrackingStatus(str, Enum):
    READY = "READY"
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    RE_DETECTING = "RE-DETECTING"
    LOST = "LOST"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class TrackingMethod(str, Enum):
    HOMOGRAPHY = "Homography"
    OPTICAL_FLOW = "OpticalFlow"
    CSRT = "CSRT"
    KCF = "KCF"
    YOLO = "YOLO"
    YOLO_SEG = "YOLO_SEG"
    LOST = "LOST"
    NONE = "None"


class ROIType(str, Enum):
    RECTANGLE = "rectangle"
    POLYGON = "polygon"


STATUS_COLORS = {
    TrackingStatus.READY: (128, 128, 128),
    TrackingStatus.INITIALIZING: (255, 165, 0),
    TrackingStatus.TRACKING: (0, 200, 0),
    TrackingStatus.RE_DETECTING: (0, 200, 255),
    TrackingStatus.LOST: (0, 0, 255),
    TrackingStatus.COMPLETED: (200, 200, 0),
    TrackingStatus.ERROR: (0, 0, 180),
}


@dataclass
class ROI:
    """Region of Interest defined by polygon points in image coordinates."""

    points: np.ndarray  # shape (N, 2) float32
    roi_type: ROIType = ROIType.POLYGON
    frame_index: int = 0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)

    @property
    def center(self) -> Tuple[float, float]:
        c = self.points.mean(axis=0)
        return float(c[0]), float(c[1])

    @property
    def area(self) -> float:
        if len(self.points) < 3:
            return 0.0
        return float(abs(cv2_contour_area(self.points)))

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return (x, y, w, h)."""
        x_min = float(self.points[:, 0].min())
        y_min = float(self.points[:, 1].min())
        x_max = float(self.points[:, 0].max())
        y_max = float(self.points[:, 1].max())
        return x_min, y_min, x_max - x_min, y_max - y_min

    def as_int_points(self) -> np.ndarray:
        return np.round(self.points).astype(np.int32)

    def to_list(self) -> List[List[float]]:
        return [[float(x), float(y)] for x, y in self.points]

    def copy(self) -> "ROI":
        return ROI(self.points.copy(), self.roi_type, self.frame_index)

    def scale_about_center(self, scale: float) -> "ROI":
        cx, cy = self.center
        pts = self.points.copy()
        pts[:, 0] = cx + (pts[:, 0] - cx) * scale
        pts[:, 1] = cy + (pts[:, 1] - cy) * scale
        return ROI(pts, self.roi_type, self.frame_index)

    @classmethod
    def from_bbox(cls, x: float, y: float, w: float, h: float, frame_index: int = 0) -> "ROI":
        pts = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.float32,
        )
        return cls(pts, ROIType.RECTANGLE, frame_index)

    @classmethod
    def from_mask(cls, mask: np.ndarray, frame_index: int = 0) -> Optional["ROI"]:
        import cv2

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 10:
            return None
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            x, y, w, h = cv2.boundingRect(contour)
            return cls.from_bbox(x, y, w, h, frame_index)
        return cls(pts, ROIType.POLYGON, frame_index)


def cv2_contour_area(points: np.ndarray) -> float:
    import cv2

    return float(cv2.contourArea(points.astype(np.float32)))


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    center_x: float
    center_y: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    polygon_points: List[List[float]]
    confidence: float
    tracking_method: str
    inlier_count: int
    reprojection_error: float
    scale: float
    status: str
    inference_ms: float = 0.0
    process_fps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    class_id: int = 0
    polygon: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


@dataclass
class TrackerState:
    status: TrackingStatus = TrackingStatus.READY
    method: TrackingMethod = TrackingMethod.NONE
    roi: Optional[ROI] = None
    confidence: float = 0.0
    inlier_count: int = 0
    reprojection_error: float = 0.0
    scale: float = 1.0
    lost_count: int = 0
    inference_ms: float = 0.0
    message: str = ""
    trail: List[Tuple[float, float]] = field(default_factory=list)
