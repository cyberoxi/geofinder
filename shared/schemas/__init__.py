"""Shared schemas and data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ROIType(str, Enum):
    RECTANGLE = "rectangle"
    POLYGON = "polygon"


class AnnotationSource(str, Enum):
    MANUAL = "manual"
    INTERPOLATED = "interpolated"
    CONFIRMED = "confirmed"


class TrackingMode(str, Enum):
    MANUAL = "manual"
    YOLO = "yolo"
    HYBRID = "hybrid"


class RuntimeState(str, Enum):
    INIT = "INIT"
    SEARCHING = "SEARCHING"
    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    SCALE_CHANGE = "SCALE_CHANGE"
    RE_DETECTING = "RE_DETECTING"
    LOST = "LOST"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    READY = "READY"


class TrackingMethod(str, Enum):
    HOMOGRAPHY = "Homography"
    OPTICAL_FLOW = "OpticalFlow"
    CSRT = "CSRT"
    KCF = "KCF"
    YOLO = "YOLO"
    YOLO_SEG = "YOLO_SEG"
    FEATURE_MATCH = "FeatureMatch"
    LOST = "LOST"
    NONE = "None"


STATUS_COLORS = {
    RuntimeState.READY: (128, 128, 128),
    RuntimeState.INIT: (255, 165, 0),
    RuntimeState.SEARCHING: (255, 200, 0),
    RuntimeState.DETECTED: (0, 220, 120),
    RuntimeState.TRACKING: (0, 200, 0),
    RuntimeState.SCALE_CHANGE: (0, 180, 255),
    RuntimeState.RE_DETECTING: (0, 200, 255),
    RuntimeState.LOST: (0, 0, 255),
    RuntimeState.COMPLETED: (200, 200, 0),
    RuntimeState.ERROR: (0, 0, 180),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ROI:
    points: np.ndarray
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
        import cv2

        if len(self.points) < 3:
            return 0.0
        return float(abs(cv2.contourArea(self.points.astype(np.float32))))

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
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
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        return cls(pts, ROIType.RECTANGLE, frame_index)

    @classmethod
    def from_mask(cls, mask: np.ndarray, frame_index: int = 0) -> Optional["ROI"]:
        import cv2

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 10:
            return None
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
        if len(approx) < 3:
            x, y, w, h = cv2.boundingRect(contour)
            return cls.from_bbox(x, y, w, h, frame_index)
        return cls(approx, ROIType.POLYGON, frame_index)


@dataclass
class VideoAsset:
    video_id: str
    relative_path: str
    fps: float = 30.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration: float = 0.0
    display_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoAsset":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Annotation:
    video_id: str
    frame_id: int
    timestamp: float
    image_width: int
    image_height: int
    roi_type: str
    polygon_points: List[List[float]]
    bounding_box: List[float]  # x, y, w, h
    annotation_source: str = AnnotationSource.MANUAL.value
    zoom_level: float = 1.0
    quality_score: float = 1.0
    annotation_id: Optional[int] = None
    class_name: str = "target_region"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Annotation":
        fields = cls.__dataclass_fields__
        return cls(**{k: d[k] for k in fields if k in d})

    def to_roi(self) -> ROI:
        rtype = ROIType(self.roi_type) if self.roi_type in ROIType._value2member_map_ else ROIType.POLYGON
        return ROI(np.asarray(self.polygon_points, dtype=np.float32), rtype, self.frame_id)


@dataclass
class ProjectMeta:
    project_name: str
    target_class: str = "target_region"
    model_type: str = "segmentation"  # segmentation | detection
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    package_version: str = "1.0.0"
    feature_method: str = "AKAZE"
    videos: List[VideoAsset] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["videos"] = [v.to_dict() if isinstance(v, VideoAsset) else v for v in self.videos]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectMeta":
        videos = [VideoAsset.from_dict(v) for v in d.get("videos", [])]
        return cls(
            project_name=d.get("project_name", "untitled"),
            target_class=d.get("target_class", "target_region"),
            model_type=d.get("model_type", "segmentation"),
            created_at=d.get("created_at", _now_iso()),
            updated_at=d.get("updated_at", _now_iso()),
            package_version=d.get("package_version", "1.0.0"),
            feature_method=d.get("feature_method", "AKAZE"),
            videos=videos,
            settings=d.get("settings", {}),
        )


@dataclass
class Manifest:
    project_name: str
    target_class: str = "target_region"
    model_type: str = "segmentation"
    model_format: str = "onnx"
    input_size: int = 640
    reference_roi: Dict[str, Any] = field(default_factory=dict)
    feature_method: str = "AKAZE"
    min_confidence: float = 0.45
    min_iou: float = 0.20
    created_at: str = field(default_factory=_now_iso)
    package_version: str = "1.0.0"
    class_names: List[str] = field(default_factory=lambda: ["target_region"])
    has_engine: bool = False
    has_onnx: bool = True
    has_pytorch: bool = False
    # Relative path of the landmark scene layout (target + landmarks in mosaic px)
    scene_layout: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Manifest":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    polygon: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    source: str = "yolo"  # "yolo" or "landmarks" (target inferred from landmark geometry)


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
class TrackerState:
    status: RuntimeState = RuntimeState.READY
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
