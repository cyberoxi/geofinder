"""ROI geometry helpers and selection utilities."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.types import ROI, ROIType


def polygon_to_bbox(points: np.ndarray) -> Tuple[float, float, float, float]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x_min = float(pts[:, 0].min())
    y_min = float(pts[:, 1].min())
    x_max = float(pts[:, 0].max())
    y_max = float(pts[:, 1].max())
    return x_min, y_min, x_max - x_min, y_max - y_min


def bbox_to_polygon(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    return float(abs(cv2.contourArea(pts)))


def polygon_center(points: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    c = pts.mean(axis=0)
    return float(c[0]), float(c[1])


def clamp_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, max(width - 1, 0))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(height - 1, 0))
    return pts


def is_valid_roi(points: Sequence[Sequence[float]], min_area: float = 25.0) -> bool:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return False
    if np.any(~np.isfinite(pts)):
        return False
    return polygon_area(pts) >= min_area


def create_roi_mask(shape_hw: Tuple[int, int], points: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(np.asarray(points, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_roi_crop(frame: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cropped_image, local_mask) for the ROI bounding box."""
    x, y, w, h = polygon_to_bbox(points)
    x0 = max(int(np.floor(x)), 0)
    y0 = max(int(np.floor(y)), 0)
    x1 = min(int(np.ceil(x + w)), frame.shape[1])
    y1 = min(int(np.ceil(y + h)), frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid ROI crop bounds")
    crop = frame[y0:y1, x0:x1].copy()
    local_pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    local_pts[:, 0] -= x0
    local_pts[:, 1] -= y0
    mask = create_roi_mask(crop.shape[:2], local_pts)
    return crop, mask


def transform_polygon(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2)


def iou_bbox(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + bh
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def scale_from_polygons(prev: np.ndarray, curr: np.ndarray) -> float:
    a0 = polygon_area(prev)
    a1 = polygon_area(curr)
    if a0 <= 1e-6:
        return 1.0
    return float(np.sqrt(a1 / a0))


class ROISelector:
    """Non-GUI helper to build ROI objects from user points."""

    @staticmethod
    def from_rectangle(
        x1: float, y1: float, x2: float, y2: float, frame_index: int = 0
    ) -> ROI:
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        if (x_max - x_min) < 2 or (y_max - y_min) < 2:
            raise ValueError("Rectangle ROI is too small")
        return ROI.from_bbox(x_min, y_min, x_max - x_min, y_max - y_min, frame_index)

    @staticmethod
    def from_polygon(points: List[List[float]], frame_index: int = 0) -> ROI:
        if not is_valid_roi(points):
            raise ValueError("Polygon ROI is invalid (need >= 3 points and sufficient area)")
        return ROI(np.asarray(points, dtype=np.float32), ROIType.POLYGON, frame_index)
