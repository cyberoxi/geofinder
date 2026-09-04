"""Optical flow and classic OpenCV trackers for inter-keyframe tracking."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.logger import get_logger
from core.roi_selector import clamp_polygon, create_roi_mask, is_valid_roi, polygon_to_bbox
from core.types import ROI

logger = get_logger("grt.optical_flow")


class OpticalFlowTracker:
    """Lucas-Kanade optical flow tracking of points inside the ROI."""

    def __init__(
        self,
        max_corners: int = 200,
        quality: float = 0.01,
        min_distance: float = 5.0,
        win_size: int = 21,
    ):
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_roi: Optional[ROI] = None
        self.initial_pts_local: Optional[np.ndarray] = None

    def _sample_points(self, gray: np.ndarray, roi: ROI) -> Optional[np.ndarray]:
        mask = create_roi_mask(gray.shape[:2], roi.points)
        pts = cv2.goodFeaturesToTrack(
            gray,
            mask=mask,
            maxCorners=self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_distance,
            blockSize=7,
        )
        return pts

    def initialize(self, frame: np.ndarray, roi: ROI) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = self._sample_points(gray, roi)
        if pts is None or len(pts) < 6:
            # Fallback: grid points inside bbox
            x, y, w, h = polygon_to_bbox(roi.points)
            xs = np.linspace(x + 0.2 * w, x + 0.8 * w, 5)
            ys = np.linspace(y + 0.2 * h, y + 0.8 * h, 5)
            grid = np.array([[xx, yy] for yy in ys for xx in xs], dtype=np.float32).reshape(-1, 1, 2)
            pts = grid
        self.prev_gray = gray
        self.prev_pts = pts
        self.prev_roi = roi.copy()
        return True

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[ROI], float]:
        if self.prev_gray is None or self.prev_pts is None or self.prev_roi is None:
            return False, None, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        if next_pts is None or status is None:
            return False, None, 0.0

        good_new = next_pts[status.ravel() == 1]
        good_old = self.prev_pts[status.ravel() == 1]
        if len(good_new) < 4:
            return False, None, 0.0

        # Estimate similarity / affine transform from point motion
        M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
        if M is None:
            # Translation-only fallback
            shift = (good_new - good_old).mean(axis=0).reshape(2)
            pts = self.prev_roi.points.copy()
            pts[:, 0] += float(shift[0])
            pts[:, 1] += float(shift[1])
            conf = min(1.0, len(good_new) / max(len(self.prev_pts), 1))
        else:
            ones = np.ones((len(self.prev_roi.points), 1), dtype=np.float32)
            homo = np.hstack([self.prev_roi.points.astype(np.float32), ones])
            pts = (M @ homo.T).T
            inl = int(inliers.ravel().sum()) if inliers is not None else len(good_new)
            conf = min(1.0, 0.4 + 0.6 * (inl / max(len(good_new), 1)))

        h, w = frame.shape[:2]
        if not np.all(np.isfinite(pts)):
            return False, None, 0.0
        # Soft validity: keep if some area remains inside frame
        clamped = clamp_polygon(pts, w, h)
        if not is_valid_roi(clamped, min_area=10.0):
            # Still return unclamped if mostly out — caller decides LOST
            if polygon_visible_ratio(pts, w, h) < 0.05:
                return False, None, conf

        roi = ROI(pts.astype(np.float32), self.prev_roi.roi_type, self.prev_roi.frame_index)
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        self.prev_roi = roi
        # Refresh points occasionally when too few remain
        if len(self.prev_pts) < 12:
            refreshed = self._sample_points(gray, roi)
            if refreshed is not None and len(refreshed) >= 6:
                self.prev_pts = refreshed
        return True, roi, float(conf)

    def reinitialize(self, frame: np.ndarray, roi: ROI) -> bool:
        return self.initialize(frame, roi)

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None
        self.prev_roi = None


def polygon_visible_ratio(points: np.ndarray, width: int, height: int) -> float:
    from core.roi_selector import polygon_area

    full = polygon_area(points)
    if full <= 1e-6:
        return 0.0
    clamped = clamp_polygon(points, width, height)
    return polygon_area(clamped) / full


class BoxTracker:
    """CSRT / KCF tracker wrapper with scale-aware bbox → polygon mapping."""

    def __init__(self, tracker_type: str = "CSRT"):
        self.tracker_type = tracker_type.upper()
        self._tracker = None
        self._roi: Optional[ROI] = None
        self._init_bbox: Optional[Tuple[float, float, float, float]] = None

    def _create(self):
        name = self.tracker_type
        creators = {}
        if hasattr(cv2, "TrackerCSRT_create"):
            creators["CSRT"] = cv2.TrackerCSRT_create
        if hasattr(cv2, "TrackerKCF_create"):
            creators["KCF"] = cv2.TrackerKCF_create
        # OpenCV contrib legacy API
        if hasattr(cv2, "legacy"):
            if hasattr(cv2.legacy, "TrackerCSRT_create"):
                creators.setdefault("CSRT", cv2.legacy.TrackerCSRT_create)
            if hasattr(cv2.legacy, "TrackerKCF_create"):
                creators.setdefault("KCF", cv2.legacy.TrackerKCF_create)
        if name not in creators:
            # Prefer CSRT, else KCF, else fail
            if "CSRT" in creators:
                name = "CSRT"
            elif "KCF" in creators:
                name = "KCF"
            else:
                raise RuntimeError("Neither CSRT nor KCF tracker is available in this OpenCV build")
        self.tracker_type = name
        return creators[name]()

    def initialize(self, frame: np.ndarray, roi: ROI) -> bool:
        self._tracker = self._create()
        x, y, w, h = roi.bbox
        bbox = (float(x), float(y), float(max(w, 2)), float(max(h, 2)))
        ok = self._tracker.init(frame, bbox)
        self._roi = roi.copy()
        self._init_bbox = bbox
        return bool(ok)

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[ROI], float]:
        if self._tracker is None or self._roi is None or self._init_bbox is None:
            return False, None, 0.0
        ok, bbox = self._tracker.update(frame)
        if not ok:
            return False, None, 0.0
        x, y, w, h = bbox
        # Map original polygon relative to initial bbox into new bbox (scale + translate)
        ix, iy, iw, ih = self._init_bbox
        if iw < 1 or ih < 1:
            return False, None, 0.0
        sx, sy = w / iw, h / ih
        pts = self._roi.points.copy()
        # Use relative coords from init bbox origin of original polygon at init time
        # Store was absolute; recompute from current using scale around bbox
        rel = np.empty_like(pts)
        rel[:, 0] = (pts[:, 0] - ix) / iw
        rel[:, 1] = (pts[:, 1] - iy) / ih
        # After first update, _roi may already be transformed — keep using ratio of corners to bbox
        # Safer: rebuild from rectangle if original was rectangle-like, else affine map bbox corners
        new_pts = np.column_stack([x + rel[:, 0] * w, y + rel[:, 1] * h]).astype(np.float32)
        # Update reference so subsequent relative mapping stays consistent
        self._roi = ROI(new_pts, self._roi.roi_type, self._roi.frame_index)
        self._init_bbox = (float(x), float(y), float(w), float(h))
        conf = 0.7
        return True, self._roi.copy(), conf

    def reinitialize(self, frame: np.ndarray, roi: ROI) -> bool:
        return self.initialize(frame, roi)

    def reset(self) -> None:
        self._tracker = None
        self._roi = None
        self._init_bbox = None
