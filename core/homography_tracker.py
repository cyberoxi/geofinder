"""Homography-based ROI propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from core.feature_tracker import FeatureSet, FeatureTracker
from core.logger import get_logger
from core.roi_selector import (
    clamp_polygon,
    is_valid_roi,
    polygon_area,
    scale_from_polygons,
    transform_polygon,
)
from core.types import ROI

logger = get_logger("grt.homography")


@dataclass
class HomographyResult:
    success: bool
    H: Optional[np.ndarray]
    polygon: Optional[np.ndarray]
    inliers: int
    reprojection_error: float
    scale: float
    message: str = ""


class HomographyTracker:
    def __init__(
        self,
        feature_tracker: Optional[FeatureTracker] = None,
        min_inliers: int = 12,
        max_reprojection_error: float = 4.0,
        min_scale: float = 0.15,
        max_scale: float = 8.0,
        max_area_change: float = 4.0,
        ransac_thresh: float = 3.0,
    ):
        self.features = feature_tracker or FeatureTracker()
        self.min_inliers = min_inliers
        self.max_reprojection_error = max_reprojection_error
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_area_change = max_area_change
        self.ransac_thresh = ransac_thresh

        self.ref_roi: Optional[ROI] = None
        self.ref_features: Optional[FeatureSet] = None
        self.ref_frame: Optional[np.ndarray] = None
        self.current_roi: Optional[ROI] = None
        self.initial_area: float = 0.0

    def initialize(self, frame: np.ndarray, roi: ROI) -> bool:
        if not is_valid_roi(roi.points):
            logger.error("Cannot initialize HomographyTracker with invalid ROI")
            return False
        self.ref_roi = roi.copy()
        self.current_roi = roi.copy()
        self.ref_frame = frame.copy()
        self.ref_features = self.features.extract_roi(frame, roi.points)
        self.initial_area = max(polygon_area(roi.points), 1.0)
        if not self.ref_features.keypoints:
            logger.warning("No features found in initial ROI")
            return False
        logger.info("HomographyTracker initialized with %d features", len(self.ref_features.keypoints))
        return True

    def update_keyframe(self, frame: np.ndarray, roi: ROI) -> None:
        self.ref_roi = roi.copy()
        self.current_roi = roi.copy()
        self.ref_frame = frame.copy()
        self.ref_features = self.features.extract_roi(frame, roi.points)

    def estimate(
        self,
        frame: np.ndarray,
        prev_roi: Optional[ROI] = None,
        use_full_frame: bool = False,
    ) -> HomographyResult:
        if self.ref_features is None or self.ref_roi is None:
            return HomographyResult(False, None, None, 0, 1e9, 1.0, "Not initialized")

        if use_full_frame:
            curr_feats = self.features.extract(frame)
        else:
            search_roi = prev_roi or self.current_roi or self.ref_roi
            # Expand search region for zoom / camera motion
            expanded = search_roi.scale_about_center(1.8)
            h, w = frame.shape[:2]
            expanded_pts = clamp_polygon(expanded.points, w, h)
            curr_feats = self.features.extract_roi(frame, expanded_pts)

        pts_ref, pts_cur, matches = self.features.match_points(self.ref_features, curr_feats)
        if len(matches) < self.min_inliers:
            # Fall back to matching against previous full-ish region features
            if self.ref_frame is not None and prev_roi is not None:
                prev_feats = self.features.extract_roi(self.ref_frame, self.ref_roi.points)
                curr_feats = self.features.extract(frame)
                pts_ref, pts_cur, matches = self.features.match_points(prev_feats, curr_feats)

        if len(matches) < max(4, self.min_inliers // 2):
            return HomographyResult(
                False, None, None, len(matches), 1e9, 1.0, "Insufficient matches"
            )

        H, mask = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, self.ransac_thresh)
        if H is None or mask is None:
            return HomographyResult(False, None, None, 0, 1e9, 1.0, "Homography failed")

        inliers = int(mask.ravel().sum())
        if inliers < self.min_inliers:
            return HomographyResult(
                False, H, None, inliers, 1e9, 1.0, f"Too few inliers ({inliers})"
            )

        # Reprojection error on inliers
        inl = mask.ravel().astype(bool)
        proj = cv2.perspectiveTransform(pts_ref[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.linalg.norm(proj - pts_cur[inl], axis=1).mean()) if inl.any() else 1e9
        if err > self.max_reprojection_error:
            return HomographyResult(
                False, H, None, inliers, err, 1.0, f"Reprojection error too high ({err:.2f})"
            )

        warped = transform_polygon(self.ref_roi.points, H)
        h, w = frame.shape[:2]
        # Allow partial out-of-frame; clamp for drawing/validity soft check
        if not np.all(np.isfinite(warped)):
            return HomographyResult(False, H, None, inliers, err, 1.0, "Non-finite polygon")

        area = polygon_area(clamp_polygon(warped, w, h))
        area_ratio = area / self.initial_area if self.initial_area > 0 else 1.0
        if area_ratio > self.max_area_change or area_ratio < (1.0 / self.max_area_change):
            return HomographyResult(
                False, H, None, inliers, err, 1.0, f"Area change invalid ({area_ratio:.2f})"
            )

        scale = scale_from_polygons(self.ref_roi.points, warped)
        if scale < self.min_scale or scale > self.max_scale:
            return HomographyResult(
                False, H, None, inliers, err, scale, f"Scale out of range ({scale:.2f})"
            )

        self.current_roi = ROI(warped, self.ref_roi.roi_type, self.ref_roi.frame_index)
        return HomographyResult(True, H, warped, inliers, err, scale, "OK")

    def reset(self) -> None:
        self.ref_roi = None
        self.ref_features = None
        self.ref_frame = None
        self.current_roi = None
        self.initial_area = 0.0
