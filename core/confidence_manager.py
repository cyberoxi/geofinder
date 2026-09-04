"""Confidence scoring and detection selection logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from core.roi_selector import iou_bbox, polygon_center, polygon_to_bbox
from core.types import Detection, ROI


@dataclass
class ConfidenceBreakdown:
    total: float
    track_score: float
    inlier_score: float
    reproj_score: float
    scale_score: float
    detection_score: float


class ConfidenceManager:
    def __init__(
        self,
        confidence_threshold: float = 0.45,
        low_confidence_threshold: float = 0.30,
        min_inliers: int = 12,
        max_reprojection_error: float = 4.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.min_inliers = min_inliers
        self.max_reprojection_error = max_reprojection_error

    def score_homography(
        self, inliers: int, reprojection_error: float, scale: float, base: float = 0.5
    ) -> ConfidenceBreakdown:
        inlier_score = min(1.0, inliers / max(self.min_inliers * 2, 1))
        reproj_score = float(
            np.clip(1.0 - (reprojection_error / max(self.max_reprojection_error * 2, 1e-6)), 0, 1)
        )
        # Prefer scales near previous (~1 relative to ref handled upstream); mild penalty extremes
        scale_score = float(np.clip(1.0 - abs(np.log2(max(scale, 1e-3))) / 4.0, 0, 1))
        total = 0.25 * base + 0.35 * inlier_score + 0.25 * reproj_score + 0.15 * scale_score
        return ConfidenceBreakdown(total, base, inlier_score, reproj_score, scale_score, 0.0)

    def score_tracker(self, tracker_conf: float, scale: float = 1.0) -> ConfidenceBreakdown:
        scale_score = float(np.clip(1.0 - abs(np.log2(max(scale, 1e-3))) / 4.0, 0, 1))
        total = 0.75 * float(np.clip(tracker_conf, 0, 1)) + 0.25 * scale_score
        return ConfidenceBreakdown(total, tracker_conf, 0, 0, scale_score, 0.0)

    def score_detection(
        self,
        det: Detection,
        predicted: Optional[ROI],
        prev_scale: float = 1.0,
    ) -> float:
        conf = float(det.confidence)
        if predicted is None:
            return conf
        iou = iou_bbox(det.bbox, predicted.bbox)
        pcx, pcy = predicted.center
        dx = det.bbox[0] + det.bbox[2] / 2 - pcx
        dy = det.bbox[1] + det.bbox[3] / 2 - pcy
        dist = float(np.hypot(dx, dy))
        diag = float(np.hypot(predicted.bbox[2], predicted.bbox[3]) + 1e-6)
        dist_score = float(np.clip(1.0 - dist / diag, 0, 1))
        scale = float(np.sqrt((det.bbox[2] * det.bbox[3]) / max(predicted.area, 1.0)))
        scale_score = float(np.clip(1.0 - abs(scale - prev_scale), 0, 1))
        return 0.35 * conf + 0.35 * iou + 0.20 * dist_score + 0.10 * scale_score

    def select_detection(
        self,
        detections: List[Detection],
        predicted: Optional[ROI],
        conf_threshold: float,
        iou_threshold: float = 0.1,
        prev_scale: float = 1.0,
    ) -> Optional[Detection]:
        candidates: List[Tuple[float, Detection]] = []
        for det in detections:
            if det.confidence < conf_threshold:
                continue
            if predicted is not None:
                iou = iou_bbox(det.bbox, predicted.bbox)
                if iou < iou_threshold:
                    # Allow far detections only if predicted is weak / missing overlap entirely
                    # when IoU is zero but centers close — still require min IoU for hybrid
                    continue
            score = self.score_detection(det, predicted, prev_scale)
            candidates.append((score, det))
        if not candidates:
            # If prediction missing (YOLO-only mode), pick highest confidence
            if predicted is None:
                valid = [d for d in detections if d.confidence >= conf_threshold]
                if not valid:
                    return None
                return max(valid, key=lambda d: d.confidence)
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def is_low(self, confidence: float) -> bool:
        return confidence < self.confidence_threshold

    def is_critical(self, confidence: float) -> bool:
        return confidence < self.low_confidence_threshold
