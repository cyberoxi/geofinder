"""Feature-based matching utilities (ORB / AKAZE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.logger import get_logger
from core.roi_selector import create_roi_mask

logger = get_logger("grt.features")


@dataclass
class FeatureSet:
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    image_shape: Tuple[int, int]


class FeatureTracker:
    """Extract and match scale-tolerant features for ROI tracking."""

    def __init__(
        self,
        detector_name: str = "AKAZE",
        max_features: int = 2000,
        lowe_ratio: float = 0.75,
    ):
        self.detector_name = detector_name.upper()
        self.max_features = max_features
        self.lowe_ratio = lowe_ratio
        self._detector = self._build_detector()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _build_detector(self):
        if self.detector_name == "ORB":
            return cv2.ORB_create(nfeatures=self.max_features)
        # Default AKAZE — robust and Jetson-friendly
        return cv2.AKAZE_create()

    def extract(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> FeatureSet:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kps, desc = self._detector.detectAndCompute(gray, mask)
        if kps is None:
            kps = []
        if self.detector_name == "AKAZE" and len(kps) > self.max_features:
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[: self.max_features]
            kps, desc = self._detector.compute(gray, kps)
        return FeatureSet(list(kps or []), desc, gray.shape[:2])

    def extract_roi(self, image: np.ndarray, polygon: np.ndarray) -> FeatureSet:
        mask = create_roi_mask(image.shape[:2], polygon)
        return self.extract(image, mask)

    def match(
        self, desc1: Optional[np.ndarray], desc2: Optional[np.ndarray]
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        try:
            knn = self._matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error as exc:
            logger.warning("Feature matching failed: %s", exc)
            return []
        good: List[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)
        return good

    def match_points(
        self, feats1: FeatureSet, feats2: FeatureSet
    ) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
        matches = self.match(feats1.descriptors, feats2.descriptors)
        if not matches:
            return (
                np.zeros((0, 2), np.float32),
                np.zeros((0, 2), np.float32),
                [],
            )
        pts1 = np.float32([feats1.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([feats2.keypoints[m.trainIdx].pt for m in matches])
        return pts1, pts2, matches
