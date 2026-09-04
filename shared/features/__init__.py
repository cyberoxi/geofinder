"""Feature extraction, matching, and reference bank I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from shared.geometry import create_roi_mask, extract_roi_crop, polygon_to_bbox


@dataclass
class FeatureSet:
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    image_shape: Tuple[int, int]


class FeatureExtractor:
    def __init__(self, detector_name: str = "AKAZE", max_features: int = 2000, lowe_ratio: float = 0.75):
        self.detector_name = detector_name.upper()
        self.max_features = max_features
        self.lowe_ratio = lowe_ratio
        self._detector = cv2.ORB_create(nfeatures=max_features) if self.detector_name == "ORB" else cv2.AKAZE_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> FeatureSet:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kps, desc = self._detector.detectAndCompute(gray, mask)
        kps = list(kps or [])
        if self.detector_name == "AKAZE" and len(kps) > self.max_features:
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[: self.max_features]
            kps, desc = self._detector.compute(gray, kps)
            kps = list(kps or [])
        return FeatureSet(kps, desc, gray.shape[:2])

    def extract_roi(self, image: np.ndarray, polygon: np.ndarray) -> FeatureSet:
        return self.extract(image, create_roi_mask(image.shape[:2], polygon))

    def match(self, desc1: Optional[np.ndarray], desc2: Optional[np.ndarray]) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        try:
            knn = self._matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)
        return good

    def match_points(self, f1: FeatureSet, f2: FeatureSet) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
        matches = self.match(f1.descriptors, f2.descriptors)
        if not matches:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), []
        pts1 = np.float32([f1.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float32([f2.keypoints[m.trainIdx].pt for m in matches])
        return pts1, pts2, matches


def save_descriptors(path: str | Path, keypoints: List[cv2.KeyPoint], descriptors: Optional[np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.zeros((0, 2), np.float32)
    sizes = np.array([kp.size for kp in keypoints], dtype=np.float32) if keypoints else np.zeros((0,), np.float32)
    angles = np.array([kp.angle for kp in keypoints], dtype=np.float32) if keypoints else np.zeros((0,), np.float32)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32) if keypoints else np.zeros((0,), np.float32)
    np.savez_compressed(
        path,
        points=pts,
        sizes=sizes,
        angles=angles,
        responses=responses,
        descriptors=descriptors if descriptors is not None else np.zeros((0, 61), np.uint8),
    )


def load_descriptors(path: str | Path) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    pts = data["points"]
    sizes = data["sizes"]
    angles = data["angles"]
    responses = data["responses"]
    desc = data["descriptors"]
    kps = [
        cv2.KeyPoint(float(p[0]), float(p[1]), float(s), float(a), float(r))
        for p, s, a, r in zip(pts, sizes, angles, responses)
    ]
    if desc is None or len(desc) == 0:
        return kps, None
    return kps, desc


def build_reference_bank(
    frame: np.ndarray,
    polygon: np.ndarray,
    out_dir: str | Path,
    feature_method: str = "AKAZE",
    scales: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
) -> dict:
    """Write reference image, mask, descriptors, multi-scale templates."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    templates = out / "templates"
    templates.mkdir(exist_ok=True)

    mask = create_roi_mask(frame.shape[:2], polygon)
    cv2.imwrite(str(out / "reference_image.jpg"), frame)
    cv2.imwrite(str(out / "roi_mask.png"), mask)

    extractor = FeatureExtractor(feature_method)
    feats = extractor.extract_roi(frame, polygon)
    save_descriptors(out / "descriptors.npz", feats.keypoints, feats.descriptors)

    crop, _ = extract_roi_crop(frame, polygon)
    for s in scales:
        if s == 1.0:
            cv2.imwrite(str(templates / "template_1.00.jpg"), crop)
            continue
        h, w = crop.shape[:2]
        nw, nh = max(8, int(w * s)), max(8, int(h * s))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        cv2.imwrite(str(templates / f"template_{s:.2f}.jpg"), resized)

    # Optional color histogram of masked region
    hist = cv2.calcHist([frame], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    np.save(out / "color_hist.npy", hist)

    x, y, w, h = polygon_to_bbox(polygon)
    return {
        "num_keypoints": len(feats.keypoints),
        "bbox": [x, y, w, h],
        "aspect": float(w / max(h, 1e-6)),
        "area": float(w * h),
        "scales": list(scales),
        "feature_method": feature_method,
    }


def verify_detection_with_features(
    frame: np.ndarray,
    det_polygon: np.ndarray,
    ref_descriptors_path: str | Path,
    feature_method: str = "AKAZE",
    min_inliers: int = 8,
) -> Tuple[bool, int, float]:
    """Match ROI region against stored reference descriptors. Returns (ok, inliers, score)."""
    kps_ref, desc_ref = load_descriptors(ref_descriptors_path)
    if desc_ref is None or len(kps_ref) < 4:
        return False, 0, 0.0
    extractor = FeatureExtractor(feature_method)
    feats = extractor.extract_roi(frame, det_polygon)
    ref_set = FeatureSet(kps_ref, desc_ref, (0, 0))
    pts1, pts2, matches = extractor.match_points(ref_set, feats)
    if len(matches) < 4:
        return False, len(matches), 0.0
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return False, 0, 0.0
    inliers = int(mask.ravel().sum())
    score = min(1.0, inliers / max(min_inliers * 2, 1))
    return inliers >= min_inliers, inliers, score
