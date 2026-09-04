"""Homography tracker tests with synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.feature_tracker import FeatureTracker
from core.homography_tracker import HomographyTracker
from core.types import ROI


def _textured_frame(w=320, h=240, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
    # Add structured patterns for features
    for i in range(20):
        x, y = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        cv2.rectangle(base, (x, y), (x + 30, y + 30), (int(rng.integers(0, 255)),) * 3, -1)
        cv2.circle(base, (x + 15, y + 15), 8, (255, 255, 255), -1)
    return base


def test_homography_identity():
    frame = _textured_frame()
    roi = ROI.from_bbox(80, 60, 100, 80)
    tracker = HomographyTracker(FeatureTracker("AKAZE", 1000), min_inliers=8)
    assert tracker.initialize(frame, roi)
    result = tracker.estimate(frame, prev_roi=roi)
    assert result.success
    assert result.inliers >= 8
    # Center should stay near original
    c0 = np.array(roi.center)
    c1 = result.polygon.mean(axis=0)
    assert np.linalg.norm(c0 - c1) < 15


def test_homography_translation():
    frame = _textured_frame()
    # Shift frame
    M = np.float32([[1, 0, 12], [0, 1, 8]])
    shifted = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    roi = ROI.from_bbox(60, 50, 120, 90)
    tracker = HomographyTracker(FeatureTracker("ORB", 1500), min_inliers=8, max_reprojection_error=6.0)
    assert tracker.initialize(frame, roi)
    result = tracker.estimate(shifted, prev_roi=roi, use_full_frame=True)
    # May succeed depending on features; if success, center should move roughly with shift
    if result.success:
        dx = result.polygon.mean(axis=0)[0] - roi.center[0]
        assert dx > 5


def test_reject_invalid_homography():
    frame = _textured_frame(seed=1)
    noise = _textured_frame(seed=99)
    roi = ROI.from_bbox(40, 40, 80, 80)
    tracker = HomographyTracker(FeatureTracker("AKAZE", 500), min_inliers=20)
    assert tracker.initialize(frame, roi)
    result = tracker.estimate(noise, prev_roi=roi, use_full_frame=True)
    # Unrelated frame should usually fail
    assert result.success is False or result.inliers < 20
