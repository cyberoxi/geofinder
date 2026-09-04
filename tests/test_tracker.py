"""Tracker / confidence / smoothing tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.confidence_manager import ConfidenceManager
from core.kalman_filter import EMAFilter, GeometrySmoother
from core.types import Detection, ROI


def test_confidence_filter_and_select():
    mgr = ConfidenceManager(confidence_threshold=0.45)
    predicted = ROI.from_bbox(100, 100, 50, 50)
    dets = [
        Detection(bbox=(10, 10, 40, 40), confidence=0.9),  # far
        Detection(bbox=(105, 105, 48, 48), confidence=0.2),  # low conf
        Detection(bbox=(102, 98, 52, 55), confidence=0.8),  # good
    ]
    best = mgr.select_detection(dets, predicted, conf_threshold=0.45, iou_threshold=0.1)
    assert best is not None
    assert best.confidence == pytest.approx(0.8)


def test_ema_smoothing():
    ema = EMAFilter(alpha=0.5)
    a = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    b = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32)
    out1 = ema.update(a)
    out2 = ema.update(b)
    assert np.allclose(out1, a)
    assert out2[0, 0] == pytest.approx(5.0)


def test_geometry_smoother_rejects_jump():
    sm = GeometrySmoother(method="ema", ema_alpha=0.5, max_center_jump_px=20)
    r1 = ROI.from_bbox(0, 0, 40, 40)
    r2 = ROI.from_bbox(200, 200, 40, 40)
    assert sm.filter(r1) is not None
    assert sm.filter(r2) is None


def test_homography_confidence_score():
    mgr = ConfidenceManager(min_inliers=10, max_reprojection_error=4.0)
    br = mgr.score_homography(inliers=20, reprojection_error=1.0, scale=1.1)
    assert 0.0 < br.total <= 1.0
