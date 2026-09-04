"""Unit tests for ROI geometry helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.roi_selector import (
    ROISelector,
    bbox_to_polygon,
    is_valid_roi,
    polygon_area,
    polygon_center,
    polygon_to_bbox,
)
from core.types import ROI, ROIType


def test_polygon_to_bbox():
    pts = np.array([[10, 20], [40, 20], [40, 50], [10, 50]], dtype=np.float32)
    x, y, w, h = polygon_to_bbox(pts)
    assert x == 10 and y == 20 and w == 30 and h == 30


def test_bbox_to_polygon_roundtrip():
    poly = bbox_to_polygon(5, 7, 20, 10)
    x, y, w, h = polygon_to_bbox(poly)
    assert (x, y, w, h) == (5, 7, 20, 10)


def test_roi_center_area():
    roi = ROI.from_bbox(0, 0, 100, 50)
    cx, cy = roi.center
    assert cx == pytest.approx(50)
    assert cy == pytest.approx(25)
    assert roi.area == pytest.approx(5000)


def test_invalid_roi():
    assert not is_valid_roi([[0, 0], [1, 1]])
    assert is_valid_roi([[0, 0], [100, 0], [100, 100], [0, 100]])


def test_selector_rectangle():
    roi = ROISelector.from_rectangle(10, 10, 50, 40)
    assert roi.roi_type == ROIType.RECTANGLE
    assert polygon_area(roi.points) > 0


def test_selector_polygon():
    roi = ROISelector.from_polygon([[0, 0], [80, 0], [80, 60], [0, 60]])
    assert len(roi.points) == 4
