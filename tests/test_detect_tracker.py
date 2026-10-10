"""Jetson DetectionTracker: report a target only when the detector confirms it, and follow it."""

from __future__ import annotations

import numpy as np

from jetson_runtime.runtime.detect_tracker import DetectionTracker
from shared.config import default_runtime_config
from shared.schemas import ROI, Detection, RuntimeState


class FakeYolo:
    mode = "detection"
    confidence_threshold = 0.35
    last_inference_ms = 5.0
    localizer = None

    def __init__(self, boxes):
        self.boxes = boxes  # frame index -> (x, y, w, h) or None
        self.i = -1

    def predict_raw(self, frame):
        self.i += 1
        b = self.boxes(self.i)
        return [] if b is None else [Detection(bbox=b, confidence=0.9, class_id=0)]

    def detection_to_roi(self, det, frame_index=0):
        return ROI.from_bbox(*det.bbox, frame_index)


def _run(boxes, n=20):
    tracker = DetectionTracker(default_runtime_config(), FakeYolo(boxes))
    frame = np.zeros((480, 640, 3), np.uint8)
    return tracker, [tracker.update(frame, i).roi for i in range(n)]


def test_no_detections_reports_nothing():
    tracker, rois = _run(lambda i: None)
    assert all(r is None for r in rois)
    assert tracker.state.status == RuntimeState.SEARCHING
    assert not tracker.is_tracking


def test_follows_moving_detection():
    tracker, rois = _run(lambda i: (100 + 5 * i, 200, 60, 40))
    assert tracker.is_tracking and tracker.state.status == RuntimeState.TRACKING
    cx, _ = rois[-1].center
    assert abs(cx - (100 + 5 * 19 + 30)) < 1.0  # box is where the detector put it, not frozen


def test_drops_target_after_detector_stops():
    tracker, rois = _run(lambda i: (100, 200, 60, 40) if i < 8 else None, n=60)
    assert rois[7] is not None
    assert rois[-1] is None and not tracker.is_tracking
