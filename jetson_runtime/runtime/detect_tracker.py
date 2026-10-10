"""Detection-first tracker for Jetson: the same logic as the desktop Live Test.

Every processed frame runs YOLO (geometry / aux filters live in ``YOLODetector``), then the
shared :class:`TargetStabilizer` decides: confirm → lock → coast → search. A target is only
reported while the stabilizer is locked or coasting, so nothing is drawn or sent over serial
for a frame the model did not actually confirm.

Optional: with ``tracking.detect_every_n`` > 1, optical flow bridges the frames between
detector runs while locked (faster on slow boards, less exact).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from jetson_runtime.detectors.yolo_detector import YOLODetector
from jetson_runtime.trackers.optical_flow_tracker import OpticalFlowTracker
from shared.logging import get_logger
from shared.schemas import FrameResult, ROI, RuntimeState, TrackerState, TrackingMethod
from shared.target_stabilizer import CONFIRMING, COASTING, LOCKED, Candidate, TargetStabilizer

logger = get_logger("jetson.detect")


class DetectionTracker:
    def __init__(self, config: Dict[str, Any], yolo: YOLODetector):
        self.config = config
        tcfg = config.get("tracking", {})
        self.yolo = yolo
        # Not the legacy ``yolo_every_n_frames`` (30): old packages must also detect every frame
        self.detect_every_n = max(1, int(tcfg.get("detect_every_n", 1)))
        self.adaptive_skip = bool(tcfg.get("adaptive_skip", True))
        self.target_fps = float(tcfg.get("target_fps", 15.0))
        self.skip_frames = 0
        self.stabilizer = TargetStabilizer(config.get("stabilizer"))
        self.optical_flow = OpticalFlowTracker()
        self.filter_state = "searching"
        self.state = TrackerState(status=RuntimeState.SEARCHING)
        self._last_detect = -(10**9)
        self._flow_ok = False
        self.stats = {"tracked": 0, "lost": 0, "yolo": 0, "landmarks": 0, "predicted": 0,
                      "optical_flow": 0, "searching": 0, "redetect": 0}

    # Kept for API parity with MultiScaleHybridTracker; detection decides from the first frame.
    def initialize(self, frame: np.ndarray, roi: ROI) -> bool:
        logger.info("Initial ROI ignored in detection mode — the model locates the target")
        return False

    def initialize_from_search(self, frame: np.ndarray) -> bool:
        return False

    @property
    def awaiting_confirmation(self) -> bool:
        return self.filter_state == CONFIRMING

    @property
    def is_tracking(self) -> bool:
        return self.filter_state in (LOCKED, COASTING) and self.state.roi is not None

    def _method(self) -> TrackingMethod:
        return TrackingMethod.YOLO_SEG if self.yolo.mode == "segmentation" else TrackingMethod.YOLO

    def _detect(self, frame: np.ndarray, idx: int) -> None:
        dets = self.yolo.predict(frame)
        self.state.inference_ms = self.yolo.last_inference_ms
        min_conf = self.yolo.confidence_threshold
        cands = [
            Candidate(self.yolo.detection_to_roi(d, idx).points, float(d.confidence),
                      getattr(d, "source", "yolo"), int(getattr(d, "n_landmarks", 0)))
            for d in dets
        ]
        st = self.stabilizer.update(cands, min_conf, idx)
        self.filter_state = st.state
        self._last_detect = idx
        if st.state in (LOCKED, COASTING) and st.polygon is not None:
            roi = ROI(st.polygon, frame_index=idx)
            self.state.roi = roi
            self.state.confidence = float(st.confidence)
            self.state.method = self._method() if st.source != "predicted" else TrackingMethod.NONE
            self.state.status = RuntimeState.TRACKING if st.state == LOCKED else RuntimeState.RE_DETECTING
            self.state.message = st.source
            self.stats[st.source if st.source in self.stats else "yolo"] += 1
            if self.detect_every_n > 1 and st.source != "predicted":
                self._flow_ok = self.optical_flow.initialize(frame, roi)
            return
        # searching / confirming: no target is reported
        self.state.roi = None
        self.state.confidence = 0.0
        self.state.method = TrackingMethod.NONE
        self.state.status = RuntimeState.SEARCHING
        self.state.message = f"confirming {st.progress}" if st.state == CONFIRMING else "searching"
        self._flow_ok = False
        self.stats["searching"] += 1

    def _bridge(self, frame: np.ndarray, idx: int) -> bool:
        """Between detector runs: move the lock with optical flow. False → detect now."""
        if not self._flow_ok or self.state.roi is None:
            return False
        ok, roi, conf = self.optical_flow.update(frame)
        if not ok or roi is None or not np.all(np.isfinite(roi.points)):
            return False
        self.state.roi = roi
        self.state.method = TrackingMethod.OPTICAL_FLOW
        self.stabilizer.follow(roi.points, idx)
        self.stats["optical_flow"] += 1
        return True

    def update(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> TrackerState:
        idx = int(frame_idx) if frame_idx is not None else self._last_detect + 1
        due = idx - self._last_detect >= self.detect_every_n
        if due or self.filter_state not in (LOCKED, COASTING) or not self._bridge(frame, idx):
            self._detect(frame, idx)
        if self.state.roi is not None:
            self.stats["tracked"] += 1
            self.state.trail.append(self.state.roi.center)
            trail_len = int(self.config.get("export", {}).get("trail_length", 60))
            self.state.trail = self.state.trail[-trail_len:]
        else:
            self.stats["lost"] += 1
            self.state.trail = []
        if self.adaptive_skip and self.state.inference_ms > 0:
            est_fps = 1000.0 / max(self.state.inference_ms, 1)
            self.skip_frames = 1 if est_fps < self.target_fps * 0.5 else 0
        return self.state

    def to_frame_result(self, frame_id: int, timestamp: float, process_fps: float = 0.0) -> FrameResult:
        roi = self.state.roi
        x, y, w, h = roi.bbox if roi is not None else (0, 0, 0, 0)
        cx, cy = roi.center if roi is not None else (0, 0)
        return FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            center_x=cx,
            center_y=cy,
            bbox_x=x,
            bbox_y=y,
            bbox_width=w,
            bbox_height=h,
            polygon_points=roi.to_list() if roi is not None else [],
            confidence=self.state.confidence,
            tracking_method=self.state.method.value,
            inlier_count=0,
            reprojection_error=0.0,
            scale=1.0,
            status=self.state.status.value,
            inference_ms=self.state.inference_ms,
            process_fps=process_fps,
        )
