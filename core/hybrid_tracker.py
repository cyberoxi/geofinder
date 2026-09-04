"""Hybrid / Manual / YOLO tracking orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.confidence_manager import ConfidenceManager
from core.feature_tracker import FeatureTracker
from core.homography_tracker import HomographyTracker
from core.kalman_filter import GeometrySmoother
from core.logger import get_logger
from core.optical_flow_tracker import BoxTracker, OpticalFlowTracker
from core.roi_selector import is_valid_roi, scale_from_polygons
from core.types import (
    FrameResult,
    ROI,
    ROIType,
    TrackingMethod,
    TrackingMode,
    TrackingStatus,
    TrackerState,
)
from core.yolo_detector import YOLODetector

logger = get_logger("grt.hybrid")


class HybridTracker:
    """
    Modes:
      - manual: ROI + features/homography/optical-flow/CSRT (no YOLO required)
      - yolo:   YOLO detect + tracker follow
      - hybrid: ROI init + trackers + periodic/urgent YOLO re-detect
    """

    def __init__(self, config: Dict[str, Any], yolo: Optional[YOLODetector] = None):
        self.config = config
        tcfg = config.get("tracking", {})
        ycfg = config.get("yolo", {})
        scfg = config.get("smoothing", {})

        self.mode = TrackingMode(tcfg.get("mode", "hybrid"))
        self.tracker_type = tcfg.get("tracker_type", "optical_flow").lower()
        self.keyframe_interval = int(tcfg.get("keyframe_interval", 15))
        self.feature_match_interval = int(tcfg.get("feature_match_interval", 5))
        self.lost_frames_threshold = int(tcfg.get("lost_frames_threshold", 15))
        self.yolo_every_n = int(ycfg.get("every_n_frames", 30))

        self.features = FeatureTracker(
            detector_name=tcfg.get("feature_detector", "AKAZE"),
            max_features=int(tcfg.get("max_features", 2000)),
            lowe_ratio=float(tcfg.get("lowe_ratio", 0.75)),
        )
        self.homography = HomographyTracker(
            feature_tracker=self.features,
            min_inliers=int(tcfg.get("min_inliers", 12)),
            max_reprojection_error=float(tcfg.get("max_reprojection_error", 4.0)),
            min_scale=float(tcfg.get("min_scale", 0.15)),
            max_scale=float(tcfg.get("max_scale", 8.0)),
            max_area_change=float(tcfg.get("max_area_change", 4.0)),
        )
        self.optical_flow = OpticalFlowTracker()
        self.box_tracker: Optional[BoxTracker] = None
        if self.tracker_type in ("csrt", "kcf"):
            try:
                self.box_tracker = BoxTracker(self.tracker_type.upper())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Box tracker unavailable (%s), using optical flow", exc)
                self.tracker_type = "optical_flow"

        self.confidence_mgr = ConfidenceManager(
            confidence_threshold=float(tcfg.get("confidence_threshold", 0.45)),
            low_confidence_threshold=float(tcfg.get("low_confidence_threshold", 0.30)),
            min_inliers=int(tcfg.get("min_inliers", 12)),
            max_reprojection_error=float(tcfg.get("max_reprojection_error", 4.0)),
        )
        self.smoother = GeometrySmoother(
            method=scfg.get("method", "ema"),
            ema_alpha=float(scfg.get("ema_alpha", 0.35)),
            max_center_jump_px=float(scfg.get("max_center_jump_px", 80.0)),
            max_scale_jump=float(scfg.get("max_scale_jump", 0.35)),
        )
        self.smoothing_enabled = bool(scfg.get("enabled", True))

        self.yolo = yolo
        self.state = TrackerState()
        self._frame_idx = 0
        self._initial_roi: Optional[ROI] = None
        self._initialized = False
        self.stats = {
            "tracked": 0,
            "lost": 0,
            "redetect": 0,
            "homography": 0,
            "optical_flow": 0,
            "yolo": 0,
            "csrt": 0,
            "kcf": 0,
        }

    def set_mode(self, mode: str) -> None:
        self.mode = TrackingMode(mode)

    def initialize(self, frame: np.ndarray, roi: Optional[ROI] = None) -> bool:
        self.state.status = TrackingStatus.INITIALIZING
        self.smoother.reset()
        self._frame_idx = 0

        if self.mode == TrackingMode.YOLO:
            if self.yolo is None or not self.yolo.available:
                self.state.status = TrackingStatus.ERROR
                self.state.message = "YOLO model not available"
                return False
            dets = self.yolo.predict(frame)
            best = self.confidence_mgr.select_detection(
                dets,
                None,
                self.confidence_mgr.confidence_threshold,
                iou_threshold=0.0,
            )
            if best is None:
                self.state.status = TrackingStatus.LOST
                self.state.message = "YOLO found no target in initial frame"
                return False
            roi = self.yolo.detection_to_roi(best, 0)
            self.state.confidence = best.confidence
            self.state.method = (
                TrackingMethod.YOLO_SEG if self.yolo.mode == "segmentation" else TrackingMethod.YOLO
            )
            self.stats["yolo"] += 1
            self.stats["redetect"] += 1

        if roi is None or not is_valid_roi(roi.points):
            self.state.status = TrackingStatus.ERROR
            self.state.message = "Invalid or missing ROI"
            return False

        self._initial_roi = roi.copy()
        ok_h = self.homography.initialize(frame, roi)
        ok_of = self.optical_flow.initialize(frame, roi)
        ok_box = True
        if self.box_tracker is not None:
            ok_box = self.box_tracker.initialize(frame, roi)

        if not (ok_h or ok_of or ok_box):
            self.state.status = TrackingStatus.ERROR
            self.state.message = "Failed to initialize trackers (insufficient features)"
            return False

        self.state.roi = roi.copy()
        self.state.scale = 1.0
        self.state.lost_count = 0
        self.state.status = TrackingStatus.TRACKING
        if self.state.method == TrackingMethod.NONE:
            self.state.method = TrackingMethod.HOMOGRAPHY if ok_h else TrackingMethod.OPTICAL_FLOW
        if self.state.confidence <= 0:
            self.state.confidence = 0.85
        self._initialized = True
        self.state.trail = [roi.center]
        logger.info("Tracker initialized mode=%s method=%s", self.mode.value, self.state.method.value)
        return True

    def reinitialize(self, frame: np.ndarray, roi: ROI) -> bool:
        self.homography.initialize(frame, roi)
        self.optical_flow.initialize(frame, roi)
        if self.box_tracker is not None:
            self.box_tracker.initialize(frame, roi)
        self.state.roi = roi.copy()
        self.state.lost_count = 0
        self.state.status = TrackingStatus.TRACKING
        return True

    def _maybe_yolo(self, frame: np.ndarray, force: bool = False) -> Optional[ROI]:
        if self.mode == TrackingMode.MANUAL:
            return None
        if self.yolo is None or not self.yolo.available:
            return None
        if not force and self.yolo_every_n > 0 and (self._frame_idx % self.yolo_every_n) != 0:
            return None

        self.state.status = TrackingStatus.RE_DETECTING
        dets = self.yolo.predict(frame)
        self.state.inference_ms = self.yolo.last_inference_ms
        best = self.confidence_mgr.select_detection(
            dets,
            self.state.roi,
            conf_threshold=self.confidence_mgr.confidence_threshold,
            iou_threshold=float(self.config.get("yolo", {}).get("iou_threshold", 0.45)) * 0.2,
            prev_scale=self.state.scale,
        )
        if best is None and self.mode == TrackingMode.YOLO and self.state.roi is None:
            best = self.confidence_mgr.select_detection(
                dets, None, self.confidence_mgr.confidence_threshold, iou_threshold=0.0
            )
        if best is None:
            return None
        roi = self.yolo.detection_to_roi(best, self._frame_idx)
        self.state.confidence = float(best.confidence)
        self.state.method = (
            TrackingMethod.YOLO_SEG if self.yolo.mode == "segmentation" else TrackingMethod.YOLO
        )
        self.stats["yolo"] += 1
        self.stats["redetect"] += 1
        return roi

    def _track_interframe(self, frame: np.ndarray) -> Tuple[bool, Optional[ROI], TrackingMethod, float]:
        # Prefer optical flow / box tracker between keyframes
        if self.tracker_type in ("csrt", "kcf") and self.box_tracker is not None:
            ok, roi, conf = self.box_tracker.update(frame)
            if ok and roi is not None:
                method = TrackingMethod.CSRT if self.tracker_type == "csrt" else TrackingMethod.KCF
                self.stats[self.tracker_type] += 1
                return True, roi, method, conf

        ok, roi, conf = self.optical_flow.update(frame)
        if ok and roi is not None:
            self.stats["optical_flow"] += 1
            return True, roi, TrackingMethod.OPTICAL_FLOW, conf
        return False, None, TrackingMethod.LOST, 0.0

    def update(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> TrackerState:
        if frame_idx is not None:
            self._frame_idx = frame_idx
        else:
            self._frame_idx += 1

        if not self._initialized and self.mode != TrackingMode.YOLO:
            self.state.status = TrackingStatus.ERROR
            self.state.message = "Tracker not initialized"
            return self.state

        if not self._initialized and self.mode == TrackingMode.YOLO:
            if not self.initialize(frame, None):
                return self.state

        force_yolo = (
            self.state.status == TrackingStatus.LOST
            or self.confidence_mgr.is_critical(self.state.confidence)
            or self.state.lost_count >= max(3, self.lost_frames_threshold // 3)
        )

        # Periodic or urgent YOLO
        yolo_roi = None
        if self.mode in (TrackingMode.HYBRID, TrackingMode.YOLO):
            yolo_roi = self._maybe_yolo(frame, force=force_yolo)

        roi: Optional[ROI] = None
        method = TrackingMethod.LOST
        conf = 0.0
        inliers = 0
        reproj = 0.0

        if yolo_roi is not None:
            roi = yolo_roi
            method = self.state.method
            conf = self.state.confidence
            self.reinitialize(frame, roi)
        else:
            # Homography on keyframe / match interval
            do_h = (
                self._frame_idx % max(1, self.feature_match_interval) == 0
                or self.tracker_type == "homography"
                or force_yolo
            )
            if do_h:
                hres = self.homography.estimate(frame, prev_roi=self.state.roi)
                if hres.success and hres.polygon is not None:
                    rtype = self._initial_roi.roi_type if self._initial_roi else ROIType.POLYGON
                    roi = ROI(hres.polygon, rtype)
                    method = TrackingMethod.HOMOGRAPHY
                    breakdown = self.confidence_mgr.score_homography(
                        hres.inliers, hres.reprojection_error, hres.scale
                    )
                    conf = breakdown.total
                    inliers = hres.inliers
                    reproj = hres.reprojection_error
                    self.state.scale = hres.scale
                    self.stats["homography"] += 1
                    # Refresh keyframe periodically
                    if self._frame_idx % max(1, self.keyframe_interval) == 0:
                        self.homography.update_keyframe(frame, roi)
                        self.optical_flow.reinitialize(frame, roi)
                        if self.box_tracker is not None:
                            self.box_tracker.reinitialize(frame, roi)

            if roi is None:
                ok, troi, tmethod, tconf = self._track_interframe(frame)
                if ok and troi is not None:
                    roi = troi
                    method = tmethod
                    if self.state.roi is not None:
                        self.state.scale = scale_from_polygons(self._initial_roi.points, roi.points) if self._initial_roi else 1.0
                    breakdown = self.confidence_mgr.score_tracker(tconf, self.state.scale)
                    conf = breakdown.total

            # Last resort: YOLO if not already tried this frame
            if roi is None and self.mode == TrackingMode.HYBRID and not force_yolo:
                yolo_roi = self._maybe_yolo(frame, force=True)
                if yolo_roi is not None:
                    roi = yolo_roi
                    method = self.state.method
                    conf = self.state.confidence
                    self.reinitialize(frame, roi)

        if roi is None or not np.all(np.isfinite(roi.points)):
            self.state.lost_count += 1
            self.stats["lost"] += 1
            if self.state.lost_count >= self.lost_frames_threshold:
                self.state.status = TrackingStatus.LOST
                self.state.method = TrackingMethod.LOST
                self.state.confidence = 0.0
            else:
                self.state.status = TrackingStatus.RE_DETECTING
            return self.state

        # Smoothing
        if self.smoothing_enabled:
            smoothed = self.smoother.filter(roi)
            if smoothed is None:
                # Rejected jump — keep previous if available
                if self.state.roi is not None:
                    self.state.lost_count += 1
                    self.state.status = TrackingStatus.RE_DETECTING
                    return self.state
            else:
                roi = smoothed

        self.state.roi = roi
        self.state.method = method
        self.state.confidence = float(conf)
        self.state.inlier_count = int(inliers)
        self.state.reprojection_error = float(reproj)
        self.state.lost_count = 0
        self.state.status = (
            TrackingStatus.RE_DETECTING
            if self.confidence_mgr.is_low(conf)
            else TrackingStatus.TRACKING
        )
        self.stats["tracked"] += 1
        cx, cy = roi.center
        self.state.trail.append((cx, cy))
        trail_len = int(self.config.get("export", {}).get("trail_length", 60))
        if len(self.state.trail) > trail_len:
            self.state.trail = self.state.trail[-trail_len:]
        return self.state

    def to_frame_result(self, frame_id: int, timestamp: float, process_fps: float = 0.0) -> FrameResult:
        roi = self.state.roi
        if roi is None:
            return FrameResult(
                frame_id=frame_id,
                timestamp=timestamp,
                center_x=0,
                center_y=0,
                bbox_x=0,
                bbox_y=0,
                bbox_width=0,
                bbox_height=0,
                polygon_points=[],
                confidence=self.state.confidence,
                tracking_method=self.state.method.value,
                inlier_count=self.state.inlier_count,
                reprojection_error=self.state.reprojection_error,
                scale=self.state.scale,
                status=self.state.status.value,
                inference_ms=self.state.inference_ms,
                process_fps=process_fps,
            )
        x, y, w, h = roi.bbox
        cx, cy = roi.center
        return FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            center_x=cx,
            center_y=cy,
            bbox_x=x,
            bbox_y=y,
            bbox_width=w,
            bbox_height=h,
            polygon_points=roi.to_list(),
            confidence=self.state.confidence,
            tracking_method=self.state.method.value,
            inlier_count=self.state.inlier_count,
            reprojection_error=self.state.reprojection_error,
            scale=self.state.scale,
            status=self.state.status.value,
            inference_ms=self.state.inference_ms,
            process_fps=process_fps,
        )

    def reset(self) -> None:
        self.homography.reset()
        self.optical_flow.reset()
        if self.box_tracker is not None:
            self.box_tracker.reset()
        self.smoother.reset()
        self.state = TrackerState()
        self._initialized = False
        self._initial_roi = None
        self._frame_idx = 0
