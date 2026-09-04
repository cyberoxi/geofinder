"""Hybrid multi-scale tracking for Jetson runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from jetson_runtime.detectors.confidence_manager import ConfidenceManager
from jetson_runtime.detectors.yolo_detector import YOLODetector
from jetson_runtime.trackers.homography_tracker import HomographyTracker
from jetson_runtime.trackers.kalman_filter import GeometrySmoother
from jetson_runtime.trackers.optical_flow_tracker import BoxTracker, OpticalFlowTracker
from shared.features import FeatureExtractor, verify_detection_with_features
from shared.geometry import is_valid_roi, scale_from_polygons
from shared.logging import get_logger
from shared.schemas import (
    FrameResult,
    ROI,
    RuntimeState,
    TrackingMethod,
    TrackerState,
)

logger = get_logger("jetson.hybrid")


class MultiScaleHybridTracker:
    def __init__(
        self,
        config: Dict[str, Any],
        yolo: Optional[YOLODetector] = None,
        reference_descriptors: Optional[Path] = None,
        feature_method: str = "AKAZE",
    ):
        self.config = config
        tcfg = config.get("tracking", {})
        ycfg = config.get("yolo", {})
        scfg = config.get("smoothing", {})

        self.yolo = yolo
        self.reference_descriptors = Path(reference_descriptors) if reference_descriptors else None
        self.feature_method = feature_method
        self.yolo_every_n = int(tcfg.get("yolo_every_n_frames", ycfg.get("every_n_frames", 30)))
        self.keyframe_interval = int(tcfg.get("keyframe_interval", 15))
        self.feature_match_interval = int(tcfg.get("feature_match_interval", 5))
        self.lost_frames_threshold = int(tcfg.get("lost_frames_threshold", 15))
        self.max_scale_jump = float(tcfg.get("max_scale_jump", 0.35))
        self.multi_scale = list(ycfg.get("multi_scale_factors", [0.75, 1.0, 1.25]))
        self.adaptive_skip = bool(tcfg.get("adaptive_skip", True))
        self.target_fps = float(tcfg.get("target_fps", 15.0))
        self.skip_frames = 0

        self.features = FeatureExtractor(
            detector_name=tcfg.get("feature_detector", feature_method),
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
        self.box_tracker = None
        tracker_type = tcfg.get("tracker_type", "optical_flow").lower()
        if tracker_type in ("csrt", "kcf"):
            try:
                self.box_tracker = BoxTracker(tracker_type.upper())
            except Exception:  # noqa: BLE001
                pass

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
        self.min_iou = float(tcfg.get("min_iou", 0.20))

        self.state = TrackerState(status=RuntimeState.INIT)
        self._frame_idx = 0
        self._initial_roi: Optional[ROI] = None
        self._initialized = False
        self.stats = {
            "tracked": 0,
            "lost": 0,
            "redetect": 0,
            "scale_change": 0,
            "homography": 0,
            "optical_flow": 0,
            "yolo": 0,
            "searching": 0,
        }

    def initialize_from_search(self, frame: np.ndarray) -> bool:
        self.state.status = RuntimeState.SEARCHING
        self.stats["searching"] += 1
        roi = self._search_yolo(frame, multi_scale=True)
        if roi is None:
            # Feature-only global search fallback is limited; stay SEARCHING
            self.state.status = RuntimeState.SEARCHING
            self.state.message = "Target not found in initial search"
            return False
        return self.initialize(frame, roi)

    def initialize(self, frame: np.ndarray, roi: ROI) -> bool:
        self.state.status = RuntimeState.INIT
        if not is_valid_roi(roi.points):
            self.state.status = RuntimeState.ERROR
            self.state.message = "Invalid ROI"
            return False
        self._initial_roi = roi.copy()
        self.homography.initialize(frame, roi)
        self.optical_flow.initialize(frame, roi)
        if self.box_tracker:
            self.box_tracker.initialize(frame, roi)
        self.smoother.reset()
        self.state.roi = roi.copy()
        self.state.scale = 1.0
        self.state.confidence = 0.85
        self.state.lost_count = 0
        self.state.status = RuntimeState.DETECTED
        self.state.method = TrackingMethod.YOLO if self.yolo and self.yolo.available else TrackingMethod.HOMOGRAPHY
        self.state.trail = [roi.center]
        self._initialized = True
        self._frame_idx = 0
        return True

    def reinitialize(self, frame: np.ndarray, roi: ROI) -> None:
        self.homography.initialize(frame, roi)
        self.optical_flow.initialize(frame, roi)
        if self.box_tracker:
            self.box_tracker.initialize(frame, roi)
        self.state.roi = roi.copy()
        self.state.lost_count = 0
        self.state.status = RuntimeState.TRACKING

    def _verify_with_reference(self, frame: np.ndarray, roi: ROI) -> Tuple[bool, int, float]:
        if self.reference_descriptors is None or not self.reference_descriptors.exists():
            return True, 0, 1.0
        return verify_detection_with_features(
            frame, roi.points, self.reference_descriptors, self.feature_method, min_inliers=8
        )

    def _search_yolo(self, frame: np.ndarray, multi_scale: bool = False) -> Optional[ROI]:
        if self.yolo is None or not self.yolo.available:
            return None
        scales = self.multi_scale if multi_scale else [1.0]
        h, w = frame.shape[:2]
        candidates = []
        for s in scales:
            if abs(s - 1.0) < 1e-3:
                img = frame
            else:
                img = cv2.resize(frame, (int(w * s), int(h * s)))
            dets = self.yolo.predict(img)
            self.state.inference_ms = self.yolo.last_inference_ms
            for d in dets:
                if abs(s - 1.0) > 1e-3:
                    x, y, bw, bh = d.bbox
                    d.bbox = (x / s, y / s, bw / s, bh / s)
                    if d.polygon is not None:
                        d.polygon = d.polygon / s
                roi = self.yolo.detection_to_roi(d, self._frame_idx)
                ok, inl, score = self._verify_with_reference(frame, roi)
                combined = 0.5 * d.confidence + 0.5 * score
                if self.state.roi is not None:
                    from shared.geometry import iou_bbox

                    combined += 0.2 * iou_bbox(d.bbox, self.state.roi.bbox)
                if ok or self.reference_descriptors is None:
                    candidates.append((combined, roi, d.confidence, inl))
        if not candidates:
            # If no feature verify, still take best YOLO via confidence manager
            dets = self.yolo.predict(frame)
            best = self.confidence_mgr.select_detection(
                dets, self.state.roi, self.confidence_mgr.confidence_threshold, iou_threshold=self.min_iou * 0.5
            )
            if best is None and self.state.roi is None:
                best = self.confidence_mgr.select_detection(
                    dets, None, self.confidence_mgr.confidence_threshold, iou_threshold=0.0
                )
            if best is None:
                return None
            roi = self.yolo.detection_to_roi(best, self._frame_idx)
            self.state.confidence = best.confidence
            self.state.method = TrackingMethod.YOLO_SEG if self.yolo.mode == "segmentation" else TrackingMethod.YOLO
            self.stats["yolo"] += 1
            self.stats["redetect"] += 1
            return roi

        candidates.sort(key=lambda x: x[0], reverse=True)
        combined, roi, conf, inl = candidates[0]
        self.state.confidence = float(conf)
        self.state.inlier_count = int(inl)
        self.state.method = TrackingMethod.YOLO_SEG if self.yolo and self.yolo.mode == "segmentation" else TrackingMethod.YOLO
        self.stats["yolo"] += 1
        self.stats["redetect"] += 1
        return roi

    def update(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> TrackerState:
        if frame_idx is not None:
            self._frame_idx = frame_idx
        else:
            self._frame_idx += 1

        if not self._initialized:
            self.initialize_from_search(frame)
            if not self._initialized:
                self.state.status = RuntimeState.SEARCHING
                return self.state

        force_yolo = (
            self.state.status in (RuntimeState.LOST, RuntimeState.SEARCHING, RuntimeState.RE_DETECTING)
            or self.confidence_mgr.is_critical(self.state.confidence)
            or self.state.lost_count >= max(3, self.lost_frames_threshold // 3)
            or self.state.status == RuntimeState.SCALE_CHANGE
        )
        periodic = self.yolo_every_n > 0 and (self._frame_idx % self.yolo_every_n == 0)

        roi: Optional[ROI] = None
        method = TrackingMethod.LOST
        conf = 0.0
        inliers = 0
        reproj = 0.0

        if force_yolo or periodic:
            self.state.status = RuntimeState.RE_DETECTING if force_yolo else self.state.status
            yolo_roi = self._search_yolo(frame, multi_scale=force_yolo or self.state.status == RuntimeState.SCALE_CHANGE)
            if yolo_roi is not None:
                roi = yolo_roi
                method = self.state.method
                conf = self.state.confidence
                inliers = self.state.inlier_count
                self.reinitialize(frame, roi)

        if roi is None:
            do_h = self._frame_idx % max(1, self.feature_match_interval) == 0
            if do_h:
                hres = self.homography.estimate(frame, prev_roi=self.state.roi)
                if hres.success and hres.polygon is not None:
                    roi = ROI(hres.polygon, self._initial_roi.roi_type if self._initial_roi else None)
                    method = TrackingMethod.HOMOGRAPHY
                    br = self.confidence_mgr.score_homography(hres.inliers, hres.reprojection_error, hres.scale)
                    conf = br.total
                    inliers = hres.inliers
                    reproj = hres.reprojection_error
                    prev_scale = self.state.scale
                    self.state.scale = hres.scale
                    self.stats["homography"] += 1
                    if abs(self.state.scale - prev_scale) / max(prev_scale, 1e-3) > self.max_scale_jump:
                        self.state.status = RuntimeState.SCALE_CHANGE
                        self.stats["scale_change"] += 1
                    if self._frame_idx % max(1, self.keyframe_interval) == 0:
                        self.homography.update_keyframe(frame, roi)
                        self.optical_flow.reinitialize(frame, roi)

        if roi is None:
            if self.box_tracker is not None:
                ok, troi, tconf = self.box_tracker.update(frame)
                if ok and troi is not None:
                    roi, method, conf = troi, TrackingMethod.CSRT, tconf
            if roi is None:
                ok, troi, tconf = self.optical_flow.update(frame)
                if ok and troi is not None:
                    roi, method, conf = troi, TrackingMethod.OPTICAL_FLOW, tconf
                    self.stats["optical_flow"] += 1
                    if self._initial_roi is not None:
                        self.state.scale = scale_from_polygons(self._initial_roi.points, roi.points)

        if roi is None and not force_yolo:
            yolo_roi = self._search_yolo(frame, multi_scale=True)
            if yolo_roi is not None:
                roi = yolo_roi
                method = self.state.method
                conf = self.state.confidence
                self.reinitialize(frame, roi)

        if roi is None or not np.all(np.isfinite(roi.points)):
            self.state.lost_count += 1
            self.stats["lost"] += 1
            if self.state.lost_count >= self.lost_frames_threshold:
                self.state.status = RuntimeState.LOST
                self.state.method = TrackingMethod.LOST
                self.state.confidence = 0.0
            else:
                self.state.status = RuntimeState.RE_DETECTING
            return self.state

        if self.smoothing_enabled:
            smoothed = self.smoother.filter(roi)
            if smoothed is None:
                self.state.lost_count += 1
                self.state.status = RuntimeState.RE_DETECTING
                return self.state
            roi = smoothed

        # Scale-change detection from area
        if self._initial_roi is not None:
            new_scale = scale_from_polygons(self._initial_roi.points, roi.points)
            if abs(new_scale - self.state.scale) / max(self.state.scale, 1e-3) > self.max_scale_jump:
                self.state.status = RuntimeState.SCALE_CHANGE
                self.stats["scale_change"] += 1
            self.state.scale = new_scale

        self.state.roi = roi
        self.state.method = method
        self.state.confidence = float(conf)
        self.state.inlier_count = int(inliers)
        self.state.reprojection_error = float(reproj)
        self.state.lost_count = 0
        if self.state.status != RuntimeState.SCALE_CHANGE:
            self.state.status = (
                RuntimeState.RE_DETECTING if self.confidence_mgr.is_low(conf) else RuntimeState.TRACKING
            )
        self.stats["tracked"] += 1
        self.state.trail.append(roi.center)
        trail_len = int(self.config.get("export", {}).get("trail_length", 60))
        if len(self.state.trail) > trail_len:
            self.state.trail = self.state.trail[-trail_len:]

        # Adaptive skip suggestion
        if self.adaptive_skip and self.state.inference_ms > 0:
            # If inference is slow, allow caller to skip frames
            est_fps = 1000.0 / max(self.state.inference_ms, 1)
            self.skip_frames = 1 if est_fps < self.target_fps * 0.5 else 0
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
