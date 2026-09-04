"""Jetson runtime processing pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from jetson_runtime.detectors.yolo_detector import YOLODetector
from jetson_runtime.exporters.result_exporter import ResultExporter, draw_overlay
from jetson_runtime.package_loader.loader import PackageLoader
from jetson_runtime.runtime.hybrid_tracker import MultiScaleHybridTracker
from shared.logging import get_logger, setup_logging
from shared.schemas import ROI, RuntimeState
from shared.video_reader import VideoReader

logger = get_logger("jetson.pipeline")

ProgressCb = Callable[[int, int, dict], None]
FrameCb = Callable[[Any, dict], None]


class FileVideoSource:
    def __init__(self, path: str, prefer_gstreamer: bool = True):
        self.reader = VideoReader(path, prefer_gstreamer=prefer_gstreamer)

    def seek(self, idx: int):
        return self.reader.seek(idx)

    def read(self):
        return self.reader.read()

    def release(self):
        self.reader.release()

    @property
    def info(self):
        return self.reader.info


class RTSPVideoSource:
    """Stub for future RTSP support."""

    def __init__(self, url: str):
        raise NotImplementedError("RTSP source is reserved for a future release. Use file video for now.")


def build_detector_from_package(loader: PackageLoader) -> Optional[YOLODetector]:
    engine, onnx, pt = loader.model_candidates()
    ycfg = loader.runtime_config.get("yolo", {})
    model_path = str(engine or onnx or pt or "")
    if not model_path:
        logger.warning("No model in package — feature/homography-only mode")
        return None
    fallbacks = [str(p) for p in (onnx, pt) if p and str(p) != model_path]
    return YOLODetector(
        model_path=model_path,
        mode=ycfg.get("mode", loader.manifest.model_type if loader.manifest else "segmentation"),
        device=ycfg.get("device", "cuda"),
        imgsz=int(ycfg.get("imgsz", loader.manifest.input_size if loader.manifest else 640)),
        confidence_threshold=float(loader.manifest.min_confidence if loader.manifest else 0.45),
        iou_threshold=float(loader.manifest.min_iou if loader.manifest else 0.2),
        use_fp16=bool(ycfg.get("use_fp16", True)),
        warmup_iterations=int(ycfg.get("warmup_iterations", 5)),
        class_names=list(loader.manifest.class_names) if loader.manifest else ["target_region"],
        fallback_paths=fallbacks,
    )


def run_runtime(
    package_path: str,
    video_path: str,
    output_dir: Optional[str] = None,
    prefer_gstreamer: bool = False,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    initial_roi: Optional[ROI] = None,
    end_frame: int = -1,
) -> Dict[str, Any]:
    setup_logging()
    loader = PackageLoader(package_path)
    root = loader.load()
    yolo = build_detector_from_package(loader)
    tracker = MultiScaleHybridTracker(
        loader.runtime_config,
        yolo=yolo,
        reference_descriptors=loader.descriptors_path(),
        feature_method=loader.manifest.feature_method if loader.manifest else "AKAZE",
    )

    source = FileVideoSource(video_path, prefer_gstreamer=prefer_gstreamer)
    assert source.info is not None
    out_dir = Path(output_dir) if output_dir else root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = out_dir / "failed_frames"
    exporter = ResultExporter(str(out_dir), str(failed_dir), loader.runtime_config)
    if loader.runtime_config.get("export", {}).get("save_annotated_video", True):
        exporter.start_video(None, source.info.width, source.info.height, source.info.fps)

    ok, frame = source.seek(0)
    if not ok or frame is None:
        source.release()
        raise RuntimeError("Cannot read video")

    if initial_roi is not None:
        tracker.initialize(frame, initial_roi)
    else:
        ok_init = tracker.initialize_from_search(frame)
        if not ok_init and loader.manifest and loader.manifest.reference_roi.get("points"):
            ref = loader.manifest.reference_roi
            pts = ref["points"]
            # Scale reference polygon if frame size differs
            rw, rh = int(ref.get("width") or frame.shape[1]), int(ref.get("height") or frame.shape[0])
            sx = frame.shape[1] / max(rw, 1)
            sy = frame.shape[0] / max(rh, 1)
            scaled = [[p[0] * sx, p[1] * sy] for p in pts]
            from shared.schemas import ROI
            import numpy as np

            roi = ROI(np.asarray(scaled, dtype=np.float32), frame_index=0)
            if not tracker.initialize(frame, roi):
                logger.warning("Reference ROI init failed; continuing SEARCHING")
        elif not ok_init:
            logger.warning("Initial search failed; continuing with SEARCHING state")

    processed = 0
    fps_list = []
    t0 = time.perf_counter()
    idx = 0
    last = end_frame if end_frame >= 0 else (source.info.frame_count - 1 if source.info.frame_count > 0 else 10**9)

    while True:
        if stop_flag and stop_flag():
            break
        if idx > last:
            break
        t_frame = time.perf_counter()
        state = tracker.update(frame, frame_idx=idx)
        dt = time.perf_counter() - t_frame
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_list.append(fps)
        result = tracker.to_frame_result(idx, source.reader.timestamp_of(idx), process_fps=fps)
        exporter.add_result(result)
        annotated = draw_overlay(frame, result, trail=state.trail)
        exporter.write_frame(annotated)
        if state.status == RuntimeState.LOST and loader.runtime_config.get("export", {}).get("save_failed_frames", True):
            exporter.save_failed_frame(frame, idx)
        processed += 1
        if progress_cb:
            progress_cb(processed, max(1, last + 1), result.to_dict())
        if frame_cb:
            frame_cb(annotated, result.to_dict())

        step = 1 + max(0, int(getattr(tracker, "skip_frames", 0)))
        next_idx = idx + step
        if next_idx > last:
            break
        ok, frame = source.seek(next_idx)
        idx = next_idx
        if not ok or frame is None:
            break

    elapsed = time.perf_counter() - t0
    tracker.state.status = RuntimeState.COMPLETED
    stats = {
        **tracker.stats,
        "processed_frames": processed,
        "elapsed_sec": elapsed,
        "avg_fps": processed / elapsed if elapsed > 0 else 0.0,
        "min_fps": min(fps_list) if fps_list else 0.0,
        "yolo": yolo.get_info() if yolo else {"available": False},
        "package": loader.manifest.to_dict() if loader.manifest else {},
    }
    paths = exporter.finalize(loader.runtime_config, stats, video_path=video_path)
    source.release()
    return {"paths": paths, "stats": stats, "package_root": str(root)}
