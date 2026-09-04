"""Headless / shared processing pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from core.config import load_config
from core.hybrid_tracker import HybridTracker
from core.logger import get_logger, setup_logging
from core.project_io import load_project, roi_from_project
from core.result_exporter import ResultExporter, draw_overlay
from core.types import ROI, TrackingStatus
from core.video_reader import VideoReader
from core.yolo_detector import YOLODetector

logger = get_logger("grt.pipeline")


ProgressCallback = Callable[[int, int, dict], None]


def build_yolo(config: Dict[str, Any]) -> YOLODetector:
    y = config.get("yolo", {})
    models_dir = Path(config.get("paths", {}).get("models_dir", "models"))
    model_path = y.get("model_path") or str(models_dir / "best.engine")
    fallbacks = [
        y.get("fallback_model_path") or str(models_dir / "best.onnx"),
        y.get("pytorch_model_path") or str(models_dir / "best.pt"),
    ]
    return YOLODetector(
        model_path=model_path,
        mode=y.get("mode", "detection"),
        device=y.get("device", "cuda"),
        imgsz=int(y.get("imgsz", 640)),
        confidence_threshold=float(y.get("confidence_threshold", 0.45)),
        iou_threshold=float(y.get("iou_threshold", 0.45)),
        target_class=int(y.get("target_class", 0)),
        use_fp16=bool(y.get("use_fp16", True)),
        warmup_iterations=int(y.get("warmup_iterations", 5)),
        class_names=list(y.get("class_names", ["ground_region"])),
        fallback_paths=fallbacks,
    )


def run_pipeline(
    video_path: str,
    roi: Optional[ROI] = None,
    config: Optional[Dict[str, Any]] = None,
    project_path: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    frame_cb: Optional[Callable[[np.ndarray, dict], None]] = None,
) -> Dict[str, Any]:
    """
    Process a video end-to-end. Returns paths and stats.
    """
    config = config or load_config()
    log_cfg = config.get("logging", {})
    setup_logging(
        log_file=log_cfg.get("file"),
        level=log_cfg.get("level", "INFO"),
        max_bytes=int(log_cfg.get("max_bytes", 5_242_880)),
        backup_count=int(log_cfg.get("backup_count", 3)),
    )

    if project_path:
        proj = load_project(project_path)
        video_path = proj.get("video_path", video_path)
        roi = roi_from_project(proj)
        if "tracker_type" in proj:
            config.setdefault("tracking", {})["tracker_type"] = proj["tracker_type"]
        if "confidence_threshold" in proj:
            config.setdefault("tracking", {})["confidence_threshold"] = proj["confidence_threshold"]
        if proj.get("model_path"):
            config.setdefault("yolo", {})["model_path"] = proj["model_path"]

    prefer_gst = bool(config.get("video", {}).get("prefer_gstreamer", True))
    reader = VideoReader(video_path, prefer_gstreamer=prefer_gst)
    assert reader.info is not None

    start_frame = int(config.get("video", {}).get("start_frame", 0))
    end_frame = int(config.get("video", {}).get("end_frame", -1))
    skip = int(config.get("video", {}).get("skip_frames", 0))
    if end_frame < 0:
        end_frame = reader.frame_count - 1 if reader.frame_count > 0 else 10**9

    if roi is None and config.get("tracking", {}).get("mode") != "yolo":
        raise ValueError("ROI is required unless tracking mode is 'yolo'")

    yolo = None
    mode = config.get("tracking", {}).get("mode", "hybrid")
    if mode in ("hybrid", "yolo") and config.get("yolo", {}).get("enabled", True):
        yolo = build_yolo(config)

    tracker = HybridTracker(config, yolo=yolo)
    exporter = ResultExporter(
        config["paths"]["output_dir"],
        config["paths"]["failed_frames_dir"],
        config,
    )

    # Seek to initial ROI frame or start
    init_idx = roi.frame_index if roi is not None else start_frame
    init_idx = max(init_idx, start_frame)
    ok, frame = reader.seek(init_idx)
    if not ok or frame is None:
        reader.release()
        raise RuntimeError(f"Cannot read initial frame {init_idx}")

    if not tracker.initialize(frame, roi):
        reader.release()
        raise RuntimeError(tracker.state.message or "Tracker initialization failed")

    if config.get("export", {}).get("save_annotated_video", True):
        exporter.start_video(None, reader.width, reader.height, reader.fps)

    processed = 0
    t_start = time.perf_counter()
    idx = init_idx

    # Process current init frame first
    while True:
        if stop_flag and stop_flag():
            break
        if idx > end_frame:
            break

        t0 = time.perf_counter()
        state = tracker.update(frame, frame_idx=idx)
        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        ts = reader.timestamp_of(idx)
        result = tracker.to_frame_result(idx, ts, process_fps=fps)
        exporter.add_result(result)

        annotated = draw_overlay(
            frame,
            result,
            trail=state.trail,
            draw_bbox=config.get("gui", {}).get("overlay_bbox", True),
            draw_polygon=config.get("gui", {}).get("overlay_polygon", True),
            draw_center=config.get("gui", {}).get("overlay_center", True),
            draw_trail=config.get("gui", {}).get("overlay_trail", True),
        )
        exporter.write_frame(annotated)

        if state.status == TrackingStatus.LOST and config.get("export", {}).get("save_failed_frames", True):
            exporter.save_failed_frame(frame, idx)
        if config.get("export", {}).get("save_roi_crops", False) and state.roi is not None:
            exporter.save_roi_crop(frame, result)

        processed += 1
        if progress_cb:
            total = max(1, end_frame - init_idx + 1)
            progress_cb(processed, total, result.to_dict())
        if frame_cb:
            frame_cb(annotated, result.to_dict())

        # Advance
        step = 1 + max(0, skip)
        target = idx + step
        if target > end_frame:
            break
        if skip == 0:
            ok, frame = reader.read()
            idx += 1
        else:
            ok, frame = reader.seek(target)
            idx = target
        if not ok or frame is None:
            break

    elapsed = time.perf_counter() - t_start
    tracker.state.status = TrackingStatus.COMPLETED
    stats = {
        **tracker.stats,
        "processed_frames": processed,
        "elapsed_sec": elapsed,
        "avg_fps": processed / elapsed if elapsed > 0 else 0.0,
        "video": reader.info.__dict__ if reader.info else {},
        "yolo": yolo.get_info() if yolo else {"available": False},
    }
    paths = exporter.finalize(config, stats, video_path=video_path)
    reader.release()
    logger.info("Pipeline completed: %s", paths)
    return {"paths": paths, "stats": stats}
