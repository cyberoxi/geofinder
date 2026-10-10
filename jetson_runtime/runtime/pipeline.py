"""Jetson runtime processing pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from jetson_runtime.detectors.yolo_detector import YOLODetector
from jetson_runtime.exporters.result_exporter import ResultExporter, draw_overlay
from jetson_runtime.package_loader.loader import PackageLoader
from jetson_runtime.runtime.detect_tracker import DetectionTracker
from jetson_runtime.runtime.hybrid_tracker import MultiScaleHybridTracker
from jetson_runtime.serial_out import (
    SerialOffsetSender,
    image_to_target_offset,
    merge_serial_config,
)
from jetson_runtime.runtime.sources import FrameSource, InputSourceSpec, open_input_source
from shared.localization import draw_localization, put_outlined
from shared.logging import get_logger, setup_logging
from shared.schemas import ROI, RuntimeState

logger = get_logger("jetson.pipeline")

ProgressCb = Callable[[int, int, dict], None]
FrameCb = Callable[[Any, dict], None]


class RTSPVideoSource:
    """Stub for future RTSP support."""

    def __init__(self, url: str):
        raise NotImplementedError("RTSP source is reserved for a future release. Use file video for now.")


def _pick_model_path(engine, onnx, pt) -> tuple[str, list[str]]:
    """Prefer TensorRT, then GPU ONNX, else PyTorch (CUDA) over CPU-only ONNX."""
    if engine is not None:
        primary = engine
    elif onnx is not None and pt is not None:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            gpu_ort = any(p in providers for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider"))
        except Exception:  # noqa: BLE001
            gpu_ort = False
        # CPU ORT is much slower than .pt on Orin CUDA; Live Test on desktop uses .pt.
        primary = onnx if gpu_ort else pt
    else:
        primary = onnx or pt
    if primary is None:
        return "", []
    fallbacks = [str(p) for p in (engine, onnx, pt) if p is not None and p != primary]
    return str(primary), fallbacks


def build_detector_from_package(loader: PackageLoader) -> Optional[YOLODetector]:
    engine, onnx, pt = loader.model_candidates()
    ycfg = loader.runtime_config.get("yolo", {})
    model_path, fallbacks = _pick_model_path(engine, onnx, pt)
    if not model_path:
        logger.warning("No model in package — feature/homography-only mode")
        return None
    localizer = None
    scene_rel = getattr(loader.manifest, "scene_layout", "") if loader.manifest else ""
    if scene_rel and loader.root is not None and (loader.root / scene_rel).exists():
        from shared.landmarks import SceneLayout, TargetLocalizer

        layout = SceneLayout.load(loader.root / scene_rel)
        n_classes = len(loader.manifest.class_names) if loader.manifest else 0
        if n_classes and len(layout.classes) != n_classes:
            # Same guard as the desktop Live Test: mismatched class ids would misplace the target
            logger.warning("Landmarks off: scene has %d classes, model %d", len(layout.classes), n_classes)
        else:
            localizer = TargetLocalizer(layout)
            logger.info("Landmark localizer enabled (%d landmarks)", layout.num_landmarks)
    return YOLODetector(
        model_path=model_path,
        mode=ycfg.get("mode", loader.manifest.model_type if loader.manifest else "segmentation"),
        device=ycfg.get("device", "cuda"),
        imgsz=int(ycfg.get("imgsz", loader.manifest.input_size if loader.manifest else 640)),
        confidence_threshold=float(loader.manifest.min_confidence if loader.manifest else 0.45),
        # NMS IoU — same as the desktop Live Test (Ultralytics default). manifest.min_iou is a
        # tracking threshold; using it for NMS (0.2) suppressed valid overlapping detections.
        iou_threshold=float(ycfg.get("nms_iou", 0.7)),
        use_fp16=bool(ycfg.get("use_fp16", True)),
        warmup_iterations=int(ycfg.get("warmup_iterations", 5)),
        class_names=list(loader.manifest.class_names) if loader.manifest else ["target_region"],
        fallback_paths=fallbacks,
        localizer=localizer,
        stabilizer_cfg=loader.runtime_config.get("stabilizer"),
    )


def _draw_footer(img: np.ndarray, result) -> None:
    """Runtime stats under the desktop-style overlay (bottom-left)."""
    text = f"frame {result.frame_id}  {result.process_fps:.1f} fps  infer {result.inference_ms:.0f} ms"
    put_outlined(img, text, (12, img.shape[0] - 14), (230, 230, 230), 0.6, 2)


def _resolve_source_spec(
    video_path: Optional[str],
    prefer_gstreamer: bool,
    source: Optional[Any],
) -> InputSourceSpec:
    if isinstance(source, InputSourceSpec):
        return source
    if isinstance(source, dict):
        return InputSourceSpec.from_dict(source)
    if video_path:
        return InputSourceSpec.from_file(video_path, prefer_gstreamer=prefer_gstreamer)
    raise ValueError("Provide video_path or source (file / imx477 / usb)")


def run_runtime(
    package_path: str,
    video_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    prefer_gstreamer: bool = False,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    initial_roi: Optional[ROI] = None,
    end_frame: int = -1,
    serial_override: Optional[Dict[str, Any]] = None,
    source: Optional[Any] = None,
    min_conf: Optional[float] = None,
    min_conf_cb: Optional[Callable[[], Optional[float]]] = None,
) -> Dict[str, Any]:
    """``min_conf`` overrides the package's min confidence; ``min_conf_cb`` is polled every frame
    so a GUI slider can change it while running (both 0–1)."""
    setup_logging()
    loader = PackageLoader(package_path)
    root = loader.load()
    yolo = build_detector_from_package(loader)
    if yolo is not None and min_conf is not None:
        yolo.set_confidence(min_conf)
    if yolo is not None:
        logger.info("Min confidence %.0f%%", yolo.confidence_threshold * 100)
        if not yolo.available:
            # Package has a model but the runtime stack failed (missing ultralytics/torch,
            # bad ONNX/engine, …). Do not silently search forever with empty detections.
            raise RuntimeError(
                "YOLO model could not be loaded — nothing will be detected.\n"
                "مدل YOLO بارگذاری نشد؛ هیچ هدفی شناسایی نمی‌شود.\n\n"
                "On Jetson (venv):\n"
                "  pip install --pre torch torchvision "
                "--extra-index-url https://download.pytorch.org/whl/cu132\n"
                "  pip install ultralytics onnx\n\n"
                "Then restart the app with: source .venv/bin/activate"
            )
        # Same per-frame detect + stabilizer logic as the desktop Live Test
        tracker = DetectionTracker(loader.runtime_config, yolo)
    else:
        tracker = MultiScaleHybridTracker(
            loader.runtime_config,
            yolo=None,
            reference_descriptors=loader.descriptors_path(),
            feature_method=loader.manifest.feature_method if loader.manifest else "AKAZE",
        )

    spec = _resolve_source_spec(video_path, prefer_gstreamer, source)
    frame_source: FrameSource = open_input_source(spec)
    assert frame_source.info is not None
    source_label = frame_source.label
    logger.info("Input source: %s (live=%s)", source_label, frame_source.is_live)

    out_dir = Path(output_dir) if output_dir else root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = out_dir / "failed_frames"
    exporter = ResultExporter(str(out_dir), str(failed_dir), loader.runtime_config)
    if loader.runtime_config.get("export", {}).get("save_annotated_video", True):
        exporter.start_video(
            None, frame_source.info.width, frame_source.info.height, frame_source.info.fps
        )

    serial_cfg = merge_serial_config(loader.runtime_config.get("serial"), serial_override)
    serial = SerialOffsetSender(serial_cfg)
    serial.open()

    ok, frame = frame_source.seek(0) if not frame_source.is_live else frame_source.read()
    if not ok or frame is None:
        serial.close()
        frame_source.release()
        raise RuntimeError(f"Cannot read frames from {source_label}")

    if isinstance(tracker, DetectionTracker):
        # No guessed start ROI: the detector + stabilizer decide from frame 0
        if initial_roi is not None:
            tracker.initialize(frame, initial_roi)
    elif initial_roi is not None:
        tracker.initialize(frame, initial_roi)
    else:
        # Feature-only mode (no model): may start from the package reference polygon
        ok_init = tracker.initialize_from_search(frame)
        # While the temporal filter is confirming a candidate, keep searching instead of guessing a ROI
        if tracker.awaiting_confirmation:
            logger.info("Target candidate seen — confirming over the next frames")
        elif not ok_init and loader.manifest and loader.manifest.reference_roi.get("points"):
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
    if end_frame >= 0:
        last = end_frame
    elif frame_source.is_live or frame_source.info.frame_count <= 0:
        last = 10**9
    else:
        last = frame_source.info.frame_count - 1
    tracking_ok = {RuntimeState.TRACKING, RuntimeState.SCALE_CHANGE}
    progress_total = max(1, last + 1) if last < 10**9 else 0

    try:
        while True:
            if stop_flag and stop_flag():
                break
            if idx > last:
                break
            if yolo is not None and min_conf_cb is not None:
                want = min_conf_cb()
                if want is not None and abs(want - yolo.confidence_threshold) > 1e-6:
                    yolo.set_confidence(want)
                    logger.info("Min confidence → %.0f%%", yolo.confidence_threshold * 100)
            t_frame = time.perf_counter()
            state = tracker.update(frame, frame_idx=idx)
            dt = time.perf_counter() - t_frame
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_list.append(fps)
            result = tracker.to_frame_result(
                idx, frame_source.reader.timestamp_of(idx), process_fps=fps
            )
            exporter.add_result(result)

            fh, fw = frame.shape[:2]
            info = result.to_dict()
            is_tracking = (
                tracker.is_tracking
                if isinstance(tracker, DetectionTracker)
                else state.status in tracking_ok and state.roi is not None
            )
            off = tracker.loc.get("offset") if isinstance(tracker, DetectionTracker) else None
            if is_tracking and off is not None:
                # Target centroid − image centre: the same numbers the overlay shows (desktop parity)
                dx, dy, dist = off["dx"], off["dy"], off["distance_px"]
            elif is_tracking:
                dx, dy, dist = image_to_target_offset(fw, fh, result.center_x, result.center_y)
            else:
                dx, dy, dist = 0.0, 0.0, 0.0
            info["offset_dx"] = dx
            info["offset_dy"] = dy
            info["offset_dist"] = dist
            info["source"] = source_label
            if yolo is not None:
                info["min_conf"] = yolo.confidence_threshold
            serial.send_offset(dx, dy, distance=dist, status=result.status, tracking=is_tracking)

            if isinstance(tracker, DetectionTracker):
                # Same overlay as the desktop Live Test: target / aux objects / rejected / confirming,
                # centre cross, arrow to the target (frame edge when off-screen), centre distance
                annotated = draw_localization(frame, tracker.dets, tracker.loc, yolo.class_names)
                _draw_footer(annotated, result)
            else:
                annotated = draw_overlay(frame, result, trail=state.trail)
            exporter.write_frame(annotated)
            if state.status == RuntimeState.LOST and loader.runtime_config.get("export", {}).get("save_failed_frames", True):
                exporter.save_failed_frame(frame, idx)
            processed += 1
            if progress_cb:
                progress_cb(processed, progress_total or processed, info)
            if frame_cb:
                frame_cb(annotated, info)

            step = 1 + max(0, int(getattr(tracker, "skip_frames", 0)))
            next_idx = idx + step
            if next_idx > last:
                break
            if frame_source.is_live:
                # Drop intermediate frames when skip_frames > 0
                ok, frame = False, None
                for _ in range(step):
                    ok, frame = frame_source.read()
                    if not ok or frame is None:
                        break
            else:
                ok, frame = frame_source.seek(next_idx)
            idx = next_idx
            if not ok or frame is None:
                break
    finally:
        serial.close()

    elapsed = time.perf_counter() - t0
    tracker.state.status = RuntimeState.COMPLETED
    stats = {
        **tracker.stats,
        "processed_frames": processed,
        "elapsed_sec": elapsed,
        "avg_fps": processed / elapsed if elapsed > 0 else 0.0,
        "min_fps": min(fps_list) if fps_list else 0.0,
        "input": spec.to_dict(),
        "yolo": yolo.get_info() if yolo else {"available": False},
        "package": loader.manifest.to_dict() if loader.manifest else {},
        "serial": {k: serial_cfg[k] for k in ("enabled", "port", "baudrate") if k in serial_cfg},
        "stabilizer": dict(tracker.stabilizer.cfg),
    }
    paths = exporter.finalize(loader.runtime_config, stats, video_path=source_label)
    frame_source.release()
    return {"paths": paths, "stats": stats, "package_root": str(root)}
