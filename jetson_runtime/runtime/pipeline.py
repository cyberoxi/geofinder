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


_CUDA_OK: Optional[bool] = None


def _cuda_usable() -> bool:
    """True only if a real CUDA alloc works (Jetson often reports available then NvRm fails)."""
    global _CUDA_OK
    if _CUDA_OK is not None:
        return _CUDA_OK
    try:
        import torch

        if not torch.cuda.is_available():
            _CUDA_OK = False
            return False
        torch.zeros(1, device="cuda")
        _CUDA_OK = True
    except Exception:  # noqa: BLE001
        _CUDA_OK = False
    return bool(_CUDA_OK)


def _pick_model_path(engine, onnx, pt) -> tuple[str, list[str]]:
    """Prefer TensorRT / CUDA PyTorch when GPU works; on CPU prefer ONNX over .pt."""
    cuda = _cuda_usable()
    if cuda:
        if engine is not None:
            primary = engine
        elif pt is not None:
            # .pt on Orin CUDA beats CPU ONNX; GPU ORT is optional.
            primary = pt
        else:
            primary = onnx
    else:
        # Engine / CUDA .pt cannot run; ONNX via ORT-CPU is the least-bad fallback.
        if engine is not None:
            logger.warning("CUDA unavailable — skipping TensorRT engine (use ONNX/CPU)")
        primary = onnx or pt
        if primary is not None:
            logger.warning(
                "CUDA unavailable (nvidia-smi / NvRmGpuLibOpen) — running on CPU. "
                "Reboot the Jetson for normal FPS; CPU is often only a few FPS."
            )
    if primary is None:
        return "", []
    fallbacks = [str(p) for p in (engine, onnx, pt) if p is not None and p != primary]
    return str(primary), fallbacks


def _apply_cpu_speed_defaults(loader: PackageLoader, yolo: YOLODetector) -> None:
    """When stuck on CPU, detect less often and shrink .pt imgsz so optical flow can carry FPS."""
    if yolo.device != "cpu":
        return
    tcfg = loader.runtime_config.setdefault("tracking", {})
    # ~220ms/infer on Orin CPU → need sparse YOLO + frame skip for usable wall FPS.
    every = max(int(tcfg.get("detect_every_n", 2)), 6)
    tcfg["detect_every_n"] = every
    tcfg["target_fps"] = max(float(tcfg.get("target_fps", 20.0)), 15.0)
    tcfg["adaptive_skip"] = True
    if yolo.backend == "pytorch" and yolo.imgsz > 320:
        # Static ONNX/engine inputs must stay at export size; .pt can letterbox smaller.
        logger.info("CPU PyTorch: imgsz %d → 320 for speed", yolo.imgsz)
        yolo.imgsz = 320
    logger.info("CPU speed mode: detect_every_n=%d imgsz=%d (reboot for GPU/CUDA)", every, yolo.imgsz)


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
    # If CUDA is down, force cpu so Ultralytics does not keep retrying a dead GPU.
    device = ycfg.get("device", "cuda")
    if str(device).lower() in ("cuda", "0", "gpu", "tensorrt") and not _cuda_usable():
        device = "cpu"
    yolo = YOLODetector(
        model_path=model_path,
        mode=ycfg.get("mode", loader.manifest.model_type if loader.manifest else "segmentation"),
        device=device,
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
    if yolo.available:
        _apply_cpu_speed_defaults(loader, yolo)
    return yolo


def _draw_footer(img: np.ndarray, result, avg_fps: float = 0.0) -> None:
    """Runtime stats under the desktop-style overlay (bottom-left)."""
    text = (
        f"frame {result.frame_id}  {avg_fps:.1f} fps  "
        f"infer {result.inference_ms:.0f} ms"
    )
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
    stabilizer_override: Optional[Dict[str, Any]] = None,
    stabilizer_cb: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    image_adjust: Optional[Dict[str, Any]] = None,
    image_adjust_cb: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """``min_conf`` overrides the package's min confidence; ``min_conf_cb`` is polled every frame
    so a GUI slider can change it while running (both 0–1).

    ``stabilizer_override`` / ``stabilizer_cb`` mirror the desktop Live Test filter toggles
    (geometry / temporal / aux-consistency).

    ``image_adjust`` / ``image_adjust_cb``: live colour normalisation of frames before detection
    (shared.image_adjust); falls back to the package ``runtime.yaml`` ``image_adjust`` section.
    """
    setup_logging()
    loader = PackageLoader(package_path)
    root = loader.load()
    if stabilizer_override:
        from shared.target_stabilizer import merge_stabilizer_config

        loader.runtime_config["stabilizer"] = merge_stabilizer_config(stabilizer_override)
        logger.info(
            "Stabilizer override: geometry=%s temporal=%s aux=%s",
            loader.runtime_config["stabilizer"].get("geometry_check"),
            loader.runtime_config["stabilizer"].get("temporal"),
            loader.runtime_config["stabilizer"].get("landmark_consistency"),
        )
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
    # Do not hard-fail on soft/blank AE frames — Camera Test / Import Source already
    # preview the same feed; keep Run usable and log a warning instead.
    frame_source: FrameSource = open_input_source(spec, require_scene=False)
    assert frame_source.info is not None
    source_label = frame_source.label
    logger.info("Input source: %s (live=%s)", source_label, frame_source.is_live)

    out_dir = Path(output_dir) if output_dir else root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = out_dir / "failed_frames"
    exporter = ResultExporter(str(out_dir), str(failed_dir), loader.runtime_config)
    export_cfg = loader.runtime_config.get("export", {})
    # Software mp4 encode every frame is a major FPS tax on Jetson; skip for live
    # camera and GUI preview runs (CLI file batch still writes annotated video).
    save_video = bool(export_cfg.get("save_annotated_video", True))
    if save_video and (frame_source.is_live or frame_cb is not None):
        logger.info("Annotated video export disabled for live/GUI run (keeps FPS high)")
        save_video = False
    if save_video:
        exporter.start_video(
            None, frame_source.info.width, frame_source.info.height, frame_source.info.fps
        )

    from shared.image_adjust import ImageAdjuster, describe_adjust

    color_ref = loader.color_stats()
    adjuster = ImageAdjuster(image_adjust if image_adjust is not None else loader.runtime_config.get("image_adjust"),
                             reference=color_ref)
    logger.info("%s", describe_adjust(adjuster.cfg, color_ref is not None))

    serial_cfg = merge_serial_config(loader.runtime_config.get("serial"), serial_override)
    serial = SerialOffsetSender(serial_cfg)
    serial.open()

    ok, frame = frame_source.seek(0) if not frame_source.is_live else frame_source.read()
    if not ok or frame is None:
        serial.close()
        frame_source.release()
        raise RuntimeError(f"Cannot read frames from {source_label}")
    if frame_source.is_live:
        from shared.video_reader import frame_lacks_scene, imx477_blank_frame_hint

        if frame_lacks_scene(frame):
            logger.warning("%s", imx477_blank_frame_hint())

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
    last_ui = 0.0
    # Lower preview rate on CPU so Qt resize/paint does not eat detect time
    ui_hz = 10.0 if (yolo is not None and yolo.device == "cpu") else 18.0
    ui_interval = 1.0 / ui_hz
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
            if stabilizer_cb is not None:
                want_stab = stabilizer_cb()
                if want_stab is not None and hasattr(tracker, "set_stabilizer"):
                    # Cheap change detect: compare the three master switches + key ints.
                    cur = getattr(tracker, "stabilizer", None)
                    cur_cfg = cur.cfg if cur is not None else {}
                    keys = (
                        "geometry_check",
                        "temporal",
                        "landmark_consistency",
                        "confirm_hits",
                        "confirm_window",
                        "min_landmarks_acquire",
                    )
                    if any(want_stab.get(k) != cur_cfg.get(k) for k in keys):
                        tracker.set_stabilizer(want_stab)
                        logger.info(
                            "Filters → geometry=%s temporal=%s aux=%s",
                            want_stab.get("geometry_check"),
                            want_stab.get("temporal"),
                            want_stab.get("landmark_consistency"),
                        )
            if image_adjust_cb is not None and adjuster.configure(image_adjust_cb()):
                logger.info("%s", describe_adjust(adjuster.cfg, color_ref is not None))
            t_frame = time.perf_counter()
            raw_frame = frame
            if adjuster.enabled:
                frame = adjuster.apply(frame)  # the detector sees the colour-normalised frame
            state = tracker.update(frame, frame_idx=idx)
            dt = time.perf_counter() - t_frame
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_list.append(fps)
            # Wall-clock average (instant 1/infer looks like ~4 FPS on CPU detect frames).
            wall_fps = (processed + 1) / max(time.perf_counter() - t0, 1e-6)
            result = tracker.to_frame_result(
                idx, frame_source.reader.timestamp_of(idx), process_fps=wall_fps
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

            now = time.perf_counter()
            want_ui = frame_cb is not None and (now - last_ui) >= ui_interval
            if save_video or want_ui:
                shown = frame if (not adjuster.enabled or adjuster.cfg["show_adjusted"]) else raw_frame
                if isinstance(tracker, DetectionTracker):
                    # Same overlay as the desktop Live Test: target / aux objects / rejected /
                    # confirming, centre cross, arrow to the target, centre distance
                    annotated = draw_localization(shown, tracker.dets, tracker.loc, yolo.class_names)
                    _draw_footer(annotated, result, avg_fps=wall_fps)
                else:
                    annotated = draw_overlay(shown, result, trail=state.trail)
                if save_video:
                    exporter.write_frame(annotated)
                if want_ui:
                    last_ui = now
                    frame_cb(annotated, info)
            # JPEG dumps on LOST are a large I/O tax; skip for live/GUI (same as video export).
            if (
                state.status == RuntimeState.LOST
                and export_cfg.get("save_failed_frames", True)
                and not frame_source.is_live
                and frame_cb is None
            ):
                exporter.save_failed_frame(frame, idx)
            processed += 1
            if progress_cb:
                progress_cb(processed, progress_total or processed, info)

            step = 1 + max(0, int(getattr(tracker, "skip_frames", 0)))
            next_idx = idx + step
            if next_idx > last:
                break
            # Always advance with sequential read — CAP_PROP seek-per-frame is very
            # slow on Jetson OpenCV backends and was the main file-run FPS bottleneck.
            ok, frame = False, None
            for _ in range(step):
                ok, frame = frame_source.read()
                if not ok or frame is None:
                    break
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
        "image_adjust": dict(adjuster.cfg),
    }
    paths = exporter.finalize(loader.runtime_config, stats, video_path=source_label)
    frame_source.release()
    return {"paths": paths, "stats": stats, "package_root": str(root)}
