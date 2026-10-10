"""Run a trained model on a video: detect target + landmarks, infer target from landmarks.

Used to test a model on the desktop before shipping it to the Jetson, and to
score it automatically on held-out auto-labeled videos.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from shared.landmarks import SceneLayout, TargetLocalizer, polygon_iou
from shared.localization import (  # noqa: F401 — re-exported for the desktop GUI
    CENTER_COLOR,
    GT_COLOR,
    INFERRED_COLOR,
    LANDMARK_COLOR,
    PENDING_COLOR,
    PREDICTED_COLOR,
    REJECT_COLOR,
    TARGET_COLOR,
    det_polygon as _det_polygon,
    draw_localization,
    put_outlined as _put_outlined,
    stabilize_frame,
    target_center_offset,
)
from shared.logging import get_logger
from shared.schemas import Detection
from shared.video_reader import VideoReader

logger = get_logger("desktop.inference")

ProgressCb = Callable[[int, int, str], None]


def _detections_from_result(r0: Any, frame_shape) -> List[Detection]:
    dets: List[Detection] = []
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    polys = None
    masks = getattr(r0, "masks", None)
    if masks is not None and getattr(masks, "xy", None) is not None:
        polys = masks.xy
    for i, (b, c, k) in enumerate(zip(xyxy, confs, clss)):
        x1, y1, x2, y2 = map(float, b)
        d = Detection(bbox=(x1, y1, x2 - x1, y2 - y1), confidence=float(c), class_id=int(k))
        if polys is not None and i < len(polys) and len(polys[i]) >= 3:
            d.polygon = np.asarray(polys[i], np.float32)
        dets.append(d)
    return dets


def localize_frame(
    dets: List[Detection],
    localizer: Optional[TargetLocalizer],
    frame_shape,
    target_conf: float = 0.35,
) -> Dict[str, Any]:
    """Decide the target polygon for one frame: direct detection first, landmarks second."""
    targets = [d for d in dets if d.class_id == 0]
    best = max(targets, key=lambda d: d.confidence) if targets else None
    # Prefer landmark-only transform when target is absent so off-screen location still works.
    est = None
    if localizer is not None:
        est = localizer.estimate(dets, frame_shape, include_target=best is not None)

    if best is not None and best.confidence >= target_conf:
        poly = _det_polygon(best)
        source, conf = "yolo", best.confidence
    elif est is not None:
        poly = est.polygon
        source, conf = "landmarks", est.confidence
    elif best is not None:
        poly = _det_polygon(best)
        source, conf = "yolo_low", best.confidence
    else:
        poly, source, conf = None, "none", 0.0

    offset = target_center_offset(poly, frame_shape)
    landmarks_used = (
        list(est.landmarks_used) if est is not None and source == "landmarks" else []
    )
    return {
        "polygon": poly,
        "source": source,
        "confidence": float(conf),
        "estimate": est,
        "offset": offset,
        "landmarks_used": landmarks_used,
    }


def run_on_video(
    weights: str | Path,
    video: str | Path,
    *,
    scene_json: Optional[str | Path] = None,
    out_video: Optional[str | Path] = None,
    labels_json: Optional[str | Path] = None,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
    every_n: int = 1,
    progress_cb: Optional[ProgressCb] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Returns metrics.  With ``labels_json`` (auto-labeled ground truth) it also
    reports IoU / centre error of the final target estimate.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)] if isinstance(model.names, dict) else list(model.names)
    localizer = None
    if scene_json and Path(scene_json).exists():
        layout = SceneLayout.load(scene_json)
        if len(layout.classes) == len(names):
            localizer = TargetLocalizer(layout)
        else:
            logger.warning("Scene has %d classes but model has %d — landmark inference disabled", len(layout.classes), len(names))
    gt = json.loads(Path(labels_json).read_text(encoding="utf-8")) if labels_json else None

    reader = VideoReader(str(video), prefer_gstreamer=False)
    info = reader.info
    assert info is not None
    writer = None
    if out_video:
        Path(out_video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, info.fps / every_n), (info.width, info.height))

    total = info.frame_count if info.frame_count > 0 else 0
    stats = {"frames": 0, "yolo": 0, "landmarks": 0, "none": 0, "yolo_low": 0}
    ious: List[float] = []
    center_err: List[float] = []
    gt_frames = gt_located = 0
    fid = -1
    try:
        while True:
            if stop_flag and stop_flag():
                break
            ok, frame = reader.read()
            if not ok or frame is None:
                break
            fid += 1
            if fid % every_n:
                continue
            r = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
            dets = _detections_from_result(r[0], frame.shape) if r else []
            loc = localize_frame(dets, localizer, frame.shape)
            stats["frames"] += 1
            stats[loc["source"]] += 1
            if gt is not None and fid < len(gt["frames"]):
                gt_target = [np.asarray(p, np.float32) for c, p in gt["frames"][fid] if int(c) == 0]
                if gt_target:
                    gt_frames += 1
                    if loc["polygon"] is not None:
                        gt_located += 1
                        h, w = frame.shape[:2]
                        ious.append(polygon_iou(loc["polygon"], gt_target[0], w, h))
                        center_err.append(float(np.linalg.norm(loc["polygon"].mean(axis=0) - gt_target[0].mean(axis=0))))
            if writer is not None:
                img = draw_localization(frame, dets, loc, names)
                if gt is not None and fid < len(gt["frames"]):
                    for c, p in gt["frames"][fid]:
                        if int(c) == 0:
                            cv2.polylines(img, [np.int32(p).reshape(-1, 1, 2)], True, GT_COLOR, 1, cv2.LINE_AA)
                writer.write(img)
            if progress_cb and (stats["frames"] % 10 == 1):
                progress_cb(fid, total, f"Inference frame {fid}/{total}")
    finally:
        reader.release()
        if writer is not None:
            writer.release()

    n = max(1, stats["frames"])
    metrics: Dict[str, Any] = {
        "video": str(video),
        "frames": stats["frames"],
        "target_detected_rate": stats["yolo"] / n,
        "inferred_from_landmarks_rate": stats["landmarks"] / n,
        "located_rate": (stats["yolo"] + stats["landmarks"] + stats["yolo_low"]) / n,
        "sources": stats,
        "landmark_localizer": localizer is not None,
        "output_video": str(out_video) if out_video else None,
    }
    if gt is not None:
        metrics.update(
            {
                "gt_frames_with_target": gt_frames,
                "recall": gt_located / max(1, gt_frames),
                "mean_iou": float(np.mean(ious)) if ious else 0.0,
                "iou_at_0.5": float(np.mean([i >= 0.5 for i in ious])) if ious else 0.0,
                "mean_center_error_px": float(np.mean(center_err)) if center_err else None,
            }
        )
    return metrics
