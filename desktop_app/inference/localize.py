"""Run a trained model on a video: detect target + landmarks, infer target from landmarks.

Used to test a model on the desktop before shipping it to the Jetson, and to
score it automatically on held-out auto-labeled videos.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from shared.landmarks import SceneLayout, TargetLocalizer, polygon_iou
from shared.logging import get_logger
from shared.schemas import Detection, ROI
from shared.target_filter import filter_frame
from shared.target_stabilizer import Candidate, TargetStabilizer
from shared.video_reader import VideoReader

logger = get_logger("desktop.inference")

ProgressCb = Callable[[int, int, str], None]

TARGET_COLOR = (0, 220, 0)
INFERRED_COLOR = (255, 0, 255)
LANDMARK_COLOR = (0, 160, 255)
GT_COLOR = (255, 255, 255)
CENTER_COLOR = (80, 200, 255)
PENDING_COLOR = (0, 220, 255)  # yellow: seen but not yet confirmed
REJECT_COLOR = (60, 60, 255)  # red: dropped by a filter
PREDICTED_COLOR = (200, 200, 0)  # cyan: coasting on prediction


def target_center_offset(polygon: Optional[np.ndarray], frame_shape) -> Optional[Dict[str, Any]]:
    """Pixel offset of the target centroid from the image center."""
    if polygon is None or len(polygon) < 1:
        return None
    h, w = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    tc = np.asarray(polygon, np.float32).reshape(-1, 2).mean(axis=0)
    dx = float(tc[0] - cx)
    dy = float(tc[1] - cy)
    return {
        "dx": dx,
        "dy": dy,
        "distance_px": float(math.hypot(dx, dy)),
        "angle_deg": float(math.degrees(math.atan2(dy, dx))),
        "target_xy": (float(tc[0]), float(tc[1])),
        "center_xy": (cx, cy),
        "in_frame": bool(0 <= tc[0] < w and 0 <= tc[1] < h),
    }


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


def _det_polygon(d: Detection) -> np.ndarray:
    if d.polygon is not None and len(d.polygon) >= 3:
        return np.asarray(d.polygon, np.float32)
    return ROI.from_bbox(*d.bbox).points


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


def stabilize_frame(
    dets: List[Detection],
    localizer: Optional[TargetLocalizer],
    frame_shape,
    min_conf: float,
    stabilizer: TargetStabilizer,
    frame_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Like :func:`localize_frame` but with the false-target filters (geometry, aux, temporal)."""
    flt = filter_frame(dets, localizer, frame_shape, stabilizer.cfg, min_conf)
    est = flt.estimate
    cands = [Candidate(_det_polygon(d), float(d.confidence), "yolo") for d in flt.targets]
    if est is not None:
        cands.append(Candidate(est.polygon, est.confidence, "landmarks", flt.n_landmarks))
    st = stabilizer.update(cands, min_conf, frame_idx)
    tracking = st.tracking and st.polygon is not None
    poly = st.polygon if tracking else None
    source = st.source if tracking else "none"
    return {
        "polygon": poly,
        "source": source,
        "confidence": float(st.confidence) if tracking else 0.0,
        "estimate": est,
        "offset": target_center_offset(poly, frame_shape),
        "landmarks_used": list(est.landmarks_used) if est is not None and source == "landmarks" else [],
        "state": st.state,
        "tracking": tracking,
        "progress": st.progress,
        "reason": st.reason,
        "pending_polygon": st.polygon if st.state == "confirming" else None,
        "rejected": flt.rejected,
        "min_conf": float(min_conf),
    }


def _put_outlined(img: np.ndarray, text: str, org: tuple, color, scale: float = 0.7, thickness: int = 2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_localization(frame: np.ndarray, dets: List[Detection], loc: Dict[str, Any], names: List[str]) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    cv2.drawMarker(img, center, CENTER_COLOR, cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    cv2.circle(img, center, 10, CENTER_COLOR, 1, cv2.LINE_AA)

    rejected = {id(d): why for d, why in loc.get("rejected") or []}
    for d in dets:
        x, y, bw, bh = map(int, d.bbox)
        if id(d) in rejected:
            cv2.rectangle(img, (x, y), (x + bw, y + bh), REJECT_COLOR, 2)
            cv2.line(img, (x, y), (x + bw, y + bh), REJECT_COLOR, 1, cv2.LINE_AA)
            cv2.line(img, (x + bw, y), (x, y + bh), REJECT_COLOR, 1, cv2.LINE_AA)
            name = names[d.class_id] if 0 <= d.class_id < len(names) else str(d.class_id)
            _put_outlined(img, f"X {name} {d.confidence:.2f} {rejected[id(d)]}", (x, max(14, y - 4)),
                          REJECT_COLOR, 0.45, 1)
            continue
        if d.class_id == 0 or d.confidence < loc.get("min_conf", 0.0):
            continue
        cv2.rectangle(img, (x, y), (x + bw, y + bh), LANDMARK_COLOR, 2)
        name = names[d.class_id] if 0 <= d.class_id < len(names) else str(d.class_id)
        cv2.putText(
            img, f"{name} {d.confidence:.2f}", (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, LANDMARK_COLOR, 1, cv2.LINE_AA,
        )

    pending = loc.get("pending_polygon")
    if pending is not None:
        pts = np.round(pending).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, PENDING_COLOR, 2, cv2.LINE_AA)
        tc = pending.mean(axis=0)
        _put_outlined(img, f"confirming {loc.get('progress', '')}", (int(tc[0]) - 50, int(tc[1])),
                      PENDING_COLOR, 0.55, 1)

    poly = loc.get("polygon")
    offset = loc.get("offset") or target_center_offset(poly, frame.shape)
    if poly is not None:
        color = {"landmarks": INFERRED_COLOR, "predicted": PREDICTED_COLOR}.get(loc["source"], TARGET_COLOR)
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.polylines(img, [pts], True, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color, 3, cv2.LINE_AA)
        tc = poly.mean(axis=0)
        tip = (int(round(tc[0])), int(round(tc[1])))
        # Line / arrow from image center to target (works on- and off-screen).
        if offset is not None and offset["distance_px"] > 2:
            if offset["in_frame"]:
                cv2.arrowedLine(img, center, tip, color, 2, cv2.LINE_AA, tipLength=0.12)
            else:
                v = tc.astype(float) - np.array([w / 2.0, h / 2.0])
                edge = np.array([w / 2.0, h / 2.0]) + v / max(1e-6, np.linalg.norm(v)) * 0.42 * min(w, h)
                cv2.arrowedLine(img, center, tuple(map(int, edge)), color, 3, cv2.LINE_AA, tipLength=0.18)
            if offset["in_frame"]:
                cv2.drawMarker(img, tip, (255, 255, 255), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
                cv2.drawMarker(img, tip, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    label = {
        "yolo": "TARGET (detected)",
        "yolo_low": "target? (low conf)",
        "landmarks": "TARGET (from landmarks)",
        "predicted": "TARGET (predicted)",
        "none": "confirming…" if loc.get("state") == "confirming" else "searching…",
    }[loc["source"]]
    state = loc.get("state")
    state_txt = f"  [{state.upper()}{' ' + loc['reason'] if loc.get('reason') else ''}]" if state else ""
    _put_outlined(img, f"{label} {loc['confidence']:.2f}{state_txt}", (12, 28), (255, 255, 255), 0.75, 2)

    if offset is not None:
        dist = offset["distance_px"]
        dx, dy = offset["dx"], offset["dy"]
        via = {"landmarks": "landmarks", "predicted": "predicted"}.get(loc["source"], "direct")
        dist_txt = f"center dist {dist:.0f}px  dx={dx:+.0f} dy={dy:+.0f}  ({via})"
        via_color = {"direct": CENTER_COLOR, "predicted": PREDICTED_COLOR}.get(via, INFERRED_COLOR)
        _put_outlined(img, dist_txt, (12, 58), via_color, 0.65, 2)
        if loc.get("landmarks_used"):
            used = ",".join(str(i) for i in loc["landmarks_used"])
            _put_outlined(img, f"aux objects: {used}", (12, 84), LANDMARK_COLOR, 0.55, 1)
    elif loc["source"] == "none":
        _put_outlined(img, "no target / no landmarks for distance", (12, 58), (180, 180, 180), 0.6, 2)

    return img


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
