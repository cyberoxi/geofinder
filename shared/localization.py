"""Target localization shared by the desktop Live Test and the Jetson runtime.

``stabilize_frame`` turns one frame of detections into the target decision (filters + temporal
stabilizer); ``draw_localization`` draws it: target / landmarks / rejected / pending, the image
centre, the arrow to the target (or to the frame edge when it is off-screen) and the centre distance.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from shared.landmarks import TargetLocalizer
from shared.schemas import Detection, ROI
from shared.target_filter import filter_frame
from shared.target_stabilizer import Candidate, TargetStabilizer

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


def det_polygon(d: Detection) -> np.ndarray:
    if d.polygon is not None and len(d.polygon) >= 3:
        return np.asarray(d.polygon, np.float32)
    return ROI.from_bbox(*d.bbox).points


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
    cands = [Candidate(det_polygon(d), float(d.confidence), "yolo") for d in flt.targets]
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


def put_outlined(img: np.ndarray, text: str, org: tuple, color, scale: float = 0.7, thickness: int = 2):
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
            put_outlined(img, f"X {name} {d.confidence:.2f} {rejected[id(d)]}", (x, max(14, y - 4)),
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
        put_outlined(img, f"confirming {loc.get('progress', '')}", (int(tc[0]) - 50, int(tc[1])),
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
    put_outlined(img, f"{label} {loc['confidence']:.2f}{state_txt}", (12, 28), (255, 255, 255), 0.75, 2)

    if offset is not None:
        dist = offset["distance_px"]
        dx, dy = offset["dx"], offset["dy"]
        via = {"landmarks": "landmarks", "predicted": "predicted"}.get(loc["source"], "direct")
        dist_txt = f"center dist {dist:.0f}px  dx={dx:+.0f} dy={dy:+.0f}  ({via})"
        via_color = {"direct": CENTER_COLOR, "predicted": PREDICTED_COLOR}.get(via, INFERRED_COLOR)
        put_outlined(img, dist_txt, (12, 58), via_color, 0.65, 2)
        if loc.get("landmarks_used"):
            used = ",".join(str(i) for i in loc["landmarks_used"])
            put_outlined(img, f"aux objects: {used}", (12, 84), LANDMARK_COLOR, 0.55, 1)
    elif loc["source"] == "none":
        put_outlined(img, "no target / no landmarks for distance", (12, 58), (180, 180, 180), 0.6, 2)

    return img
