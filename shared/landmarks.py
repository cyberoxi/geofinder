"""Scene layout (target + surrounding landmarks) and landmark-based target localization.

A *scene layout* stores every labelled object (class 0 = target region,
classes 1..K = landmarks) as a polygon in the satellite mosaic pixel frame.
Because the relative geometry is fixed on the ground, any camera view of the
scene is (approximately) a similarity transform of the mosaic.  When YOLO sees
one or more landmarks, :class:`TargetLocalizer` estimates that transform and
projects the target polygon into the frame — even when the target itself is
too small, occluded or outside the field of view.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

TARGET_CLASS_ID = 0


def polygon_area_abs(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    return float(abs(cv2.contourArea(pts)))


def clip_polygon_to_frame(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Intersect a convex polygon with the frame rectangle (returns (N,2) float32, maybe empty)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return np.zeros((0, 2), np.float32)
    frame = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    # intersectConvexConvex requires consistent orientation
    if cv2.contourArea(pts, oriented=True) < 0:
        pts = pts[::-1].copy()
    area, inter = cv2.intersectConvexConvex(pts, frame)
    if area <= 0 or inter is None:
        return np.zeros((0, 2), np.float32)
    out = inter.reshape(-1, 2).astype(np.float32)
    out[:, 0] = np.clip(out[:, 0], 0, width - 1)
    out[:, 1] = np.clip(out[:, 1], 0, height - 1)
    return out


def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)


@dataclass
class SceneLayout:
    """All objects of a scene in mosaic pixel coordinates."""

    classes: List[str]
    polygons: List[List[List[float]]]  # index == class id
    mosaic_size: Tuple[int, int] = (0, 0)  # (w, h)
    meters_per_px: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_landmarks(self) -> int:
        return max(0, len(self.classes) - 1)

    def polygon(self, class_id: int) -> np.ndarray:
        return np.asarray(self.polygons[class_id], dtype=np.float32).reshape(-1, 2)

    def center(self, class_id: int) -> np.ndarray:
        return self.polygon(class_id).mean(axis=0)

    def size(self, class_id: int) -> float:
        """Characteristic size = sqrt(area) in mosaic px."""
        return math.sqrt(max(polygon_area_abs(self.polygon(class_id)), 1e-6))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "classes": list(self.classes),
            "polygons": [[[float(x), float(y)] for x, y in poly] for poly in self.polygons],
            "mosaic_size": [int(self.mosaic_size[0]), int(self.mosaic_size[1])],
            "meters_per_px": float(self.meters_per_px),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneLayout":
        ms = d.get("mosaic_size") or (0, 0)
        return cls(
            classes=list(d["classes"]),
            polygons=[list(map(list, p)) for p in d["polygons"]],
            mosaic_size=(int(ms[0]), int(ms[1])),
            meters_per_px=float(d.get("meters_per_px", 0.0)),
            meta=dict(d.get("meta") or {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SceneLayout":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class LocalizationResult:
    polygon: np.ndarray  # target polygon in frame px
    confidence: float
    landmarks_used: List[int]
    transform: np.ndarray  # 2x3 mosaic → frame
    scale: float  # frame px per mosaic px
    angle_deg: float
    in_view: bool  # target polygon intersects the frame


class TargetLocalizer:
    """Estimate the target polygon from landmark detections using the scene layout."""

    def __init__(
        self,
        layout: SceneLayout,
        min_confidence: float = 0.30,
        max_scale_disagreement: float = 1.8,
    ):
        self.layout = layout
        self.min_confidence = min_confidence
        self.max_scale_disagreement = max_scale_disagreement
        self._last_angle = 0.0  # radians; used when only one landmark is visible

    @staticmethod
    def _det_center_size(det: Any) -> Tuple[np.ndarray, float]:
        poly = getattr(det, "polygon", None)
        if poly is not None and len(poly) >= 3:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            area = polygon_area_abs(pts)
            m = cv2.moments(pts)
            if m["m00"] > 1e-6:
                c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], np.float32)
            else:
                c = pts.mean(axis=0)
            if area > 1.0:
                return c, math.sqrt(area)
        x, y, w, h = det.bbox
        return np.array([x + w / 2.0, y + h / 2.0], np.float32), math.sqrt(max(w * h, 1.0))

    def _best_per_class(self, detections: Sequence[Any]) -> Dict[int, Any]:
        best: Dict[int, Any] = {}
        n_cls = len(self.layout.classes)
        for d in detections:
            cid = int(getattr(d, "class_id", -1))
            if cid < 0 or cid >= n_cls or float(d.confidence) < self.min_confidence:
                continue
            if cid not in best or float(d.confidence) > float(best[cid].confidence):
                best[cid] = d
        return best

    def estimate(
        self,
        detections: Sequence[Any],
        frame_shape: Tuple[int, ...],
        include_target: bool = True,
    ) -> Optional[LocalizationResult]:
        """
        Return the projected target polygon, or None when no usable landmark is seen.

        ``include_target``: also use a target detection as a correspondence
        (improves the transform when target + landmarks are visible).
        """
        best = self._best_per_class(detections)
        landmark_ids = [c for c in best if c != TARGET_CLASS_ID]
        if not landmark_ids:
            return None
        ids = list(landmark_ids)
        if include_target and TARGET_CLASS_ID in best:
            ids.append(TARGET_CLASS_ID)

        src = np.array([self.layout.center(c) for c in ids], np.float32)
        dst_sizes = []
        dst = []
        for c in ids:
            center, size = self._det_center_size(best[c])
            dst.append(center)
            dst_sizes.append(size)
        dst = np.asarray(dst, np.float32)
        ratios = np.array([dst_sizes[i] / max(self.layout.size(c), 1e-6) for i, c in enumerate(ids)])
        confs = np.array([float(best[c].confidence) for c in ids])

        M: Optional[np.ndarray] = None
        used = ids
        if len(ids) >= 2:
            if len(ids) >= 3:
                M, inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=max(8.0, 0.08 * max(frame_shape[:2]))
                )
                if M is not None and inliers is not None:
                    mask = inliers.ravel().astype(bool)
                    if mask.sum() >= 2:
                        used = [c for c, m in zip(ids, mask) if m]
            else:
                M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            if M is not None:
                s = math.hypot(M[0, 0], M[1, 0])
                # Sanity: transform scale must agree with observed object sizes
                med_ratio = float(np.median(ratios))
                if not (med_ratio / self.max_scale_disagreement <= s <= med_ratio * self.max_scale_disagreement):
                    M = None
                    used = ids
        if M is None:
            # Single strongest landmark: scale from its size, rotation from history
            order = np.argsort(-confs)
            k = int(next((i for i in order if ids[i] != TARGET_CLASS_ID), order[0]))
            s = float(ratios[k])
            a = self._last_angle
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * s
            t = dst[k] - R @ src[k]
            M = np.hstack([R, t.reshape(2, 1)]).astype(np.float64)
            used = [ids[k]]
            conf = float(confs[k]) * 0.6
        else:
            idx = [ids.index(c) for c in used]
            n_lm = sum(1 for c in used if c != TARGET_CLASS_ID)
            conf = float(np.mean(confs[idx])) * min(1.0, 0.55 + 0.2 * n_lm)
            self._last_angle = math.atan2(M[1, 0], M[0, 0])

        target = apply_affine(self.layout.polygon(TARGET_CLASS_ID), M)
        h, w = frame_shape[:2]
        in_view = len(clip_polygon_to_frame(target, w, h)) >= 3
        scale = math.hypot(M[0, 0], M[1, 0])
        return LocalizationResult(
            polygon=target,
            confidence=float(max(0.0, min(1.0, conf))),
            landmarks_used=[c for c in used if c != TARGET_CLASS_ID],
            transform=M,
            scale=scale,
            angle_deg=math.degrees(math.atan2(M[1, 0], M[0, 0])),
            in_view=in_view,
        )


def polygon_iou(a: np.ndarray, b: np.ndarray, width: int, height: int) -> float:
    """Mask IoU of two polygons, clipped to the frame."""
    ma = np.zeros((height, width), np.uint8)
    mb = np.zeros((height, width), np.uint8)
    pa = np.round(np.asarray(a, np.float32).reshape(-1, 2)).astype(np.int32)
    pb = np.round(np.asarray(b, np.float32).reshape(-1, 2)).astype(np.int32)
    if len(pa) >= 3:
        cv2.fillPoly(ma, [pa], 1)
    if len(pb) >= 3:
        cv2.fillPoly(mb, [pb], 1)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return inter / union if union > 0 else 0.0
