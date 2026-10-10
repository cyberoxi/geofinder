"""Per-frame geometric filtering of target / aux-object (landmark) detections.

Two checks, both switchable from the ``stabilizer`` config section:

* ``landmark_consistency`` — keep only aux objects whose relative positions and
  sizes agree with the scene layout (rejects a false aux object, and picks the
  right one when a class is detected more than once).
* ``geometry_check`` — when at least two consistent aux objects predict where the
  target must be, drop target detections that are too far from it or the wrong size.

Everything here is scale-invariant (works on any resized image), so the Jetson
detector can run it on each multi-scale pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from shared.landmarks import (
    TARGET_CLASS_ID,
    LocalizationResult,
    SceneLayout,
    TargetLocalizer,
    polygon_area_abs,
)

Rejected = Tuple[Any, str]  # (detection, short reason)


@dataclass
class FrameFilterResult:
    targets: List[Any]  # accepted target detections, best first
    landmarks: List[Any]  # aux-object detections kept (one per class)
    estimate: Optional[LocalizationResult]  # target predicted from the kept aux objects
    rejected: List[Rejected] = field(default_factory=list)

    @property
    def n_landmarks(self) -> int:
        return len(self.estimate.landmarks_used) if self.estimate is not None else 0


def polygon_center_size(poly: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = np.asarray(poly, np.float32).reshape(-1, 2)
    area = polygon_area_abs(pts) if len(pts) >= 3 else 0.0
    if area <= 1.0:
        span = pts.max(axis=0) - pts.min(axis=0)
        area = max(float(span[0] * span[1]), 1.0)
    return pts.mean(axis=0), math.sqrt(area)


def _scale_ok(s: float, ratio: float, tol: float) -> bool:
    return ratio / tol <= s <= ratio * tol


def consistent_landmarks(
    dets: Sequence[Any],
    layout: SceneLayout,
    frame_shape: Tuple[int, ...],
    min_conf: float,
    scale_tol: float = 1.8,
) -> Tuple[List[Any], List[Rejected]]:
    """
    Pick the largest set of aux-object detections (one per class) that fits the
    layout under one similarity transform.  Hypotheses come from every pair of
    detections of different classes; other classes vote as inliers.
    """
    n_cls = len(layout.classes)
    info = []
    for d in dets:
        cid = int(getattr(d, "class_id", -1))
        if cid == TARGET_CLASS_ID or cid < 0 or cid >= n_cls or float(d.confidence) < min_conf:
            continue
        c, s = TargetLocalizer._det_center_size(d)
        info.append((d, cid, c, s, s / max(layout.size(cid), 1e-6)))
    if not info:
        return [], []
    classes = {cid for _, cid, _, _, _ in info}
    if len(classes) < 2:
        best = max(info, key=lambda e: float(e[0].confidence))
        return [best[0]], [(e[0], "duplicate aux") for e in info if e is not best]

    tol_px = max(8.0, 0.08 * max(frame_shape[:2]))
    best_set: Optional[Dict[int, int]] = None
    best_key = (0, 0.0)
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            di, ci, pi, _, ri = info[i]
            dj, cj, pj, _, rj = info[j]
            if ci == cj:
                continue
            vs = layout.center(cj) - layout.center(ci)
            vd = pj - pi
            ns = float(np.linalg.norm(vs))
            if ns < 1e-6:
                continue
            s = float(np.linalg.norm(vd)) / ns
            if not (_scale_ok(s, ri, scale_tol) and _scale_ok(s, rj, scale_tol)):
                continue
            a = math.atan2(vd[1], vd[0]) - math.atan2(vs[1], vs[0])
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * s
            t = pi - R @ layout.center(ci)
            chosen = {ci: i, cj: j}
            for k, (dk, ck, pk, _, rk) in enumerate(info):
                if ck in (ci, cj):
                    continue
                proj = R @ layout.center(ck) + t
                dist = float(np.linalg.norm(pk - proj))
                if dist > max(tol_px, 0.5 * s * layout.size(ck)) or not _scale_ok(s, rk, scale_tol):
                    continue
                prev = chosen.get(ck)
                if prev is None or float(dk.confidence) > float(info[prev][0].confidence):
                    chosen[ck] = k
            key = (len(chosen), sum(float(info[k][0].confidence) for k in chosen.values()))
            if key > best_key:
                best_key, best_set = key, chosen
    if best_set is None:
        # No pair agrees on scale — trust only the strongest single aux object
        best = max(info, key=lambda e: float(e[0].confidence))
        return [best[0]], [(e[0], "aux scale mismatch") for e in info if e is not best]
    keep = set(best_set.values())
    kept = [info[k][0] for k in sorted(keep)]
    rejected = [(e[0], "aux off-layout") for k, e in enumerate(info) if k not in keep]
    return kept, rejected


def filter_frame(
    dets: Sequence[Any],
    localizer: Optional[TargetLocalizer],
    frame_shape: Tuple[int, ...],
    cfg: Dict[str, Any],
    min_conf: float,
) -> FrameFilterResult:
    """Apply the enabled per-frame checks; targets come back best-first."""
    targets = sorted(
        (d for d in dets if int(d.class_id) == TARGET_CLASS_ID), key=lambda d: -float(d.confidence)
    )
    others = [d for d in dets if int(d.class_id) != TARGET_CLASS_ID]
    rejected: List[Rejected] = []
    landmarks: List[Any] = []
    est = None
    if localizer is not None:
        if cfg.get("landmark_consistency", True):
            landmarks, rej = consistent_landmarks(
                others, localizer.layout, frame_shape, min_conf, localizer.max_scale_disagreement
            )
            rejected.extend(rej)
        else:
            landmarks = others
        est = localizer.estimate(landmarks, frame_shape, include_target=False)

    if cfg.get("geometry_check", True) and targets and est is not None and len(est.landmarks_used) >= 2:
        exp_c, exp_s = polygon_center_size(est.polygon)
        max_err = float(cfg.get("geometry_max_center_err", 0.8))
        max_ratio = max(1.01, float(cfg.get("geometry_max_size_ratio", 2.0)))
        scored = []
        for t in targets:
            c, s = TargetLocalizer._det_center_size(t)
            err = float(np.linalg.norm(c - exp_c)) / max(exp_s, 1e-6)
            ratio = s / max(exp_s, 1e-6)
            if err > max_err:
                rejected.append((t, f"off-geometry {err:.1f}x"))
            elif not (1.0 / max_ratio <= ratio <= max_ratio):
                rejected.append((t, f"size {ratio:.1f}x"))
            else:
                scored.append((float(t.confidence) - 0.5 * err, t))
        scored.sort(key=lambda e: -e[0])
        targets = [t for _, t in scored]
    return FrameFilterResult(targets=targets, landmarks=landmarks, estimate=est, rejected=rejected)
