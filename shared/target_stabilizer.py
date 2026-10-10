"""Temporal target stabilizer: N-of-M confirmation, hysteresis and a motion gate.

State machine (one call to :meth:`TargetStabilizer.update` per processed frame)::

    searching ─► confirming ─► locked ◄─► coasting ─► searching

* **Confirmation** — a position must be seen ``confirm_hits`` times within the last
  ``confirm_window`` frames (at confidence ≥ ``min_conf``) before it is locked.
* **Hysteresis** — once locked, detections down to ``min_conf × hold_conf_ratio``
  keep the lock alive.
* **Gate** — while locked, only candidates near the predicted position (constant
  velocity) are accepted; a far-away detection never moves the lock, however confident.
* **Coasting** — when the detector misses, the lock follows the prediction for up
  to ``coast_frames`` frames, then drops back to searching.

``tracking`` in the result is True only for locked / coasting — use it to gate
serial output so an unconfirmed detection never reaches the controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from shared.config import DEFAULT_RUNTIME
from shared.target_filter import polygon_center_size

SEARCHING, CONFIRMING, LOCKED, COASTING = "searching", "confirming", "locked", "coasting"
_MAX_PENDING = 8


def merge_stabilizer_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = dict(DEFAULT_RUNTIME["stabilizer"])
    for k, v in (cfg or {}).items():
        if k not in out or v is None:
            continue
        if isinstance(out[k], bool):
            out[k] = v.strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)
        else:
            out[k] = type(out[k])(v)
    out["confirm_hits"] = max(1, out["confirm_hits"])
    out["confirm_window"] = max(out["confirm_hits"], out["confirm_window"])
    out["hold_conf_ratio"] = min(1.0, max(0.1, out["hold_conf_ratio"]))
    return out


def detector_conf(min_conf: float, cfg: Dict[str, Any]) -> float:
    """Confidence to run the detector at, so hysteresis can see weaker hits."""
    if cfg.get("temporal", True):
        return max(0.01, min_conf * float(cfg.get("hold_conf_ratio", 1.0)))
    return min_conf


@dataclass
class Candidate:
    polygon: np.ndarray
    confidence: float
    source: str = "yolo"  # "yolo" | "landmarks"
    n_landmarks: int = 0
    score: Optional[float] = None  # ranking score; defaults to confidence


@dataclass
class StabilizedTarget:
    state: str
    polygon: Optional[np.ndarray]
    confidence: float
    source: str  # yolo | landmarks | predicted | none
    chosen: Optional[int]  # index of the accepted candidate this frame
    tracking: bool  # safe to drive serial / tracker
    progress: str = ""  # "2/3" while confirming
    reason: str = ""


@dataclass
class _Track:
    center: np.ndarray
    size: float
    polygon: np.ndarray
    confidence: float
    source: str
    last_idx: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    hits: Deque[bool] = field(default_factory=deque)
    matched: Optional[Tuple[int, Candidate]] = None

    def absorb(self, cand: Candidate, idx: int) -> None:
        c, s = polygon_center_size(cand.polygon)
        dt = max(1, idx - self.last_idx)
        v = (c - self.center) / dt
        lim = max(self.size, 1.0)
        n = float(np.linalg.norm(v))
        if n > lim:
            v = v * (lim / n)
        self.velocity = 0.6 * self.velocity + 0.4 * v
        self.center, self.size = c, s
        self.polygon = np.asarray(cand.polygon, np.float32)
        self.confidence, self.source, self.last_idx = float(cand.confidence), cand.source, idx


def _new_track(cand: Candidate, idx: int, window: int) -> _Track:
    c, s = polygon_center_size(cand.polygon)
    return _Track(c, s, np.asarray(cand.polygon, np.float32), float(cand.confidence), cand.source,
                  idx, hits=deque(maxlen=window))


def _rank(c: Candidate) -> Tuple[bool, float]:
    return (c.source != "landmarks", c.confidence if c.score is None else c.score)


class TargetStabilizer:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = merge_stabilizer_config(cfg)
        self.reset()

    def configure(self, cfg: Optional[Dict[str, Any]]) -> None:
        self.cfg = merge_stabilizer_config(cfg)
        self.reset()

    def reset(self) -> None:
        self._lock: Optional[_Track] = None
        self._pending: List[_Track] = []
        self._idx = -1

    @property
    def locked(self) -> bool:
        return self._lock is not None

    def _gate(self, size: float) -> float:
        return max(float(self.cfg["gate_min_px"]), float(self.cfg["gate_ratio"]) * size)

    def _advance(self, frame_idx: Optional[int]) -> Tuple[int, bool]:
        idx = self._idx + 1 if frame_idx is None else int(frame_idx)
        new = idx != self._idx
        self._idx = idx
        return idx, new

    def follow(self, polygon: np.ndarray, frame_idx: Optional[int] = None) -> None:
        """An external tracker moved the target (Jetson homography / flow): keep the lock on it."""
        if self._lock is None or polygon is None:
            return
        idx, _ = self._advance(frame_idx)
        self._lock.absorb(Candidate(polygon, self._lock.confidence, self._lock.source), idx)

    def update(
        self, candidates: List[Candidate], min_conf: float, frame_idx: Optional[int] = None
    ) -> StabilizedTarget:
        idx, new_frame = self._advance(frame_idx)
        cfg = self.cfg
        hold = min_conf * float(cfg["hold_conf_ratio"]) if cfg["temporal"] else min_conf
        need_lm = int(cfg["min_landmarks_acquire"]) if cfg["landmark_consistency"] else 1
        eligible: List[Tuple[int, Candidate, bool]] = []
        for i, c in enumerate(candidates):
            if c.source == "landmarks":
                eligible.append((i, c, c.n_landmarks >= need_lm))
            elif c.confidence >= hold:
                eligible.append((i, c, c.confidence >= min_conf))
        if not cfg["temporal"]:
            strong = [(i, c) for i, c, ok in eligible if ok]
            if not strong:
                return StabilizedTarget(SEARCHING, None, 0.0, "none", None, False)
            i, c = max(strong, key=lambda e: _rank(e[1]))
            return StabilizedTarget(LOCKED, c.polygon, c.confidence, c.source, i, True)
        if self._lock is not None:
            res = self._update_locked(eligible, idx, new_frame)
            if res is not None:
                return res
            new_frame = False  # pending already advanced for this frame
        return self._update_searching(eligible, idx, new_frame)

    def _update_locked(self, eligible, idx: int, new_frame: bool) -> Optional[StabilizedTarget]:
        L = self._lock
        assert L is not None
        dt = max(0, idx - L.last_idx)
        pred = L.center + L.velocity * dt
        R = self._gate(L.size)
        inside, outside = [], []
        for i, c, strong in eligible:
            d = float(np.linalg.norm(polygon_center_size(c.polygon)[0] - pred))
            if d <= R:
                inside.append((i, c, d))
            elif strong:
                outside.append((i, c, strong))
        # Far-away strong hits only build pending hypotheses (used if the lock is lost)
        self._feed_pending(outside, idx, new_frame)
        if inside:
            i, c, d = max(inside, key=lambda e: (e[1].source != "landmarks", e[1].confidence - 0.5 * e[2] / R))
            L.absorb(c, idx)
            return StabilizedTarget(LOCKED, c.polygon, c.confidence, c.source, i, True)
        if dt <= int(self.cfg["coast_frames"]):
            shift = (L.velocity * dt).astype(np.float32)
            reason = f"gated {len(outside)} far hit(s)" if outside else "detector miss"
            return StabilizedTarget(COASTING, L.polygon + shift, L.confidence, "predicted", None, True,
                                    reason=reason)
        self._lock = None
        return None

    def _feed_pending(self, strong, idx: int, new_frame: bool) -> None:
        win = int(self.cfg["confirm_window"])
        for h in self._pending:
            h.matched = None
        for i, c, _ in sorted(strong, key=lambda e: _rank(e[1]), reverse=True):
            cc = polygon_center_size(c.polygon)[0]
            best, best_d = None, None
            for h in self._pending:
                d = float(np.linalg.norm(cc - (h.center + h.velocity * max(0, idx - h.last_idx))))
                if d <= self._gate(h.size) and (best_d is None or d < best_d):
                    best, best_d = h, d
            if best is None:
                best = _new_track(c, idx, win)
                self._pending.append(best)
            elif best.matched is not None:
                continue  # duplicate of a better candidate at the same place (multi-scale / aux)
            else:
                best.absorb(c, idx)
            best.matched = (i, c)
        for h in self._pending:
            hit = h.matched is not None
            if new_frame or not h.hits:
                h.hits.append(hit)
            elif hit:
                h.hits[-1] = True
        self._pending = [h for h in self._pending if any(h.hits)]
        if len(self._pending) > _MAX_PENDING:
            self._pending.sort(key=lambda h: (sum(h.hits), h.confidence), reverse=True)
            del self._pending[_MAX_PENDING:]

    def _update_searching(self, eligible, idx: int, new_frame: bool) -> StabilizedTarget:
        self._feed_pending([e for e in eligible if e[2]], idx, new_frame)
        need = int(self.cfg["confirm_hits"])
        if not self._pending:
            return StabilizedTarget(SEARCHING, None, 0.0, "none", None, False)
        best = max(self._pending, key=lambda h: (sum(h.hits), h.matched is not None, h.confidence))
        n = sum(best.hits)
        if n >= need and best.matched is not None:
            i, c = best.matched
            best.hits.clear()
            self._lock, self._pending = best, []
            return StabilizedTarget(LOCKED, c.polygon, c.confidence, c.source, i, True,
                                    progress=f"{n}/{need}", reason="confirmed")
        return StabilizedTarget(CONFIRMING, best.polygon, best.confidence, best.source, None, False,
                                progress=f"{n}/{need}")
