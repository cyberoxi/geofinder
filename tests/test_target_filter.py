"""False-target filters: aux consistency, geometry check, temporal stabilizer."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.landmarks import SceneLayout, TargetLocalizer, apply_affine  # noqa: E402
from shared.schemas import Detection  # noqa: E402
from shared.target_filter import consistent_landmarks, filter_frame  # noqa: E402
from shared.target_stabilizer import (  # noqa: E402
    COASTING,
    CONFIRMING,
    LOCKED,
    SEARCHING,
    Candidate,
    TargetStabilizer,
    merge_stabilizer_config,
)

FRAME = (1080, 1920, 3)


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _layout() -> SceneLayout:
    return SceneLayout(
        classes=["target_region", "landmark_1", "landmark_2", "landmark_3"],
        polygons=[
            _rect(900, 900, 1100, 1100),
            _rect(400, 500, 600, 700),
            _rect(1400, 800, 1700, 1100),
            _rect(1000, 1500, 1200, 1700),
        ],
        mosaic_size=(2000, 2000),
    )


def _M(scale=0.5, angle=20.0, tx=400.0, ty=-50.0):
    a = math.radians(angle)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * scale
    return np.hstack([R, np.array([[tx], [ty]])])


def _det_poly(poly, cid, conf=0.9) -> Detection:
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    return Detection(bbox=(x0, y0, x1 - x0, y1 - y0), confidence=conf, class_id=cid, polygon=poly)


def _det(layout, cid, M, conf=0.9, shift=(0.0, 0.0)) -> Detection:
    poly = apply_affine(layout.polygon(cid), M) + np.asarray(shift, np.float32)
    return _det_poly(poly.astype(np.float32), cid, conf)


def _box(cx, cy, size=100.0):
    h = size / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]], np.float32)


def test_false_aux_object_is_dropped():
    layout, M = _layout(), _M()
    good = [_det(layout, c, M) for c in (1, 2, 3)]
    fake = _det(layout, 2, M, conf=0.95, shift=(-500, 300))  # second "landmark_2" in the wrong place
    kept, rejected = consistent_landmarks(good + [fake], layout, FRAME, 0.3)
    assert fake not in kept and len(kept) == 3
    assert any(d is fake for d, _ in rejected)


def test_geometry_check_rejects_far_target():
    layout, M = _layout(), _M()
    loc = TargetLocalizer(layout)
    true_t = _det(layout, 0, M, conf=0.7)
    false_t = _det(layout, 0, M, conf=0.95, shift=(600, 0))
    dets = [true_t, false_t] + [_det(layout, c, M) for c in (1, 2, 3)]
    cfg = merge_stabilizer_config({})
    res = filter_frame(dets, loc, FRAME, cfg, 0.5)
    assert res.targets == [true_t]
    assert any(d is false_t for d, _ in res.rejected)
    # Disabled → both pass, highest confidence first (old behaviour)
    res_off = filter_frame(dets, loc, FRAME, {**cfg, "geometry_check": False}, 0.5)
    assert res_off.targets[0] is false_t


def test_needs_n_of_m_before_lock():
    st = TargetStabilizer({"confirm_hits": 3, "confirm_window": 5})
    for i in range(2):
        r = st.update([Candidate(_box(500, 500), 0.9)], 0.5)
        assert r.state == CONFIRMING and not r.tracking
    r = st.update([Candidate(_box(505, 500), 0.9)], 0.5)
    assert r.state == LOCKED and r.tracking and r.chosen == 0


def test_single_frame_flicker_never_locks():
    st = TargetStabilizer({"confirm_hits": 3, "confirm_window": 5})
    for i in range(20):
        cands = [Candidate(_box(200 + 300 * (i % 5), 400), 0.95)] if i % 2 == 0 else []
        assert not st.update(cands, 0.5).tracking


def _locked(cfg=None) -> TargetStabilizer:
    st = TargetStabilizer({"confirm_hits": 2, "confirm_window": 3, **(cfg or {})})
    st.update([Candidate(_box(500, 500), 0.9)], 0.5)
    assert st.update([Candidate(_box(500, 500), 0.9)], 0.5).state == LOCKED
    return st


def test_gate_ignores_far_jump_and_coasts():
    st = _locked({"coast_frames": 3})
    r = st.update([Candidate(_box(1500, 500), 0.99)], 0.5)  # confident but far
    assert r.state == COASTING and r.chosen is None and r.tracking
    # Near detection wins over the far one
    r = st.update([Candidate(_box(1500, 500), 0.99), Candidate(_box(510, 505), 0.6)], 0.5)
    assert r.state == LOCKED and r.chosen == 1


def test_lock_expires_after_coast_then_relocks_elsewhere():
    st = _locked({"coast_frames": 2})
    states = [st.update([Candidate(_box(1500, 500), 0.9)], 0.5).state for _ in range(4)]
    assert states[:2] == [COASTING, COASTING]
    assert LOCKED in states[2:]  # far target persisted → confirmed as the new lock


def test_hysteresis_keeps_weaker_hits():
    st = _locked({"hold_conf_ratio": 0.6})
    assert st.update([Candidate(_box(505, 500), 0.35)], 0.5).state == LOCKED  # 0.35 ≥ 0.5×0.6
    st2 = TargetStabilizer({"confirm_hits": 1})
    assert st2.update([Candidate(_box(505, 500), 0.35)], 0.5).state == SEARCHING  # can't acquire


def test_single_aux_object_cannot_acquire():
    st = TargetStabilizer({"temporal": False})
    r = st.update([Candidate(_box(500, 500), 0.5, "landmarks", n_landmarks=1)], 0.5)
    assert not r.tracking
    r = st.update([Candidate(_box(500, 500), 0.5, "landmarks", n_landmarks=2)], 0.5)
    assert r.tracking
    st_off = TargetStabilizer({"temporal": False, "landmark_consistency": False})
    assert st_off.update([Candidate(_box(500, 500), 0.5, "landmarks", n_landmarks=1)], 0.5).tracking


def test_same_frame_duplicates_count_once():
    st = TargetStabilizer({"confirm_hits": 3, "confirm_window": 5})
    for _ in range(2):  # multi-scale passes of one frame
        r = st.update([Candidate(_box(500, 500), 0.9), Candidate(_box(503, 500), 0.8)], 0.5, frame_idx=7)
    assert r.state == CONFIRMING and r.progress == "1/3"


def test_config_merge_parses_strings():
    cfg = merge_stabilizer_config({"temporal": "false", "confirm_hits": "4", "confirm_window": 2, "junk": 1})
    assert cfg["temporal"] is False and cfg["confirm_hits"] == 4 and cfg["confirm_window"] == 4
    assert "junk" not in cfg
