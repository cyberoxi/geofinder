"""Landmark-based target localization + synthetic label geometry."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desktop_app.satellite.synth import MosaicCamera, Pose, frame_labels, pan_path, zoom_path  # noqa: E402
from shared.landmarks import SceneLayout, TargetLocalizer, apply_affine, polygon_iou  # noqa: E402
from shared.schemas import Detection  # noqa: E402


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


def _similarity(scale: float, angle_deg: float, tx: float, ty: float) -> np.ndarray:
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * scale
    return np.hstack([R, np.array([[tx], [ty]])])


def _det(layout: SceneLayout, cid: int, M: np.ndarray, conf: float = 0.9) -> Detection:
    poly = apply_affine(layout.polygon(cid), M)
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    return Detection(bbox=(x0, y0, x1 - x0, y1 - y0), confidence=conf, class_id=cid, polygon=poly)


def test_localizer_recovers_target_from_two_landmarks():
    layout = _layout()
    M = _similarity(0.5, 35.0, 300.0, -100.0)
    loc = TargetLocalizer(layout)
    est = loc.estimate([_det(layout, 1, M), _det(layout, 2, M)], (1080, 1920, 3))
    assert est is not None
    gt = apply_affine(layout.polygon(0), M)
    assert polygon_iou(est.polygon, gt, 1920, 1080) > 0.9
    assert abs(est.angle_deg - 35.0) < 1.0
    assert abs(est.scale - 0.5) < 0.02


def test_localizer_target_off_screen_reports_not_in_view():
    layout = _layout()
    # Landmark 2 visible, target shifted far left of the frame
    M = _similarity(1.0, 0.0, -1200.0, -500.0)  # target x in [-300,-100]
    est = TargetLocalizer(layout).estimate([_det(layout, 2, M), _det(layout, 3, M)], (720, 1280, 3))
    assert est is not None
    assert not est.in_view


def test_localizer_single_landmark_uses_size_for_scale():
    layout = _layout()
    M = _similarity(0.8, 0.0, 50.0, 20.0)
    est = TargetLocalizer(layout).estimate([_det(layout, 1, M)], (1080, 1920, 3))
    assert est is not None
    gt = apply_affine(layout.polygon(0), M)
    assert polygon_iou(est.polygon, gt, 1920, 1080) > 0.7


def test_localizer_ignores_target_only_and_low_confidence():
    layout = _layout()
    M = _similarity(1.0, 0.0, 0.0, 0.0)
    loc = TargetLocalizer(layout, min_confidence=0.3)
    assert loc.estimate([_det(layout, 0, M)], (2000, 2000, 3)) is None
    assert loc.estimate([_det(layout, 1, M, conf=0.1)], (2000, 2000, 3)) is None


def test_ransac_rejects_wrong_landmark():
    layout = _layout()
    M = _similarity(0.6, -20.0, 200.0, 100.0)
    bad = _det(layout, 3, _similarity(0.6, -20.0, 900.0, 600.0))  # misplaced detection
    est = TargetLocalizer(layout).estimate(
        [_det(layout, 1, M), _det(layout, 2, M), bad, _det(layout, 0, M, conf=0.8)], (1080, 1920, 3)
    )
    gt = apply_affine(layout.polygon(0), M)
    assert est is not None and polygon_iou(est.polygon, gt, 1920, 1080) > 0.85


def test_synthetic_labels_match_rendered_geometry():
    layout = _layout()
    mosaic = np.zeros((2000, 2000, 3), np.uint8)
    for cid in range(len(layout.classes)):
        x0, y0 = layout.polygon(cid).min(axis=0).astype(int)
        x1, y1 = layout.polygon(cid).max(axis=0).astype(int)
        mosaic[y0:y1, x0:x1] = 255
    cam = MosaicCamera(mosaic, (640, 360))
    img, M = cam.render(Pose(1000, 1000, 600, 30.0))
    labels = frame_labels(layout, M, 640, 360)
    target = [np.asarray(p) for c, p in labels if c == 0]
    assert target, "target must be labeled when centred"
    mask = np.zeros((360, 640), np.uint8)
    import cv2

    cv2.fillPoly(mask, [np.round(target[0]).astype(np.int32)], 1)
    inside = img[..., 0][mask > 0]
    assert inside.mean() > 200  # the labeled region really is the bright target


def test_camera_paths_stay_inside_mosaic():
    mosaic = np.zeros((1500, 2500, 3), np.uint8)
    cam = MosaicCamera(mosaic, (640, 360))
    rng = random.Random(1)
    for poses in (
        zoom_path(cam, (1250, 750), 100, 40, rng, angle=47, start_offset=0.2, end_jitter=0.2),
        pan_path(cam, (1250, 750), 100, 40, rng, angle=120, view_factor=6, length_factor=4),
    ):
        for p in poses:
            assert p.view_h >= 16
            assert p.view_h <= cam.max_view_h(p.cx, p.cy, p.angle) + 1e-6


def test_clip_concave_polygon():
    from shared.landmarks import clip_polygon_to_frame, polygon_area_abs

    # U-shape, partly outside on the right
    u = np.array([[10, 10], [150, 10], [150, 90], [110, 90], [110, 40], [50, 40], [50, 90], [10, 90]], np.float32)
    clipped = clip_polygon_to_frame(u, 131, 200)  # x_max = 130
    import cv2

    m = np.zeros((200, 131), np.uint8)
    cv2.fillPoly(m, [np.round(clipped).astype(np.int32)], 1)
    full = np.zeros((200, 200), np.uint8)
    cv2.fillPoly(full, [u.astype(np.int32)], 1)
    expect = full[:, :131]
    inter = np.logical_and(m, expect).sum() / np.logical_or(m, expect).sum()
    assert inter > 0.95
    assert len(clip_polygon_to_frame(u + 500, 100, 100)) == 0


def test_render_scene_videos_json_with_numpy_values(tmp_path):
    """Regression: numpy float32 values (from editor / mosaic geometry) must not break label JSON."""
    import json

    from desktop_app.satellite import render_scene_videos

    rng = np.random.default_rng(0)
    mosaic = rng.integers(0, 255, (800, 1000, 3), dtype=np.uint8)
    layout = SceneLayout(
        classes=["target_region", "landmark_1_building"],
        polygons=[
            np.float32([[450, 350], [550, 350], [550, 450], [450, 450]]).tolist(),
            [[np.float32(200), np.float32(200)], [np.float32(280), np.float32(200)], [np.float32(240), np.float32(280)]],
        ],
        mosaic_size=(1000, 800),
        meters_per_px=np.float32(0.9),
    )
    res = render_scene_videos(mosaic, layout, tmp_path, n_videos=1, duration_s=0.5, fps=4, out_size=(320, 180))
    data = json.loads(res["videos"][0]["labels"].read_text(encoding="utf-8"))
    assert len(data["frames"]) == 2 and all(isinstance(s, float) for s in data["scale"])
    layout.save(tmp_path / "scene.json")  # must not raise


def test_align_from_points_and_coverage():
    import cv2

    from desktop_app.satellite.align import align_from_points, covered_classes, warp_to_mosaic

    layout = _layout()
    mosaic = np.zeros((2000, 2000, 3), np.uint8)
    # "other satellite" image = left half of the mosaic, scaled by 0.5
    img = np.full((1000, 500, 3), 200, np.uint8)
    src = [[0, 0], [499, 0], [499, 999], [0, 999]]
    dst = [[0, 0], [998, 0], [998, 1998], [0, 1998]]
    al = align_from_points(src, dst)
    assert al.reproj_error < 1e-3
    warped, mask = warp_to_mosaic(img, al, mosaic.shape)
    assert warped.shape == mosaic.shape and 0.45 < (mask > 0).mean() < 0.52
    keep = covered_classes(layout, mask)
    assert 1 in keep  # landmark_1 at x 400..600 → mostly inside the left half
    assert 2 not in keep  # landmark_2 at x 1400..1700 → outside
