"""Comprehensive tests for dual-app architecture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop_app.annotation.interpolate import interpolate_between_keyframes
from desktop_app.dataset.builder import DatasetBuilder, _split_videos
from desktop_app.packaging.builder import PackageBuilder
from desktop_app.project_manager.manager import ProjectManager
from shared.geometry import ROISelector, clamp_polygon, is_valid_roi, polygon_to_bbox
from shared.package_utils import validate_package
from shared.schemas import Annotation, AnnotationSource
from jetson_runtime.trackers.homography_tracker import HomographyTracker
from jetson_runtime.detectors.confidence_manager import ConfidenceManager
from jetson_runtime.trackers.kalman_filter import GeometrySmoother
from jetson_runtime.exporters.result_exporter import ResultExporter
from shared.schemas import Detection, FrameResult, ROI
from shared.features import FeatureExtractor


def _make_video(path: Path, n=30, w=320, h=240):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    rng = np.random.default_rng(0)
    patch = rng.integers(40, 200, (60, 60, 3), dtype=np.uint8)
    for i in range(8):
        cv2.circle(patch, (8 + i * 6, 20), 3, (0, 255, 255), -1)
    for i in range(n):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        scale = 1.0 + 0.03 * i
        pw = int(60 * scale)
        ph = int(60 * scale)
        resized = cv2.resize(patch, (pw, ph))
        x, y = 40 + i, 30 + i // 2
        frame[y : y + ph, x : x + pw] = resized[: min(ph, h - y), : min(pw, w - x)]
        writer.write(frame)
    writer.release()


@pytest.fixture
def project(tmp_path):
    pm = ProjectManager.create(tmp_path / "proj", "test_proj")
    v1 = tmp_path / "a.mp4"
    v2 = tmp_path / "b.mp4"
    _make_video(v1, 25)
    _make_video(v2, 20)
    a1 = pm.add_video(v1)
    a2 = pm.add_video(v2)
    # annotations on video 1 at frames 0 and 10
    for fid, zoom in [(0, 1.0), (10, 1.3)]:
        roi = ROISelector.from_rectangle(40, 30, 100, 90, fid)
        ann = Annotation(
            video_id=a1.video_id,
            frame_id=fid,
            timestamp=fid / 15.0,
            image_width=320,
            image_height=240,
            roi_type="rectangle",
            polygon_points=roi.to_list(),
            bounding_box=list(roi.bbox),
            annotation_source=AnnotationSource.MANUAL.value,
            zoom_level=zoom,
            class_name="target_region",
        )
        pm.upsert_annotation(ann)
    # one annotation on video 2
    roi = ROISelector.from_rectangle(40, 30, 100, 90, 0)
    pm.upsert_annotation(
        Annotation(
            video_id=a2.video_id,
            frame_id=0,
            timestamp=0,
            image_width=320,
            image_height=240,
            roi_type="rectangle",
            polygon_points=roi.to_list(),
            bounding_box=list(roi.bbox),
            annotation_source=AnnotationSource.MANUAL.value,
            class_name="target_region",
        )
    )
    return pm


def test_polygon_to_bbox():
    x, y, w, h = polygon_to_bbox(np.array([[10, 10], [30, 10], [30, 40], [10, 40]], np.float32))
    assert (x, y, w, h) == (10, 10, 20, 30)


def test_project_save_load(tmp_path):
    pm = ProjectManager.create(tmp_path / "p", "n")
    v = tmp_path / "v.mp4"
    _make_video(v, 5)
    pm.add_video(v)
    pm.close()
    pm2 = ProjectManager.open(tmp_path / "p")
    assert pm2.meta.project_name == "n"
    assert len(pm2.meta.videos) == 1
    pm2.close()


def test_multi_frame_annotation_and_interpolate(project):
    anns = project.list_annotations(project.meta.videos[0].video_id, include_interpolated=False)
    assert len(anns) >= 2
    generated = interpolate_between_keyframes(anns)
    assert len(generated) >= 5
    assert all(g.annotation_source == "interpolated" for g in generated)


def test_dataset_and_data_yaml(project):
    out = DatasetBuilder(project, model_type="segmentation", every_n=1, augment_per_image=0).build("ds")
    assert (out / "data.yaml").exists()
    data = (out / "data.yaml").read_text()
    assert "train" in data and "names" in data
    # video-based split mapping exists
    split = json.loads((out / "split_by_video.json").read_text())
    assert "split_map" in split


def test_split_by_video_no_leak():
    m = _split_videos(["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"])
    assert set(m.values()) <= {"train", "val", "test"}
    assert len(m) == 10


def test_package_and_checksum(project):
    for a in project.list_annotations(project.meta.videos[0].video_id, include_interpolated=False):
        project.upsert_annotation(a)
    zip_path = PackageBuilder(project).build()
    assert zip_path.exists()
    result = validate_package(zip_path)
    assert result["ok"] is True
    assert "manifest" in result


def test_homography_synthetic():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for i in range(30):
        cv2.rectangle(frame, (i * 8, i * 5), (i * 8 + 20, i * 5 + 20), (255, 255, 255), -1)
    roi = ROI.from_bbox(40, 40, 80, 80)
    ht = HomographyTracker(FeatureExtractor("ORB", 1000), min_inliers=6)
    assert ht.initialize(frame, roi)
    r = ht.estimate(frame, prev_roi=roi)
    assert r.success


def test_reject_bad_homography():
    a = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    b = np.random.default_rng(99).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    ht = HomographyTracker(FeatureExtractor("AKAZE", 400), min_inliers=15)
    ht.initialize(a, ROI.from_bbox(20, 20, 40, 40))
    r = ht.estimate(b, use_full_frame=True)
    assert r.success is False or r.inliers < 15


def test_scale_change_and_smoother():
    sm = GeometrySmoother(max_center_jump_px=30)
    assert sm.filter(ROI.from_bbox(0, 0, 40, 40)) is not None
    assert sm.filter(ROI.from_bbox(200, 200, 40, 40)) is None


def test_select_detection():
    mgr = ConfidenceManager()
    pred = ROI.from_bbox(100, 100, 50, 50)
    dets = [
        Detection((10, 10, 20, 20), 0.9),
        Detection((105, 105, 48, 48), 0.7),
    ]
    best = mgr.select_detection(dets, pred, 0.45, iou_threshold=0.1)
    assert best is not None and best.confidence == pytest.approx(0.7)


def test_export_csv_json(tmp_path):
    exp = ResultExporter(str(tmp_path / "o"), str(tmp_path / "f"), {"export": {"save_csv": True, "save_json": True, "save_center_path": True}})
    exp.add_result(
        FrameResult(0, 0, 1, 2, 0, 0, 10, 10, [[0, 0], [10, 0], [10, 10], [0, 10]], 0.9, "Homography", 10, 1.0, 1.0, "TRACKING")
    )
    paths = exp.finalize({}, {"tracked": 1})
    assert Path(paths["csv"]).exists() and Path(paths["json"]).exists()


def test_clamp_out_of_frame():
    pts = np.array([[-10, -5], [1000, 5], [1000, 900], [-10, 900]], np.float32)
    c = clamp_polygon(pts, 320, 240)
    assert c[:, 0].min() >= 0 and c[:, 1].max() <= 239


def test_corrupt_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    from shared.video_reader import VideoReader

    with pytest.raises(Exception):
        VideoReader(str(bad), prefer_gstreamer=False)


def test_runtime_without_engine(project, tmp_path):
    zip_path = PackageBuilder(project).build()
    from jetson_runtime.runtime.pipeline import run_runtime

    video = project.resolve_video(project.meta.videos[0].video_id)
    result = run_runtime(str(zip_path), str(video), output_dir=str(tmp_path / "out"), end_frame=12, prefer_gstreamer=False)
    assert result["stats"]["processed_frames"] >= 5
    assert Path(result["paths"]["csv"]).exists()


def test_valid_roi_checks():
    assert not is_valid_roi([[0, 0], [1, 1]])
    assert is_valid_roi([[0, 0], [50, 0], [50, 50], [0, 50]])
