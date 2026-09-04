"""End-to-end tests with synthetic zoom/motion video."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.pipeline import run_pipeline
from core.roi_selector import ROISelector
from core.types import ROI
from core.hybrid_tracker import HybridTracker


def make_synthetic_video(path: Path, n_frames: int = 40, w: int = 320, h: int = 240) -> ROI:
    """Create a video where a textured patch zooms and translates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 15.0, (w, h))
    rng = np.random.default_rng(42)
    patch = rng.integers(30, 220, size=(80, 80, 3), dtype=np.uint8)
    cv2.putText(patch, "ROI", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # Add unique markers
    for i in range(8):
        cv2.circle(patch, (10 + i * 8, 10 + (i % 3) * 20), 3, (0, 255, 255), -1)

    init_roi = None
    for i in range(n_frames):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        # Noise background for false features
        frame ^= rng.integers(0, 25, size=frame.shape, dtype=np.uint8)
        scale = 1.0 + 0.04 * i  # zoom in
        tx = 40 + i * 1.5
        ty = 30 + i * 0.8
        pw = int(80 * scale)
        ph = int(80 * scale)
        resized = cv2.resize(patch, (pw, ph))
        x1, y1 = int(tx), int(ty)
        x2, y2 = min(w, x1 + pw), min(h, y1 + ph)
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = resized[: y2 - y1, : x2 - x1]
        if i == 0:
            init_roi = ROISelector.from_rectangle(x1, y1, x1 + 80, y1 + 80, frame_index=0)
        writer.write(frame)
    writer.release()
    assert init_roi is not None
    return init_roi


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("vid")
    path = root / "synthetic_zoom.mp4"
    roi = make_synthetic_video(path)
    return path, roi


def test_end_to_end_manual(synthetic, tmp_path):
    video, roi = synthetic
    config = load_config()
    config["tracking"]["mode"] = "manual"
    config["yolo"]["enabled"] = False
    config["paths"]["output_dir"] = str(tmp_path / "out")
    config["paths"]["failed_frames_dir"] = str(tmp_path / "failed")
    config["video"]["prefer_gstreamer"] = False
    config["video"]["end_frame"] = 25
    result = run_pipeline(str(video), roi=roi, config=config)
    assert result["stats"]["processed_frames"] > 5
    assert result["stats"]["tracked"] >= 1
    assert Path(result["paths"]["csv"]).exists()


def test_scale_change_optical_flow(synthetic):
    video, roi = synthetic
    cap = cv2.VideoCapture(str(video))
    ok, frame0 = cap.read()
    assert ok
    tracker = HybridTracker(
        {
            "tracking": {
                "mode": "manual",
                "tracker_type": "optical_flow",
                "feature_detector": "ORB",
                "min_inliers": 8,
                "confidence_threshold": 0.3,
                "feature_match_interval": 3,
                "keyframe_interval": 8,
                "lost_frames_threshold": 20,
            },
            "yolo": {"enabled": False, "every_n_frames": 999},
            "smoothing": {"enabled": True, "method": "ema", "ema_alpha": 0.4, "max_center_jump_px": 120, "max_scale_jump": 0.8},
            "export": {"trail_length": 30},
        },
        yolo=None,
    )
    assert tracker.initialize(frame0, roi)
    scales = []
    for i in range(1, 20):
        ok, frame = cap.read()
        if not ok:
            break
        state = tracker.update(frame, frame_idx=i)
        scales.append(state.scale)
    cap.release()
    # Scale should generally increase due to zoom
    assert max(scales) > 1.05 or tracker.stats["tracked"] > 5


def test_yolo_disabled_graceful(synthetic, tmp_path):
    video, roi = synthetic
    config = load_config()
    config["tracking"]["mode"] = "hybrid"
    config["yolo"]["enabled"] = True
    config["yolo"]["model_path"] = str(tmp_path / "missing.engine")
    config["yolo"]["fallback_model_path"] = str(tmp_path / "missing.onnx")
    config["yolo"]["pytorch_model_path"] = str(tmp_path / "missing.pt")
    config["paths"]["output_dir"] = str(tmp_path / "out2")
    config["paths"]["failed_frames_dir"] = str(tmp_path / "failed2")
    config["video"]["prefer_gstreamer"] = False
    config["video"]["end_frame"] = 10
    # Should still run using manual trackers when YOLO missing
    result = run_pipeline(str(video), roi=roi, config=config)
    assert result["stats"]["processed_frames"] >= 5
