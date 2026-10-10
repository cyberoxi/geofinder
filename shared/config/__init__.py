"""Config helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)


DEFAULT_RUNTIME = {
    "tracking": {
        "mode": "hybrid",
        "tracker_type": "optical_flow",
        "feature_detector": "AKAZE",
        "max_features": 2000,
        "keyframe_interval": 15,
        "feature_match_interval": 5,
        "yolo_every_n_frames": 30,
        "lowe_ratio": 0.75,
        "min_inliers": 12,
        "max_reprojection_error": 4.0,
        "min_scale": 0.15,
        "max_scale": 8.0,
        "max_area_change": 4.0,
        "max_scale_jump": 0.35,
        "lost_frames_threshold": 15,
        "confidence_threshold": 0.45,
        "low_confidence_threshold": 0.30,
        "min_iou": 0.20,
        "adaptive_skip": True,
        "target_fps": 15.0,
    },
    "yolo": {
        "mode": "segmentation",
        "imgsz": 640,
        "device": "cuda",
        "use_fp16": True,
        "warmup_iterations": 5,
        "multi_scale_factors": [0.75, 1.0, 1.25],
    },
    "smoothing": {"enabled": True, "method": "ema", "ema_alpha": 0.35, "max_center_jump_px": 80.0},
    "export": {
        "save_annotated_video": True,
        "save_csv": True,
        "save_json": True,
        "save_failed_frames": True,
        "trail_length": 60,
        "video_codec": "mp4v",
    },
    # Jetson UART: pixel offset from image centre to target centre (+x right, +y down)
    "serial": {
        "enabled": False,
        "port": "/dev/ttyTHS1",
        "baudrate": 115200,
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "timeout": 0.1,
        "write_every_n": 1,
        "send_when_lost": False,
        "line_format": "{dx},{dy}\n",
    },
    # False-target suppression (shared/target_filter.py + shared/target_stabilizer.py).
    # Edited on the desktop (Tools → Detection Filters) and shipped inside the package.
    "stabilizer": {
        # 1) Target must agree with the position predicted from aux objects (landmarks)
        "geometry_check": True,
        "geometry_max_center_err": 0.8,  # × expected target size
        "geometry_max_size_ratio": 2.0,  # detected / expected size outside [1/r, r] → reject
        # 2) Temporal confirmation (N of M) + hysteresis + motion gate
        "temporal": True,
        "confirm_hits": 3,
        "confirm_window": 5,
        "hold_conf_ratio": 0.7,  # once locked, keep down to min_conf × this
        "gate_ratio": 1.5,  # gate radius × target size around the predicted position
        "gate_min_px": 40.0,
        "coast_frames": 8,  # frames to keep the lock on prediction when the detector misses
        # 3) Aux objects must fit the scene layout; one alone cannot acquire the target
        "landmark_consistency": True,
        "min_landmarks_acquire": 2,
    },
}


def default_runtime_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_RUNTIME)
