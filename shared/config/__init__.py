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
}


def default_runtime_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_RUNTIME)
