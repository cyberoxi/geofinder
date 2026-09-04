"""Project save/load helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.types import ROI, ROIType


def save_project(path: str | Path, data: Dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return out


def load_project(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def project_from_roi(
    video_path: str,
    roi: ROI,
    model_path: str = "",
    tracker_type: str = "optical_flow",
    confidence_threshold: float = 0.45,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = {
        "video_path": video_path,
        "initial_frame": int(roi.frame_index),
        "roi_type": roi.roi_type.value,
        "roi_points": roi.to_list(),
        "model_path": model_path,
        "tracker_type": tracker_type,
        "confidence_threshold": confidence_threshold,
    }
    if extras:
        data.update(extras)
    return data


def roi_from_project(data: Dict[str, Any]) -> ROI:
    points: List[List[float]] = data["roi_points"]
    roi_type = ROIType(data.get("roi_type", "polygon"))
    return ROI(points, roi_type, int(data.get("initial_frame", 0)))
