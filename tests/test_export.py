"""Export CSV/JSON tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.result_exporter import ResultExporter
from core.types import FrameResult


def test_export_csv_json(tmp_path):
    exporter = ResultExporter(str(tmp_path / "out"), str(tmp_path / "failed"), {"export": {"save_csv": True, "save_json": True, "save_center_path": True}})
    for i in range(3):
        exporter.add_result(
            FrameResult(
                frame_id=i,
                timestamp=i / 30.0,
                center_x=10 + i,
                center_y=20 + i,
                bbox_x=1,
                bbox_y=2,
                bbox_width=30,
                bbox_height=40,
                polygon_points=[[1, 2], [31, 2], [31, 42], [1, 42]],
                confidence=0.8,
                tracking_method="Homography",
                inlier_count=20,
                reprojection_error=1.2,
                scale=1.0 + i * 0.1,
                status="TRACKING",
            )
        )
    paths = exporter.finalize({"tracking": {"mode": "manual"}}, {"tracked": 3}, video_path="x.mp4")
    assert Path(paths["csv"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["center_path"]).exists()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["num_frames"] == 3
