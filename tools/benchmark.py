#!/usr/bin/env python3
"""Benchmark Jetson runtime or legacy manual pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", default=None, help="Deployment package for runtime benchmark")
    p.add_argument("--video", required=True)
    p.add_argument("--roi", default="40,30,120,110")
    p.add_argument("--output", default="data/output/benchmark.json")
    p.add_argument("--end-frame", type=int, default=40)
    args = p.parse_args()

    t0 = time.perf_counter()
    if args.package:
        from jetson_runtime.runtime.pipeline import run_runtime

        result = run_runtime(args.package, args.video, end_frame=args.end_frame, prefer_gstreamer=False)
        stats = result.get("stats", {})
        report = {
            "mode": "package_runtime",
            "average_fps": stats.get("avg_fps"),
            "minimum_fps": stats.get("min_fps"),
            "successful_frames": stats.get("tracked"),
            "lost_frames": stats.get("lost"),
            "re_detections": stats.get("redetect"),
            "scale_changes": stats.get("scale_change"),
            "elapsed_sec": time.perf_counter() - t0,
            "paths": result.get("paths"),
            "raw_stats": stats,
        }
    else:
        from core.roi_selector import ROISelector
        from core.pipeline import run_pipeline
        from core.config import load_config

        parts = [float(x) for x in args.roi.split(",")]
        roi = ROISelector.from_rectangle(*parts)
        config = load_config()
        config["tracking"]["mode"] = "manual"
        config["yolo"]["enabled"] = False
        config["video"]["prefer_gstreamer"] = False
        config["video"]["end_frame"] = args.end_frame
        result = run_pipeline(args.video, roi=roi, config=config)
        stats = result.get("stats", {})
        report = {
            "mode": "manual_legacy",
            "average_fps": stats.get("avg_fps"),
            "successful_frames": stats.get("tracked"),
            "lost_frames": stats.get("lost"),
            "elapsed_sec": time.perf_counter() - t0,
            "paths": result.get("paths"),
            "raw_stats": stats,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
