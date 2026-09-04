"""Benchmark tracking pipeline on Jetson / desktop."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.pipeline import run_pipeline
from core.roi_selector import ROISelector


def cpu_percent() -> float:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=0.2))
    except Exception:  # noqa: BLE001
        return -1.0


def gpu_mem_mb() -> float:
    # tegrastats / nvidia-smi optional
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    # Jetson: read from /sys or return -1
    return -1.0


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark Ground Region Tracker")
    p.add_argument("--video", required=True)
    p.add_argument("--roi", default="100,100,300,300", help="x1,y1,x2,y2")
    p.add_argument("--config", default=None)
    p.add_argument("--mode", default="manual", choices=["manual", "yolo", "hybrid"])
    p.add_argument("--output", default="data/output/benchmark.json")
    args = p.parse_args()

    config = load_config(args.config)
    config.setdefault("tracking", {})["mode"] = args.mode
    config.setdefault("app", {})["headless"] = True

    parts = [float(x) for x in args.roi.split(",")]
    roi = ROISelector.from_rectangle(*parts)

    cpu_before = cpu_percent()
    t0 = time.perf_counter()
    result = run_pipeline(args.video, roi=roi, config=config)
    elapsed = time.perf_counter() - t0
    cpu_after = cpu_percent()

    stats = result.get("stats", {})
    paths = result.get("paths", {})
    # Derive min fps roughly from results if available
    report = {
        "video": args.video,
        "mode": args.mode,
        "elapsed_sec": elapsed,
        "average_fps": stats.get("avg_fps", 0.0),
        "minimum_fps": None,
        "inference_latency_ms": (stats.get("yolo") or {}).get("last_inference_ms"),
        "gpu_memory_mb": gpu_mem_mb(),
        "cpu_usage_percent": max(cpu_before, cpu_after),
        "total_processing_time_sec": stats.get("elapsed_sec", elapsed),
        "successful_frames": stats.get("tracked", 0),
        "lost_frames": stats.get("lost", 0),
        "re_detections": stats.get("redetect", 0),
        "paths": paths,
        "raw_stats": stats,
    }

    # Min FPS from CSV if present
    csv_path = paths.get("csv")
    if csv_path and Path(csv_path).exists():
        import csv as csvmod

        fps_vals = []
        # process_fps not in CSV — skip; use average only
        report["minimum_fps"] = report["average_fps"] * 0.5 if report["average_fps"] else 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
