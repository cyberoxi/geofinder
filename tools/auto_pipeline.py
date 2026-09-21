#!/usr/bin/env python3
"""
Fully automatic: satellite region → labeled videos (target + landmarks) →
dataset → YOLO training → ONNX → Jetson package → evaluation.

Example:
    python tools/auto_pipeline.py --project data/projects/tehran \\
        --south 35.6885 --west 51.3880 --north 35.6899 --east 51.3900 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop_app.pipeline.auto import AutoPipeline, AutoPipelineConfig  # noqa: E402
from desktop_app.project_manager.manager import ProjectManager  # noqa: E402
from desktop_app.satellite import generate_satellite_videos  # noqa: E402
from shared.logging import setup_logging  # noqa: E402


def import_satellite_result(pm: ProjectManager, result: dict) -> list:
    """Add generated videos + their labels + scene layout to the project."""
    added = []
    for item in result.get("videos", []):
        asset = pm.add_video(item["path"])
        pm.attach_auto_labels(asset.video_id, item["labels"])
        added.append(asset.video_id)
    if result.get("scene"):
        pm.set_scene(result["scene"], result.get("scene_preview"), result.get("mosaic"))
    return added


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True, help="Project folder (created if missing)")
    p.add_argument("--name", default=None, help="Project name (new projects)")
    p.add_argument("--south", type=float, required=True)
    p.add_argument("--west", type=float, required=True)
    p.add_argument("--north", type=float, required=True)
    p.add_argument("--east", type=float, required=True)
    p.add_argument("--zoom", type=int, default=17, help="Tile download zoom")
    p.add_argument("--zoom-out", type=float, default=12.0, help="Widest view = this × region")
    p.add_argument("--videos", type=int, default=6, help="Number of varied training videos")
    p.add_argument("--landmarks", type=int, default=4, help="Landmarks to discover around the target")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-type", default="segmentation", choices=["segmentation", "detection"])
    p.add_argument("--cache", default=str(ROOT / "data" / "satellite_cache"))
    p.add_argument("--skip-train", action="store_true", help="Only generate videos + labels")
    args = p.parse_args()

    root = Path(args.project)
    if (root / "project.json").exists():
        pm = ProjectManager.open(root)
    else:
        pm = ProjectManager.create(root, args.name or root.name)

    lat = (args.south + args.north) / 2
    lon = (args.west + args.east) / 2
    result = generate_satellite_videos(
        lat=lat,
        lon=lon,
        south=args.south,
        west=args.west,
        north=args.north,
        east=args.east,
        zoom=args.zoom,
        zoom_out_factor=args.zoom_out,
        out_dir=pm.exports_dir / "satellite",
        duration_s=args.duration,
        fps=args.fps,
        cache_dir=args.cache,
        n_videos=args.videos,
        num_landmarks=args.landmarks,
        target_class=pm.meta.target_class if pm.meta else "target_region",
        progress_cb=lambda d, t, m: print(f"[satellite {d:3d}%] {m}", flush=True) if d % 5 == 0 else None,
    )
    added = import_satellite_result(pm, result)
    print(f"Added {len(added)} auto-labeled videos; classes: {pm.class_names()}")
    pm.close()
    if args.skip_train:
        return 0

    cfg = AutoPipelineConfig(
        model_type=args.model_type,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )
    summary = AutoPipeline(root, cfg, progress_cb=lambda s, m: print(f"[{s}] {m}", flush=True)).run()
    print(json.dumps({k: summary.get(k) for k in ("package", "onnx", "evaluation", "summary_path")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
