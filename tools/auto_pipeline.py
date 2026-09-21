#!/usr/bin/env python3
"""
Fully automatic: satellite region → labeled videos (target + landmarks) →
dataset → YOLO training → ONNX → Jetson package → evaluation.

Examples:
    python tools/auto_pipeline.py --project data/projects/tehran \\
        --south 35.6885 --west 51.3880 --north 35.6899 --east 51.3900 --epochs 40

    # other source + 2 extra archive dates + night / thermal videos
    python tools/auto_pipeline.py --project p --south .. --west .. --north .. --east .. \\
        --source esri_clarity --extra-dates 2 --night 0.25 --ir 0.15

    # your own satellite image (target rectangle in image pixels, ground resolution in m/px)
    python tools/auto_pipeline.py --project p --image site.tif --target 1200 800 1500 1050 --gsd 0.5
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
from desktop_app.satellite import (  # noqa: E402
    prepare_scene,
    prepare_scene_from_image,
    read_world_file_mpp,
    render_scene_videos,
)
from desktop_app.satellite.tiles import (  # noqa: E402
    list_wayback_releases,
    mosaic_layers,
    pick_wayback_dates,
    register_wayback_layer,
)
from shared.logging import setup_logging  # noqa: E402


def import_satellite_result(pm: ProjectManager, result: dict) -> list:
    """Add generated videos + their labels + scene layout to the project."""
    added = []
    for item in result.get("videos", []):
        asset = pm.add_video(item["path"])
        pm.attach_auto_labels(asset.video_id, item["labels"])
        added.append(asset.video_id)
    if result.get("scene"):
        pm.set_scene(result["scene"], result.get("scene_preview"), result.get("mosaic"), result.get("candidates"))
    return added


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # Windows consoles (cp1252) cannot print progress arrows etc.
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True, help="Project folder (created if missing)")
    p.add_argument("--name", default=None, help="Project name (new projects)")
    p.add_argument("--south", type=float)
    p.add_argument("--west", type=float)
    p.add_argument("--north", type=float)
    p.add_argument("--east", type=float)
    p.add_argument("--source", default="satellite", help=f"Imagery source: {', '.join(mosaic_layers())}")
    p.add_argument("--extra-dates", type=int, default=2, help="Extra Esri Wayback dates (seasons/years) to train on")
    p.add_argument("--image", help="Use your own satellite/aerial image instead of downloading tiles")
    p.add_argument("--target", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="Target rect in image px")
    p.add_argument("--gsd", type=float, help="Own image ground resolution (m/px); read from world file if omitted")
    p.add_argument("--zoom", type=int, default=17, help="Tile download zoom")
    p.add_argument("--zoom-out", type=float, default=20.0, help="Widest view = this × region")
    p.add_argument("--videos", type=int, default=6, help="Number of varied training videos")
    p.add_argument("--landmarks", type=int, default=8, help="Landmarks to discover around the target")
    p.add_argument("--night", type=float, default=0.25, help="Fraction of videos as simulated night")
    p.add_argument("--ir", type=float, default=0.15, help="Fraction of videos as thermal-like IR")
    p.add_argument("--no-sam", action="store_true", help="Classical landmark detection only (no FastSAM)")
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

    out_dir = pm.exports_dir / "satellite"
    target_class = pm.meta.target_class if pm.meta else "target_region"
    prog = lambda d, t, m: print(f"[satellite {d:3d}%] {m}", flush=True) if d % 5 == 0 else None  # noqa: E731
    if args.image:
        if not args.target:
            p.error("--image needs --target X0 Y0 X1 Y1 (target rectangle in image pixels)")
        gsd = args.gsd or read_world_file_mpp(args.image)
        if not gsd:
            p.error("--image needs --gsd (metres per pixel) — no world file found")
        prepared = prepare_scene_from_image(
            args.image, tuple(args.target), meters_per_px=gsd, out_dir=out_dir, num_landmarks=args.landmarks,
            target_class=target_class, use_sam=not args.no_sam, progress_cb=prog,
        )
    else:
        if None in (args.south, args.west, args.north, args.east):
            p.error("give --south --west --north --east, or --image")
        extra = []
        if args.extra_dates > 0:
            rel = list_wayback_releases(args.cache)
            extra = [register_wayback_layer(r) for r in pick_wayback_dates(rel, args.extra_dates)]
        prepared = prepare_scene(
            south=args.south, west=args.west, north=args.north, east=args.east, zoom=args.zoom,
            zoom_out_factor=args.zoom_out, out_dir=out_dir, cache_dir=args.cache, num_landmarks=args.landmarks,
            target_class=target_class, use_sam=not args.no_sam, layer=args.source, extra_layers=extra, progress_cb=prog,
        )
    print("Landmarks:", ", ".join(prepared.layout.classes[1:]))
    result = render_scene_videos(
        prepared.all_mosaics(), prepared.layout, out_dir, n_videos=args.videos, night_fraction=args.night,
        ir_fraction=args.ir, duration_s=args.duration, fps=args.fps, progress_cb=prog,
    )
    result.update(prepared.paths())
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
    summary = AutoPipeline(
        root, cfg, progress_cb=lambda s, m, pct=None: print(f"[{s}{'' if pct is None else f' {pct:5.1f}%'}] {m}", flush=True)
    ).run()
    print(json.dumps({k: summary.get(k) for k in ("package", "onnx", "evaluation", "summary_path")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
