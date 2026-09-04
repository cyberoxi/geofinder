#!/usr/bin/env python3
"""Ground Region Tracker — entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import env_override, load_config
from core.logger import setup_logging
from core.pipeline import run_pipeline
from core.project_io import load_project, roi_from_project
from core.roi_selector import ROISelector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ground Region Tracker for Jetson Orin Nano")
    p.add_argument("--config", default=None, help="Path to YAML config")
    p.add_argument("--video", default=None, help="Input video path")
    p.add_argument("--project", default=None, help="Project JSON path")
    p.add_argument("--headless", action="store_true", help="Run without GUI")
    p.add_argument("--mode", choices=["manual", "yolo", "hybrid"], default=None)
    p.add_argument(
        "--roi",
        default=None,
        help="Rectangle ROI as x1,y1,x2,y2 (for headless manual/hybrid)",
    )
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--end-frame", type=int, default=None)
    return p.parse_args()


def run_gui(config: dict) -> int:
    from gui.qt_compat import QApplication, QT_API
    from gui.main_window import MainWindow
    from core.logger import get_logger

    get_logger("grt").info("Using Qt API: %s", QT_API)
    app = QApplication(sys.argv)
    app.setApplicationName(config.get("app", {}).get("name", "Ground Region Tracker"))
    win = MainWindow(config)
    win.show()
    from gui.qt_compat import qt_exec

    return qt_exec(app)


def run_headless(args: argparse.Namespace, config: dict) -> int:
    roi = None
    video = args.video
    if args.project:
        proj = load_project(args.project)
        video = video or proj["video_path"]
        roi = roi_from_project(proj)
    if args.roi:
        parts = [float(x) for x in args.roi.split(",")]
        if len(parts) != 4:
            raise SystemExit("--roi must be x1,y1,x2,y2")
        roi = ROISelector.from_rectangle(*parts, frame_index=args.start_frame or 0)
    if not video:
        raise SystemExit("Headless mode requires --video or --project")
    if args.mode:
        config.setdefault("tracking", {})["mode"] = args.mode
    if args.start_frame is not None:
        config.setdefault("video", {})["start_frame"] = args.start_frame
    if args.end_frame is not None:
        config.setdefault("video", {})["end_frame"] = args.end_frame

    result = run_pipeline(video_path=video, roi=roi, config=config, project_path=None)
    print("Done.")
    for k, v in result.get("paths", {}).items():
        print(f"  {k}: {v}")
    stats = result.get("stats", {})
    print(f"  avg_fps: {stats.get('avg_fps', 0):.2f}")
    print(f"  tracked: {stats.get('tracked', 0)} lost: {stats.get('lost', 0)}")
    return 0


def main() -> int:
    args = parse_args()
    config = env_override(load_config(args.config))
    log_cfg = config.get("logging", {})
    setup_logging(
        log_file=log_cfg.get("file"),
        level=log_cfg.get("level", "INFO"),
        max_bytes=int(log_cfg.get("max_bytes", 5_242_880)),
        backup_count=int(log_cfg.get("backup_count", 3)),
    )

    headless = args.headless or config.get("app", {}).get("headless", False)
    if headless:
        return run_headless(args, config)
    return run_gui(config)


if __name__ == "__main__":
    raise SystemExit(main())
