#!/usr/bin/env python3
"""Jetson Orin Nano Runtime entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Jetson Ground Region Runtime")
    p.add_argument("--package", required=False, help="Deployment package ZIP or folder")
    p.add_argument("--video", default=None, help="Input video file (source=file)")
    p.add_argument(
        "--source",
        choices=("file", "imx477", "usb"),
        default=None,
        help="Input kind: file | imx477 (CSI) | usb (webcam). Default: file if --video set",
    )
    p.add_argument("--sensor-id", type=int, default=0, help="IMX477 CSI sensor-id (default 0)")
    p.add_argument("--camera-index", type=int, default=0, help="USB webcam /dev/videoN index")
    p.add_argument("--width", type=int, default=None, help="Capture width (live cameras)")
    p.add_argument("--height", type=int, default=None, help="Capture height (live cameras)")
    p.add_argument("--fps", type=int, default=30, help="Capture FPS (live cameras)")
    p.add_argument("--flip", type=int, default=0, help="nvvidconv flip-method for CSI")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--end-frame", type=int, default=-1)
    p.add_argument("--min-conf", type=float, default=None,
                   help="Min detection confidence in %% (1-99); default = package value (Live Test)")
    p.add_argument("--serial", action="store_true", help="Enable UART offset send (dx,dy)")
    p.add_argument("--serial-port", default=None, help="Serial device, e.g. /dev/ttyTHS1 or /dev/ttyUSB0")
    p.add_argument("--serial-baud", type=int, default=None, help="Baud rate (default 115200)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.headless:
        kind = args.source
        if kind is None:
            kind = "file" if args.video else None
        if not args.package or not kind:
            raise SystemExit(
                "--package and input required in headless mode "
                "(--video for file, or --source imx477|usb)"
            )
        if kind == "file" and not args.video:
            raise SystemExit("--video required when --source file")

        from jetson_runtime.runtime.pipeline import run_runtime
        from jetson_runtime.runtime.sources import InputSourceSpec

        if kind == "file":
            spec = InputSourceSpec.from_file(args.video, prefer_gstreamer=False)
        elif kind == "imx477":
            spec = InputSourceSpec(
                kind="imx477",
                sensor_id=args.sensor_id,
                width=args.width or 1920,
                height=args.height or 1080,
                fps=args.fps,
                flip_method=args.flip,
            )
        else:
            spec = InputSourceSpec(
                kind="usb",
                usb_index=args.camera_index,
                width=args.width or 1280,
                height=args.height or 720,
                fps=args.fps,
            )

        serial_override = None
        if args.serial or args.serial_port or args.serial_baud:
            serial_override = {"enabled": True}
            if args.serial_port:
                serial_override["port"] = args.serial_port
            if args.serial_baud:
                serial_override["baudrate"] = args.serial_baud

        result = run_runtime(
            args.package,
            source=spec,
            output_dir=args.output,
            end_frame=args.end_frame,
            serial_override=serial_override,
            min_conf=args.min_conf / 100.0 if args.min_conf is not None else None,
        )
        print("Done.")
        for k, v in result.get("paths", {}).items():
            print(f"  {k}: {v}")
        stats = result.get("stats", {})
        print(f"  avg_fps={stats.get('avg_fps', 0):.2f} tracked={stats.get('tracked')} lost={stats.get('lost')}")
        return 0

    from jetson_runtime.gui.main_window import main as gui_main

    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
