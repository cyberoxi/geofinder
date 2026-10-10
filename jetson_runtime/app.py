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
    p.add_argument("--video", default=None, help="Input video")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--end-frame", type=int, default=-1)
    p.add_argument("--serial", action="store_true", help="Enable UART offset send (dx,dy)")
    p.add_argument("--serial-port", default=None, help="Serial device, e.g. /dev/ttyTHS1 or /dev/ttyUSB0")
    p.add_argument("--serial-baud", type=int, default=None, help="Baud rate (default 115200)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.headless:
        if not args.package or not args.video:
            raise SystemExit("--package and --video required in headless mode")
        from jetson_runtime.runtime.pipeline import run_runtime

        serial_override = None
        if args.serial or args.serial_port or args.serial_baud:
            serial_override = {"enabled": True}
            if args.serial_port:
                serial_override["port"] = args.serial_port
            if args.serial_baud:
                serial_override["baudrate"] = args.serial_baud

        result = run_runtime(
            args.package,
            args.video,
            output_dir=args.output,
            end_frame=args.end_frame,
            serial_override=serial_override,
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
