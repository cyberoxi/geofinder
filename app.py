#!/usr/bin/env python3
"""Thin launcher — prefer desktop_app or jetson_runtime entrypoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description="Ground Region Tracker dual-app launcher")
    p.add_argument("app", nargs="?", choices=["desktop", "jetson", "legacy"], default="jetson")
    p.add_argument("--legacy-headless", action="store_true")
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.app == "desktop":
        from desktop_app.app import main as desktop_main

        return desktop_main() if callable(desktop_main) else __import__("desktop_app.gui.main_window", fromlist=["main"]).main()
    if args.app == "legacy":
        # Keep old monolithic app available
        import runpy

        sys.path.insert(0, str(ROOT))
        # Execute old entry by importing previous pipeline via jetson for headless
        from jetson_runtime.app import main as jmain

        return jmain()
    from jetson_runtime.app import main as jmain

    return jmain()


if __name__ == "__main__":
    raise SystemExit(main())
