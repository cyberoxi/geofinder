#!/usr/bin/env python3
"""Generate synthetic zoom/pan videos from Esri World Imagery tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from desktop_app.satellite import generate_satellite_videos  # noqa: E402
from shared.logging import setup_logging  # noqa: E402


def main() -> int:
    setup_logging()
    p = argparse.ArgumentParser(description="Satellite imagery → zoom/pan video")
    p.add_argument("--lat", type=float, default=35.6892, help="Center latitude")
    p.add_argument("--lon", type=float, default=51.3890, help="Center longitude")
    p.add_argument("--span", type=float, default=0.02, help="BBox span in degrees")
    p.add_argument("--south", type=float, default=None)
    p.add_argument("--west", type=float, default=None)
    p.add_argument("--north", type=float, default=None)
    p.add_argument("--east", type=float, default=None)
    p.add_argument("--zoom", type=int, default=17, help="Tile zoom 1..20")
    p.add_argument(
        "--zoom-out",
        type=float,
        default=12.0,
        help="Zoom flyover starts this many× wider than the region (default 12)",
    )
    p.add_argument(
        "--modes",
        default="both",
        help="zoom, pan, or both (comma-separated also ok)",
    )
    p.add_argument("--out", default="data/output", help="Output directory")
    p.add_argument("--duration", type=float, default=8.0, help="Seconds per clip")
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--cache", default="data/satellite_cache")
    args = p.parse_args()

    modes = [m.strip() for m in args.modes.replace(",", " ").split() if m.strip()]
    paths = generate_satellite_videos(
        lat=args.lat,
        lon=args.lon,
        span=args.span,
        south=args.south,
        west=args.west,
        north=args.north,
        east=args.east,
        zoom=args.zoom,
        modes=modes,
        out_dir=args.out,
        duration_s=args.duration,
        fps=args.fps,
        out_size=(args.width, args.height),
        cache_dir=args.cache,
        zoom_out_factor=args.zoom_out,
    )
    for key, path in paths.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
