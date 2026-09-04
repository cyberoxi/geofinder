#!/usr/bin/env python3
"""Extract frames from a video for labeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.video_reader import VideoReader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--every", type=int, default=10)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(args.video, prefer_gstreamer=False)
    idx = 0
    saved = 0
    while True:
        ok, frame = reader.read()
        if not ok or frame is None:
            break
        if idx % args.every == 0:
            cv2.imwrite(str(out / f"frame_{idx:06d}.jpg"), frame)
            saved += 1
        idx += 1
    reader.release()
    print(f"Saved {saved} frames to {out}")


if __name__ == "__main__":
    main()
