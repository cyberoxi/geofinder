#!/usr/bin/env python3
"""Export YOLO weights to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        out.replace(target)
        out = target
    print(f"ONNX: {out}")


if __name__ == "__main__":
    main()
