"""Convert Ultralytics YOLO .pt model to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    p.add_argument("--weights", required=True, help="Path to .pt weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--output", default=None, help="Output .onnx path")
    p.add_argument("--dynamic", action="store_true", help="Dynamic input shapes")
    p.add_argument("--simplify", action="store_true", default=True)
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    out_path = Path(out)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        out_path.replace(target)
        out_path = target
    print(f"ONNX exported: {out_path}")


if __name__ == "__main__":
    main()
