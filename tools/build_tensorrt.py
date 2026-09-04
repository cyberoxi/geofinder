#!/usr/bin/env python3
"""Build TensorRT engine on Jetson."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", default="model.engine")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--workspace", type=int, default=2048)
    args = p.parse_args()
    fp16 = not args.no_fp16
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec).exists():
        # Fallback ultralytics
        from ultralytics import YOLO

        model = YOLO(args.onnx)
        out = model.export(format="engine", half=fp16, device=0)
        Path(out).replace(args.output)
        print(f"Engine: {args.output}")
        return
    cmd = [trtexec, f"--onnx={args.onnx}", f"--saveEngine={args.output}", f"--memPoolSize=workspace:{args.workspace}"]
    if fp16:
        cmd.append("--fp16")
    print(" ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
