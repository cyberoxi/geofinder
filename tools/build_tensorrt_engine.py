"""Build TensorRT engine from ONNX on Jetson (device-specific)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_with_trtexec(onnx: Path, engine: Path, fp16: bool, int8: bool, workspace: int) -> None:
    import shutil
    import subprocess

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        # Common Jetson location
        candidate = Path("/usr/src/tensorrt/bin/trtexec")
        if candidate.exists():
            trtexec = str(candidate)
    if trtexec is None:
        raise RuntimeError("trtexec not found. Install TensorRT or use Ultralytics export.")

    cmd = [
        trtexec,
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        f"--memPoolSize=workspace:{workspace}",
    ]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def build_with_ultralytics(weights_or_onnx: Path, engine: Path, imgsz: int, half: bool) -> None:
    from ultralytics import YOLO

    model = YOLO(str(weights_or_onnx))
    out = model.export(format="engine", imgsz=imgsz, half=half, device=0)
    out_path = Path(out)
    engine.parent.mkdir(parents=True, exist_ok=True)
    if out_path.resolve() != engine.resolve():
        out_path.replace(engine)
    print(f"Engine exported: {engine}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build TensorRT engine for Jetson")
    p.add_argument("--onnx", help="Input ONNX model")
    p.add_argument("--weights", help="Optional .pt to export engine via Ultralytics")
    p.add_argument("--output", default="models/best.engine")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--workspace", type=int, default=2048, help="Workspace MB for trtexec")
    p.add_argument("--backend", choices=["trtexec", "ultralytics"], default="trtexec")
    args = p.parse_args()

    fp16 = False if args.no_fp16 else args.fp16
    engine = Path(args.output)
    engine.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "ultralytics":
        src = Path(args.weights or args.onnx or "")
        if not src.exists():
            raise SystemExit("Provide --weights or --onnx for ultralytics backend")
        build_with_ultralytics(src, engine, args.imgsz, half=fp16)
        return

    if not args.onnx:
        raise SystemExit("--onnx is required for trtexec backend")
    onnx = Path(args.onnx)
    if not onnx.exists():
        raise SystemExit(f"ONNX not found: {onnx}")
    try:
        build_with_trtexec(onnx, engine, fp16=fp16, int8=args.int8, workspace=args.workspace)
    except Exception as exc:  # noqa: BLE001
        print(f"trtexec failed: {exc}", file=sys.stderr)
        if args.weights or args.onnx:
            print("Falling back to Ultralytics export…")
            src = Path(args.weights) if args.weights else onnx
            build_with_ultralytics(src, engine, args.imgsz, half=fp16)
        else:
            raise


if __name__ == "__main__":
    main()
