# Models directory

Place your trained YOLO weights here:

- `best.pt` — PyTorch checkpoint (training / desktop testing)
- `best.onnx` — portable intermediate format
- `best.engine` — TensorRT engine built on the target Jetson

## Quick export (desktop with GPU)

```bash
python tools/convert_to_onnx.py --weights models/best.pt --imgsz 640 --output models/best.onnx
```

## Build engine on Jetson

```bash
python tools/build_tensorrt_engine.py --onnx models/best.onnx --output models/best.engine --fp16
```

YOLO is **optional** for Manual ROI Tracking mode.
