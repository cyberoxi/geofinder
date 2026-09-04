# Ground Region Tracker

Hybrid ground-region tracking for **NVIDIA Jetson Orin Nano**: manual ROI selection, feature matching + Homography, optical-flow / CSRT tracking, and optional YOLO re-detection (Detection or Segmentation).

## Features

- Rectangle / Polygon ROI selection on an initial frame
- AKAZE / ORB feature matching with RANSAC Homography (zoom + perspective)
- Optical Flow, CSRT, KCF inter-keyframe tracking
- Hybrid mode with periodic / urgent YOLO re-detection
- Confidence scoring, EMA / Kalman smoothing
- PySide6 GUI (non-blocking workers) + headless CLI
- GStreamer-preferred video IO with OpenCV fallback
- CSV / JSON / annotated video / failed-frame export
- ONNX + TensorRT tooling for Jetson deployment

## Tracking modes

| Mode | Description |
|------|-------------|
| **manual** | Follow the user-selected ROI with Homography + Flow/CSRT. **No YOLO training required.** |
| **yolo** | Detect a trained class, then track. Needed for new videos without manual ROI. |
| **hybrid** (default) | Manual ROI init + trackers + YOLO correction when confidence drops. |

---

## Target environment (Jetson Orin Nano)

| Component | Notes |
|-----------|--------|
| Device | NVIDIA Jetson Orin Nano |
| OS | Ubuntu (JetPack) |
| JetPack / L4T | R39.x (see `/etc/nv_tegra_release`) |
| Python | 3.10+ (3.12 tested) |
| CUDA / cuDNN / TensorRT | Bundled with JetPack (TensorRT 10.x on this image) |
| OpenCV | 4.x with GStreamer if available |

Set max performance on Jetson:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## Install (Jetson)

```bash
cd /path/to/project
python3 -m venv .venv --system-site-packages   # keeps JetPack CV/CUDA packages visible
source .venv/bin/activate
pip install -U pip
pip install PyYAML pytest psutil
# GUI (choose one):
pip install PySide6
# or: sudo apt install python3-pyqt5
```

The application auto-detects **PySide6** or **PyQt5** via `gui/qt_compat.py`.
### Optional: YOLO + TensorRT runtime

Install JetPack-matched PyTorch first (from NVIDIA), then:

```bash
pip install ultralytics onnx
# TensorRT Python is usually already available via JetPack
```

Place models under `models/`:

- `best.pt` (dev), `best.onnx`, `best.engine` (production)

---

## Install (desktop training PC)

Use a machine with an NVIDIA GPU (8–12 GB VRAM recommended), Ubuntu or Windows:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics opencv-python PyYAML pytest
```

Train **on the desktop**, export ONNX there, copy ONNX to Jetson, build the TensorRT engine **on the Jetson**.

---

## Run GUI

```bash
python app.py
# or
python app.py --config config/default_config.yaml
```

### Workflow

1. **File → Open Video**
2. Seek to the frame where the ground region is visible
3. Toolbar: **Rect ROI** or **Poly ROI** (polygon: left-click points, right-click / double-click to close)
4. **Confirm ROI**
5. Choose mode in the side panel (`hybrid` / `manual` / `yolo`)
6. **Run** — processing runs in a worker thread; GUI stays responsive
7. Outputs appear under `data/output/`

### Headless

```bash
python app.py --headless --video data/input/sample.mp4 --roi 100,80,260,220 --mode manual
python app.py --headless --project data/projects/my_project.json
```

---

## Configuration

All defaults live in [`config/default_config.yaml`](config/default_config.yaml). Important keys:

- `tracking.mode`, `tracker_type`, `feature_detector`, `min_inliers`, `lowe_ratio`
- `yolo.model_path`, `every_n_frames`, `imgsz` (default **640**), `device` (`cuda` / `tensorrt` / `cpu`)
- `smoothing.method` (`ema` / `kalman`)
- `video.prefer_gstreamer`, `skip_frames`, `start_frame`, `end_frame`

---

## Training YOLO (when needed)

**Not required** if you only track the manually selected ROI (Manual mode).

**Required** if you want automatic discovery of the same region on new videos (YOLO / Hybrid with re-detect).

### Dataset

1. Extract frames from real project videos (zoom low/high, lighting, shadow, angle, partial crop, motion blur).
2. Aim for **1000–3000** images to start.
3. Label with CVAT / Label Studio / Roboflow.
4. Use **Segmentation** if exact ground shape matters; else **Detection**.
5. Split ≈ 70% train / 20% val / 10% test.
6. Augment: scale, perspective, limited rotation, brightness, contrast, blur, noise, shadow, crop.

YOLO label layout:

```text
dataset/
  images/{train,val,test}/...
  labels/{train,val,test}/...
  data.yaml
```

Validate:

```bash
python tools/validate_dataset.py --data /path/to/dataset
```

### Train (desktop GPU)

```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 device=0
# or segmentation:
yolo segment train data=dataset/data.yaml model=yolov8n-seg.pt imgsz=640 epochs=100 device=0
```

Evaluate Precision / Recall / mAP, then test on held-out videos.

### Export ONNX (desktop)

```bash
python tools/convert_to_onnx.py --weights runs/detect/train/weights/best.pt --imgsz 640 --output models/best.onnx
```

### Build TensorRT engine (on Jetson)

```bash
python tools/build_tensorrt_engine.py --onnx models/best.onnx --output models/best.engine --fp16
# optional INT8 (needs calibration data for best accuracy):
python tools/build_tensorrt_engine.py --onnx models/best.onnx --output models/best.engine --int8
```

Engines are **device / JetPack specific** — rebuild after JetPack upgrades.

Production should prefer `.engine` (FP16). Use `.pt` only for debugging.

---

## Benchmark

```bash
python tools/benchmark.py --video data/input/sample.mp4 --roi 80,60,200,180 --mode manual --output data/output/benchmark.json
```

Reports average FPS, inference latency, CPU usage, GPU memory (if available), tracked / lost / re-detect counts.

---

## Tests

```bash
pip install pytest
pytest -q
```

Synthetic zoom/motion video is generated inside the e2e tests.

Generate a sample clip for manual GUI trials:

```bash
python -c "from tests.test_e2e import make_synthetic_video; from pathlib import Path; make_synthetic_video(Path('data/input/sample.mp4'), 60, 640, 480)"
```

---

## Project layout

```text
app.py
config/default_config.yaml
gui/          # PySide6 UI + workers
core/         # video, ROI, trackers, YOLO, export, pipeline
models/       # weights + model_config.yaml
tools/        # ONNX, TensorRT, benchmark, dataset validation
tests/
data/{input,output,failed_frames,projects}
logs/
```

---

## Suggested settings for top-down zoom videos

- Mode: `hybrid` (or `manual` if no model yet)
- Feature detector: `AKAZE`
- Tracker: `optical_flow`
- `feature_match_interval`: 3–5
- `keyframe_interval`: 10–15
- `min_inliers`: 10–15
- `max_scale`: 8+
- YOLO every N frames: 20–40; lower when tracking is unstable
- Inference size: **640** on Orin Nano; drop to 480 if GPU memory is tight
- Enable smoothing (`ema_alpha` ≈ 0.3–0.4)

---

## FAQ

**GUI freezes during processing?**  
Processing runs in `ProcessingWorker` (QThread). If the UI still stalls, reduce resolution / enable skip frames / use Manual mode without YOLO.

**GStreamer fails to open the file?**  
The reader falls back to OpenCV automatically (`prefer_gstreamer: false` to force).

**TensorRT engine incompatible?**  
Rebuild on this Jetson. The detector falls back to ONNX / PT paths listed in config.

**Not enough features in ROI?**  
Enlarge the ROI, switch to ORB, or lower `min_inliers`. Hybrid mode can still recover with YOLO if a model is present.

**Offline use?**  
Yes. After dependencies and models are installed, no network is required at runtime.

---

## What runs where

| Task | Machine |
|------|---------|
| Dataset labeling, YOLO training, ONNX export | Desktop GPU PC |
| TensorRT engine build, GUI tracking, benchmark | Jetson Orin Nano |
| Manual / Hybrid tracking without YOLO | Either (Jetson target) |

---

## License

Project code provided for the Ground Region Tracker deployment described above.
