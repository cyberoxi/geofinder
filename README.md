# Ground Region Tracker — Dual App Architecture

Two applications share a common core:

| App | Path | Role |
|-----|------|------|
| **Desktop Dataset & Deployment Manager** | `desktop_app/` | Annotate, build YOLO dataset, train, export ONNX, build Jetson package |
| **Jetson Orin Nano Runtime** | `jetson_runtime/` | Offline import package, detect/track target region under zoom |

Shared code lives in `shared/` (geometry, features, schemas, config, logging, video I/O, package validation).

```mermaid
flowchart LR
  Desktop[desktop_app] --> Package[deployment_package.zip]
  Package --> Jetson[jetson_runtime]
  Desktop --> Shared[shared]
  Jetson --> Shared
```

---

## Install — Desktop (GPU PC)

```bash
cd /path/to/project
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-desktop.txt
python desktop_app/app.py
```

## Install — Jetson Orin Nano

Verified context: JetPack **R39.x**, TensorRT **10.x**, OpenCV **4.x**, Python **3.12**.

```bash
sudo nvpmodel -m 0
sudo jetson_clocks

cd /path/to/project
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -U pip
pip install -r requirements-jetson.txt
# Optional YOLO runtime after JetPack-matched torch:
# pip install ultralytics onnx
```

---

## Desktop workflow

1. **New Project** — choose an empty folder (creates `project.json`, `annotations.db`, `videos/`, …).
2. **Add Video** — one or more reference videos (copied under `videos/` with relative paths).
3. **Annotate** — seek frames, draw Polygon/Rectangle ROI (image coordinates). Save multiple keyframes.
4. **Interpolate** — fill frames between keyframes (yellow); Confirm to promote.
5. **Build Dataset** — YOLO layout with **video-based** 70/20/10 split; default **segmentation**.
6. **Train YOLO** — non-blocking worker (`yolov8n-seg.pt` default, imgsz 640).
7. **Export ONNX** — `runs/model.onnx` or `tools/export_onnx.py`.
8. **Build Jetson Package** — ZIP with manifest, reference bank (AKAZE), checksums, runtime.yaml.

### Automatic pipeline: satellite region → Jetson package

Select a region on a satellite map. Everything after that is automatic:

1. **Imagery.** Esri World Imagery tiles (no API key) are fetched for an area *Zoom depth ×* larger than the selection.
2. **Exact target polygon.** The selection is projected into mosaic pixels (Web Mercator).
3. **Landmark discovery.** *K* distinctive patches around the target are picked automatically (stadiums, large roofs, parks and so on). They are scored by rarity against the whole mosaic, local uniqueness (no look-alike nearby) and texture, and spread around the target. See `scene/scene_preview.jpg`.
4. **Varied labeled videos.** The generator renders a canonical zoom-in plus rotated, off-centre and deep zooms, low passes where the target leaves the view, and lighting, haze and blur changes. The target and every landmark are projected into each frame through the same affine used to render it, so **labels are exact with no manual annotation**. Labels sit next to each video as `*.labels.json` and are shown on the annotation canvas.
5. **Dataset.** Multi-class YOLO data: class 0 = target, 1..K = landmarks. Split is by video, background frames are included, and augmentation applies to train only.
6. **Train → ONNX → package → evaluation.** The pipeline scores the model on a held-out video and writes `exports/evaluation/*_localized.mp4`.

**GUI:** **Tools → Generate Satellite Video…** → select the region → keep *Auto-label target + landmarks* and *Then run full pipeline* checked → **Generate Video**. You can also press **▶ Run Full Pipeline** at any time, and use **Test Model on Video…** to run the model on any clip.

**CLI (headless):**

```bash
python tools/auto_pipeline.py --project data/projects/my_site \
    --south 35.6885 --west 51.3880 --north 35.6899 --east 51.3900 \
    --videos 6 --landmarks 4 --epochs 40 --imgsz 640
```

**Locating the target from landmarks (runtime).** The package carries `reference/scene.json` (target and landmark layout). On the Jetson, when YOLO does not see the target directly (too small, occluded or out of frame) but sees landmarks, `shared.landmarks.TargetLocalizer` fits a similarity transform:
- 2 or more landmarks: RANSAC fit of scale, rotation and shift.
- 1 landmark: scale from its size, heading from the last estimate.

The target polygon is then projected through that transform. When it is off-screen, the overlay draws an arrow toward it. Colors: green = target detected, magenta = target inferred from landmarks, orange = landmarks.

> Synthetic satellite footage is not real drone video. Add real clips of the site (annotated or tested with *Test Model on Video…*) before relying on it in the field.

Legacy clips without labels: `python tools/satellite_video.py --lat 35.6892 --lon 51.3890 --span 0.02 --modes both --out data/output`. Tiles cache under `data/satellite_cache/`. Attribution: Esri World Imagery.

### Dataset layout

```text
datasets/dataset_v1/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── data.yaml
└── split_by_video.json
```

### Sample training CLI

```bash
yolo segment train data=datasets/dataset_v1/data.yaml model=yolov8n-seg.pt imgsz=640 epochs=100 batch=8 device=0
python tools/export_onnx.py --weights runs/train/weights/best.pt --output models/model.onnx
```

---

## Deployment package

```text
deployment_package/
├── manifest.json
├── model/{best.pt, model.onnx, model.engine?}
├── reference/{reference_image.jpg, roi_mask.png, descriptors.npz, templates/}
├── config/runtime.yaml
├── labels/classes.txt
├── checksums/sha256.json
└── README_DEPLOYMENT.md
```

### Sample `manifest.json`

```json
{
  "project_name": "field_plot_a",
  "target_class": "target_region",
  "model_type": "segmentation",
  "model_format": "onnx",
  "input_size": 640,
  "reference_roi": {"points": [[...]], "width": 1920, "height": 1080},
  "feature_method": "AKAZE",
  "min_confidence": 0.45,
  "min_iou": 0.20,
  "package_version": "1.0.0"
}
```

### Transfer

```bash
scp deployment_package.zip user@JETSON_IP:/home/user/packages/
```

On Jetson:

```bash
unzip deployment_package.zip -d deployment_package
# Prefer building engine on-device:
python tools/build_tensorrt.py --onnx deployment_package/model/model.onnx --output deployment_package/model/model.engine --fp16
python jetson_runtime/app.py --package deployment_package.zip --video /path/new.mp4 --headless
# or GUI:
python jetson_runtime/app.py
```

---

## Jetson Runtime algorithm

States: `INIT → SEARCHING → DETECTED → TRACKING ⇄ SCALE_CHANGE / RE_DETECTING → LOST → SEARCHING → COMPLETED`

- YOLO (+ reference AKAZE verification) for search / re-detect
- Homography + Optical Flow / CSRT between detections
- Multi-scale inference on `SCALE_CHANGE` / after `LOST`
- EMA/Kalman smoothing; adaptive frame skip when FPS drops
- Production path: TensorRT FP16; fallback ONNX → PyTorch (emergency)

Outputs: annotated video, CSV, JSON summary, lost frames.

---

## Zoom-friendly settings

- `yolo_every_n_frames`: 20–40  
- `feature_match_interval`: 3–5  
- `max_scale`: ≥ 8  
- `multi_scale_factors`: `[0.75, 1.0, 1.25]`  
- `imgsz`: 640 (drop to 480 if memory tight)

---

## Benchmark

```bash
python tools/benchmark.py --video data/input/sample.mp4 --roi 40,30,120,110 --mode manual
```

Or runtime headless stats (`avg_fps`, tracked/lost/redetect/scale_change) from `run_runtime`.

---

## Tests

```bash
pytest -q
```

---

## When is YOLO training required?

| Goal | Need YOLO? |
|------|------------|
| Track the same manually selected ROI in one video | No (features + trackers) |
| Find the **same physical plot** in new videos | YOLO + reference features (package) |
| Find **any similar region type** | YOLO on diverse dataset |

Train on desktop GPU (8–12 GB VRAM recommended). Build TensorRT **on the Jetson**.

---

## FAQ / limits

- Engines are JetPack/device specific — rebuild after upgrades.
- RTSP is stubbed for a future release (file video supported).
- Without a model, runtime initializes from package reference ROI and tracks via Homography/OF.
- GStreamer may fail on software-encoded MP4; OpenCV fallback is automatic.

## Entrypoints

```bash
python desktop_app/app.py
python jetson_runtime/app.py
python jetson_runtime/app.py --package pkg.zip --video new.mp4 --headless
python app.py desktop   # launcher
python app.py jetson
```
