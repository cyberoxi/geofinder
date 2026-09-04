# Deployment Package: demo_zoom

Target class: `target_region`
Model type: segmentation
Model format: none

## Transfer to Jetson

```bash
scp pkg_20260904_111818.zip user@JETSON_IP:/home/user/packages/
```

## On Jetson

```bash
unzip pkg_20260904_111818.zip
python3 jetson_runtime/app.py --package pkg_20260904_111818
# Build TensorRT engine if missing:
python3 tools/build_tensorrt.py --onnx model/model.onnx --output model/model.engine --fp16
```
