# Deployment Package: target_2

Target class: `target_region`
Model type: segmentation
Model format: onnx

## Transfer to Jetson

```bash
scp pkg_20261009_180306.zip user@JETSON_IP:/home/user/packages/
```

## On Jetson

```bash
unzip pkg_20261009_180306.zip
python3 jetson_runtime/app.py --package pkg_20261009_180306
# Build TensorRT engine if missing:
python3 tools/build_tensorrt.py --onnx model/model.onnx --output model/model.engine --fp16
```
