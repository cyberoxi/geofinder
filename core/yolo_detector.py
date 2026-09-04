"""YOLO detection / segmentation wrapper with CUDA / TensorRT / CPU backends."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.logger import get_logger
from core.types import Detection, ROI

logger = get_logger("grt.yolo")


class YOLODetector:
    """
    Ultralytics YOLO wrapper.

    Backend selection:
      - tensorrt / .engine  → TensorRT
      - .onnx               → ONNX Runtime / Ultralytics
      - .pt                 → PyTorch (dev/test)
      - cpu                 → force CPU
    """

    def __init__(
        self,
        model_path: str,
        mode: str = "detection",
        device: str = "cuda",
        imgsz: int = 640,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        target_class: int = 0,
        use_fp16: bool = True,
        warmup_iterations: int = 5,
        class_names: Optional[List[str]] = None,
        fallback_paths: Optional[List[str]] = None,
    ):
        self.mode = mode.lower()
        self.device_pref = device
        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_class = target_class
        self.use_fp16 = use_fp16
        self.warmup_iterations = warmup_iterations
        self.class_names = class_names or ["ground_region"]
        self.model = None
        self.backend = "none"
        self.device = "cpu"
        self.load_ms = 0.0
        self.last_inference_ms = 0.0
        self.available = False
        self._model_path = model_path
        self._fallback_paths = fallback_paths or []
        self._load_model()

    def _resolve_path(self) -> Optional[Path]:
        candidates = [self._model_path, *self._fallback_paths]
        for c in candidates:
            if not c:
                continue
            p = Path(c)
            if p.exists():
                return p
        return None

    def _pick_device(self, path: Path) -> str:
        pref = self.device_pref.lower()
        if pref == "cpu":
            return "cpu"
        if path.suffix.lower() == ".engine" or pref == "tensorrt":
            return "0"  # Ultralytics uses CUDA device for TRT engines
        # Probe CUDA
        try:
            import torch

            if torch.cuda.is_available() and pref in ("cuda", "tensorrt", "0", "gpu"):
                return "0"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _load_model(self) -> None:
        path = self._resolve_path()
        if path is None:
            logger.warning(
                "YOLO model not found at %s (fallbacks=%s). Detector disabled.",
                self._model_path,
                self._fallback_paths,
            )
            self.available = False
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not installed — YOLO disabled")
            self.available = False
            return

        t0 = time.perf_counter()
        try:
            self.model = YOLO(str(path))
            self.device = self._pick_device(path)
            suffix = path.suffix.lower()
            if suffix == ".engine":
                self.backend = "tensorrt"
            elif suffix == ".onnx":
                self.backend = "onnx"
            else:
                self.backend = "pytorch"
            self.available = True
            self.load_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "Loaded YOLO model %s backend=%s device=%s load=%.1fms",
                path,
                self.backend,
                self.device,
                self.load_ms,
            )
            self.warmup()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load YOLO model %s: %s", path, exc)
            self.model = None
            self.available = False
            # TensorRT engine may be incompatible with this device
            if path.suffix.lower() == ".engine":
                for fb in self._fallback_paths:
                    fb_path = Path(fb) if fb else None
                    if fb_path and fb_path.exists() and fb_path.suffix.lower() != ".engine":
                        logger.info("Trying fallback model: %s", fb_path)
                        self._model_path = str(fb_path)
                        self._fallback_paths = [p for p in self._fallback_paths if p != fb]
                        self._load_model()
                        return

    def warmup(self) -> None:
        if not self.available or self.model is None:
            return
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(max(1, self.warmup_iterations)):
            try:
                self.model.predict(
                    dummy,
                    imgsz=self.imgsz,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    half=self.use_fp16 and self.device != "cpu",
                    verbose=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warmup issue: %s", exc)
                break
        logger.info("YOLO warmup complete (%d iters)", self.warmup_iterations)

    def predict(self, frame: np.ndarray) -> List[Detection]:
        if not self.available or self.model is None:
            return []
        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                half=self.use_fp16 and self.device != "cpu",
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("YOLO inference failed: %s", exc)
            self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
            return []
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        if not results:
            return detections
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        masks = getattr(r0, "masks", None)

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        mask_data = None
        if self.mode == "segmentation" and masks is not None and masks.data is not None:
            mask_data = masks.data.cpu().numpy()

        for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
            if int(cls_id) != int(self.target_class) and self.target_class >= 0:
                # If model is single-class custom, still accept class 0 mismatches lightly
                if len(self.class_names) > 1:
                    continue
            x1, y1, x2, y2 = map(float, box)
            det = Detection(
                bbox=(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                confidence=float(conf),
                class_id=int(cls_id),
            )
            if mask_data is not None and i < len(mask_data):
                m = mask_data[i]
                # Resize mask to frame size
                mh, mw = frame.shape[:2]
                m_resized = cv2.resize(m.astype(np.float32), (mw, mh), interpolation=cv2.INTER_LINEAR)
                binary = (m_resized > 0.5).astype(np.uint8) * 255
                det.mask = binary
                roi = ROI.from_mask(binary)
                if roi is not None:
                    det.polygon = roi.points
            detections.append(det)
        return detections

    def detection_to_roi(self, det: Detection, frame_index: int = 0) -> ROI:
        if det.polygon is not None and len(det.polygon) >= 3:
            return ROI(det.polygon, frame_index=frame_index)
        x, y, w, h = det.bbox
        return ROI.from_bbox(x, y, w, h, frame_index)

    def get_info(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "backend": self.backend,
            "device": self.device,
            "model_path": self._model_path,
            "load_ms": self.load_ms,
            "last_inference_ms": self.last_inference_ms,
            "imgsz": self.imgsz,
            "mode": self.mode,
        }
