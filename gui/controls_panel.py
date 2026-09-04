"""Side controls / settings panel."""

from __future__ import annotations

from typing import Any, Dict

from gui.qt_compat import (
    Signal,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QCheckBox,
    QVBoxLayout,
    QWidget,
)


class ControlsPanel(QWidget):
    settings_changed = Signal(dict)

    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config
        layout = QVBoxLayout(self)

        # Tracking mode
        mode_box = QGroupBox("Tracking")
        form = QFormLayout(mode_box)
        self.mode = QComboBox()
        self.mode.addItems(["hybrid", "manual", "yolo"])
        self.mode.setCurrentText(config.get("tracking", {}).get("mode", "hybrid"))
        form.addRow("Mode", self.mode)

        self.tracker_type = QComboBox()
        self.tracker_type.addItems(["optical_flow", "csrt", "kcf", "homography"])
        self.tracker_type.setCurrentText(config.get("tracking", {}).get("tracker_type", "optical_flow"))
        form.addRow("Tracker", self.tracker_type)

        self.feature_detector = QComboBox()
        self.feature_detector.addItems(["AKAZE", "ORB"])
        self.feature_detector.setCurrentText(config.get("tracking", {}).get("feature_detector", "AKAZE"))
        form.addRow("Features", self.feature_detector)

        self.min_inliers = QSpinBox()
        self.min_inliers.setRange(4, 200)
        self.min_inliers.setValue(int(config.get("tracking", {}).get("min_inliers", 12)))
        form.addRow("Min Inliers", self.min_inliers)

        self.conf_thr = QDoubleSpinBox()
        self.conf_thr.setRange(0.05, 0.99)
        self.conf_thr.setSingleStep(0.05)
        self.conf_thr.setValue(float(config.get("tracking", {}).get("confidence_threshold", 0.45)))
        form.addRow("Confidence", self.conf_thr)

        self.smoothing = QDoubleSpinBox()
        self.smoothing.setRange(0.05, 1.0)
        self.smoothing.setSingleStep(0.05)
        self.smoothing.setValue(float(config.get("smoothing", {}).get("ema_alpha", 0.35)))
        form.addRow("Smooth α", self.smoothing)

        self.skip_frames = QSpinBox()
        self.skip_frames.setRange(0, 30)
        self.skip_frames.setValue(int(config.get("video", {}).get("skip_frames", 0)))
        form.addRow("Skip Frames", self.skip_frames)
        layout.addWidget(mode_box)

        # YOLO
        yolo_box = QGroupBox("YOLO")
        yform = QFormLayout(yolo_box)
        self.yolo_mode = QComboBox()
        self.yolo_mode.addItems(["detection", "segmentation"])
        self.yolo_mode.setCurrentText(config.get("yolo", {}).get("mode", "detection"))
        yform.addRow("Type", self.yolo_mode)

        self.yolo_every = QSpinBox()
        self.yolo_every.setRange(1, 500)
        self.yolo_every.setValue(int(config.get("yolo", {}).get("every_n_frames", 30)))
        yform.addRow("Every N frames", self.yolo_every)

        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(int(config.get("yolo", {}).get("imgsz", 640)))
        yform.addRow("Inference size", self.imgsz)

        self.device = QComboBox()
        self.device.addItems(["cuda", "tensorrt", "cpu"])
        self.device.setCurrentText(config.get("yolo", {}).get("device", "cuda"))
        yform.addRow("Device", self.device)

        self.use_gpu = QCheckBox("Enable GPU")
        self.use_gpu.setChecked(config.get("yolo", {}).get("device", "cuda") != "cpu")
        yform.addRow(self.use_gpu)

        self.target_class = QSpinBox()
        self.target_class.setRange(0, 1000)
        self.target_class.setValue(int(config.get("yolo", {}).get("target_class", 0)))
        yform.addRow("Target class", self.target_class)

        self.iou_thr = QDoubleSpinBox()
        self.iou_thr.setRange(0.05, 0.95)
        self.iou_thr.setSingleStep(0.05)
        self.iou_thr.setValue(float(config.get("yolo", {}).get("iou_threshold", 0.45)))
        yform.addRow("IoU thr", self.iou_thr)
        layout.addWidget(yolo_box)

        # Status
        status_box = QGroupBox("Status")
        sform = QVBoxLayout(status_box)
        self.status_label = QLabel("READY")
        self.status_label.setStyleSheet("font-weight: bold; color: gray; font-size: 16px;")
        self.info_label = QLabel("—")
        self.info_label.setWordWrap(True)
        sform.addWidget(self.status_label)
        sform.addWidget(self.info_label)
        layout.addWidget(status_box)
        layout.addStretch(1)

        for w in (
            self.mode,
            self.tracker_type,
            self.feature_detector,
            self.min_inliers,
            self.conf_thr,
            self.smoothing,
            self.skip_frames,
            self.yolo_mode,
            self.yolo_every,
            self.imgsz,
            self.device,
            self.use_gpu,
            self.target_class,
            self.iou_thr,
        ):
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._emit)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._emit)
            elif hasattr(w, "toggled"):
                w.toggled.connect(self._emit)

    def _emit(self, *_args) -> None:
        self.settings_changed.emit(self.collect())

    def collect(self) -> Dict[str, Any]:
        device = self.device.currentText()
        if not self.use_gpu.isChecked():
            device = "cpu"
        return {
            "tracking": {
                "mode": self.mode.currentText(),
                "tracker_type": self.tracker_type.currentText(),
                "feature_detector": self.feature_detector.currentText(),
                "min_inliers": self.min_inliers.value(),
                "confidence_threshold": self.conf_thr.value(),
            },
            "smoothing": {"ema_alpha": self.smoothing.value(), "enabled": True},
            "video": {"skip_frames": self.skip_frames.value()},
            "yolo": {
                "mode": self.yolo_mode.currentText(),
                "every_n_frames": self.yolo_every.value(),
                "imgsz": self.imgsz.value(),
                "device": device,
                "target_class": self.target_class.value(),
                "iou_threshold": self.iou_thr.value(),
            },
        }

    def set_status(self, status: str, detail: str = "") -> None:
        colors = {
            "READY": "gray",
            "INITIALIZING": "orange",
            "TRACKING": "limegreen",
            "RE-DETECTING": "gold",
            "LOST": "red",
            "COMPLETED": "deepskyblue",
            "ERROR": "crimson",
        }
        self.status_label.setText(status)
        self.status_label.setStyleSheet(
            f"font-weight: bold; color: {colors.get(status, 'white')}; font-size: 16px;"
        )
        if detail:
            self.info_label.setText(detail)
