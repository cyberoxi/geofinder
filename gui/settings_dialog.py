"""Full settings dialog."""

from __future__ import annotations

from typing import Any, Dict

from gui.qt_compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.config import deep_update


class SettingsDialog(QDialog):
    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 480)
        self._config = config
        self._result = None

        tabs = QTabWidget()
        tabs.addTab(self._tracking_tab(), "Tracking")
        tabs.addTab(self._yolo_tab(), "YOLO / Device")
        tabs.addTab(self._export_tab(), "Export")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    def _tracking_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        t = self._config.get("tracking", {})
        self.lowe = QDoubleSpinBox()
        self.lowe.setRange(0.5, 0.95)
        self.lowe.setSingleStep(0.05)
        self.lowe.setValue(float(t.get("lowe_ratio", 0.75)))
        form.addRow("Lowe ratio", self.lowe)

        self.max_reproj = QDoubleSpinBox()
        self.max_reproj.setRange(0.5, 20.0)
        self.max_reproj.setValue(float(t.get("max_reprojection_error", 4.0)))
        form.addRow("Max reproj error", self.max_reproj)

        self.keyframe = QSpinBox()
        self.keyframe.setRange(1, 200)
        self.keyframe.setValue(int(t.get("keyframe_interval", 15)))
        form.addRow("Keyframe interval", self.keyframe)

        self.match_interval = QSpinBox()
        self.match_interval.setRange(1, 100)
        self.match_interval.setValue(int(t.get("feature_match_interval", 5)))
        form.addRow("Feature match interval", self.match_interval)

        self.min_scale = QDoubleSpinBox()
        self.min_scale.setRange(0.01, 1.0)
        self.min_scale.setValue(float(t.get("min_scale", 0.15)))
        form.addRow("Min scale", self.min_scale)

        self.max_scale = QDoubleSpinBox()
        self.max_scale.setRange(1.0, 20.0)
        self.max_scale.setValue(float(t.get("max_scale", 8.0)))
        form.addRow("Max scale", self.max_scale)

        self.max_area = QDoubleSpinBox()
        self.max_area.setRange(1.0, 20.0)
        self.max_area.setValue(float(t.get("max_area_change", 4.0)))
        form.addRow("Max area change", self.max_area)

        self.max_features = QSpinBox()
        self.max_features.setRange(100, 10000)
        self.max_features.setValue(int(t.get("max_features", 2000)))
        form.addRow("Max features", self.max_features)

        self.lost_thr = QSpinBox()
        self.lost_thr.setRange(1, 200)
        self.lost_thr.setValue(int(t.get("lost_frames_threshold", 15)))
        form.addRow("Lost frames thr", self.lost_thr)

        self.smooth_method = QComboBox()
        self.smooth_method.addItems(["ema", "kalman"])
        self.smooth_method.setCurrentText(self._config.get("smoothing", {}).get("method", "ema"))
        form.addRow("Smoothing", self.smooth_method)
        return w

    def _yolo_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        y = self._config.get("yolo", {})
        row = QHBoxLayout()
        self.model_path = QLineEdit(y.get("model_path", "models/best.engine"))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_model)
        row.addWidget(self.model_path)
        row.addWidget(browse)
        form.addRow("Model path", row)

        self.fp16 = QCheckBox("FP16")
        self.fp16.setChecked(bool(y.get("use_fp16", True)))
        form.addRow(self.fp16)

        self.int8 = QCheckBox("INT8 (optional)")
        self.int8.setChecked(bool(y.get("use_int8", False)))
        form.addRow(self.int8)

        self.prefer_gst = QCheckBox("Prefer GStreamer")
        self.prefer_gst.setChecked(bool(self._config.get("video", {}).get("prefer_gstreamer", True)))
        form.addRow(self.prefer_gst)
        return w

    def _export_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        e = self._config.get("export", {})
        self.save_video = QCheckBox("Annotated video")
        self.save_video.setChecked(bool(e.get("save_annotated_video", True)))
        self.save_csv = QCheckBox("CSV")
        self.save_csv.setChecked(bool(e.get("save_csv", True)))
        self.save_json = QCheckBox("JSON")
        self.save_json.setChecked(bool(e.get("save_json", True)))
        self.save_failed = QCheckBox("Failed frames")
        self.save_failed.setChecked(bool(e.get("save_failed_frames", True)))
        self.save_crops = QCheckBox("ROI crops")
        self.save_crops.setChecked(bool(e.get("save_roi_crops", False)))
        for cb in (self.save_video, self.save_csv, self.save_json, self.save_failed, self.save_crops):
            form.addRow(cb)
        return w

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO model",
            "",
            "Models (*.engine *.onnx *.pt);;All (*)",
        )
        if path:
            self.model_path.setText(path)

    def accept(self) -> None:
        override = {
            "tracking": {
                "lowe_ratio": self.lowe.value(),
                "max_reprojection_error": self.max_reproj.value(),
                "keyframe_interval": self.keyframe.value(),
                "feature_match_interval": self.match_interval.value(),
                "min_scale": self.min_scale.value(),
                "max_scale": self.max_scale.value(),
                "max_area_change": self.max_area.value(),
                "max_features": self.max_features.value(),
                "lost_frames_threshold": self.lost_thr.value(),
            },
            "smoothing": {"method": self.smooth_method.currentText()},
            "yolo": {
                "model_path": self.model_path.text(),
                "use_fp16": self.fp16.isChecked(),
                "use_int8": self.int8.isChecked(),
            },
            "video": {"prefer_gstreamer": self.prefer_gst.isChecked()},
            "export": {
                "save_annotated_video": self.save_video.isChecked(),
                "save_csv": self.save_csv.isChecked(),
                "save_json": self.save_json.isChecked(),
                "save_failed_frames": self.save_failed.isChecked(),
                "save_roi_crops": self.save_crops.isChecked(),
            },
        }
        self._result = deep_update(self._config, override)
        super().accept()

    def result_config(self) -> Dict[str, Any]:
        return self._result or self._config
