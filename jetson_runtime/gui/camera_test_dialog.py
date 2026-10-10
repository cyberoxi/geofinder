"""Live camera preview / test for Jetson (IMX477 CSI or USB webcam)."""

from __future__ import annotations

from typing import Optional

import cv2

from jetson_runtime.runtime.sources import (
    KIND_IMX477,
    KIND_USB,
    InputSourceSpec,
    open_input_source,
)
from shared.qt_compat import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QImage,
    QLabel,
    QMessageBox,
    QPixmap,
    QPushButton,
    QSpinBox,
    Qt,
    QTimer,
    QVBoxLayout,
    qt_exec,
)


class CameraTestDialog(QDialog):
    """Open CSI/USB camera and show a live preview to verify the feed."""

    def __init__(self, parent=None, initial: Optional[InputSourceSpec] = None):
        super().__init__(parent)
        self.setWindowTitle("Camera test / تست دوربین")
        self.setMinimumSize(720, 520)
        self._source = None
        self._frame_count = 0
        self._chosen: Optional[InputSourceSpec] = None
        self._initial = initial
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._build()
        self._apply_initial()
        self._on_kind_changed()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        tip = QLabel(
            "Start the camera to verify the live image (no detection).\n"
            "دوربین را روشن کنید و تصویر زنده را ببینید — بدون تشخیص هدف."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#94a3b8;")
        root.addWidget(tip)

        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItem("IMX477 CSI camera / دوربین IMX477", KIND_IMX477)
        self.kind.addItem("USB webcam / وب‌کم USB", KIND_USB)
        self.kind.currentIndexChanged.connect(self._on_kind_changed)
        form.addRow("Camera / دوربین:", self.kind)

        self.sensor_id = QSpinBox()
        self.sensor_id.setRange(0, 3)
        self.sensor_id.setToolTip("nvarguscamerasrc sensor-id (usually 0)")
        form.addRow("CSI sensor-id:", self.sensor_id)

        self.usb_index = QSpinBox()
        self.usb_index.setRange(0, 9)
        self.usb_index.setToolTip("/dev/videoN index — 0 = first USB webcam")
        form.addRow("USB camera index:", self.usb_index)

        self.width = QSpinBox()
        self.width.setRange(320, 4032)
        self.width.setSingleStep(16)
        self.width.setValue(1920)
        self.height = QSpinBox()
        self.height.setRange(240, 3040)
        self.height.setSingleStep(16)
        self.height.setValue(1080)
        self.fps = QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(30)
        form.addRow("Capture width:", self.width)
        form.addRow("Capture height:", self.height)
        form.addRow("Capture FPS:", self.fps)

        self.flip = QSpinBox()
        self.flip.setRange(0, 7)
        self.flip.setToolTip("nvvidconv flip-method (0 = none)")
        form.addRow("Flip method (CSI):", self.flip)
        root.addLayout(form)

        row = QHBoxLayout()
        self.btn_start = QPushButton("Start camera / روشن کردن")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton("Stop / توقف")
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        root.addLayout(row)

        self.view = QLabel("Camera off / دوربین خاموش")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setStyleSheet(
            "background:#0b1220;color:#64748b;border:1px solid #243041;border-radius:8px;"
        )
        root.addWidget(self.view, stretch=1)

        self.status = QLabel("Ready")
        self.status.setStyleSheet("color:#94a3b8;")
        root.addWidget(self.status)

        buttons = QDialogButtonBox()
        self.btn_use = buttons.addButton(
            "Use as input / استفاده به‌عنوان منبع", QDialogButtonBox.AcceptRole
        )
        buttons.addButton(QDialogButtonBox.Close)
        self.btn_use.clicked.connect(self._use_as_input)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _apply_initial(self) -> None:
        spec = self._initial
        if spec is None or spec.kind not in (KIND_IMX477, KIND_USB):
            return
        idx = self.kind.findData(spec.kind)
        if idx >= 0:
            self.kind.setCurrentIndex(idx)
        self.sensor_id.setValue(spec.sensor_id)
        self.usb_index.setValue(spec.usb_index)
        self.width.setValue(spec.width or 1920)
        self.height.setValue(spec.height or 1080)
        self.fps.setValue(spec.fps or 30)
        self.flip.setValue(spec.flip_method)

    def _on_kind_changed(self) -> None:
        kind = self.kind.currentData()
        is_csi = kind == KIND_IMX477
        self.sensor_id.setEnabled(is_csi and not self._timer.isActive())
        self.usb_index.setEnabled(not is_csi and not self._timer.isActive())
        self.flip.setEnabled(is_csi and not self._timer.isActive())
        if is_csi and self.width.value() < 640:
            self.width.setValue(1920)
            self.height.setValue(1080)
        if not is_csi and self.width.value() > 1920:
            self.width.setValue(1280)
            self.height.setValue(720)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.kind.setEnabled(enabled)
        self.width.setEnabled(enabled)
        self.height.setEnabled(enabled)
        self.fps.setEnabled(enabled)
        self._on_kind_changed()

    def _spec(self) -> InputSourceSpec:
        kind = str(self.kind.currentData())
        return InputSourceSpec(
            kind=kind,
            sensor_id=int(self.sensor_id.value()),
            usb_index=int(self.usb_index.value()),
            width=int(self.width.value()),
            height=int(self.height.value()),
            fps=int(self.fps.value()),
            flip_method=int(self.flip.value()),
            prefer_gstreamer=False,
        )

    def _start(self) -> None:
        self._stop()
        spec = self._spec()
        try:
            self._source = open_input_source(spec)
            ok, frame = self._source.read()
            if not ok or frame is None:
                raise RuntimeError(f"No frame from {spec.label()}")
        except Exception as exc:  # noqa: BLE001
            self._release_source()
            QMessageBox.critical(self, "Camera", str(exc))
            self.status.setText(f"Failed: {exc}")
            return
        self._frame_count = 0
        self._show_frame(frame)
        self._timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_controls_enabled(False)
        self.status.setText(f"Live: {spec.label()}")

    def _stop(self) -> None:
        self._timer.stop()
        self._release_source()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_controls_enabled(True)
        if self._frame_count:
            self.status.setText(f"Stopped — {self._frame_count} frame(s)")
        else:
            self.status.setText("Ready")

    def _release_source(self) -> None:
        if self._source is not None:
            try:
                self._source.release()
            except Exception:  # noqa: BLE001
                pass
            self._source = None

    def _tick(self) -> None:
        if self._source is None:
            return
        ok, frame = self._source.read()
        if not ok or frame is None:
            self._stop()
            self.status.setText("Lost camera stream")
            QMessageBox.warning(self, "Camera", "Lost camera stream / اتصال قطع شد")
            return
        self._show_frame(frame)

    def _show_frame(self, frame_bgr) -> None:
        self._frame_count += 1
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy()).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)
        if self._frame_count % 15 == 0:
            self.status.setText(f"Live {w}x{h} — frame {self._frame_count}")

    def _use_as_input(self) -> None:
        self._chosen = self._spec()
        self._stop()
        self.accept()

    def chosen_spec(self) -> Optional[InputSourceSpec]:
        return self._chosen

    def reject(self) -> None:
        self._stop()
        super().reject()

    def closeEvent(self, e) -> None:  # noqa: N802
        self._stop()
        e.accept()


def open_camera_test(
    parent=None, initial: Optional[InputSourceSpec] = None
) -> Optional[InputSourceSpec]:
    """Show camera test dialog. Returns spec if user chose «Use as input»."""
    dlg = CameraTestDialog(parent, initial=initial)
    if qt_exec(dlg) != QDialog.Accepted:
        return None
    return dlg.chosen_spec()
