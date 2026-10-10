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
from jetson_runtime.gui.image_controls import edit_isp_settings, load_isp_settings
from shared.video_reader import frame_lacks_scene
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
    QThread,
    QTimer,
    QVBoxLayout,
    Signal,
    qt_exec,
)


class _OpenCameraWorker(QThread):
    """Open camera off the UI thread so Argus/GStreamer cannot freeze the dialog."""

    finished_ok = Signal(object, object)  # FrameSource, first BGR frame
    finished_err = Signal(str)

    def __init__(self, spec: InputSourceSpec, parent=None):
        super().__init__(parent)
        self._spec = spec

    def run(self) -> None:
        source = None
        try:
            # Allow blank frames here so the user can see purple fog / lens issues.
            source = open_input_source(self._spec, require_scene=False)
            ok, frame = source.read()
            if not ok or frame is None:
                raise RuntimeError(f"No frame from {self._spec.label()}")
            self.finished_ok.emit(source, frame)
        except Exception as exc:  # noqa: BLE001
            if source is not None:
                try:
                    source.release()
                except Exception:  # noqa: BLE001
                    pass
            self.finished_err.emit(str(exc))


class CameraTestDialog(QDialog):
    """Open CSI/USB camera and show a live preview to verify the feed."""

    _OPEN_TIMEOUT_MS = 12000

    def __init__(self, parent=None, initial: Optional[InputSourceSpec] = None):
        super().__init__(parent)
        self.setWindowTitle("Camera test / تست دوربین")
        self.setMinimumSize(720, 520)
        self._source = None
        self._frame_count = 0
        self._chosen: Optional[InputSourceSpec] = None
        self._initial = initial
        self._worker: Optional[_OpenCameraWorker] = None
        self._open_spec: Optional[InputSourceSpec] = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._open_watchdog = QTimer(self)
        self._open_watchdog.setSingleShot(True)
        self._open_watchdog.timeout.connect(self._on_open_timeout)
        self._build()
        self._apply_initial()
        self._on_kind_changed()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        tip = QLabel(
            "Start the camera to verify the live image (no detection).\n"
            "دوربین را روشن کنید و تصویر زنده را ببینید — بدون تشخیص هدف.\n"
            "You must see a real scene before Run. Purple/gray fog or a flat "
            "field means lens cap / focus / no light — detection will never lock.\n"
            "صفحه بنفش/خاکستری = درپوش لنز، فوکوس، یا نور کم (مدل مقصر نیست)."
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
        self.sensor_id.setToolTip("CSI sensor-id / /dev/videoN (usually 0)")
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
        self.flip.setToolTip("nvvidconv flip-method (0 = none; Argus only)")
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
        self.btn_isp = QPushButton("Camera image settings… / تنظیمات تصویر")
        self.btn_isp.setToolTip("IMX477 white balance, exposure, gain, noise reduction — camera restarts to apply")
        self.btn_isp.clicked.connect(self._edit_isp)
        row.addWidget(self.btn_isp)
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
        busy = self._timer.isActive() or (self._worker is not None and self._worker.isRunning())
        is_csi = kind == KIND_IMX477
        self.btn_isp.setEnabled(is_csi)
        self.sensor_id.setEnabled(is_csi and not busy)
        self.usb_index.setEnabled(not is_csi and not busy)
        self.flip.setEnabled(is_csi and not busy)
        if is_csi and self.width.value() < 640:
            self.width.setValue(1920)
            self.height.setValue(1080)
            self.fps.setValue(60)
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
            isp=load_isp_settings() if kind == KIND_IMX477 else {},
        )

    def _edit_isp(self) -> None:
        """Change CSI image settings; a running camera re-opens so the preview shows them."""
        if edit_isp_settings(self) is None:
            return
        running = self._timer.isActive() or (self._worker is not None and self._worker.isRunning())
        if running and self.kind.currentData() == KIND_IMX477:
            self._start()

    def _start(self) -> None:
        self._stop()
        spec = self._spec()
        self._open_spec = spec
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_controls_enabled(False)
        self.status.setText(f"Opening {spec.label()}…")
        self.view.setText("Opening camera…")

        worker = _OpenCameraWorker(spec, self)
        worker.finished_ok.connect(self._on_open_ok)
        worker.finished_err.connect(self._on_open_err)
        self._worker = worker
        self._open_watchdog.start(self._OPEN_TIMEOUT_MS)
        worker.start()

    def _on_open_ok(self, source, frame) -> None:
        self._open_watchdog.stop()
        self._worker = None
        self._source = source
        self._frame_count = 0
        self._show_frame(frame)
        self._timer.start()
        label = self._open_spec.label() if self._open_spec else "camera"
        backend = ""
        try:
            backend = getattr(source.reader, "backend", "") or ""
        except Exception:  # noqa: BLE001
            pass
        suffix = f" [{backend}]" if backend else ""
        if frame_lacks_scene(frame):
            self.status.setText(
                f"Live: {label}{suffix} — WARNING: blank/fog frame (no scene)"
            )
            self.status.setStyleSheet("color:#fbbf24;")
        else:
            self.status.setStyleSheet("")
            self.status.setText(f"Live: {label}{suffix}")

    def _on_open_err(self, message: str) -> None:
        self._open_watchdog.stop()
        self._worker = None
        self._release_source()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_controls_enabled(True)
        self.view.setText("Camera off / دوربین خاموش")
        self.status.setText(f"Failed: {message}")
        QMessageBox.critical(self, "Camera", message)

    def _on_open_timeout(self) -> None:
        worker = self._worker
        if worker is None or not worker.isRunning():
            return
        self.status.setText("Open timed out — stopping…")
        # Cannot safely kill OpenCV mid-open; detach and reset UI.
        try:
            worker.finished_ok.disconnect(self._on_open_ok)
            worker.finished_err.disconnect(self._on_open_err)
        except Exception:  # noqa: BLE001
            pass
        worker.setParent(None)
        self._worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_controls_enabled(True)
        self.view.setText("Camera off / دوربین خاموش")
        msg = (
            "Camera open timed out (often broken Argus/GPU).\n"
            "App will skip Argus when nvidia-smi has no device and use V4L2.\n"
            "Restart the app and try again, or reboot if GPU stays down."
        )
        self.status.setText("Open timed out")
        QMessageBox.warning(self, "Camera", msg)

    def _stop(self) -> None:
        self._open_watchdog.stop()
        self._timer.stop()
        worker = self._worker
        if worker is not None:
            try:
                worker.finished_ok.disconnect(self._on_open_ok)
                worker.finished_err.disconnect(self._on_open_err)
            except Exception:  # noqa: BLE001
                pass
            if worker.isRunning():
                worker.setParent(None)
            else:
                worker.deleteLater()
            self._worker = None
        self._release_source()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_controls_enabled(True)
        self.view.setText("Camera off / دوربین خاموش")
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
        # Downscale huge CSI V4L2 frames before Qt convert (e.g. 3840x2160).
        h0, w0 = frame_bgr.shape[:2]
        max_w = max(640, self.view.width())
        if w0 > max_w:
            scale = max_w / float(w0)
            frame_bgr = cv2.resize(
                frame_bgr,
                (int(w0 * scale), int(h0 * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy()).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)
        if self._frame_count % 15 == 0:
            self.status.setText(f"Live {w0}x{h0} — frame {self._frame_count}")

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
