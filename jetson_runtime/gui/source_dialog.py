"""Select Jetson runtime input: video file, IMX477 CSI, or USB webcam."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from jetson_runtime.runtime.sources import (
    KIND_FILE,
    KIND_IMX477,
    KIND_USB,
    InputSourceSpec,
    describe_csi_hardware,
)
from shared.qt_compat import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    qt_exec,
)


class InputSourceDialog(QDialog):
    """Pick file video / IMX477 CSI / USB webcam before Run."""

    def __init__(self, parent=None, initial: Optional[InputSourceSpec] = None):
        super().__init__(parent)
        self.setWindowTitle("Input source / منبع تصویر")
        self.setMinimumWidth(480)
        self._initial = initial or InputSourceSpec()
        self._build()
        self._apply_initial()
        self._on_kind_changed()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        tip = QLabel(
            "Choose where frames come from.\n"
            "فایل ویدیو، دوربین CSI (IMX477)، یا وب‌کم USB."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#94a3b8;")
        root.addWidget(tip)

        form = QFormLayout()
        self.kind = QComboBox()
        self.kind.addItem("Video file / فایل ویدیو", KIND_FILE)
        self.kind.addItem("IMX477 CSI camera / دوربین IMX477", KIND_IMX477)
        self.kind.addItem("USB webcam / وب‌کم USB", KIND_USB)
        self.kind.currentIndexChanged.connect(self._on_kind_changed)
        form.addRow("Source / منبع:", self.kind)

        path_row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("Path to .mp4 / .avi / …")
        path_row.addWidget(self.path, stretch=1)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self.btn_browse)
        form.addRow("Video file:", path_row)

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
        self.height = QSpinBox()
        self.height.setRange(240, 3040)
        self.height.setSingleStep(16)
        self.fps = QSpinBox()
        self.fps.setRange(1, 120)
        form.addRow("Capture width:", self.width)
        form.addRow("Capture height:", self.height)
        form.addRow("Capture FPS:", self.fps)

        self.flip = QSpinBox()
        self.flip.setRange(0, 7)
        self.flip.setToolTip("nvvidconv flip-method (0 = none)")
        form.addRow("Flip method (CSI):", self.flip)

        self.csi_status = QLabel(describe_csi_hardware())
        self.csi_status.setWordWrap(True)
        self.csi_status.setStyleSheet("color:#fbbf24;")
        form.addRow("CSI status:", self.csi_status)

        root.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _apply_initial(self) -> None:
        spec = self._initial
        idx = self.kind.findData(spec.kind)
        if idx >= 0:
            self.kind.setCurrentIndex(idx)
        if spec.path:
            self.path.setText(spec.path)
        self.sensor_id.setValue(spec.sensor_id)
        self.usb_index.setValue(spec.usb_index)
        self.width.setValue(spec.width or 1920)
        self.height.setValue(spec.height or 1080)
        self.fps.setValue(spec.fps or 30)
        self.flip.setValue(spec.flip_method)

    def _on_kind_changed(self) -> None:
        kind = self.kind.currentData()
        is_file = kind == KIND_FILE
        is_csi = kind == KIND_IMX477
        is_usb = kind == KIND_USB
        self.path.setEnabled(is_file)
        self.btn_browse.setEnabled(is_file)
        self.sensor_id.setEnabled(is_csi)
        self.usb_index.setEnabled(is_usb)
        self.flip.setEnabled(is_csi)
        # Resolution applies to live cameras; file uses its own size.
        self.width.setEnabled(not is_file)
        self.height.setEnabled(not is_file)
        self.fps.setEnabled(not is_file)
        if is_csi and self.width.value() < 640:
            self.width.setValue(1920)
            self.height.setValue(1080)
        if is_usb and self.width.value() > 1920 and self._initial.kind != KIND_USB:
            self.width.setValue(1280)
            self.height.setValue(720)

    def _browse(self) -> None:
        start = self.path.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Video",
            start,
            "Videos (*.mp4 *.avi *.mkv *.mov);;All (*)",
        )
        if path:
            self.path.setText(path)

    def spec(self) -> InputSourceSpec:
        kind = str(self.kind.currentData())
        path = self.path.text().strip() or None
        if kind == KIND_FILE and not path:
            raise ValueError("Select a video file")
        return InputSourceSpec(
            kind=kind,
            path=path,
            sensor_id=int(self.sensor_id.value()),
            usb_index=int(self.usb_index.value()),
            width=int(self.width.value()),
            height=int(self.height.value()),
            fps=int(self.fps.value()),
            flip_method=int(self.flip.value()),
            prefer_gstreamer=False,
        )


def choose_input_source(
    parent=None, initial: Optional[InputSourceSpec] = None
) -> Optional[InputSourceSpec]:
    from shared.qt_compat import QMessageBox

    dlg = InputSourceDialog(parent, initial=initial)
    if qt_exec(dlg) != QDialog.Accepted:
        return None
    try:
        return dlg.spec()
    except ValueError as exc:
        QMessageBox.warning(parent, "Input", str(exc))
        return None
