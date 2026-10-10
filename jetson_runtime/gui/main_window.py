"""Jetson Runtime GUI."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.icons import get_icon
from shared.logging import setup_logging
from shared.package_utils import validate_package
from shared.theme import apply_theme, style_brand_label
from shared.usb_transfer import (
    USB_PACKAGE_DIR,
    find_usb_packages,
    import_package_from_usb,
)
from shared.qt_compat import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSize,
    QSlider,
    QSpinBox,
    QSplitter,
    QTimer,
    QToolBar,
    Qt,
    QThread,
    QVBoxLayout,
    QWidget,
    QAction,
    QImage,
    QPixmap,
    Signal,
    qt_exec,
)
from shared.video_reader import VideoReader
from jetson_runtime.gui.camera_test_dialog import open_camera_test
from jetson_runtime.gui.confidence_control import ConfidenceControl
from jetson_runtime.gui.serial_dialog import edit_serial_settings, load_serial_settings
from jetson_runtime.gui.source_dialog import choose_input_source
from jetson_runtime.runtime.pipeline import run_runtime
from jetson_runtime.runtime.sources import KIND_FILE, InputSourceSpec, open_input_source


class ProcessWorker(QThread):
    progress = Signal(int, int, dict)
    frame_updated = Signal(object, dict)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        package: str,
        source_spec: InputSourceSpec,
        serial_cfg: Optional[dict] = None,
        min_conf_cb=None,
    ):
        super().__init__()
        self.package = package
        self.source_spec = source_spec
        self.serial_cfg = serial_cfg
        self.min_conf_cb = min_conf_cb
        self._stop = False
        self._last_ui = 0.0
        self._last_progress = 0.0

    def request_stop(self):
        self._stop = True

    def _emit_frame(self, fr, info: dict) -> None:
        # Cap preview refresh so Qt paint/scale does not starve the detect thread.
        now = time.perf_counter()
        if now - self._last_ui < 1.0 / 18.0:
            return
        self._last_ui = now
        h, w = fr.shape[:2]
        max_w = 1280
        if w > max_w:
            fr = cv2.resize(fr, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
        self.frame_updated.emit(fr, info)

    def _emit_progress(self, c: int, t: int, info: dict) -> None:
        now = time.perf_counter()
        if now - self._last_progress < 0.2 and c < t:
            return
        self._last_progress = now
        self.progress.emit(c, t, info)

    def run(self):
        try:
            result = run_runtime(
                self.package,
                source=self.source_spec,
                prefer_gstreamer=False,
                progress_cb=self._emit_progress,
                frame_cb=self._emit_frame,
                stop_flag=lambda: self._stop,
                serial_override=self.serial_cfg,
                min_conf=self.min_conf_cb() if self.min_conf_cb else None,
                min_conf_cb=self.min_conf_cb,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class JetsonMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("سامانه تنگسیری")
        self.resize(1280, 800)
        self.package_path: Optional[str] = None
        self.input_spec: Optional[InputSourceSpec] = None
        self.reader: Optional[VideoReader] = None
        self.worker: Optional[ProcessWorker] = None
        self.local_packages_dir = Path.home() / "geofinder_packages"
        self.local_packages_dir.mkdir(parents=True, exist_ok=True)
        self._imported_usb_keys: Set[str] = set()
        self._usb_busy = False
        self._build()
        self._start_usb_watcher()

    def _tb_action(self, text: str, icon_name: str, slot, tip: str = "") -> QAction:
        act = QAction(get_icon(icon_name), text, self)
        act.triggered.connect(slot)
        if tip:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        return act

    def _build(self):
        m = self.menuBar().addMenu("&File")
        m.addAction(self._tb_action("Import Package…", "package", self.import_package))
        m.addAction(self._tb_action("Scan USB…", "usb", self.scan_usb_now))
        m.addAction(self._tb_action("Input Source…", "open", self.choose_input))
        m.addAction(self._tb_action("Open Video File…", "open", self.open_video))
        m.addAction(
            self._tb_action(
                "Camera Test…",
                "add_video",
                self.test_camera,
                "Start CSI/USB camera and show live preview",
            )
        )
        m.addAction(
            QAction(
                "Open RTSP…",
                self,
                triggered=lambda: QMessageBox.information(self, "RTSP", "RTSP reserved for future release."),
            )
        )
        m.addSeparator()
        m.addAction(QAction("Serial settings…", self, triggered=self.open_serial_settings))

        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(22, 22))
        tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(tb)
        brand = QLabel()
        style_brand_label(brand)
        tb.addWidget(brand)
        tb.addSeparator()
        tb.addAction(self._tb_action("Package", "package", self.import_package, "Import deployment package"))
        tb.addAction(self._tb_action("USB", "usb", self.scan_usb_now, "Scan USB for trained package"))
        tb.addAction(
            self._tb_action(
                "Input",
                "open",
                self.choose_input,
                "Video file / IMX477 CSI / USB webcam",
            )
        )
        tb.addAction(
            self._tb_action(
                "Cam Test",
                "add_video",
                self.test_camera,
                "Start camera and show live preview",
            )
        )
        tb.addSeparator()
        tb.addAction(self._tb_action("Run", "run", self.start, "Start processing"))
        tb.addAction(self._tb_action("Stop", "stop_proc", self.stop, "Stop processing"))
        tb.addAction(self._tb_action("Reset", "reset", self.reset, "Reset to first frame"))
        tb.addSeparator()
        tb.addAction(
            self._tb_action("Serial", "settings", self.open_serial_settings, "Serial port for dx,dy offset")
        )

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split)

        self.view = QLabel("Import a package and choose input source")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setStyleSheet(
            "background:#0b1220;color:#64748b;border:1px solid #243041;border-radius:8px;"
        )
        split.addWidget(self.view)

        side = QWidget()
        side.setMaximumWidth(320)
        sv = QVBoxLayout(side)
        self.info = QLabel("No package\nWaiting for USB…")
        self.info.setWordWrap(True)
        self.info.setStyleSheet(
            "background:#151b24;border:1px solid #243041;border-radius:8px;padding:10px;"
        )
        sv.addWidget(self.info)
        self.status = QLabel("READY")
        self.status.setStyleSheet("font-size:18px;font-weight:bold;color:#94a3b8;")
        sv.addWidget(self.status)
        self.usb_status = QLabel(f"USB: watching for {USB_PACKAGE_DIR}/pkg_*.zip")
        self.usb_status.setWordWrap(True)
        self.usb_status.setStyleSheet("color:#94a3b8;font-size:12px;")
        sv.addWidget(self.usb_status)
        for text, icon_name, slot in [
            ("Import Package", "package", self.import_package),
            ("Scan USB", "usb", self.scan_usb_now),
            ("Input Source", "open", self.choose_input),
            ("Camera Test", "add_video", self.test_camera),
            ("Serial settings", "settings", self.open_serial_settings),
            ("Start Processing", "run", self.start),
            ("Stop Processing", "stop_proc", self.stop),
            ("Reset", "reset", self.reset),
        ]:
            b = QPushButton(f"  {text}")
            b.setIcon(get_icon(icon_name))
            b.setIconSize(QSize(16, 16))
            b.clicked.connect(slot)
            if "Start" in text:
                b.setObjectName("primaryButton")
            sv.addWidget(b)
        sv.addStretch(1)
        split.addWidget(side)

        # Same live threshold as the desktop Live Test (drag while processing)
        self.conf = ConfidenceControl()
        layout.addWidget(self.conf)

        tl = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._seek)
        self.frame_spin = QSpinBox()
        self.frame_spin.setFixedWidth(90)
        tl.addWidget(self.slider, stretch=1)
        tl.addWidget(QLabel("Frame"))
        tl.addWidget(self.frame_spin)
        layout.addLayout(tl)
        self._refresh_serial_status()

    def open_serial_settings(self) -> None:
        edit_serial_settings(self)
        self._refresh_serial_status()

    def _refresh_serial_status(self) -> None:
        cfg = load_serial_settings()
        if cfg.get("enabled"):
            self.statusBar().showMessage(
                f"Serial ON — {cfg.get('port')} @ {cfg.get('baudrate')} (dx,dy = target − image centre)"
            )
        else:
            self.statusBar().showMessage("Ready — USB watcher active · Serial OFF")

    def _start_usb_watcher(self) -> None:
        self._usb_timer = QTimer(self)
        self._usb_timer.setInterval(3000)
        self._usb_timer.timeout.connect(self._poll_usb_packages)
        self._usb_timer.start()
        QTimer.singleShot(800, self._poll_usb_packages)

    def scan_usb_now(self) -> None:
        imported = self._poll_usb_packages(force_notify=True)
        if not imported and not self._usb_busy:
            QMessageBox.information(
                self,
                "USB",
                f"No new package found on USB.\nLooking for {USB_PACKAGE_DIR}/pkg_*.zip",
            )

    def _usb_key(self, path: Path) -> str:
        try:
            st = path.stat()
            return f"{path.name}:{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            return path.name

    def _poll_usb_packages(self, force_notify: bool = False) -> bool:
        if self._usb_busy:
            return False
        self._usb_busy = True
        try:
            packages = find_usb_packages()
            self.usb_status.setText(
                f"USB: {len(packages)} package(s) seen — watching {USB_PACKAGE_DIR}/"
            )
            for usb_zip in packages:
                key = self._usb_key(usb_zip)
                if key in self._imported_usb_keys:
                    continue
                try:
                    local = import_package_from_usb(usb_zip, self.local_packages_dir)
                    result = validate_package(local)
                    if not result.get("ok"):
                        self._imported_usb_keys.add(key)
                        self.usb_status.setText(f"USB: invalid package {usb_zip.name}")
                        if force_notify:
                            QMessageBox.warning(
                                self,
                                "USB",
                                f"Found {usb_zip.name} but validation failed.",
                            )
                        continue
                    self._imported_usb_keys.add(key)
                    self._set_package(str(local), from_usb=True, source=usb_zip)
                    QMessageBox.information(
                        self,
                        "USB",
                        "Package transferred from USB.\n\n"
                        f"Source: {usb_zip}\n"
                        f"Saved to: {local}\n\n"
                        "منتقل شد / Transferred successfully",
                    )
                    return True
                except Exception as exc:  # noqa: BLE001
                    self._imported_usb_keys.add(key)
                    self.usb_status.setText(f"USB error: {exc}")
                    if force_notify:
                        QMessageBox.warning(self, "USB", str(exc))
            return False
        finally:
            self._usb_busy = False

    def _refresh_info_panel(self, *, package_note: str = "") -> None:
        lines: list[str] = []
        if self.package_path:
            lines.append(f"Package:\n{self.package_path}")
        else:
            lines.append("No package\nWaiting for USB…")
        if package_note:
            lines.append(package_note)
        if self.input_spec is not None:
            lines.append(f"\nInput:\n{self.input_spec.label()}")
            if self.reader and self.input_spec.kind == KIND_FILE:
                lines.append(f"\n{self.reader.width}x{self.reader.height} @ {self.reader.fps:.1f}")
            elif self.input_spec.kind != KIND_FILE:
                lines.append(
                    f"\nCapture {self.input_spec.width}x{self.input_spec.height}"
                    f" @ {self.input_spec.fps} fps"
                )
        self.info.setText("".join(lines))

    def _set_package(self, path: str, *, from_usb: bool = False, source: Optional[Path] = None) -> None:
        self.package_path = path
        self.conf.set_package(path)
        note = ""
        if from_usb and source is not None:
            note = f"\nImported from USB:\n{source}\nStatus: transferred"
        self._refresh_info_panel(package_note=note)
        self.statusBar().showMessage(
            f"{'USB transfer OK' if from_usb else 'Package selected'}: {path}"
        )
        if from_usb:
            self.status.setText("USB IMPORTED")
            self.status.setStyleSheet("font-size:18px;font-weight:bold;color:#22c55e;")

    def import_package(self):
        path, _ = QFileDialog.getOpenFileName(self, "Deployment package", "", "ZIP (*.zip);;All (*)")
        if not path:
            return
        self._set_package(path)

    def choose_input(self):
        spec = choose_input_source(self, initial=self.input_spec)
        if spec is None:
            return
        self._apply_input_spec(spec)

    def test_camera(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Camera Test",
                "Stop processing before testing the camera.\n"
                "قبل از تست دوربین، پردازش را متوقف کنید.",
            )
            return
        # Release any open file reader so the camera can claim the device.
        if self.reader:
            self.reader.release()
            self.reader = None
        initial = self.input_spec
        if initial is not None and initial.kind == KIND_FILE:
            initial = None
        spec = open_camera_test(self, initial=initial)
        if spec is None:
            return
        self._apply_input_spec(spec)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Video", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if not path:
            return
        self._apply_input_spec(InputSourceSpec.from_file(path, prefer_gstreamer=False))

    def _apply_input_spec(self, spec: InputSourceSpec) -> None:
        if self.reader:
            self.reader.release()
            self.reader = None
        try:
            if spec.kind == KIND_FILE:
                if not spec.path:
                    raise ValueError("No video path")
                self.reader = VideoReader(spec.path, prefer_gstreamer=False)
                n = max((self.reader.frame_count or 1) - 1, 0)
                self.slider.setEnabled(True)
                self.frame_spin.setEnabled(True)
                self.slider.setRange(0, n)
                self.frame_spin.setRange(0, n)
                self.input_spec = spec
                self._seek(0)
            else:
                src = open_input_source(spec)
                try:
                    ok, frame = src.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"No frame from {spec.label()}")
                    self.slider.setEnabled(False)
                    self.frame_spin.setEnabled(False)
                    self.slider.setRange(0, 0)
                    self.frame_spin.setRange(0, 0)
                    self.input_spec = spec
                    self._show(frame)
                finally:
                    src.release()
        except Exception as exc:  # noqa: BLE001
            self.input_spec = None
            QMessageBox.critical(self, "Input", str(exc))
            return
        self._refresh_info_panel()
        self.statusBar().showMessage(f"Input ready: {spec.label()}")

    def _seek(self, v: int):
        if not self.reader:
            return
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(v)
        self.frame_spin.blockSignals(False)
        ok, frame = self.reader.seek(v)
        if ok and frame is not None:
            self._show(frame)

    def _show(self, frame_bgr, info=None):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        # FastTransformation: SmoothTransformation was a large per-frame cost on Jetson
        self.view.setPixmap(pix.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        if info:
            self.status.setText(str(info.get("status", "")))
            color = {"TRACKING": "limegreen", "LOST": "red", "RE_DETECTING": "gold", "SEARCHING": "orange", "SCALE_CHANGE": "deepskyblue"}.get(
                info.get("status"), "white"
            )
            self.status.setStyleSheet(f"font-size:18px;font-weight:bold;color:{color};")

    def start(self):
        if not self.package_path or self.input_spec is None:
            QMessageBox.warning(
                self,
                "Run",
                "Import package and choose input source first\n"
                "(video file / IMX477 / USB webcam)",
            )
            return
        # Release preview capture so the worker can reopen the same camera.
        if self.reader:
            self.reader.release()
            self.reader = None
        serial_cfg = load_serial_settings()
        self.worker = ProcessWorker(
            self.package_path, self.input_spec, serial_cfg=serial_cfg, min_conf_cb=self.conf.fraction
        )
        self.worker.frame_updated.connect(lambda fr, info: self._show(fr, info))
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()
        self.status.setText("PROCESSING")

    def _on_finished(self, result: dict) -> None:
        QMessageBox.information(self, "Done", json.dumps(result.get("paths", {}), indent=2))
        # Re-open file preview after a file run
        if self.input_spec and self.input_spec.kind == KIND_FILE and self.input_spec.path:
            try:
                self.reader = VideoReader(self.input_spec.path, prefer_gstreamer=False)
                self._seek(0)
            except Exception:  # noqa: BLE001
                pass

    def _on_progress(self, c: int, t: int, info: dict) -> None:
        dx = info.get("offset_dx")
        dy = info.get("offset_dy")
        live = self.input_spec is not None and self.input_spec.kind != KIND_FILE
        prog = f"{c}" if live or t <= 0 else f"{c}/{t}"
        if info.get("min_conf") is not None:
            prog += f"  conf≥{info['min_conf'] * 100:.0f}%"
        if dx is not None and dy is not None:
            self.statusBar().showMessage(
                f"{prog} {info.get('status')}  offset dx={dx:.0f} dy={dy:.0f}"
            )
        else:
            self.statusBar().showMessage(f"{prog} {info.get('status')}")

    def stop(self):
        if self.worker:
            self.worker.request_stop()

    def reset(self):
        self.stop()
        self.status.setText("READY")
        if self.reader and self.input_spec and self.input_spec.kind == KIND_FILE:
            self._seek(0)

    def closeEvent(self, e):  # noqa: N802
        if hasattr(self, "_usb_timer"):
            self._usb_timer.stop()
        self.stop()
        if self.reader:
            self.reader.release()
        e.accept()


def main():
    setup_logging()
    from shared.app_icon import apply_app_icon, ensure_windows_app_id

    ensure_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("سامانه تنگسیری")
    apply_theme(app)
    win = JetsonMainWindow()
    apply_app_icon(app, win)
    win.show()
    return qt_exec(app)


if __name__ == "__main__":
    raise SystemExit(main())
