"""Jetson Runtime GUI."""

from __future__ import annotations

import json
import sys
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
from jetson_runtime.gui.serial_dialog import edit_serial_settings, load_serial_settings
from jetson_runtime.runtime.pipeline import run_runtime


class ProcessWorker(QThread):
    progress = Signal(int, int, dict)
    frame_updated = Signal(object, dict)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, package: str, video: str, serial_cfg: Optional[dict] = None):
        super().__init__()
        self.package = package
        self.video = video
        self.serial_cfg = serial_cfg
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            result = run_runtime(
                self.package,
                self.video,
                prefer_gstreamer=False,
                progress_cb=lambda c, t, info: self.progress.emit(c, t, info),
                frame_cb=lambda fr, info: self.frame_updated.emit(fr, info),
                stop_flag=lambda: self._stop,
                serial_override=self.serial_cfg,
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
        self.video_path: Optional[str] = None
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
        m.addAction(self._tb_action("Open Video…", "open", self.open_video))
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
        tb.addAction(self._tb_action("Video", "open", self.open_video, "Open video"))
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

        self.view = QLabel("Import a package and open a video")
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
            ("Open Video", "open", self.open_video),
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

    def _set_package(self, path: str, *, from_usb: bool = False, source: Optional[Path] = None) -> None:
        self.package_path = path
        lines = [f"Package:\n{path}"]
        if from_usb and source is not None:
            lines.append(f"\nImported from USB:\n{source}")
            lines.append("\nStatus: transferred")
        video_line = ""
        if self.video_path:
            video_line = f"\nVideo: {self.video_path}"
        self.info.setText("".join(lines) + video_line)
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

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Video", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if not path:
            return
        self.video_path = path
        if self.reader:
            self.reader.release()
        self.reader = VideoReader(path, prefer_gstreamer=False)
        n = max((self.reader.frame_count or 1) - 1, 0)
        self.slider.setRange(0, n)
        self.frame_spin.setRange(0, n)
        self._seek(0)
        self.info.setText(self.info.text() + f"\nVideo: {path}\n{self.reader.width}x{self.reader.height} @ {self.reader.fps:.1f}")

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
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(qimg.copy()).scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if info:
            self.status.setText(str(info.get("status", "")))
            color = {"TRACKING": "limegreen", "LOST": "red", "RE_DETECTING": "gold", "SEARCHING": "orange", "SCALE_CHANGE": "deepskyblue"}.get(
                info.get("status"), "white"
            )
            self.status.setStyleSheet(f"font-size:18px;font-weight:bold;color:{color};")

    def start(self):
        if not self.package_path or not self.video_path:
            QMessageBox.warning(self, "Run", "Import package and open video first")
            return
        serial_cfg = load_serial_settings()
        self.worker = ProcessWorker(self.package_path, self.video_path, serial_cfg=serial_cfg)
        self.worker.frame_updated.connect(lambda fr, info: self._show(fr, info))
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(lambda r: QMessageBox.information(self, "Done", json.dumps(r.get("paths", {}), indent=2)))
        self.worker.failed.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()
        self.status.setText("PROCESSING")

    def _on_progress(self, c: int, t: int, info: dict) -> None:
        dx = info.get("offset_dx")
        dy = info.get("offset_dy")
        if dx is not None and dy is not None:
            self.statusBar().showMessage(
                f"{c}/{t} {info.get('status')}  offset dx={dx:.0f} dy={dy:.0f}"
            )
        else:
            self.statusBar().showMessage(f"{c}/{t} {info.get('status')}")

    def stop(self):
        if self.worker:
            self.worker.request_stop()

    def reset(self):
        self.stop()
        self.status.setText("READY")
        if self.reader:
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
    app = QApplication(sys.argv)
    app.setApplicationName("سامانه تنگسیری")
    apply_theme(app)
    win = JetsonMainWindow()
    win.show()
    return qt_exec(app)


if __name__ == "__main__":
    raise SystemExit(main())
