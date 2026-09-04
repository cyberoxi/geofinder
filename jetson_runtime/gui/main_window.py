"""Jetson Runtime GUI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.logging import setup_logging
from shared.qt_compat import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
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
from jetson_runtime.runtime.pipeline import run_runtime


class ProcessWorker(QThread):
    progress = Signal(int, int, dict)
    frame_updated = Signal(object, dict)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, package: str, video: str):
        super().__init__()
        self.package = package
        self.video = video
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
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class JetsonMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jetson Orin Nano Runtime")
        self.resize(1280, 800)
        self.package_path: Optional[str] = None
        self.video_path: Optional[str] = None
        self.reader: Optional[VideoReader] = None
        self.worker: Optional[ProcessWorker] = None
        self._build()

    def _build(self):
        m = self.menuBar().addMenu("File")
        m.addAction(QAction("Import Package…", self, triggered=self.import_package))
        m.addAction(QAction("Open Video…", self, triggered=self.open_video))
        m.addAction(QAction("Open RTSP…", self, triggered=lambda: QMessageBox.information(self, "RTSP", "RTSP reserved for future release.")))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split)

        self.view = QLabel("Import a package and open a video")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setStyleSheet("background:#111;color:#aaa;")
        split.addWidget(self.view)

        side = QWidget()
        sv = QVBoxLayout(side)
        self.info = QLabel("No package")
        self.info.setWordWrap(True)
        sv.addWidget(self.info)
        self.status = QLabel("READY")
        self.status.setStyleSheet("font-size:18px;font-weight:bold;color:gray;")
        sv.addWidget(self.status)
        for text, slot in [
            ("Import Package", self.import_package),
            ("Open Video", self.open_video),
            ("Start Processing", self.start),
            ("Stop Processing", self.stop),
            ("Reset", self.reset),
        ]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            sv.addWidget(b)
        sv.addStretch(1)
        split.addWidget(side)

        tl = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._seek)
        self.frame_spin = QSpinBox()
        tl.addWidget(self.slider)
        tl.addWidget(self.frame_spin)
        layout.addLayout(tl)
        self.statusBar().showMessage("Ready")

    def import_package(self):
        path, _ = QFileDialog.getOpenFileName(self, "Deployment package", "", "ZIP (*.zip);;All (*)")
        if not path:
            return
        self.package_path = path
        self.info.setText(f"Package:\n{path}")
        self.statusBar().showMessage(f"Package selected: {path}")

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
        self.worker = ProcessWorker(self.package_path, self.video_path)
        self.worker.frame_updated.connect(lambda fr, info: self._show(fr, info))
        self.worker.progress.connect(lambda c, t, info: self.statusBar().showMessage(f"{c}/{t} {info.get('status')}"))
        self.worker.finished_ok.connect(lambda r: QMessageBox.information(self, "Done", json.dumps(r.get("paths", {}), indent=2)))
        self.worker.failed.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()
        self.status.setText("PROCESSING")

    def stop(self):
        if self.worker:
            self.worker.request_stop()

    def reset(self):
        self.stop()
        self.status.setText("READY")
        if self.reader:
            self._seek(0)

    def closeEvent(self, e):  # noqa: N802
        self.stop()
        if self.reader:
            self.reader.release()
        e.accept()


def main():
    setup_logging()
    app = QApplication(sys.argv)
    win = JetsonMainWindow()
    win.show()
    return qt_exec(app)


if __name__ == "__main__":
    raise SystemExit(main())
