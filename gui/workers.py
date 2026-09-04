"""Background workers for non-blocking GUI processing."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from gui.qt_compat import QMutex, QMutexLocker, QThread, Signal

from core.pipeline import run_pipeline
from core.types import ROI
from core.video_reader import VideoReader


class VideoLoadWorker(QThread):
    loaded = Signal(object)  # VideoReader or error handled via failed
    failed = Signal(str)
    frame_ready = Signal(object, int)  # frame, index

    def __init__(self, path: str, prefer_gstreamer: bool = True, parent=None):
        super().__init__(parent)
        self.path = path
        self.prefer_gstreamer = prefer_gstreamer

    def run(self) -> None:
        try:
            reader = VideoReader(self.path, prefer_gstreamer=self.prefer_gstreamer)
            ok, frame = reader.seek(0)
            self.loaded.emit(reader)
            if ok and frame is not None:
                self.frame_ready.emit(frame, 0)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ProcessingWorker(QThread):
    progress = Signal(int, int, dict)
    frame_updated = Signal(object, dict)
    finished_ok = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        video_path: str,
        roi: Optional[ROI],
        config: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.roi = roi
        self.config = config
        self._stop = False
        self._mutex = QMutex()

    def request_stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._stop = True

    def _should_stop(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._stop

    def run(self) -> None:
        try:
            self.status.emit("INITIALIZING")

            def progress_cb(cur, total, info):
                self.progress.emit(cur, total, info)

            def frame_cb(frame, info):
                self.frame_updated.emit(frame, info)
                self.status.emit(str(info.get("status", "TRACKING")))

            result = run_pipeline(
                video_path=self.video_path,
                roi=self.roi,
                config=self.config,
                progress_cb=progress_cb,
                stop_flag=self._should_stop,
                frame_cb=frame_cb,
            )
            self.status.emit("COMPLETED")
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.status.emit("ERROR")
            self.failed.emit(str(exc))
