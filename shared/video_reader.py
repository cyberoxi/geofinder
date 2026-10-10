"""Video reader with GStreamer preference and OpenCV fallback for Jetson."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from shared.logging import get_logger

logger = get_logger("shared.video")


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    backend: str

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def _gstreamer_pipeline(path: str) -> str:
    # Hardware-accelerated decode when available on Jetson; falls back gracefully.
    return (
        f"filesrc location={path} ! "
        "qtdemux ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false"
    )


def _soft_gstreamer_pipeline(path: str) -> str:
    return (
        f"filesrc location={path} ! decodebin ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=false"
    )


class VideoReader:
    """Seekable video reader with GStreamer / OpenCV backends (or live camera)."""

    def __init__(self, path: str, prefer_gstreamer: bool = True):
        self.path = str(Path(path).resolve())
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._cap: Optional[cv2.VideoCapture] = None
        self.backend = "opencv"
        self.info: Optional[VideoInfo] = None
        self._current_index = -1
        self._is_camera = False
        self._open(prefer_gstreamer)

    @classmethod
    def from_camera(cls, index: int = 0) -> "VideoReader":
        """Open a connected webcam / capture device by index (0 = default)."""
        obj = cls.__new__(cls)
        obj.path = f"camera:{int(index)}"
        obj._cap = None
        obj.backend = "camera"
        obj.info = None
        obj._current_index = -1
        obj._is_camera = True
        obj._open_camera(int(index))
        return obj

    def _try_open(self, source, api=cv2.CAP_ANY, *, rewind: bool = True) -> bool:
        cap = cv2.VideoCapture(source, api)
        if not cap.isOpened():
            cap.release()
            return False
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return False
        if rewind:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._cap = cap
        return True

    def _open_camera(self, index: int) -> None:
        apis: list[tuple[str, int]] = []
        if sys.platform == "win32":
            apis.append(("dshow", cv2.CAP_DSHOW))
        apis.append(("any", cv2.CAP_ANY))
        if sys.platform == "linux":
            apis.append(("v4l2", getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)))

        opened = False
        for name, api in apis:
            try:
                if self._try_open(index, api, rewind=False):
                    self.backend = f"camera-{name}"
                    opened = True
                    logger.info("Opened camera %s via %s", index, name)
                    break
            except Exception as exc:  # noqa: BLE001
                logger.debug("Camera open failed (%s index=%s): %s", name, index, exc)

        if not opened:
            raise RuntimeError(
                f"Cannot open camera index {index} (not connected or in use by another app)"
            )

        assert self._cap is not None
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 30.0
        self.info = VideoInfo(self.path, width, height, fps, 0, self.backend)
        self._current_index = -1

    def _open(self, prefer_gstreamer: bool) -> None:
        opened = False
        if prefer_gstreamer:
            for name, pipe, api in (
                ("gstreamer-nv", _gstreamer_pipeline(self.path), cv2.CAP_GSTREAMER),
                ("gstreamer-soft", _soft_gstreamer_pipeline(self.path), cv2.CAP_GSTREAMER),
            ):
                try:
                    if self._try_open(pipe, api):
                        self.backend = name
                        opened = True
                        logger.info("Opened video with %s: %s", name, self.path)
                        break
                except Exception as exc:  # noqa: BLE001
                    logger.debug("GStreamer open failed (%s): %s", name, exc)

        if not opened:
            if not self._try_open(self.path, cv2.CAP_FFMPEG):
                if not self._try_open(self.path):
                    raise RuntimeError(
                        f"Cannot open video (unsupported codec or corrupt file): {self.path}"
                    )
            self.backend = "opencv"
            logger.info("Opened video with OpenCV: %s", self.path)

        assert self._cap is not None
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 1e-3:
            fps = 30.0
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.info = VideoInfo(self.path, width, height, fps, frame_count, self.backend)
        self._current_index = -1

    @property
    def width(self) -> int:
        return self.info.width if self.info else 0

    @property
    def height(self) -> int:
        return self.info.height if self.info else 0

    @property
    def fps(self) -> float:
        return self.info.fps if self.info else 0.0

    @property
    def frame_count(self) -> int:
        return self.info.frame_count if self.info else 0

    @property
    def current_index(self) -> int:
        return self._current_index

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        self._current_index += 1
        return True, frame

    def seek(self, frame_index: int) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        frame_index = max(0, int(frame_index))
        if self.info and self.info.frame_count > 0:
            frame_index = min(frame_index, self.info.frame_count - 1)

        # GStreamer seek can be unreliable; use sequential read when needed.
        if self.backend.startswith("gstreamer"):
            if frame_index < self._current_index or self._current_index < 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._current_index = -1
            while self._current_index < frame_index:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    return False, None
                self._current_index += 1
            return True, frame

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        self._current_index = frame_index
        return True, frame

    def timestamp_of(self, frame_index: int) -> float:
        if self.fps <= 0:
            return 0.0
        return frame_index / self.fps

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args) -> None:
        self.release()
