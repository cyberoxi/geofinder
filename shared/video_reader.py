"""Video reader with GStreamer preference and OpenCV fallback for Jetson."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from shared.logging import get_logger

logger = get_logger("shared.video")


def jetson_argus_available() -> bool:
    """True when NVIDIA GPU/CUDA is usable (Argus/NVMM pipelines need this)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False
    if "No devices" in out or "couldn't communicate" in out:
        return False
    return bool(proc.stdout.strip())


def jetson_gpu_power_hint() -> str:
    """Short hint when CSI needs GPU but nvidia-smi has no device."""
    powered = Path("/sys/devices/platform/bus@0/17000000.gpu/gpu_powered_on")
    rail = Path("/sys/devices/platform/bus@0/17000000.gpu/is_railgated")
    bits: list[str] = []
    try:
        if powered.is_file() and powered.read_text().strip() == "off":
            bits.append("GPU powered off")
        if rail.is_file() and rail.read_text().strip() == "yes":
            bits.append("railgated")
    except OSError:
        pass
    detail = f" ({', '.join(bits)})" if bits else ""
    return (
        f"Jetson GPU/CUDA unavailable{detail} — IMX477 needs Argus/ISP "
        "(not raw V4L2). Fix: sudo sh -c 'echo on > "
        "/sys/devices/platform/bus@0/17000000.gpu/power/control' "
        "then ensure /etc/ld.so.conf.d/000-nvidia-gpu-libs.conf points to "
        "openrm (not nvgpu), sudo ldconfig, reboot. Check: nvidia-smi -L"
    )


def _frame_is_solid_green(frame: np.ndarray) -> bool:
    """True for classic broken CSI preview (solid / near-solid green)."""
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        return False
    # Sample to keep this cheap on 4K frames.
    sample = frame[::16, ::16]
    b, g, r = sample[:, :, 0], sample[:, :, 1], sample[:, :, 2]
    if float(g.std()) > 8.0:
        return False
    return float(g.mean()) > 40.0 and float(b.mean()) < 15.0 and float(r.mean()) < 15.0


def frame_lacks_scene(frame: np.ndarray) -> bool:
    """True when the frame is green/blank/noise-only (no usable scene for YOLO).

    IMX477 sometimes streams a purple/gray fog with sensor grain when the lens
    is capped, completely defocused, or getting almost no light. Argus is up,
    so open succeeds, but detection always stays searching.
    """
    if frame is None or frame.ndim < 2:
        return True
    if _frame_is_solid_green(frame):
        return True
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    # Downscale so Canny cost stays tiny on 4K.
    h, w = gray.shape[:2]
    if max(h, w) > 480:
        scale = 480.0 / float(max(h, w))
        gray = cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    std = float(gray.std())
    if std < 10.0:
        return True
    edges = cv2.Canny(gray, 30, 100)
    # Real scenes (even a desk) have thousands of edge pixels at 480px; fog has ~0.
    return int(np.count_nonzero(edges)) < 400


def imx477_blank_frame_hint() -> str:
    return (
        "IMX477 stream has no scene content (blank / purple fog / noise only). "
        "YOLO cannot detect anything — this is not a model bug.\n"
        "تصویر دوربین خالی یا مه بنفش است؛ مدل چیزی برای تشخیص ندارد.\n\n"
        "Check: remove lens cap / protective film, turn the focus ring, "
        "point at a lit scene, then use Camera Test — you must see a real "
        "picture (desk, floor, screen), not a flat purple/gray field."
    )


def _normalize_imx477_mode(width: int, height: int, fps: int) -> Tuple[int, int, int]:
    """Snap to IMX477 Argus sensor modes: 1080p@60 or 4K@30 only."""
    w, h = int(width), int(height)
    if w >= 2560 or h >= 1440:
        return 3840, 2160, 30
    return 1920, 1080, 60


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


def imx477_gstreamer_pipeline(
    sensor_id: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    flip_method: int = 0,
) -> str:
    """Argus CSI pipeline for IMX477 (and similar) on Jetson."""
    width, height, fps = _normalize_imx477_mode(width, height, fps)
    return (
        f"nvarguscamerasrc sensor-id={int(sensor_id)} ! "
        f"video/x-raw(memory:NVMM), width=(int){int(width)}, height=(int){int(height)}, "
        f"framerate=(fraction){int(fps)}/1, format=(string)NV12 ! "
        f"nvvidconv flip-method={int(flip_method)} ! "
        f"video/x-raw, width=(int){int(width)}, height=(int){int(height)}, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )


def usb_v4l2_gstreamer_pipeline(
    index: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> str:
    """USB webcam via V4L2 + GStreamer (common on Jetson Linux)."""
    return (
        f"v4l2src device=/dev/video{int(index)} ! "
        f"video/x-raw, width=(int){int(width)}, height=(int){int(height)}, "
        f"framerate=(fraction){int(fps)}/1 ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
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
    def _blank_live(cls, path: str) -> "VideoReader":
        obj = cls.__new__(cls)
        obj.path = path
        obj._cap = None
        obj.backend = "camera"
        obj.info = None
        obj._current_index = -1
        obj._is_camera = True
        return obj

    @classmethod
    def from_camera(
        cls,
        index: int = 0,
        *,
        width: int = 0,
        height: int = 0,
        fps: int = 30,
    ) -> "VideoReader":
        """Open a USB webcam / V4L2 device by index (0 = default)."""
        obj = cls._blank_live(f"usb:{int(index)}")
        obj._open_usb_camera(int(index), width=width, height=height, fps=fps)
        return obj

    @classmethod
    def from_imx477(
        cls,
        sensor_id: int = 0,
        *,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        flip_method: int = 0,
        require_scene: bool = False,
    ) -> "VideoReader":
        """Open Jetson CSI IMX477 via Argus (requires working GPU/ISP)."""
        obj = cls._blank_live(f"imx477:{int(sensor_id)}")
        sid = int(sensor_id)
        width, height, fps = _normalize_imx477_mode(width, height, fps)

        if not jetson_argus_available():
            raise RuntimeError(jetson_gpu_power_hint())

        pipe = imx477_gstreamer_pipeline(
            sensor_id=sid,
            width=width,
            height=height,
            fps=fps,
            flip_method=flip_method,
        )
        if obj._try_open(pipe, cv2.CAP_GSTREAMER, rewind=False):
            obj.backend = "imx477-argus"
            # AE/AWB need a short warm-up; first frames are often purple fog.
            last = obj._warmup_live(n=25)
            obj._finish_live_info(width=width, height=height, fps=float(fps))
            if require_scene and (last is None or frame_lacks_scene(last)):
                obj.release()
                raise RuntimeError(imx477_blank_frame_hint())
            logger.info(
                "Opened IMX477 CSI sensor-id=%s %sx%s@%s (Argus)",
                sid,
                width,
                height,
                fps,
            )
            return obj

        raise RuntimeError(
            f"Cannot open CSI (sensor-id={sid}, {width}x{height}@{fps}). "
            "Argus reported no usable camera. Enable the matching IMX overlay "
            "in Jetson-IO and reboot; also try sensor-id 1 if the ribbon is on CAM1. "
            "Raw V4L2 is not used for IMX477 (shows a green screen without ISP)."
        )

    def _warmup_live(self, n: int = 20) -> Optional[np.ndarray]:
        """Discard early AE frames; return the last good BGR frame (or None)."""
        last: Optional[np.ndarray] = None
        for _ in range(max(0, int(n))):
            ok, frame = self.read()
            if not ok or frame is None:
                break
            last = frame
        return last

    def _try_open(self, source, api=cv2.CAP_ANY, *, rewind: bool = True) -> bool:
        cap = cv2.VideoCapture(source, api)
        if not cap.isOpened():
            cap.release()
            return False
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return False
        if _frame_is_solid_green(frame):
            cap.release()
            logger.warning("Rejected solid-green camera frame (broken CSI/ISP path)")
            return False
        if rewind:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._cap = cap
        return True

    def _finish_live_info(self, *, width: int = 0, height: int = 0, fps: float = 30.0) -> None:
        assert self._cap is not None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width or 640
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height or 480
        f = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if f <= 1e-3:
            f = float(fps) if fps > 1e-3 else 30.0
        self.info = VideoInfo(self.path, w, h, f, 0, self.backend)
        self._current_index = -1

    def _open_usb_camera(
        self,
        index: int,
        *,
        width: int = 0,
        height: int = 0,
        fps: int = 30,
    ) -> None:
        attempts: list[tuple[str, object, int]] = []
        if sys.platform == "linux":
            w = width or 1280
            h = height or 720
            # Prefer plain V4L2 — GStreamer/NV plugins can hang when GPU is down.
            attempts.append(("v4l2", index, getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)))
            if jetson_argus_available():
                attempts.append(
                    (
                        "gstreamer-v4l2",
                        usb_v4l2_gstreamer_pipeline(index, width=w, height=h, fps=fps),
                        cv2.CAP_GSTREAMER,
                    )
                )
        if sys.platform == "win32":
            attempts.append(("dshow", index, cv2.CAP_DSHOW))
        attempts.append(("any", index, cv2.CAP_ANY))

        opened = False
        for name, source, api in attempts:
            try:
                if self._try_open(source, api, rewind=False):
                    self.backend = f"camera-{name}"
                    opened = True
                    logger.info("Opened USB camera %s via %s", index, name)
                    break
            except Exception as exc:  # noqa: BLE001
                logger.debug("USB camera open failed (%s index=%s): %s", name, index, exc)

        if not opened:
            raise RuntimeError(
                f"Cannot open USB camera index {index} "
                "(not connected, wrong index, or in use by another app)"
            )

        assert self._cap is not None
        if width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._finish_live_info(width=width, height=height, fps=float(fps))

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
    def is_live(self) -> bool:
        return self._is_camera

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
        # Live cameras have no timeline — grab the next frame.
        if self._is_camera:
            return self.read()

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
