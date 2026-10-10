"""Video / camera input sources for Jetson runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from shared.video_reader import VideoInfo, VideoReader


KIND_FILE = "file"
KIND_IMX477 = "imx477"
KIND_USB = "usb"

VALID_KINDS = (KIND_FILE, KIND_IMX477, KIND_USB)


def describe_csi_hardware() -> str:
    """Short CSI/Argus status for the input dialog and open errors."""
    videos = sorted(p.name for p in Path("/dev").glob("video*"))
    sensors: list[str] = []
    root = Path("/proc/device-tree")
    if root.is_dir():
        for tag in ("imx219", "imx477", "imx708", "ov5693"):
            sensors.extend(p.name for p in root.rglob(f"*{tag}*"))
    sensors = list(dict.fromkeys(sensors))
    if sensors and videos:
        shown = ", ".join(sensors[:4])
        nodes = ", ".join(f"/dev/{n}" for n in videos[:4])
        return f"CSI OK: {shown}; {nodes}"
    if sensors and not videos:
        return (
            "Sensor in device tree but no /dev/video*. Restart nvargus-daemon, "
            "or check the CSI ribbon (contacts toward the board)."
        )
    return (
        "Argus: no CSI camera (no overlay, no /dev/video*). "
        "On Orin Nano run: sudo /opt/nvidia/jetson-io/jetson-io.py → "
        "Configure 24pin CSI → IMX219 or IMX477 (CAM0=A, CAM1=C) → Save and reboot."
    )


@dataclass
class InputSourceSpec:
    """How to open the runtime video input."""

    kind: str = KIND_FILE
    path: Optional[str] = None
    sensor_id: int = 0
    usb_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    flip_method: int = 0
    prefer_gstreamer: bool = False

    def label(self) -> str:
        if self.kind == KIND_FILE:
            return self.path or "(no file)"
        if self.kind == KIND_IMX477:
            return f"IMX477 CSI (sensor-id={self.sensor_id})"
        if self.kind == KIND_USB:
            return f"USB webcam (index={self.usb_index})"
        return self.kind

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sensor_id": self.sensor_id,
            "usb_index": self.usb_index,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "flip_method": self.flip_method,
            "prefer_gstreamer": self.prefer_gstreamer,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "InputSourceSpec":
        data = data or {}
        kind = str(data.get("kind") or KIND_FILE).lower()
        if kind not in VALID_KINDS:
            raise ValueError(f"Unknown input kind {kind!r}; expected one of {VALID_KINDS}")
        return cls(
            kind=kind,
            path=data.get("path"),
            sensor_id=int(data.get("sensor_id", 0)),
            usb_index=int(data.get("usb_index", data.get("camera_index", 0))),
            width=int(data.get("width", 1920 if kind == KIND_IMX477 else 1280)),
            height=int(data.get("height", 1080 if kind == KIND_IMX477 else 720)),
            fps=int(data.get("fps", 30)),
            flip_method=int(data.get("flip_method", 0)),
            prefer_gstreamer=bool(data.get("prefer_gstreamer", False)),
        )

    @classmethod
    def from_file(cls, path: str, *, prefer_gstreamer: bool = False) -> "InputSourceSpec":
        return cls(kind=KIND_FILE, path=path, prefer_gstreamer=prefer_gstreamer)


@dataclass
class FrameSource:
    """Uniform seek/read wrapper around file or live VideoReader."""

    reader: VideoReader
    is_live: bool = False
    label: str = ""

    @property
    def info(self) -> Optional[VideoInfo]:
        return self.reader.info

    def seek(self, idx: int) -> Tuple[bool, Optional[Any]]:
        if self.is_live:
            return self.read()
        return self.reader.seek(idx)

    def read(self) -> Tuple[bool, Optional[Any]]:
        return self.reader.read()

    def release(self) -> None:
        self.reader.release()


def open_input_source(spec: InputSourceSpec) -> FrameSource:
    """Open file / IMX477 CSI / USB webcam according to *spec*."""
    kind = spec.kind
    if kind == KIND_FILE:
        if not spec.path:
            raise ValueError("File input requires a video path")
        reader = VideoReader(spec.path, prefer_gstreamer=spec.prefer_gstreamer)
        return FrameSource(reader=reader, is_live=False, label=spec.label())

    if kind == KIND_IMX477:
        try:
            reader = VideoReader.from_imx477(
                sensor_id=spec.sensor_id,
                width=spec.width,
                height=spec.height,
                fps=spec.fps,
                flip_method=spec.flip_method,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} {describe_csi_hardware()}") from exc
        return FrameSource(reader=reader, is_live=True, label=spec.label())

    if kind == KIND_USB:
        reader = VideoReader.from_camera(
            spec.usb_index,
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
        )
        return FrameSource(reader=reader, is_live=True, label=spec.label())

    raise ValueError(f"Unknown input kind: {kind}")
