"""Render zoom-flyover and pan videos from a satellite mosaic."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import cv2
import numpy as np

from shared.logging import get_logger

logger = get_logger("desktop.satellite.video")

ATTRIBUTION = "Esri World Imagery"
ProgressCb = Callable[[int, int, str], None]
FrameCb = Callable[[np.ndarray], None]


def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _write_video(
    frames: Iterable[np.ndarray],
    output: Path,
    fps: float,
    size: Tuple[int, int],
    *,
    total_frames: Optional[int] = None,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
    progress_label: str = "Rendering",
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writers = [
        ("mp4v", output),
        ("avc1", output.with_suffix(".mp4")),
        ("XVID", output.with_suffix(".avi")),
    ]
    writer = None
    out_path = output
    for fourcc_name, path in writers:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        candidate = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if candidate.isOpened():
            writer = candidate
            out_path = path
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"Could not open VideoWriter for {output}")

    count = 0
    total = total_frames or 0
    preview_every = max(1, (total // 30) if total else 5)
    for frame in frames:
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1
        if frame_cb and (count == 1 or count % preview_every == 0 or (total and count == total)):
            frame_cb(frame.copy())
        if progress_cb and total and (count == 1 or count % 5 == 0 or count == total):
            progress_cb(count, total, f"{progress_label} {count}/{total}")
    writer.release()
    if count == 0:
        raise RuntimeError("No frames written")
    logger.info("Wrote %d frames -> %s", count, out_path)
    return out_path


def _overlay_attribution(frame: np.ndarray, text: str = ATTRIBUTION) -> np.ndarray:
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 8
    x, y = pad, out.shape[0] - pad
    cv2.rectangle(out, (x - 4, y - th - 6), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _crop_resize(
    mosaic: np.ndarray,
    cx: float,
    cy: float,
    crop_w: float,
    crop_h: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    h, w = mosaic.shape[:2]
    crop_w = max(8.0, min(crop_w, float(w)))
    crop_h = max(8.0, min(crop_h, float(h)))
    x0 = int(round(cx - crop_w / 2))
    y0 = int(round(cy - crop_h / 2))
    x0 = max(0, min(x0, w - int(crop_w)))
    y0 = max(0, min(y0, h - int(crop_h)))
    x1 = min(w, x0 + int(round(crop_w)))
    y1 = min(h, y0 + int(round(crop_h)))
    patch = mosaic[y0:y1, x0:x1]
    return cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def render_zoom_video(
    mosaic: np.ndarray,
    output: str | Path,
    *,
    duration_s: float = 8.0,
    fps: float = 24.0,
    out_size: Tuple[int, int] = (1280, 720),
    end_scale: float = 1.0 / 12.0,
    center: Optional[Tuple[float, float]] = None,
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
) -> Path:
    """
    Zoom-in flyover: starts on the full mosaic (wide view) and shrinks the crop
    toward ``center`` with ease-in-out until ``end_scale`` of the start crop.
    Pass end_scale ≈ 1/zoom_out_factor when the mosaic was expanded around the
    target region so the final frame matches that selection.
    """
    h, w = mosaic.shape[:2]
    out_w, out_h = out_size
    aspect = out_w / out_h
    cx, cy = center if center else (w / 2.0, h / 2.0)

    # Start: largest crop fitting mosaic with output aspect
    if w / h >= aspect:
        start_h = float(h)
        start_w = start_h * aspect
    else:
        start_w = float(w)
        start_h = start_w / aspect

    end_h = max(32.0, min(start_h * end_scale, h * 0.95))
    end_w = end_h * aspect
    if end_w > w:
        end_w = float(w) * 0.95
        end_h = end_w / aspect

    n = max(2, int(round(duration_s * fps)))

    def frames():
        for i in range(n):
            t = _ease_in_out(i / (n - 1))
            cw = start_w + (end_w - start_w) * t
            ch = start_h + (end_h - start_h) * t
            frame = _crop_resize(mosaic, cx, cy, cw, ch, out_w, out_h)
            if i == 0 or i == n - 1:
                frame = _overlay_attribution(frame)
            yield frame

    return _write_video(
        frames(),
        Path(output),
        fps,
        out_size,
        total_frames=n,
        progress_cb=progress_cb,
        frame_cb=frame_cb,
        progress_label="Rendering zoom",
    )


def render_pan_video(
    mosaic: np.ndarray,
    output: str | Path,
    *,
    duration_s: float = 8.0,
    fps: float = 24.0,
    out_size: Tuple[int, int] = (1280, 720),
    view_scale: float = 0.45,
    direction: str = "lr",
    progress_cb: Optional[ProgressCb] = None,
    frame_cb: Optional[FrameCb] = None,
) -> Path:
    """
    Horizontal (or diagonal) pan at fixed zoom across the mosaic.
    direction: 'lr' left-to-right, 'diag' NW-to-SE.
    """
    h, w = mosaic.shape[:2]
    out_w, out_h = out_size
    aspect = out_w / out_h

    if w / h >= aspect:
        full_h = float(h)
        full_w = full_h * aspect
    else:
        full_w = float(w)
        full_h = full_w / aspect

    crop_h = max(32.0, full_h * view_scale)
    crop_w = crop_h * aspect
    if crop_w > w:
        crop_w = float(w) * 0.95
        crop_h = crop_w / aspect

    margin_x = max(0.0, (w - crop_w) / 2)
    margin_y = max(0.0, (h - crop_h) / 2)

    if direction == "diag":
        start = (crop_w / 2, crop_h / 2)
        end = (w - crop_w / 2, h - crop_h / 2)
    else:
        start = (crop_w / 2, h / 2.0)
        end = (w - crop_w / 2, h / 2.0)
        # Keep y within safe band
        start = (start[0], max(crop_h / 2, min(start[1], h - crop_h / 2)))
        end = (end[0], max(crop_h / 2, min(end[1], h - crop_h / 2)))

    # Clamp start/end so crop stays inside
    def clamp_c(cx: float, cy: float) -> Tuple[float, float]:
        return (
            max(crop_w / 2, min(cx, w - crop_w / 2)),
            max(crop_h / 2, min(cy, h - crop_h / 2)),
        )

    start = clamp_c(*start)
    end = clamp_c(*end)
    n = max(2, int(round(duration_s * fps)))

    def frames():
        for i in range(n):
            t = _ease_in_out(i / (n - 1))
            cx = start[0] + (end[0] - start[0]) * t
            cy = start[1] + (end[1] - start[1]) * t
            frame = _crop_resize(mosaic, cx, cy, crop_w, crop_h, out_w, out_h)
            if i == 0 or i == n - 1:
                frame = _overlay_attribution(frame)
            yield frame

    _ = margin_x, margin_y  # retained for readability / future padding
    return _write_video(
        frames(),
        Path(output),
        fps,
        out_size,
        total_frames=n,
        progress_cb=progress_cb,
        frame_cb=frame_cb,
        progress_label="Rendering pan",
    )
