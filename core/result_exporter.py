"""Export annotated video, CSV, JSON, failed frames, trails."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.logger import get_logger
from core.types import STATUS_COLORS, FrameResult, TrackingStatus

logger = get_logger("grt.export")


def draw_overlay(
    frame: np.ndarray,
    result: FrameResult,
    trail: Optional[Sequence[Tuple[float, float]]] = None,
    draw_bbox: bool = True,
    draw_polygon: bool = True,
    draw_center: bool = True,
    draw_trail: bool = True,
) -> np.ndarray:
    out = frame.copy()
    try:
        status = TrackingStatus(result.status)
    except ValueError:
        status = TrackingStatus.ERROR
    color = STATUS_COLORS.get(status, (255, 255, 255))

    if draw_polygon and result.polygon_points:
        pts = np.array(result.polygon_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

    if draw_bbox and result.bbox_width > 0 and result.bbox_height > 0:
        x, y = int(result.bbox_x), int(result.bbox_y)
        w, h = int(result.bbox_width), int(result.bbox_height)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)

    if draw_center:
        cx, cy = int(result.center_x), int(result.center_y)
        cv2.circle(out, (cx, cy), 4, color, -1, cv2.LINE_AA)

    if draw_trail and trail and len(trail) > 1:
        for i in range(1, len(trail)):
            p1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            p2 = (int(trail[i][0]), int(trail[i][1]))
            cv2.line(out, p1, p2, (0, 255, 255), 1, cv2.LINE_AA)

    lines = [
        f"Frame: {result.frame_id}",
        f"Status: {result.status}",
        f"Method: {result.tracking_method}",
        f"Conf: {result.confidence:.2f}",
        f"Inliers: {result.inlier_count}",
        f"Scale: {result.scale:.2f}",
        f"FPS: {result.process_fps:.1f}",
        f"Infer: {result.inference_ms:.1f}ms",
    ]
    y0 = 24
    for i, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text,
            (10, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


class ResultExporter:
    def __init__(self, output_dir: str, failed_frames_dir: str, config: Optional[Dict[str, Any]] = None):
        self.output_dir = Path(output_dir)
        self.failed_frames_dir = Path(failed_frames_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failed_frames_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.results: List[FrameResult] = []
        self._writer: Optional[cv2.VideoWriter] = None
        self._video_path: Optional[Path] = None
        self._session = datetime.now().strftime("%Y%m%d_%H%M%S")

    def start_video(self, path: Optional[str], width: int, height: int, fps: float) -> Path:
        codec = self.config.get("export", {}).get("video_codec", "mp4v")
        if path:
            self._video_path = Path(path)
        else:
            self._video_path = self.output_dir / f"annotated_{self._session}.mp4"
        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self._video_path), fourcc, max(fps, 1.0), (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for {self._video_path}")
        return self._video_path

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def close_video(self) -> Optional[Path]:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self._video_path

    def add_result(self, result: FrameResult) -> None:
        self.results.append(result)

    def save_failed_frame(self, frame: np.ndarray, frame_id: int) -> Path:
        path = self.failed_frames_dir / f"lost_{self._session}_{frame_id:06d}.jpg"
        cv2.imwrite(str(path), frame)
        return path

    def save_roi_crop(self, frame: np.ndarray, result: FrameResult) -> Optional[Path]:
        if not result.polygon_points:
            return None
        from core.roi_selector import extract_roi_crop

        try:
            crop, _ = extract_roi_crop(frame, np.array(result.polygon_points, dtype=np.float32))
        except Exception:  # noqa: BLE001
            return None
        path = self.output_dir / "roi_crops" / f"roi_{self._session}_{result.frame_id:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        return path

    def export_csv(self, path: Optional[str] = None) -> Path:
        out = Path(path) if path else self.output_dir / f"results_{self._session}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "frame_id",
            "timestamp",
            "center_x",
            "center_y",
            "bbox_x",
            "bbox_y",
            "bbox_width",
            "bbox_height",
            "polygon_points",
            "confidence",
            "tracking_method",
            "inlier_count",
            "reprojection_error",
            "scale",
            "status",
        ]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.results:
                row = {k: getattr(r, k) for k in fields}
                row["polygon_points"] = json.dumps(r.polygon_points)
                writer.writerow(row)
        logger.info("CSV exported: %s (%d rows)", out, len(self.results))
        return out

    def export_center_path(self, path: Optional[str] = None) -> Path:
        out = Path(path) if path else self.output_dir / f"center_path_{self._session}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "timestamp", "center_x", "center_y", "status"])
            for r in self.results:
                writer.writerow([r.frame_id, r.timestamp, r.center_x, r.center_y, r.status])
        return out

    def export_json(
        self,
        settings: Dict[str, Any],
        stats: Dict[str, Any],
        path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> Path:
        out = Path(path) if path else self.output_dir / f"summary_{self._session}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now().isoformat(),
            "video_path": video_path,
            "settings": settings,
            "stats": stats,
            "num_frames": len(self.results),
            "results_preview": [r.to_dict() for r in self.results[:5]],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("JSON summary exported: %s", out)
        return out

    def finalize(
        self,
        settings: Dict[str, Any],
        stats: Dict[str, Any],
        video_path: Optional[str] = None,
    ) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        v = self.close_video()
        if v:
            paths["annotated_video"] = str(v)
        if self.config.get("export", {}).get("save_csv", True):
            paths["csv"] = str(self.export_csv())
        if self.config.get("export", {}).get("save_center_path", True):
            paths["center_path"] = str(self.export_center_path())
        if self.config.get("export", {}).get("save_json", True):
            paths["json"] = str(self.export_json(settings, stats, video_path=video_path))
        return paths
