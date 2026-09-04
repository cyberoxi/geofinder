"""Desktop project manager: portable folder + SQLite annotations + project.json."""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from shared.geometry import is_valid_roi, polygon_area, polygon_to_bbox
from shared.schemas import Annotation, AnnotationSource, ProjectMeta, VideoAsset
from shared.video_reader import VideoReader


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    roi_type TEXT NOT NULL,
    polygon_points TEXT NOT NULL,
    bounding_box TEXT NOT NULL,
    annotation_source TEXT NOT NULL,
    zoom_level REAL NOT NULL DEFAULT 1.0,
    quality_score REAL NOT NULL DEFAULT 1.0,
    class_name TEXT NOT NULL DEFAULT 'target_region',
    UNIQUE(video_id, frame_id, annotation_source)
);
CREATE INDEX IF NOT EXISTS idx_ann_video_frame ON annotations(video_id, frame_id);
"""


class ProjectManager:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.meta_path = self.root / "project.json"
        self.db_path = self.root / "annotations.db"
        self.videos_dir = self.root / "videos"
        self.datasets_dir = self.root / "datasets"
        self.runs_dir = self.root / "runs"
        self.packages_dir = self.root / "packages"
        self.exports_dir = self.root / "exports"
        self.meta: Optional[ProjectMeta] = None
        self._conn: Optional[sqlite3.Connection] = None

    @classmethod
    def create(cls, root: str | Path, project_name: str, target_class: str = "target_region") -> "ProjectManager":
        pm = cls(root)
        pm.root.mkdir(parents=True, exist_ok=True)
        for d in (pm.videos_dir, pm.datasets_dir, pm.runs_dir, pm.packages_dir, pm.exports_dir):
            d.mkdir(exist_ok=True)
        pm.meta = ProjectMeta(project_name=project_name, target_class=target_class)
        pm.save_meta()
        pm._open_db()
        return pm

    @classmethod
    def open(cls, root: str | Path) -> "ProjectManager":
        pm = cls(root)
        if not pm.meta_path.exists():
            raise FileNotFoundError(f"project.json not found in {pm.root}")
        with open(pm.meta_path, "r", encoding="utf-8") as f:
            pm.meta = ProjectMeta.from_dict(json.load(f))
        pm._open_db()
        return pm

    def _open_db(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._open_db()
        assert self._conn is not None
        return self._conn

    def save_meta(self) -> None:
        if self.meta is None:
            raise RuntimeError("No project meta")
        self.meta.updated_at = _now()
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta.to_dict(), f, indent=2, ensure_ascii=False)

    def add_video(self, source_path: str | Path, copy_file: bool = True) -> VideoAsset:
        if self.meta is None:
            raise RuntimeError("Project not loaded")
        src = Path(source_path).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        video_id = uuid.uuid4().hex[:12]
        dest_name = f"{video_id}_{src.name}"
        dest = self.videos_dir / dest_name
        if copy_file:
            shutil.copy2(src, dest)
        else:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src)

        reader = VideoReader(str(dest), prefer_gstreamer=False)
        assert reader.info is not None
        asset = VideoAsset(
            video_id=video_id,
            relative_path=f"videos/{dest_name}",
            fps=reader.info.fps,
            width=reader.info.width,
            height=reader.info.height,
            frame_count=reader.info.frame_count,
            duration=reader.info.duration,
            display_name=src.name,
        )
        reader.release()
        self.meta.videos.append(asset)
        self.save_meta()
        return asset

    def resolve_video(self, video_id: str) -> Path:
        if self.meta is None:
            raise RuntimeError("Project not loaded")
        for v in self.meta.videos:
            if v.video_id == video_id:
                return self.root / v.relative_path
        raise KeyError(video_id)

    def get_video(self, video_id: str) -> VideoAsset:
        assert self.meta
        for v in self.meta.videos:
            if v.video_id == video_id:
                return v
        raise KeyError(video_id)

    def upsert_annotation(self, ann: Annotation) -> int:
        if not is_valid_roi(ann.polygon_points):
            raise ValueError("Invalid annotation ROI")
        pts_json = json.dumps(ann.polygon_points)
        bbox_json = json.dumps(ann.bounding_box)
        cur = self.conn.execute(
            """
            INSERT INTO annotations(
                video_id, frame_id, timestamp, image_width, image_height, roi_type,
                polygon_points, bounding_box, annotation_source, zoom_level, quality_score, class_name
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(video_id, frame_id, annotation_source) DO UPDATE SET
                timestamp=excluded.timestamp,
                image_width=excluded.image_width,
                image_height=excluded.image_height,
                roi_type=excluded.roi_type,
                polygon_points=excluded.polygon_points,
                bounding_box=excluded.bounding_box,
                zoom_level=excluded.zoom_level,
                quality_score=excluded.quality_score,
                class_name=excluded.class_name
            """,
            (
                ann.video_id,
                ann.frame_id,
                ann.timestamp,
                ann.image_width,
                ann.image_height,
                ann.roi_type,
                pts_json,
                bbox_json,
                ann.annotation_source,
                ann.zoom_level,
                ann.quality_score,
                ann.class_name,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def delete_annotation(self, annotation_id: int) -> None:
        self.conn.execute("DELETE FROM annotations WHERE id=?", (annotation_id,))
        self.conn.commit()

    def delete_frame_annotations(self, video_id: str, frame_id: int, sources: Optional[Sequence[str]] = None) -> None:
        if sources:
            q = ",".join("?" * len(sources))
            self.conn.execute(
                f"DELETE FROM annotations WHERE video_id=? AND frame_id=? AND annotation_source IN ({q})",
                (video_id, frame_id, *sources),
            )
        else:
            self.conn.execute("DELETE FROM annotations WHERE video_id=? AND frame_id=?", (video_id, frame_id))
        self.conn.commit()

    def list_annotations(self, video_id: Optional[str] = None, include_interpolated: bool = True) -> List[Annotation]:
        if video_id:
            rows = self.conn.execute(
                "SELECT * FROM annotations WHERE video_id=? ORDER BY frame_id", (video_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM annotations ORDER BY video_id, frame_id").fetchall()
        out = []
        for r in rows:
            src = r["annotation_source"]
            if not include_interpolated and src == AnnotationSource.INTERPOLATED.value:
                continue
            out.append(
                Annotation(
                    annotation_id=r["id"],
                    video_id=r["video_id"],
                    frame_id=r["frame_id"],
                    timestamp=r["timestamp"],
                    image_width=r["image_width"],
                    image_height=r["image_height"],
                    roi_type=r["roi_type"],
                    polygon_points=json.loads(r["polygon_points"]),
                    bounding_box=json.loads(r["bounding_box"]),
                    annotation_source=src,
                    zoom_level=r["zoom_level"],
                    quality_score=r["quality_score"],
                    class_name=r["class_name"],
                )
            )
        return out

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
