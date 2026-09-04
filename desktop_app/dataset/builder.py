"""Build YOLO dataset from project annotations (video-based split)."""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from desktop_app.project_manager.manager import ProjectManager
from shared.geometry import (
    is_valid_roi,
    polygon_area,
    polygon_to_yolo_bbox,
    polygon_to_yolo_seg,
)
from shared.schemas import Annotation, AnnotationSource
from shared.video_reader import VideoReader


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _phash(gray: np.ndarray, size: int = 16) -> str:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()


def _augment(image: np.ndarray, points: np.ndarray, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    img = image.copy()
    pts = points.astype(np.float32).copy()
    h, w = img.shape[:2]

    # Brightness / contrast
    alpha = rng.uniform(0.7, 1.3)
    beta = rng.uniform(-30, 30)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:
        noise = rng.randint(5, 20)
        img = np.clip(img.astype(np.int16) + rng.randint(-noise, noise, img.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        # Simple shadow rectangle
        x0, y0 = rng.randint(0, w // 2), rng.randint(0, h // 2)
        x1, y1 = rng.randint(w // 2, w), rng.randint(h // 2, h)
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1].astype(np.float32) * 0.55).astype(np.uint8)
    if rng.random() < 0.3:
        angle = rng.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        ones = np.ones((len(pts), 1), np.float32)
        pts = (M @ np.hstack([pts, ones]).T).T
    if rng.random() < 0.35:
        scale = rng.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        ones = np.ones((len(pts), 1), np.float32)
        pts = (M @ np.hstack([pts, ones]).T).T

    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return img, pts


def _split_videos(video_ids: List[str], seed: int = 42) -> Dict[str, str]:
    ids = sorted(video_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    if n == 1:
        return {ids[0]: "train"}
    if n == 2:
        return {ids[0]: "train", ids[1]: "val"}
    n_test = max(1, int(round(n * 0.1)))
    n_val = max(1, int(round(n * 0.2)))
    n_train = max(1, n - n_val - n_test)
    # Adjust if overflow
    while n_train + n_val + n_test > n:
        n_train = max(1, n_train - 1)
    mapping = {}
    i = 0
    for _ in range(n_train):
        mapping[ids[i]] = "train"
        i += 1
    for _ in range(n_val):
        if i < n:
            mapping[ids[i]] = "val"
            i += 1
    while i < n:
        mapping[ids[i]] = "test"
        i += 1
    return mapping


class DatasetBuilder:
    def __init__(
        self,
        project: ProjectManager,
        model_type: str = "segmentation",
        every_n: int = 1,
        min_area: float = 100.0,
        min_blur_var: float = 30.0,
        dedupe: bool = True,
        augment_per_image: int = 0,
        class_name: str = "target_region",
    ):
        self.project = project
        self.model_type = model_type
        self.every_n = max(1, every_n)
        self.min_area = min_area
        self.min_blur_var = min_blur_var
        self.dedupe = dedupe
        self.augment_per_image = augment_per_image
        self.class_name = class_name

    def build(self, out_name: str = "dataset_v1") -> Path:
        assert self.project.meta is not None
        out = self.project.datasets_dir / out_name
        if out.exists():
            shutil.rmtree(out)
        for split in ("train", "val", "test"):
            (out / "images" / split).mkdir(parents=True)
            (out / "labels" / split).mkdir(parents=True)

        anns = [
            a
            for a in self.project.list_annotations(include_interpolated=True)
            if a.annotation_source != AnnotationSource.INTERPOLATED.value
            or a.quality_score >= 0.5
        ]
        # Prefer manual/confirmed; keep interpolated only if no manual on that frame
        by_vf: Dict[Tuple[str, int], Annotation] = {}
        for a in anns:
            key = (a.video_id, a.frame_id)
            prev = by_vf.get(key)
            if prev is None:
                by_vf[key] = a
            else:
                rank = {AnnotationSource.MANUAL.value: 2, AnnotationSource.CONFIRMED.value: 3, AnnotationSource.INTERPOLATED.value: 1}
                if rank.get(a.annotation_source, 0) >= rank.get(prev.annotation_source, 0):
                    by_vf[key] = a

        video_ids = sorted({a.video_id for a in by_vf.values()})
        split_map = _split_videos(video_ids)
        rng = random.Random(42)
        seen_hashes: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
        counts = {"train": 0, "val": 0, "test": 0}

        # Group by video for efficient seeking
        by_video: Dict[str, List[Annotation]] = {}
        for a in by_vf.values():
            by_video.setdefault(a.video_id, []).append(a)

        for video_id, items in by_video.items():
            split = split_map.get(video_id, "train")
            path = self.project.resolve_video(video_id)
            reader = VideoReader(str(path), prefer_gstreamer=False)
            items = sorted(items, key=lambda x: x.frame_id)
            for ann in items:
                if ann.frame_id % self.every_n != 0 and ann.annotation_source == AnnotationSource.INTERPOLATED.value:
                    continue
                if not is_valid_roi(ann.polygon_points, self.min_area):
                    continue
                ok, frame = reader.seek(ann.frame_id)
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if _laplacian_var(gray) < self.min_blur_var:
                    continue
                ph = _phash(gray)
                if self.dedupe and ph in seen_hashes[split]:
                    continue
                seen_hashes[split].add(ph)
                pts = np.asarray(ann.polygon_points, dtype=np.float32)
                self._write_sample(out, split, frame, pts, f"{video_id}_{ann.frame_id:06d}", counts)
                for ai in range(self.augment_per_image):
                    aug_img, aug_pts = _augment(frame, pts, rng)
                    if polygon_area(aug_pts) < self.min_area:
                        continue
                    self._write_sample(out, split, aug_img, aug_pts, f"{video_id}_{ann.frame_id:06d}_aug{ai}", counts)
            reader.release()

        data_yaml = {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: self.class_name},
            "nc": 1,
        }
        with open(out / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)
        with open(out / "split_by_video.json", "w", encoding="utf-8") as f:
            import json

            json.dump({"split_map": split_map, "counts": counts}, f, indent=2)
        return out

    def _write_sample(
        self,
        out: Path,
        split: str,
        frame: np.ndarray,
        points: np.ndarray,
        stem: str,
        counts: Dict[str, int],
    ) -> None:
        h, w = frame.shape[:2]
        img_path = out / "images" / split / f"{stem}.jpg"
        lbl_path = out / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_path), frame)
        if self.model_type == "detection":
            cx, cy, bw, bh = polygon_to_yolo_bbox(points, w, h)
            line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        else:
            coords = polygon_to_yolo_seg(points, w, h)
            line = "0 " + " ".join(f"{c:.6f}" for c in coords) + "\n"
        lbl_path.write_text(line, encoding="utf-8")
        counts[split] = counts.get(split, 0) + 1
