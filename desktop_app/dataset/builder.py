"""Build YOLO dataset from project annotations (video-based split)."""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
from shared.landmarks import clip_polygon_to_frame
from shared.schemas import Annotation, AnnotationSource
from shared.video_reader import VideoReader


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _phash(gray: np.ndarray, size: int = 16) -> str:
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return hashlib.md5(small.tobytes()).hexdigest()


def _augment(
    image: np.ndarray, polys: List[np.ndarray], rng: random.Random
) -> Tuple[np.ndarray, List[np.ndarray]]:
    img = image.copy()
    polys = [p.astype(np.float32).copy() for p in polys]
    h, w = img.shape[:2]

    def _warp(M: np.ndarray) -> None:
        for i, pts in enumerate(polys):
            if len(pts):
                ones = np.ones((len(pts), 1), np.float32)
                polys[i] = (M @ np.hstack([pts, ones]).T).T.astype(np.float32)

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
        _warp(M)
    if rng.random() < 0.35:
        scale = rng.uniform(0.85, 1.15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        _warp(M)

    out = []
    for pts in polys:
        if len(pts) >= 3 and cv2.isContourConvex(pts.reshape(-1, 1, 2)):
            out.append(clip_polygon_to_frame(pts, w, h))
        else:  # manual non-convex polygons: clamp vertices
            pts = pts.copy()
            if len(pts):
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            out.append(pts)
    return img, out


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
        auto_every_n: int = 3,
        negative_ratio: float = 0.15,
    ):
        self.project = project
        self.model_type = model_type
        self.every_n = max(1, every_n)
        self.min_area = min_area
        self.min_blur_var = min_blur_var
        self.dedupe = dedupe
        self.augment_per_image = augment_per_image
        self.class_name = class_name
        self.auto_every_n = max(1, auto_every_n)
        self.negative_ratio = negative_ratio

    def build(
        self,
        out_name: str = "dataset_v1",
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Path:
        assert self.project.meta is not None
        out = self.project.datasets_dir / out_name
        if out.exists():
            shutil.rmtree(out)
        for split in ("train", "val", "test"):
            (out / "images" / split).mkdir(parents=True)
            (out / "labels" / split).mkdir(parents=True)

        classes = self.project.class_names()
        classes[0] = self.class_name
        class_ids = {c: i for i, c in enumerate(classes)}
        auto_videos = sorted(
            v.video_id for v in self.project.meta.videos if self.project.auto_labels_path(v.video_id) is not None
        )

        anns = [
            a
            for a in self.project.list_annotations(include_interpolated=True)
            if (a.annotation_source != AnnotationSource.INTERPOLATED.value or a.quality_score >= 0.5)
            and a.video_id not in auto_videos
        ]
        # Prefer manual/confirmed; keep interpolated only if no manual on that frame
        by_vf: Dict[Tuple[str, int], Annotation] = {}
        for a in anns:
            key = (a.video_id, a.frame_id)
            prev = by_vf.get(key)
            if prev is None:
                by_vf[key] = a
            else:
                rank = {
                    AnnotationSource.MANUAL.value: 2,
                    AnnotationSource.CONFIRMED.value: 3,
                    AnnotationSource.INTERPOLATED.value: 1,
                }
                if rank.get(a.annotation_source, 0) >= rank.get(prev.annotation_source, 0):
                    by_vf[key] = a

        if not by_vf and not auto_videos:
            raise RuntimeError(
                "No usable annotations found. Generate a satellite video (auto-labeled) or save "
                "ROI annotations on video frames (and Confirm Interpolate if needed), then try again."
            )

        video_ids = sorted({a.video_id for a in by_vf.values()} | set(auto_videos))
        split_map = _split_videos(video_ids)
        rng = random.Random(42)
        seen_hashes: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
        counts = {"train": 0, "val": 0, "test": 0}
        negatives = {"train": 0, "val": 0, "test": 0}

        by_video: Dict[str, List[Annotation]] = {}
        for a in by_vf.values():
            by_video.setdefault(a.video_id, []).append(a)

        total = len(by_vf)
        done = 0
        if progress_cb:
            progress_cb(f"Building dataset… 0/{total}")

        for video_id, items in by_video.items():
            split = split_map.get(video_id, "train")
            path = self.project.resolve_video(video_id)
            if progress_cb:
                progress_cb(f"Opening video {video_id}…")
            reader = VideoReader(str(path), prefer_gstreamer=False)
            items = sorted(items, key=lambda x: x.frame_id)
            try:
                for ann in items:
                    done += 1
                    if progress_cb and (done == 1 or done % 5 == 0 or done == total):
                        progress_cb(f"Building dataset… {done}/{total} (written={sum(counts.values())})")
                    if ann.frame_id % self.every_n != 0 and ann.annotation_source == AnnotationSource.INTERPOLATED.value:
                        continue
                    if not is_valid_roi(ann.polygon_points, self.min_area):
                        continue
                    ok, frame = reader.seek(ann.frame_id)
                    if not ok or frame is None:
                        continue
                    if not self._accept_frame(frame, split, seen_hashes):
                        continue
                    pts = np.asarray(ann.polygon_points, dtype=np.float32)
                    objects = [(class_ids.get(ann.class_name, 0), pts)]
                    self._write_with_aug(out, split, frame, objects, f"{video_id}_{ann.frame_id:06d}", counts, rng)
            finally:
                reader.release()

        for vi, video_id in enumerate(auto_videos):
            split = split_map.get(video_id, "train")
            data = self.project.load_auto_labels(video_id) or {}
            label_classes = data.get("classes") or classes
            remap = {i: (0 if i == 0 else class_ids.get(c, i)) for i, c in enumerate(label_classes)}
            frames_labels = data.get("frames") or []
            path = self.project.resolve_video(video_id)
            reader = VideoReader(str(path), prefer_gstreamer=False)
            try:
                for fid, objs in enumerate(frames_labels):
                    ok, frame = reader.read()
                    if not ok or frame is None:
                        break
                    is_negative = not objs
                    step = self.auto_every_n * (2 if is_negative else 1)
                    if fid % step != 0:
                        continue
                    if is_negative and negatives[split] > self.negative_ratio * max(10, counts[split]):
                        continue
                    objects = []
                    for cid, poly in objs:
                        pts = np.asarray(poly, dtype=np.float32)
                        if polygon_area(pts) >= self.min_area * 0.5:
                            objects.append((remap.get(int(cid), int(cid)), pts))
                    if not objects and not is_negative:
                        continue  # only tiny objects — ambiguous frame
                    if not self._accept_frame(frame, split, seen_hashes):
                        continue
                    if not objects:
                        negatives[split] += 1
                    self._write_with_aug(out, split, frame, objects, f"{video_id}_{fid:06d}", counts, rng)
                    if progress_cb and fid % 30 == 0:
                        progress_cb(
                            f"Auto-labeled video {vi + 1}/{len(auto_videos)} frame {fid}/{len(frames_labels)} "
                            f"(written={sum(counts.values())})"
                        )
            finally:
                reader.release()

        if counts["val"] == 0 and counts["train"] >= 5:
            self._borrow_val_from_train(out, counts)

        if sum(counts.values()) == 0:
            raise RuntimeError(
                "No samples written. Check annotations (blur filter / area / every_n) "
                "or lower min quality requirements."
            )

        data_yaml = {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: c for i, c in enumerate(classes)},
            "nc": len(classes),
        }
        with open(out / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)
        with open(out / "split_by_video.json", "w", encoding="utf-8") as f:
            import json

            json.dump(
                {"split_map": split_map, "counts": counts, "negatives": negatives, "classes": classes}, f, indent=2
            )
        if progress_cb:
            progress_cb(f"Done — {sum(counts.values())} samples → {out}")
        return out

    def _accept_frame(self, frame: np.ndarray, split: str, seen_hashes: Dict[str, set]) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if _laplacian_var(gray) < self.min_blur_var:
            return False
        ph = _phash(gray)
        if self.dedupe and ph in seen_hashes[split]:
            return False
        seen_hashes[split].add(ph)
        return True

    def _write_with_aug(
        self,
        out: Path,
        split: str,
        frame: np.ndarray,
        objects: List[Tuple[int, np.ndarray]],
        stem: str,
        counts: Dict[str, int],
        rng: random.Random,
    ) -> None:
        self._write_sample(out, split, frame, objects, stem, counts)
        if split != "train":
            return  # keep val/test un-augmented for honest metrics
        for ai in range(self.augment_per_image):
            aug_img, aug_polys = _augment(frame, [p for _, p in objects], rng)
            aug_objs = [
                (cid, p)
                for (cid, _), p in zip(objects, aug_polys)
                if len(p) >= 3 and polygon_area(p) >= self.min_area * 0.5
            ]
            if objects and not aug_objs:
                continue
            self._write_sample(out, split, aug_img, aug_objs, f"{stem}_aug{ai}", counts)

    @staticmethod
    def _borrow_val_from_train(out: Path, counts: Dict[str, int]) -> None:
        """Single-video projects: move every 5th (non-augmented) train sample to val."""
        imgs = sorted(p for p in (out / "images" / "train").glob("*.jpg") if "_aug" not in p.stem)
        for p in imgs[::5]:
            lbl = out / "labels" / "train" / f"{p.stem}.txt"
            shutil.move(str(p), str(out / "images" / "val" / p.name))
            if lbl.exists():
                shutil.move(str(lbl), str(out / "labels" / "val" / lbl.name))
            counts["train"] -= 1
            counts["val"] += 1
            # Drop its augmented copies so val never leaks into train
            for aug in (out / "images" / "train").glob(f"{p.stem}_aug*.jpg"):
                aug.unlink()
                (out / "labels" / "train" / f"{aug.stem}.txt").unlink(missing_ok=True)
                counts["train"] -= 1

    def _write_sample(
        self,
        out: Path,
        split: str,
        frame: np.ndarray,
        objects: List[Tuple[int, np.ndarray]],
        stem: str,
        counts: Dict[str, int],
    ) -> None:
        """Write image + YOLO label file (empty label file = background/negative image)."""
        h, w = frame.shape[:2]
        img_path = out / "images" / split / f"{stem}.jpg"
        lbl_path = out / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_path), frame)
        lines = []
        for cid, points in objects:
            if self.model_type == "detection":
                cx, cy, bw, bh = polygon_to_yolo_bbox(points, w, h)
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            else:
                coords = polygon_to_yolo_seg(points, w, h)
                lines.append(f"{cid} " + " ".join(f"{c:.6f}" for c in coords))
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] = counts.get(split, 0) + 1
