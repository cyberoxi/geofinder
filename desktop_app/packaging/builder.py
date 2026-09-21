"""Build and validate Jetson deployment packages."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from desktop_app.project_manager.manager import ProjectManager
from shared.package_utils import sha256_file, validate_package
from shared.config import default_runtime_config, save_yaml
from shared.features import build_reference_bank
from shared.geometry import polygon_area
from shared.schemas import AnnotationSource, Manifest
from shared.video_reader import VideoReader


class PackageBuilder:
    def __init__(self, project: ProjectManager):
        self.project = project

    def build(
        self,
        output_zip: Optional[str | Path] = None,
        weights_pt: Optional[str | Path] = None,
        weights_onnx: Optional[str | Path] = None,
        weights_engine: Optional[str | Path] = None,
        reference_video_id: Optional[str] = None,
        reference_frame: Optional[int] = None,
    ) -> Path:
        assert self.project.meta is not None
        staging = self.project.packages_dir / f"pkg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if staging.exists():
            shutil.rmtree(staging)
        for sub in ("model", "reference/templates", "config", "labels", "checksums"):
            (staging / sub).mkdir(parents=True)

        frame, ref_points, ref_meta = self._pick_reference(reference_video_id, reference_frame)
        pts = np.asarray(ref_points, dtype=np.float32)
        ref_info = build_reference_bank(
            frame,
            pts,
            staging / "reference",
            feature_method=self.project.meta.feature_method,
        )

        # Models
        has_pt = has_onnx = has_engine = False
        if weights_pt and Path(weights_pt).exists():
            shutil.copy2(weights_pt, staging / "model" / "best.pt")
            has_pt = True
        if weights_onnx and Path(weights_onnx).exists():
            shutil.copy2(weights_onnx, staging / "model" / "model.onnx")
            has_onnx = True
        if weights_engine and Path(weights_engine).exists():
            shutil.copy2(weights_engine, staging / "model" / "model.engine")
            has_engine = True

        # Auto-pick from runs if not provided
        if not has_pt:
            candidates = list(self.project.runs_dir.glob("**/weights/best.pt"))
            if candidates:
                shutil.copy2(candidates[-1], staging / "model" / "best.pt")
                has_pt = True
        if not has_onnx:
            candidates = list(self.project.runs_dir.glob("**/*.onnx")) + list(self.project.root.glob("**/*.onnx"))
            # Prefer project-local non-package
            for c in candidates:
                if "packages" in c.parts:
                    continue
                shutil.copy2(c, staging / "model" / "model.onnx")
                has_onnx = True
                break

        model_format = "engine" if has_engine else ("onnx" if has_onnx else ("pytorch" if has_pt else "none"))
        runtime = default_runtime_config()
        runtime["yolo"]["mode"] = self.project.meta.model_type
        runtime["tracking"]["feature_detector"] = self.project.meta.feature_method
        runtime["yolo"]["model_path"] = "model/model.engine" if has_engine else (
            "model/model.onnx" if has_onnx else "model/best.pt"
        )
        save_yaml(runtime, staging / "config" / "runtime.yaml")

        class_names = self.project.class_names()
        (staging / "labels" / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
        scene_rel = ""
        if self.project.scene_path.exists():
            shutil.copy2(self.project.scene_path, staging / "reference" / "scene.json")
            scene_rel = "reference/scene.json"

        manifest = Manifest(
            project_name=self.project.meta.project_name,
            target_class=self.project.meta.target_class,
            model_type=self.project.meta.model_type,
            model_format=model_format,
            input_size=int(runtime["yolo"]["imgsz"]),
            reference_roi={
                "points": [[float(x), float(y)] for x, y in pts],
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                **ref_meta,
                **ref_info,
            },
            feature_method=self.project.meta.feature_method,
            min_confidence=0.45,
            min_iou=0.20,
            class_names=class_names,
            scene_layout=scene_rel,
            has_engine=has_engine,
            has_onnx=has_onnx,
            has_pytorch=has_pt,
        )
        (staging / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        readme = f"""# Deployment Package: {self.project.meta.project_name}

Target class: `{self.project.meta.target_class}`
Model type: {self.project.meta.model_type}
Model format: {model_format}

## Transfer to Jetson

```bash
scp {staging.name}.zip user@JETSON_IP:/home/user/packages/
```

## On Jetson

```bash
unzip {staging.name}.zip
python3 jetson_runtime/app.py --package {staging.name}
# Build TensorRT engine if missing:
python3 tools/build_tensorrt.py --onnx model/model.onnx --output model/model.engine --fp16
```
"""
        (staging / "README_DEPLOYMENT.md").write_text(readme, encoding="utf-8")

        checksums: Dict[str, str] = {}
        for path in staging.rglob("*"):
            if path.is_file() and path.name != "sha256.json":
                rel = path.relative_to(staging).as_posix()
                checksums[rel] = sha256_file(path)
        (staging / "checksums" / "sha256.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")

        # Re-hash including checksum file itself is not needed; validate excludes regenerating

        zip_path = Path(output_zip) if output_zip else self.project.packages_dir / f"{staging.name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in staging.rglob("*"):
                if path.is_file():
                    # Ensure no absolute paths in archive
                    zf.write(path, arcname=path.relative_to(staging).as_posix())
        return zip_path

    def _pick_reference(
        self, reference_video_id: Optional[str], reference_frame: Optional[int]
    ) -> Tuple[np.ndarray, List[List[float]], Dict[str, Any]]:
        """Reference frame + target polygon: manual annotation, else best auto-labeled frame."""
        anns = [
            a
            for a in self.project.list_annotations(include_interpolated=False)
            if a.annotation_source in (AnnotationSource.MANUAL.value, AnnotationSource.CONFIRMED.value)
        ]
        if anns:
            if reference_video_id:
                anns = [a for a in anns if a.video_id == reference_video_id] or anns
            if reference_frame is not None:
                preferred = [a for a in anns if a.frame_id == reference_frame]
                ann = preferred[0] if preferred else anns[0]
            else:
                ann = anns[0]
            frame = self._read_frame(ann.video_id, ann.frame_id)
            return frame, ann.polygon_points, {"frame_id": ann.frame_id, "video_id": ann.video_id}

        # Auto labels: the frame where the target is largest while fully inside the view
        best: Optional[Tuple[float, str, int, List[List[float]]]] = None
        for v in self.project.meta.videos if self.project.meta else []:
            data = self.project.load_auto_labels(v.video_id)
            if not data:
                continue
            w, h = int(data.get("width", v.width)), int(data.get("height", v.height))
            for fid, objs in enumerate(data.get("frames", [])):
                for cid, poly in objs:
                    if int(cid) != 0:
                        continue
                    pts = np.asarray(poly, np.float32)
                    if pts[:, 0].min() <= 1 or pts[:, 1].min() <= 1 or pts[:, 0].max() >= w - 2 or pts[:, 1].max() >= h - 2:
                        continue  # clipped by frame edge
                    area = polygon_area(pts) / float(w * h)
                    if area > 0.6:
                        continue  # too close — little context for features
                    if best is None or area > best[0]:
                        best = (area, v.video_id, fid, poly)
        if best is None:
            raise RuntimeError(
                "No reference available: add a manual annotation or generate an auto-labeled satellite video"
            )
        _, vid, fid, poly = best
        return self._read_frame(vid, fid), poly, {"frame_id": fid, "video_id": vid, "source": "auto"}

    def _read_frame(self, video_id: str, frame_id: int) -> np.ndarray:
        reader = VideoReader(str(self.project.resolve_video(video_id)), prefer_gstreamer=False)
        ok, frame = reader.seek(frame_id)
        reader.release()
        if not ok or frame is None:
            raise RuntimeError("Cannot read reference frame")
        return frame


def validate_package(package_path: str | Path) -> Dict[str, Any]:
    from shared.package_utils import validate_package as _validate

    return _validate(package_path)
