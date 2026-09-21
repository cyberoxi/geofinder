"""One-click pipeline: dataset → train → ONNX → Jetson package → evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from desktop_app.dataset.builder import DatasetBuilder
from desktop_app.packaging.builder import PackageBuilder
from desktop_app.project_manager.manager import ProjectManager
from desktop_app.training.trainer import TrainingRunner
from shared.logging import get_logger

logger = get_logger("desktop.pipeline")

StageCb = Callable[..., None]  # (stage, message, percent 0..100 or None)

# Overall progress share of each stage (percent range)
STAGE_RANGE = {"dataset": (0, 10), "train": (10, 88), "onnx": (88, 92), "package": (92, 95), "evaluate": (95, 100), "done": (100, 100)}


@dataclass
class AutoPipelineConfig:
    model_type: str = "segmentation"
    epochs: int = 40
    batch: int = 8
    imgsz: int = 640
    device: str = "auto"  # auto → CUDA if available, else CPU
    augment_per_image: int = 1
    auto_every_n: int = 3
    export_onnx: bool = True
    build_package: bool = True
    evaluate: bool = True


def _parse_fraction(msg: str) -> Optional[float]:
    """'… video 2/6 …' / '… 30/48 …' → rough fraction for the dataset stage."""
    import re

    m = re.search(r"video (\d+)/(\d+)", msg) or re.search(r"(\d+)/(\d+)", msg)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return a / b if b else None


def gpu_name() -> Optional[str]:
    """Name of the first NVIDIA GPU (via nvidia-smi), even if PyTorch is a CPU build."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        name = out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() else None
        return name
    except Exception:  # noqa: BLE001
        return None


def torch_cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def resolve_device(device: str) -> str:
    if device not in ("auto", "", None):
        if device in ("cpu", "mps"):
            return device
    try:
        import torch

        if torch.cuda.is_available():
            return "0" if device in ("auto", "", None) else device
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class AutoPipeline:
    def __init__(self, project_root: str | Path, cfg: AutoPipelineConfig, progress_cb: Optional[StageCb] = None):
        self.project_root = Path(project_root)
        self.cfg = cfg
        self.progress_cb = progress_cb
        self.runner: Optional[TrainingRunner] = None
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True
        if self.runner is not None:
            self.runner.request_stop()

    def _emit(self, stage: str, msg: str, fraction: Optional[float] = None) -> None:
        """fraction = progress inside the stage (0..1); mapped to an overall percentage."""
        logger.info("[%s] %s", stage, msg)
        lo, hi = STAGE_RANGE.get(stage, (0, 100))
        pct = None if fraction is None else lo + (hi - lo) * max(0.0, min(1.0, fraction))
        if pct is None and stage in STAGE_RANGE:
            pct = lo
        if self.progress_cb:
            self.progress_cb(stage, msg, pct)

    def _check_stop(self) -> None:
        if self._stop:
            raise RuntimeError("Pipeline stopped by user")

    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        summary: Dict[str, Any] = {"config": asdict(self.cfg)}
        pm = ProjectManager.open(self.project_root)
        try:
            assert pm.meta is not None
            pm.meta.model_type = self.cfg.model_type
            pm.save_meta()
            device = resolve_device(self.cfg.device)
            summary["device"] = device

            # 1. Dataset
            self._emit("dataset", "Building dataset from auto labels + manual annotations…")
            builder = DatasetBuilder(
                pm,
                model_type=self.cfg.model_type,
                augment_per_image=self.cfg.augment_per_image,
                class_name=pm.meta.target_class,
                auto_every_n=self.cfg.auto_every_n,
            )
            dataset_dir = builder.build(progress_cb=lambda m: self._emit("dataset", m, _parse_fraction(m)))
            split_info = json.loads((dataset_dir / "split_by_video.json").read_text(encoding="utf-8"))
            summary["dataset"] = {"path": str(dataset_dir), **split_info}
            self._check_stop()

            # 2. Train
            self._emit("train", f"Training {self.cfg.model_type} on {device} for {self.cfg.epochs} epochs…")
            self.runner = TrainingRunner(
                dataset_dir / "data.yaml",
                pm.runs_dir,
                model_type=self.cfg.model_type,
                epochs=self.cfg.epochs,
                batch=self.cfg.batch,
                imgsz=self.cfg.imgsz,
                device=device,
            )

            def _train_cb(d: Dict[str, Any]) -> None:
                if "status" in d:
                    self._emit("train", d["status"], 0.0)
                    return
                if d.get("type") == "batch":
                    eta = d.get("eta_sec")
                    eta_s = f" — ETA {int(eta // 60)}m{int(eta % 60):02d}s" if eta else ""
                    self._emit(
                        "train",
                        f"Epoch {d['epoch']}/{d['epochs']} batch {d['batch']}/{d['nb']}{eta_s}",
                        d.get("fraction"),
                    )
                    return
                m = d.get("metrics", {})
                key = next((k for k in m if "mAP50(" in k and "(M)" in k), None) or next(
                    (k for k in m if "mAP50(" in k), None
                )
                extra = f" {key}={m[key]:.3f}" if key else ""
                ep = d.get("epoch", 0)
                label = f"Epoch {ep}/{self.cfg.epochs}" if ep <= self.cfg.epochs else "Final validation"
                self._emit("train", f"{label}{extra}", min(1.0, ep / max(1, self.cfg.epochs)))

            train = self.runner.run(progress_cb=_train_cb)
            summary["train"] = train
            best = train.get("best_weights")
            if not best:
                raise RuntimeError("Training produced no best.pt")
            self._check_stop()

            # 3. ONNX
            onnx_path = None
            if self.cfg.export_onnx:
                self._emit("onnx", "Exporting ONNX…")
                try:
                    onnx_path = self.runner.export_onnx(best, pm.runs_dir / "model.onnx")
                    summary["onnx"] = str(onnx_path)
                except Exception as exc:  # noqa: BLE001
                    summary["onnx_error"] = str(exc)
                    self._emit("onnx", f"ONNX export failed (package will carry best.pt): {exc}")

            # 4. Package
            if self.cfg.build_package:
                self._emit("package", "Building Jetson package…")
                zip_path = PackageBuilder(pm).build(weights_pt=best, weights_onnx=onnx_path)
                summary["package"] = str(zip_path)
                self._emit("package", f"Package ready: {zip_path}", 1.0)
            self._check_stop()

            # 5. Evaluate on held-out auto-labeled videos (test split, else val)
            if self.cfg.evaluate:
                summary["evaluation"] = self._evaluate(pm, best, split_info.get("split_map", {}), device)
        finally:
            pm.close()
        summary["elapsed_sec"] = round(time.time() - t0, 1)
        out = self.project_root / "runs" / "auto_pipeline_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        summary["summary_path"] = str(out)
        self._emit("done", f"Finished in {summary['elapsed_sec']}s")
        return summary

    def _evaluate(self, pm: ProjectManager, best: str, split_map: Dict[str, str], device: str) -> list:
        from desktop_app.inference.localize import run_on_video

        held_out = [v for v, s in split_map.items() if s == "test"] or [v for v, s in split_map.items() if s == "val"]
        held_out = [v for v in held_out if pm.auto_labels_path(v) is not None][:2]
        results = []
        for k, vid in enumerate(held_out):
            self._check_stop()
            self._emit("evaluate", f"Evaluating on held-out video {vid}…", k / max(1, len(held_out)))
            out_video = pm.exports_dir / "evaluation" / f"{vid}_localized.mp4"
            m = run_on_video(
                best,
                pm.resolve_video(vid),
                scene_json=pm.scene_path if pm.scene_path.exists() else None,
                out_video=out_video,
                labels_json=pm.auto_labels_path(vid),
                imgsz=self.cfg.imgsz,
                device=device,
            )
            self._emit(
                "evaluate",
                f"{vid}: recall={m.get('recall', 0):.2f} mIoU={m.get('mean_iou', 0):.2f} "
                f"from-landmarks={m['inferred_from_landmarks_rate']:.0%}",
            )
            results.append(m)
        return results
