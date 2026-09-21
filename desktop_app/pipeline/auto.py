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

StageCb = Callable[[str, str], None]  # (stage, message)


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

    def _emit(self, stage: str, msg: str) -> None:
        logger.info("[%s] %s", stage, msg)
        if self.progress_cb:
            self.progress_cb(stage, msg)

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
            dataset_dir = builder.build(progress_cb=lambda m: self._emit("dataset", m))
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
                    self._emit("train", d["status"])
                    return
                m = d.get("metrics", {})
                key = next((k for k in m if "mAP50(" in k and "(M)" in k), None) or next(
                    (k for k in m if "mAP50(" in k), None
                )
                extra = f" {key}={m[key]:.3f}" if key else ""
                self._emit("train", f"Epoch {d.get('epoch')}/{self.cfg.epochs}{extra}")

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
                self._emit("package", f"Package ready: {zip_path}")
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
        for vid in held_out:
            self._check_stop()
            self._emit("evaluate", f"Evaluating on held-out video {vid}…")
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
