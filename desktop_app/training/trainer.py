"""YOLO training runner (desktop GPU)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from shared.config import save_yaml
from shared.logging import get_logger

logger = get_logger("desktop.training")

ProgressCb = Callable[[Dict[str, Any]], None]


class TrainingRunner:
    def __init__(
        self,
        data_yaml: str | Path,
        project_runs_dir: str | Path,
        model_type: str = "segmentation",
        base_model: str = "yolov8n-seg.pt",
        epochs: int = 50,
        batch: int = 8,
        imgsz: int = 640,
        device: str = "0",
    ):
        self.data_yaml = Path(data_yaml)
        self.runs_dir = Path(project_runs_dir)
        self.model_type = model_type
        self.base_model = base_model if model_type == "segmentation" or "seg" in base_model else (
            base_model if model_type == "detection" else "yolov8n-seg.pt"
        )
        if model_type == "detection" and "seg" in self.base_model:
            self.base_model = "yolov8n.pt"
        if model_type == "segmentation" and "seg" not in self.base_model:
            self.base_model = "yolov8n-seg.pt"
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.device = device
        self._stop = False
        self.last_metrics: Dict[str, Any] = {}
        self.best_weights: Optional[Path] = None

    def request_stop(self) -> None:
        self._stop = True

    def run(self, progress_cb: Optional[ProgressCb] = None) -> Dict[str, Any]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for training on the desktop") from exc

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "data": str(self.data_yaml),
            "model": self.base_model,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "device": self.device,
            "model_type": self.model_type,
        }
        save_yaml(cfg, self.runs_dir / "training_config.yaml")

        model = YOLO(self.base_model)

        def on_fit_epoch_end(trainer):  # noqa: ANN001
            metrics = {}
            if hasattr(trainer, "metrics") and trainer.metrics:
                metrics = {k: float(v) for k, v in dict(trainer.metrics).items() if isinstance(v, (int, float))}
            payload = {
                "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                "metrics": metrics,
                "stopped": self._stop,
            }
            self.last_metrics = payload
            if progress_cb:
                progress_cb(payload)
            if self._stop:
                trainer.stop = True

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        results = model.train(
            data=str(self.data_yaml),
            epochs=self.epochs,
            batch=self.batch,
            imgsz=self.imgsz,
            device=self.device,
            project=str(self.runs_dir),
            name="train",
            exist_ok=True,
        )
        best = self.runs_dir / "train" / "weights" / "best.pt"
        if best.exists():
            self.best_weights = best
        summary = {
            "best_weights": str(self.best_weights) if self.best_weights else None,
            "last_metrics": self.last_metrics,
            "results_dir": str(self.runs_dir / "train"),
        }
        (self.runs_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def export_onnx(self, weights: Optional[str | Path] = None, output: Optional[str | Path] = None) -> Path:
        from ultralytics import YOLO

        w = Path(weights) if weights else self.best_weights
        if w is None or not Path(w).exists():
            raise FileNotFoundError("No weights to export")
        model = YOLO(str(w))
        out = model.export(format="onnx", imgsz=self.imgsz, simplify=True)
        out_path = Path(out)
        if output:
            target = Path(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            out_path.replace(target)
            return target
        return out_path
