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
        base_model: Optional[str] = "yolov8n-seg.pt",
        epochs: int = 50,
        batch: int = 8,
        imgsz: int = 640,
        device: str = "0",
        freeze: int = 0,
    ):
        self.data_yaml = Path(data_yaml)
        self.runs_dir = Path(project_runs_dir)
        self.model_type = model_type
        # An existing .pt file (e.g. a region-atlas base model) is fine-tuned as-is
        self.custom_weights = bool(base_model) and Path(str(base_model)).suffix == ".pt" and Path(str(base_model)).is_file()
        base_model = base_model or "yolov8n-seg.pt"
        if self.custom_weights:
            self.base_model = str(Path(str(base_model)).resolve())
        else:
            self.base_model = base_model if model_type == "segmentation" or "seg" in base_model else (
                base_model if model_type == "detection" else "yolov8n-seg.pt"
            )
            if model_type == "detection" and "seg" in self.base_model:
                self.base_model = "yolov8n.pt"
            if model_type == "segmentation" and "seg" not in self.base_model:
                self.base_model = "yolov8n-seg.pt"
        self.freeze = max(0, int(freeze))
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
            raise RuntimeError(
                "ultralytics is required for training. "
                "Install: pip install torch torchvision ultralytics"
            ) from exc

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
        if self.freeze:
            cfg["freeze"] = self.freeze
        save_yaml(cfg, self.runs_dir / "training_config.yaml")

        if progress_cb:
            progress_cb({"epoch": 0, "metrics": {}, "status": f"Loading model {self.base_model}…"})

        # Prefer CPU when CUDA device requested but unavailable
        device = self.device
        if str(device) not in ("cpu", "mps"):
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning("CUDA unavailable; falling back to CPU")
                    device = "cpu"
            except Exception:
                device = "cpu"

        from shared.runtime_paths import find_bundled_file

        weights = Path(self.base_model) if self.custom_weights else (find_bundled_file(self.base_model) or Path(self.base_model))
        model = YOLO(str(weights))

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

        batch_state = {"i": 0, "t0": 0.0, "start": None}

        def on_train_epoch_start(trainer):  # noqa: ANN001
            import time as _t

            batch_state["i"] = 0
            batch_state["t0"] = _t.time()
            if batch_state["start"] is None:
                batch_state["start"] = _t.time()

        def on_train_batch_end(trainer):  # noqa: ANN001
            import time as _t

            batch_state["i"] += 1
            nb = max(1, len(getattr(trainer, "train_loader", []) or []))
            i = batch_state["i"]
            if progress_cb and (i == 1 or i % 2 == 0 or i == nb):
                epoch = int(getattr(trainer, "epoch", 0))
                epochs = int(getattr(trainer, "epochs", self.epochs) or self.epochs)
                done = (epoch + i / nb) / max(1, epochs)
                elapsed = _t.time() - (batch_state["start"] or _t.time())
                eta = elapsed / done * (1 - done) if done > 0.01 else None
                progress_cb({
                    "type": "batch",
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "batch": i,
                    "nb": nb,
                    "fraction": done,
                    "eta_sec": eta,
                })
            if self._stop:
                trainer.stop = True

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)
        if progress_cb:
            progress_cb({"epoch": 0, "metrics": {}, "status": f"Training on {device}…"})

        extra: Dict[str, Any] = {"freeze": self.freeze} if self.freeze else {}
        results = model.train(
            data=str(self.data_yaml),
            epochs=self.epochs,
            batch=self.batch,
            imgsz=self.imgsz,
            device=device,
            project=str(self.runs_dir),
            name="train",
            exist_ok=True,
            **extra,
        )
        _ = results
        best = self.runs_dir / "train" / "weights" / "best.pt"
        if best.exists():
            self.best_weights = best
        summary = {
            "best_weights": str(self.best_weights) if self.best_weights else None,
            "last_metrics": self.last_metrics,
            "results_dir": str(self.runs_dir / "train"),
            "device": device,
            "settings": {
                "epochs": self.epochs,
                "batch": self.batch,
                "imgsz": self.imgsz,
                "device": device,
                "model": self.base_model,
                "model_type": self.model_type,
                "data": str(self.data_yaml),
            },
        }
        (self.runs_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
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
