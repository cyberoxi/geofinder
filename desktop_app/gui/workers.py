"""Background QThread workers for the desktop GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from shared.qt_compat import QThread, Signal
from desktop_app.dataset.builder import DatasetBuilder
from desktop_app.pipeline.auto import AutoPipeline, AutoPipelineConfig
from desktop_app.project_manager.manager import ProjectManager
from desktop_app.satellite import generate_satellite_videos
from desktop_app.training.trainer import TrainingRunner


class SatelliteWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)
    frame_ready = Signal(object)

    def __init__(self, params: Dict[str, Any], out_dir: Path):
        super().__init__()
        self.params = params
        self.out_dir = out_dir

    def run(self):
        try:
            result = generate_satellite_videos(
                lat=self.params["lat"],
                lon=self.params["lon"],
                span=self.params.get("span", 0.02),
                south=self.params.get("south"),
                west=self.params.get("west"),
                north=self.params.get("north"),
                east=self.params.get("east"),
                zoom=self.params["zoom"],
                modes=self.params["modes"],
                out_dir=self.out_dir,
                duration_s=self.params["duration_s"],
                fps=self.params["fps"],
                zoom_out_factor=float(self.params.get("zoom_out_factor", 12)),
                auto_label=bool(self.params.get("auto_label", True)),
                num_landmarks=int(self.params.get("num_landmarks", 4)),
                n_videos=int(self.params.get("n_videos", 6)),
                cache_dir=Path("data") / "satellite_cache",
                progress_cb=lambda d, t, m: self.progress.emit(d, t, m),
                frame_cb=lambda frame: self.frame_ready.emit(frame),
            )
            result["run_pipeline"] = bool(self.params.get("run_pipeline"))
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))



class StageWorker(QThread):
    """Runs job(progress_cb, frame_cb) off the UI thread."""

    finished_ok = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)
    frame_ready = Signal(object)

    def __init__(self, job):
        super().__init__()
        self.job = job

    def run(self):
        try:
            res = self.job(lambda d, t, m: self.progress.emit(d, t, m), lambda f: self.frame_ready.emit(f))
            self.finished_ok.emit(res)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.failed.emit(str(exc))


class AutoPipelineWorker(QThread):
    """Dataset → train → ONNX → package → evaluation, off the UI thread."""

    progress = Signal(str, str, object)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, project_root: Path, cfg: AutoPipelineConfig):
        super().__init__()
        self.pipeline = AutoPipeline(
            project_root, cfg, progress_cb=lambda st, m, pct=None: self.progress.emit(st, m, pct)
        )

    def request_stop(self) -> None:
        self.pipeline.request_stop()

    def run(self):
        try:
            self.finished_ok.emit(self.pipeline.run())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, weights: Path, video: Path, scene: Optional[Path], out_video: Path, imgsz: int, device: str):
        super().__init__()
        self.args = (weights, video, scene, out_video, imgsz, device)

    def run(self):
        try:
            from desktop_app.inference.localize import run_on_video
            from desktop_app.pipeline.auto import resolve_device

            weights, video, scene, out_video, imgsz, device = self.args
            m = run_on_video(
                weights,
                video,
                scene_json=scene,
                out_video=out_video,
                imgsz=imgsz,
                device=resolve_device(device),
                progress_cb=lambda d, t, msg: self.progress.emit(msg),
            )
            self.finished_ok.emit(m)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))



class TrainWorker(QThread):
    progress = Signal(dict)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, runner: TrainingRunner):
        super().__init__()
        self.runner = runner

    def run(self):
        try:
            result = self.runner.run(progress_cb=lambda d: self.progress.emit(d))
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class DatasetWorker(QThread):
    """Build YOLO dataset off the UI thread so the window stays responsive."""

    progress = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        project_root: Path,
        model_type: str,
        every_n: int,
        augment_per_image: int,
        class_name: str,
    ):
        super().__init__()
        self.project_root = Path(project_root)
        self.model_type = model_type
        self.every_n = every_n
        self.augment_per_image = augment_per_image
        self.class_name = class_name

    def run(self):
        pm = None
        try:
            pm = ProjectManager.open(self.project_root)
            builder = DatasetBuilder(
                pm,
                model_type=self.model_type,
                every_n=self.every_n,
                augment_per_image=self.augment_per_image,
                class_name=self.class_name,
            )
            out = builder.build(progress_cb=lambda msg: self.progress.emit(msg))
            self.finished_ok.emit(str(out))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            if pm is not None:
                try:
                    pm.close()
                except Exception:
                    pass

