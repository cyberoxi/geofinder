"""Region atlas actions for DesktopMainWindow: build a base model, fine-tune a sub-region."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from shared.qt_compat import QDialog, QMessageBox, qt_exec
from desktop_app.atlas import (
    RegionAtlas,
    finalize_base_model,
    instant_subregion,
    prepare_atlas_data,
    prepare_subregion_project,
)
from desktop_app.gui.atlas_build_dialog import AtlasBuildDialog, remember_atlas
from desktop_app.gui.atlas_finetune_dialog import AtlasFinetuneDialog
from desktop_app.gui.workers import StageWorker
from desktop_app.pipeline.auto import AutoPipelineConfig


class AtlasActionsMixin:
    """Expects DesktopMainWindow: _start_pipeline, _open_project_path, _set_task_progress, log, …"""

    _atlas_worker = None
    _atlas_stop = False

    # ------------------------------------------------------------ common
    def _atlas_busy(self) -> bool:
        w = self._atlas_worker
        if w is not None and w.isRunning():
            if QMessageBox.question(self, "Region Atlas", "Atlas data preparation is running. Stop it?") == QMessageBox.Yes:
                self._atlas_stop = True
            return True
        pw = getattr(self, "_pipeline_worker", None)
        if pw is not None and pw.isRunning():
            QMessageBox.information(self, "Region Atlas", "A training pipeline is running — wait or stop it first.")
            return True
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "Region Atlas", "ultralytics / torch are not installed in this environment.")
            return True
        return False

    def _run_atlas_job(self, title: str, job: Callable, on_ok: Callable[[Any], None]) -> None:
        self._atlas_stop = False
        self.pause()
        self._set_task_progress("atlas", 0, title)
        worker = StageWorker(job)
        worker.progress.connect(lambda d, t, m: self._set_task_progress("atlas", (100.0 * d / t) if t else None, m))
        worker.finished_ok.connect(on_ok)
        worker.failed.connect(self._on_atlas_failed)
        self._atlas_worker = worker
        worker.start()
        self.statusBar().showMessage(title)

    def _on_atlas_failed(self, err: str) -> None:
        self.log(f"Region atlas failed: {err}", "error")
        self.task_progress.setRange(0, 100)
        self.task_stage.setText("Failed / خطا")
        self.status_box.setText(f"Region atlas failed: {err}")
        self.statusBar().showMessage("Region atlas failed")
        QMessageBox.critical(self, "Region Atlas", err)

    def _pipeline_cfg(self, **overrides: Any) -> AutoPipelineConfig:
        cfg = AutoPipelineConfig(
            model_type=self.model_type.currentText(),
            epochs=self.epochs.value(),
            batch=self.batch.value(),
            imgsz=self.imgsz.value(),
            device=self._device_value(),
            augment_per_image=self.aug_n.value(),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    # ------------------------------------------------------------ build atlas
    def build_region_atlas(self):
        if self._atlas_busy():
            return
        if not self.project:
            QMessageBox.information(
                self, "Region Atlas",
                "Open or create a project first.\nAtlas files are created inside that project folder.",
            )
            return
        dlg = AtlasBuildDialog(
            self,
            project_dir=self.project.root,
            model_type=self.model_type.currentText(),
            imgsz=self.imgsz.value(),
        )
        if qt_exec(dlg) != QDialog.Accepted or not dlg.result:
            return
        atlas: RegionAtlas = dlg.result["atlas"]
        epochs = int(dlg.result["epochs"])
        remember_atlas(atlas.root)
        self.log(f"Region atlas: {atlas.root}", "head")
        self.log(atlas.summary())

        def job(progress, _frame):
            return prepare_atlas_data(atlas, progress_cb=progress, should_stop=lambda: self._atlas_stop)

        def on_ok(base_root: Any) -> None:
            self.log(f"Atlas data ready — {len(atlas.landmarks)} landmarks; training base model…", "ok")
            if not self._open_project_path(Path(str(base_root))):
                return
            cfg = self._pipeline_cfg(
                model_type=atlas.model_type, epochs=epochs, imgsz=atlas.imgsz,
                export_onnx=False, build_package=False, evaluate=False,
            )

            def done(summary: Dict[str, Any]) -> None:
                best = (summary.get("train") or {}).get("best_weights")
                if best:
                    dest = finalize_base_model(atlas, best, summary)
                    self.log(f"Atlas base model saved: {dest}", "ok")
                    QMessageBox.information(
                        self, "Region Atlas",
                        f"Base model for '{atlas.name}' is ready.\n\n{dest}\n\n"
                        "Use Region Atlas → Fine-tune Sub-region to train a target inside this area quickly.",
                    )

            self._start_pipeline(cfg, on_done=done)

        self._run_atlas_job(f"Preparing atlas '{atlas.name}' (download, landmarks, videos)…", job, on_ok)

    # ------------------------------------------------------------ fine-tune
    def finetune_from_atlas(self):
        if self._atlas_busy():
            return
        dlg = AtlasFinetuneDialog(self)
        if qt_exec(dlg) != QDialog.Accepted or not dlg.result:
            return
        r = dlg.result
        atlas: RegionAtlas = r["atlas"]
        weights = atlas.base_weights_path()
        if r.get("mode") == "instant":
            self._instant_subregion(r)
            return
        self.log(f"Fine-tune sub-region '{r['project_name']}' from atlas '{atlas.name}'", "head")

        from desktop_app.gui.camera_sim_dialog import active_camera_sim

        camera_sim = active_camera_sim()
        if camera_sim:
            self.log(camera_sim.describe())

        def job(progress, _frame):
            return prepare_subregion_project(
                atlas, r["target_lonlat"], r["project_dir"], project_name=r["project_name"],
                n_videos=r["n_videos"], zoom_out_factor=r["zoom_out_factor"], freeze=r["freeze"],
                night_fraction=r["night_fraction"], ir_fraction=r["ir_fraction"], camera_sim=camera_sim,
                progress_cb=progress, should_stop=lambda: self._atlas_stop,
            )

        def on_ok(project_root: Any) -> None:
            if not self._open_project_path(Path(str(project_root))):
                return
            cfg = self._pipeline_cfg(
                model_type=atlas.model_type, epochs=int(r["epochs"]), imgsz=atlas.imgsz,
                augment_per_image=1, auto_every_n=2, base_model=str(weights), freeze=int(r["freeze"]),
            )
            self._start_pipeline(cfg)

        self._run_atlas_job("Preparing sub-region from atlas (cached tiles)…", job, on_ok)

    def _instant_subregion(self, r: Dict[str, Any]) -> None:
        """Sub-region without training: base model + new scene (+ Jetson package)."""
        atlas: RegionAtlas = r["atlas"]
        self.log(f"Instant sub-region '{r['project_name']}' from atlas '{atlas.name}' (no training)", "head")

        def job(progress, _frame):
            return instant_subregion(
                atlas, r["target_lonlat"], r["project_dir"], project_name=r["project_name"],
                zoom_out_factor=r["zoom_out_factor"], build_package=bool(r.get("build_package", True)),
                progress_cb=progress, should_stop=lambda: self._atlas_stop,
            )

        def on_ok(res: Any) -> None:
            if not self._open_project_path(Path(str(res["project"]))):
                return
            self.log(f"Model (atlas base): {res['weights']}", "ok")
            lines = [
                f"Sub-region '{r['project_name']}' is ready — no training was needed.",
                "The target is located from the atlas landmarks the base model detects.",
            ]
            if res.get("package"):
                self.log(f"Jetson package: {res['package']}", "ok")
                lines.append(f"\nJetson package:\n{res['package']}")
            elif res.get("package_error"):
                self.log(f"Jetson package failed: {res['package_error']}", "warn")
                lines.append(f"\nJetson package failed: {res['package_error']}")
            lines.append(
                "\nTry it with Test Model (Live Test). If the target is lost when the camera is very "
                "close (no landmark in view), create the sub-region again in Fine-tune mode."
            )
            self.status_box.setText("\n".join(lines))
            self.statusBar().showMessage("Instant sub-region ready")
            QMessageBox.information(self, "Region Atlas", "\n".join(lines))

        self._run_atlas_job("Creating instant sub-region (cached tiles, no training)…", job, on_ok)
