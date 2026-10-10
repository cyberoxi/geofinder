"""Pipeline / train / package actions for DesktopMainWindow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from shared.icons import get_icon
from shared.qt_compat import QDialog, QFileDialog, QInputDialog, QMessageBox, qt_exec
from desktop_app.gui.landmark_editor import LandmarkEditorDialog
from desktop_app.gui.path_designer import PathDesignerDialog
from desktop_app.gui.test_video_dialog import TestVideoDialog
from desktop_app.gui.workers import (
    AutoPipelineWorker,
    DatasetWorker,
    StageWorker,
    TrainWorker,
)
from desktop_app.packaging.builder import PackageBuilder, validate_package
from desktop_app.pipeline.auto import AutoPipelineConfig
from desktop_app.satellite import (
    load_prepared_scene,
    render_path_videos,
    render_scene_videos,
    render_test_videos,
)
from desktop_app.satellite.scene import draw_scene_preview
from desktop_app.gui.train_report_dialog import show_train_report
from desktop_app.training.trainer import TrainingRunner
from shared.usb_transfer import (
    USB_PACKAGE_DIR,
    copy_package_to_usb,
    drive_choices,
    latest_package_zip,
    list_removable_drives,
    pick_drive_for_copy,
)


class PipelineActionsMixin:
    """Expects DesktopMainWindow attributes: project, log, _set_task_progress, …"""

    def _has_trained_model(self) -> bool:
        """True if this project/target already has a completed training run."""
        if not self.project:
            return False
        runs = self.project.runs_dir
        if any(runs.glob("**/weights/best.pt")):
            return True
        return (runs / "train_summary.json").exists()

    def _confirm_replace_previous_train(self, kind: str = "train") -> bool:
        """Ask before overwriting an existing model. Returns True if OK to proceed."""
        if not self._has_trained_model():
            return True
        target = ""
        if self.project and self.project.meta:
            target = (self.project.meta.project_name or self.project.meta.target_class or "").strip()
        name = f" «{target}»" if target else ""
        if kind == "pipeline":
            title = "Pipeline — Retrain / آموزش مجدد"
            what = "اجرای کامل پایپ‌لاین (شامل آموزش)"
            what_en = "Running the full pipeline (including training)"
        else:
            title = "Train — Retrain / آموزش مجدد"
            what = "آموزش مجدد"
            what_en = "Retraining"
        msg = (
            f"این تارگت{name} قبلاً آموزش دیده است.\n"
            f"{what} مدل قبلی را جایگزین می‌کند.\n\n"
            f"This target{name} is already trained.\n"
            f"{what_en} will replace the previous model.\n\n"
            "ادامه؟ / Continue?"
        )
        return QMessageBox.question(self, title, msg) == QMessageBox.Yes

    def run_full_pipeline(self):
        if not self.project:
            QMessageBox.warning(self, "Pipeline", "Create or open a project first")
            return
        if self._pipeline_worker is not None and self._pipeline_worker.isRunning():
            if QMessageBox.question(self, "Pipeline", "Pipeline is running. Stop it?") == QMessageBox.Yes:
                self._pipeline_worker.request_stop()
            return
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "Pipeline", "ultralytics / torch are not installed in this environment.")
            return
        anns = self.project.list_annotations(include_interpolated=True)
        has_auto = any(
            self.project.auto_labels_path(v.video_id)
            for v in (self.project.meta.videos if self.project.meta else [])
        )
        if not anns and not has_auto:
            sat = self.project.exports_dir / "satellite"
            orphan_mp4s = list(sat.glob("satellite_*.mp4")) if sat.is_dir() else []
            if orphan_mp4s and QMessageBox.question(
                self,
                "Pipeline",
                f"Found {len(orphan_mp4s)} satellite video(s) in exports/satellite that are not "
                "in the project (often after Cancel during Generate).\n\n"
                "Import them with auto-labels now?",
            ) == QMessageBox.Yes:
                self._import_orphan_satellite(sat)
                has_auto = any(
                    self.project.auto_labels_path(v.video_id)
                    for v in (self.project.meta.videos if self.project.meta else [])
                )
            if not has_auto and not self.project.list_annotations(include_interpolated=True):
                QMessageBox.warning(
                    self,
                    "Pipeline",
                    "No usable annotations found.\n\n"
                    "Use Tools → Generate Satellite Video… (keep Auto-label on) and wait until "
                    "videos appear in the list with [auto-labeled], then try again.",
                )
                return
        if self._device_value() == "0" and not self._cuda:
            QMessageBox.information(
                self,
                "GPU",
                "CUDA PyTorch is not available in this environment.\n"
                "Install a CUDA build of torch, or switch Train on → CPU / Auto.",
            )
        cfg = AutoPipelineConfig(
            model_type=self.model_type.currentText(),
            epochs=self.epochs.value(),
            batch=self.batch.value(),
            imgsz=self.imgsz.value(),
            device=self._device_value(),
            augment_per_image=self.aug_n.value(),
        )
        from desktop_app.atlas import finetune_source

        src = finetune_source(self.project)
        if src and self.project.meta and cfg.model_type == self.project.meta.model_type:
            cfg.base_model, cfg.freeze = src
            self.log(f"Region-atlas sub-region: fine-tuning from {cfg.base_model}", "ok")
        if not self._confirm_replace_previous_train("pipeline"):
            return
        self._start_pipeline(cfg)

    def _start_pipeline(self, cfg: AutoPipelineConfig, on_done=None):
        """Run the automatic pipeline on the open project; ``on_done(summary)`` runs after success."""
        self._set_task_progress("dataset", 0, "Starting pipeline…")
        self._pipeline_on_done = on_done
        self.pause()
        self._pipeline_worker = AutoPipelineWorker(self.project.root, cfg)
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.finished_ok.connect(self._on_pipeline_ok)
        self._pipeline_worker.failed.connect(self._on_pipeline_failed)
        self.btn_pipeline.setText("  Stop Pipeline")
        self.btn_pipeline.setIcon(get_icon("stop_proc"))
        self._pipeline_worker.start()
        self.statusBar().showMessage("Automatic pipeline running…")


    def _on_pipeline_progress(self, stage: str, msg: str, pct=None):
        metrics_payload = None
        if isinstance(pct, dict):
            metrics_payload = pct
            pct = pct.get("pct")
        if stage == "train" and getattr(self, "train_chart", None) is not None:
            if not getattr(self, "_train_chart_armed", False):
                n_ep = int(self.epochs.value()) if hasattr(self, "epochs") else 0
                self._begin_live_train_chart(n_ep)
            if metrics_payload:
                self._feed_train_chart_progress(metrics_payload)
            elif msg:
                self._feed_train_chart_progress({"status": msg})
        self._set_task_progress(stage, pct, msg)
        pct_s = "" if pct is None else f" {float(pct):.0f}%"
        self.statusBar().showMessage(f"Pipeline: {stage}{pct_s}")


    def _on_pipeline_ok(self, summary: dict):
        self._set_task_progress("done", 100)
        hook, self._pipeline_on_done = getattr(self, "_pipeline_on_done", None), None
        if hook is not None:
            try:
                hook(summary)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Post-pipeline step failed: {exc}", "error")
        train = summary.get("train") or {}
        chart = getattr(self, "train_chart", None)
        if chart is not None:
            lm = train.get("last_metrics") or {}
            if lm.get("metrics"):
                self._feed_train_chart_progress(lm)
            chart.mark_done(True)
        self._train_chart_armed = False
        self._log_train_report(train)
        ds = summary.get("dataset") or {}
        if ds:
            self.log(f"   dataset: {ds.get('path')}  counts={ds.get('counts')}  classes={len(ds.get('classes') or [])}")
        self._log_file("ONNX model", summary.get("onnx"))
        if summary.get("onnx_error"):
            self.log(f"ONNX export failed: {summary['onnx_error']}", "warn")
        self._log_file("Jetson package", summary.get("package"))
        for m in summary.get("evaluation") or []:
            self.log(
                f"Held-out test: recall {m.get('recall', 0):.0%}, mean IoU {m.get('mean_iou', 0):.2f}, "
                f"from landmarks {m.get('inferred_from_landmarks_rate', 0):.0%}",
                "ok",
            )
            self._log_file("evaluation video", m.get("output_video"))
        self._log_file("summary", summary.get("summary_path"))
        self.log(f"Total time: {summary.get('elapsed_sec', '?')} s", "ok")
        if hasattr(self, "_refresh_train_status"):
            self._refresh_train_status()
        else:
            self.btn_pipeline.setText("  Run Full Pipeline")
            self.btn_pipeline.setIcon(get_icon("pipeline"))
        self.statusBar().showMessage("Pipeline finished")
        lines = []
        if summary.get("package"):
            lines.append(f"Jetson package:\n{summary['package']}")
        if summary.get("onnx"):
            lines.append(f"ONNX: {summary['onnx']}")
        for m in summary.get("evaluation") or []:
            lines.append(
                f"Held-out test: recall {m.get('recall', 0):.0%}, mean IoU {m.get('mean_iou', 0):.2f}, "
                f"located via landmarks {m['inferred_from_landmarks_rate']:.0%}\n  video: {m.get('output_video')}"
            )
        lines.append(f"Summary: {summary.get('summary_path')}")
        self.status_box.setText("\n".join(lines))
        applied = show_train_report(
            self,
            summary.get("train") or {},
            runs_dir=self.project.runs_dir if self.project else None,
            title="Pipeline finished — Training Report / گزارش آموزش",
        )
        self._apply_train_suggestions(applied)


    def _on_pipeline_failed(self, err: str):
        self._pipeline_on_done = None
        self.log(f"Pipeline failed: {err}", "error")
        self.task_progress.setRange(0, 100)
        self.task_stage.setText("Failed / خطا")
        if hasattr(self, "_refresh_train_status"):
            self._refresh_train_status()
        else:
            self.btn_pipeline.setText("  Run Full Pipeline")
            self.btn_pipeline.setIcon(get_icon("pipeline"))
        chart = getattr(self, "train_chart", None)
        if chart is not None and getattr(self, "_train_chart_armed", False):
            chart.mark_done(False)
            chart.set_status(f"Failed: {err}")
        self._train_chart_armed = False
        self.statusBar().showMessage("Pipeline failed")
        self.status_box.setText(f"Pipeline failed: {err}")
        QMessageBox.critical(self, "Pipeline", err)

    def _import_orphan_satellite(self, sat_dir: Path) -> int:
        """Import ``satellite_*.mp4`` (+ sidecar labels / scene) left in exports after a cancelled Generate."""
        if not self.project:
            return 0
        existing = {Path(v.display_name).name for v in (self.project.meta.videos if self.project.meta else [])}
        n = 0
        for vpath in sorted(sat_dir.glob("satellite_*.mp4")):
            if vpath.name in existing:
                continue
            asset = self.project.add_video(vpath)
            labels = vpath.with_name(vpath.stem + ".labels.json")
            if labels.exists():
                self.project.attach_auto_labels(asset.video_id, labels)
            n += 1
        scene = sat_dir / "scene.json"
        need_scene = scene.exists() and (
            not self.project.scene_path.exists() or not (self.project.scene_dir / "mosaic.jpg").exists()
        )
        if need_scene:
            prev = sat_dir / "scene_preview.jpg"
            mos = next(
                (p for p in (sat_dir / "satellite_mosaic.jpg", sat_dir / "mosaic.jpg") if p.exists()),
                None,
            )
            cand = sat_dir / "candidates.json"
            self.project.set_scene(
                scene,
                prev if prev.exists() else None,
                mos,
                cand if cand.exists() else None,
            )
        self._auto_labels_cache.clear()
        self._refresh_videos()
        if n:
            self.log(f"Imported {n} orphan satellite video(s) from {sat_dir}", "ok")
        return n

    def edit_landmarks(self):
        """Review / change the landmarks of this project's scene, then rebuild the labeled videos."""
        if not self.project:
            QMessageBox.warning(self, "Landmarks", "Create or open a project first")
            return
        if not self.project.scene_path.exists():
            QMessageBox.information(
                self, "Landmarks", "No scene yet — use Tools → Generate Satellite Video… (auto-label) first."
            )
            return
        try:
            prepared = load_prepared_scene(self.project.scene_dir, mosaic_name="mosaic.jpg")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Landmarks", str(exc))
            return
        dlg = LandmarkEditorDialog(
            prepared.mosaic, prepared.layout, prepared.candidates, self,
            extra_mosaics=prepared.extra_mosaics, extra_masks=prepared.extra_masks,
        )
        if qt_exec(dlg) != QDialog.Accepted:
            return
        layout = dlg.result_layout()
        prepared.extra_mosaics, prepared.extra_masks = dlg.result_extras()
        from desktop_app.satellite.scene import save_candidates

        save_candidates(self.project.scene_dir / "candidates.json", dlg.result_candidates())
        prepared.out_dir = self.project.scene_dir
        prepared.save_extras()
        layout.save(self.project.scene_path)
        from desktop_app.satellite.scene import draw_scene_preview

        cv2.imwrite(str(self.project.scene_dir / "scene_preview.jpg"), draw_scene_preview(prepared.mosaic, layout))
        auto_ids = [v.video_id for v in self.project.meta.videos if self.project.auto_labels_path(v.video_id)]
        ans = QMessageBox.question(
            self,
            "Landmarks",
            f"Landmarks saved ({layout.num_landmarks}).\n\n"
            f"Rebuild the {len(auto_ids)} auto-labeled videos with the new landmarks and run the full pipeline?\n"
            "(old auto-labeled videos are removed from the project)",
        )
        if ans != QMessageBox.Yes:
            return
        for vid in auto_ids:
            self.project.remove_video(vid)
        self._auto_labels_cache.clear()
        out_dir = self.project.exports_dir / f"satellite_{np.random.randint(1e9):09d}"
        n_videos = max(1, len(auto_ids) or 6)

        def job(progress, frame):
            res = render_scene_videos(
                prepared.all_mosaics(), layout, out_dir, n_videos=n_videos,
                night_fraction=0.25, ir_fraction=0.15, progress_cb=progress, frame_cb=frame,
            )
            res["run_pipeline"] = True
            return res

        self._stage_worker = StageWorker(job)
        self._stage_worker.progress.connect(lambda d, t, m: self.status_box.setText(f"{m} ({d}%)"))
        self._stage_worker.finished_ok.connect(self._on_satellite_done)
        self._stage_worker.failed.connect(lambda e: QMessageBox.critical(self, "Landmarks", e))
        self._stage_worker.start()
        self.status_box.setText("Rendering videos with the new landmarks…")


    def design_camera_path(self):
        """Design a zoom / camera movement graphically, preview it, render labeled videos from it."""
        if not self.project:
            QMessageBox.warning(self, "Camera path", "Create or open a project first")
            return
        if not self.project.scene_path.exists():
            QMessageBox.information(
                self, "Camera path", "No scene yet — use Tools → Generate Satellite Video… (auto-label) first."
            )
            return
        try:
            prepared = load_prepared_scene(self.project.scene_dir, mosaic_name="mosaic.jpg")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Camera path", str(exc))
            return
        names = [n for n, *_ in prepared.all_mosaics()]
        dlg = PathDesignerDialog(prepared.mosaic, prepared.layout, self, sources=names)
        if qt_exec(dlg) != QDialog.Accepted:
            return
        spec = dlg.result_spec()
        out_dir = self.project.exports_dir / f"paths_{np.random.randint(1e9):09d}"
        layout = prepared.layout
        mosaics = prepared.all_mosaics()

        def job(progress, frame):
            res = render_path_videos(
                mosaics, layout, spec["keys"], out_dir, lights=spec["lights"], all_sources=spec["all_sources"],
                fps=spec["fps"], name=spec["name"], progress_cb=progress, frame_cb=frame,
            )
            res["run_pipeline"] = False
            return res

        self._stage_worker = StageWorker(job)
        self._stage_worker.progress.connect(lambda d, t, m: self._set_task_progress("render", d, m))
        self._stage_worker.finished_ok.connect(self._on_satellite_done)
        self._stage_worker.failed.connect(lambda e: QMessageBox.critical(self, "Camera path", e))
        self._stage_worker.start()
        self._set_task_progress("render", 0, "Rendering camera-path videos…")

    def build_test_videos(self):
        """Build varied test videos (with / without target) from the satellite scene — target read-only."""
        if not self.project:
            QMessageBox.warning(self, "Test videos", "Create or open a project first")
            return
        if not self.project.scene_path.exists():
            QMessageBox.information(
                self,
                "Test videos",
                "No scene yet — use Tools → Generate Satellite Video… (auto-label) first.",
            )
            return
        try:
            prepared = load_prepared_scene(self.project.scene_dir, mosaic_name="mosaic.jpg")
            dlg = TestVideoDialog(prepared.mosaic, prepared.layout, self)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Test videos", str(exc))
            return
        if qt_exec(dlg) != QDialog.Accepted or not dlg.params:
            return
        params = dlg.params
        out_dir = self.project.exports_dir / f"test_videos_{np.random.randint(1e9):09d}"
        mosaic = prepared.mosaic
        layout = prepared.layout
        world_mode = params.get("empty_region_mode") == "world"
        n_with = int(params["n_with_target"])
        n_without = int(params["n_without_target"])

        def job(progress, frame):
            videos: list = []
            skipped = 0
            # With-target (+ random without when not using world map)
            scene_without = 0 if world_mode else n_without
            if n_with > 0 or scene_without > 0:
                res = render_test_videos(
                    mosaic,
                    layout,
                    out_dir,
                    n_with_target=n_with,
                    n_without_target=scene_without,
                    render_modes=params["modes"],
                    duration_s=params["duration_s"],
                    fps=params["fps"],
                    night_fraction=params["night_fraction"],
                    ir_fraction=params["ir_fraction"],
                    empty_region_mode="random",
                    empty_regions=[],
                    zoom_start=float(params.get("zoom_start", 12.0)),
                    zoom_end=float(params.get("zoom_end", 1.6)),
                    zoom_speed=float(params.get("zoom_speed", 1.0)),
                    pan_view=float(params.get("pan_view", 4.0)),
                    pan_speed=float(params.get("pan_speed", 1.0)),
                    progress_cb=progress,
                    frame_cb=frame,
                )
                videos.extend(res.get("videos") or [])
                skipped += int(res.get("skipped") or 0)
            if world_mode and n_without > 0:
                from desktop_app.satellite.test_world import render_world_test_videos

                res_w = render_world_test_videos(
                    params.get("world_regions") or [],
                    out_dir,
                    n_videos=n_without,
                    render_modes=params["modes"],
                    duration_s=params["duration_s"],
                    fps=params["fps"],
                    night_fraction=params["night_fraction"],
                    ir_fraction=params["ir_fraction"],
                    zoom_start=float(params.get("zoom_start", 12.0)),
                    zoom_end=float(params.get("zoom_end", 1.6)),
                    zoom_speed=float(params.get("zoom_speed", 1.0)),
                    pan_view=float(params.get("pan_view", 4.0)),
                    pan_speed=float(params.get("pan_speed", 1.0)),
                    tile_zoom=int(params.get("tile_zoom", 16)),
                    zoom_out_factor=float(params.get("zoom_start", 12.0)),
                    progress_cb=progress,
                    frame_cb=frame,
                )
                videos.extend(res_w.get("videos") or [])
            return {"videos": videos, "skipped": skipped, "out_dir": out_dir, "run_pipeline": False}

        self._stage_worker = StageWorker(job)
        self._stage_worker.progress.connect(lambda d, t, m: self._set_task_progress("render", d, m))
        self._stage_worker.finished_ok.connect(self._on_test_videos_done)
        self._stage_worker.failed.connect(lambda e: QMessageBox.critical(self, "Test videos", e))
        self._stage_worker.start()
        self._set_task_progress("render", 0, "Rendering test videos…")

    def _on_test_videos_done(self, result: Dict[str, Any]):
        videos = result.get("videos") or []
        n_pos = sum(1 for v in videos if v.get("has_target"))
        n_neg = sum(1 for v in videos if not v.get("has_target"))
        out = result.get("out_dir", "")
        skipped = int(result.get("skipped") or 0)
        try:
            for v in videos:
                self.project.add_video(v["path"])
            self._refresh_videos()
        except Exception:  # noqa: BLE001
            pass
        msg = (
            f"Built {len(videos)} test videos\n"
            f"  with target: {n_pos}\n"
            f"  without target: {n_neg}\n"
            f"Output:\n{out}"
        )
        if skipped:
            msg += f"\n\n({skipped} empty-region attempts skipped — mosaic too tight)"
        self.status_box.setText(msg)
        self._set_task_progress("render", 100, "Test videos ready")
        QMessageBox.information(self, "Test videos", msg)


    def test_model_on_video(self):
        """Live-play a video or webcam with the trained model; source is chosen in the dialog."""
        if not self.project:
            QMessageBox.warning(self, "Test", "Create or open a project first")
            return
        weights = sorted(self.project.runs_dir.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
        if not weights:
            QMessageBox.warning(self, "Test", "Train a model first (Run Full Pipeline)")
            return
        from desktop_app.gui.live_test_dialog import LiveTestDialog

        scene = self.project.scene_path if self.project.scene_path.exists() else None
        dlg = LiveTestDialog(
            weights[-1],
            None,
            scene,
            self.imgsz.value(),
            self._device_value(),
            self,
        )
        qt_exec(dlg)
        m = dlg.metrics
        if m and m.get("frames"):
            msg = (
                f"Source: {m.get('video', '')}\n"
                f"Frames: {m['frames']}\n"
                f"Target detected directly: {m['target_detected_rate']:.0%}\n"
                f"Target located from landmarks: {m['inferred_from_landmarks_rate']:.0%}\n"
                f"Located overall: {m['located_rate']:.0%}"
            )
            self.status_box.setText(msg)


    def build_dataset(self):
        if not self.project:
            QMessageBox.warning(self, "Dataset", "Create or open a project first")
            return
        if self._dataset_worker is not None and self._dataset_worker.isRunning():
            QMessageBox.information(self, "Dataset", "Dataset build is already running")
            return
        anns = self.project.list_annotations(include_interpolated=True)
        usable = [
            a
            for a in anns
            if a.annotation_source != AnnotationSource.INTERPOLATED.value or a.quality_score >= 0.5
        ]
        has_auto = any(self.project.auto_labels_path(v.video_id) for v in (self.project.meta.videos if self.project.meta else []))
        if not usable and not has_auto:
            QMessageBox.warning(
                self,
                "Dataset",
                "No labeled videos found.\n\n"
                "Use Tools → Generate Satellite Video… (auto-labeled) first, then Build Dataset.",
            )
            return

        if self.project.meta:
            self.project.meta.model_type = self.model_type.currentText()
            self.project.save_meta()

        self.pause()
        if self.btn_build:
            self.btn_build.setEnabled(False)
        self.status_box.setText("Building dataset in background…")
        self.statusBar().showMessage("Building dataset…")

        self._dataset_worker = DatasetWorker(
            self.project.root,
            model_type=self.model_type.currentText(),
            every_n=self.every_n.value(),
            augment_per_image=self.aug_n.value(),
            class_name=self.project.meta.target_class if self.project.meta else "target_region",
        )
        self._set_task_progress("dataset", None, "Building dataset…")
        self._dataset_worker.progress.connect(self.status_box.setText)
        self._dataset_worker.finished_ok.connect(self._on_dataset_ok)
        self._dataset_worker.failed.connect(self._on_dataset_failed)
        self._dataset_worker.start()


    def _on_dataset_ok(self, out: str):
        self._set_task_progress("done", 100)
        self.log(f"Dataset built: {out}", "ok")
        if self.btn_build:
            self.btn_build.setEnabled(True)
        self.statusBar().showMessage("Ready")
        self.status_box.setText(f"Dataset built: {out}")
        QMessageBox.information(self, "Dataset", f"Created:\n{out}")


    def _on_dataset_failed(self, err: str):
        self.task_progress.setRange(0, 100)
        self.task_stage.setText("Failed / خطا")
        if self.btn_build:
            self.btn_build.setEnabled(True)
        self.statusBar().showMessage("Ready")
        self.status_box.setText(f"Dataset failed: {err}")
        QMessageBox.critical(self, "Dataset", err)


    def train_yolo(self):
        if not self.project:
            QMessageBox.warning(self, "Train", "Create or open a project first")
            return
        if self._train_worker is not None and self._train_worker.isRunning():
            QMessageBox.information(self, "Train", "Training is already running")
            return
        datasets = sorted(self.project.datasets_dir.glob("*/data.yaml"))
        if not datasets:
            QMessageBox.warning(self, "Train", "Build a dataset first")
            return
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self,
                "Train",
                "ultralytics / torch are not installed in this environment.\n\n"
                "Install with:\n"
                "  pip install torch torchvision\n"
                "  pip install ultralytics\n\n"
                "Then restart the app.",
            )
            return
        if not self._confirm_replace_previous_train("train"):
            return

        from desktop_app.pipeline.auto import resolve_device

        device = resolve_device(self._device_value())
        if device not in ("cpu", "mps"):
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
                    self.status_box.setText("CUDA not available — using CPU")
            except Exception:
                device = "cpu"

        runner = TrainingRunner(
            datasets[-1],
            self.project.runs_dir,
            model_type=self.model_type.currentText(),
            epochs=self.epochs.value(),
            batch=self.batch.value(),
            imgsz=self.imgsz.value(),
            device=device,
        )
        self._train_worker = TrainWorker(runner)
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished_ok.connect(self._on_train_ok)
        self._train_worker.failed.connect(self._on_train_failed)
        if self.btn_train:
            self.btn_train.setEnabled(False)
        self._begin_live_train_chart(self.epochs.value())
        self._train_worker.start()
        self.status_box.setText(f"Training started on {device}… (first run may download weights)")
        self.statusBar().showMessage("Training…")

    def _show_video_canvas(self) -> None:
        tabs = getattr(self, "mid_tabs", None)
        if tabs is not None:
            tabs.setCurrentIndex(getattr(self, "_tab_video", 0))

    def _begin_live_train_chart(self, total_epochs: int = 0) -> None:
        chart = getattr(self, "train_chart", None)
        tabs = getattr(self, "mid_tabs", None)
        if chart is None or tabs is None:
            return
        self.pause()
        chart.reset(int(total_epochs or 0))
        tabs.setCurrentIndex(getattr(self, "_tab_train", 1))
        self._train_chart_armed = True

    def _feed_train_chart_progress(self, d: Optional[dict]) -> None:
        chart = getattr(self, "train_chart", None)
        if chart is None or not d:
            return
        if d.get("type") == "batch":
            chart.set_batch(
                int(d.get("epoch") or 0),
                int(d.get("epochs") or 0),
                int(d.get("batch") or 0),
                int(d.get("nb") or 0),
                float(d.get("fraction") or 0.0),
                d.get("eta_sec"),
            )
            return
        if "status" in d and not d.get("metrics"):
            chart.set_status(str(d.get("status") or ""))
            return
        metrics = d.get("metrics") or {}
        if metrics or d.get("epoch"):
            chart.add_epoch(
                int(d.get("epoch") or 0),
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                epochs=int(d.get("epochs") or 0),
            )

    def _on_train_progress(self, d: dict):
        self._feed_train_chart_progress(d)
        if d.get("type") == "batch":
            eta = d.get("eta_sec")
            eta_s = f" — ETA {int(eta // 60)}m{int(eta % 60):02d}s" if eta else ""
            self._set_task_progress(
                "train", 100.0 * float(d.get("fraction", 0.0)), f"Epoch {d['epoch']}/{d['epochs']} batch {d['batch']}/{d['nb']}{eta_s}"
            )
        elif "status" in d:
            self._set_task_progress("train", 0, d["status"])
        else:
            m = d.get("metrics", {})
            key = next((k for k in m if "mAP50(" in k), None)
            self.status_box.setText(f"Epoch {d.get('epoch')} done" + (f" — {key}={m[key]:.3f}" if key else ""))

    def _on_train_ok(self, result: dict):
        self._set_task_progress("done", 100)
        self._log_train_report(result)
        if self.btn_train:
            self.btn_train.setEnabled(True)
        self.statusBar().showMessage("Ready")
        m = (result.get("last_metrics") or {}).get("metrics") or {}
        key = next((k for k in m if "mAP50(" in k and "95" not in k), None)
        self.status_box.setText(
            f"Training done on {result.get('device', '?')}"
            + (f" — {key}={m[key]:.3f}" if key else "")
        )
        chart = getattr(self, "train_chart", None)
        if chart is not None:
            chart.mark_done(True)
            # Ensure final epoch metrics are plotted if callback missed them
            lm = result.get("last_metrics") or {}
            if lm.get("metrics"):
                self._feed_train_chart_progress(lm)
        self._train_chart_armed = False
        if hasattr(self, "_refresh_train_status"):
            self._refresh_train_status()
        applied = show_train_report(
            self,
            result,
            runs_dir=self.project.runs_dir if self.project else None,
        )
        self._apply_train_suggestions(applied)

    def _apply_train_suggestions(self, suggested: Optional[Dict[str, int]]) -> None:
        if not suggested:
            return
        if "epochs" in suggested:
            self.epochs.setValue(int(suggested["epochs"]))
        if "batch" in suggested:
            self.batch.setValue(int(suggested["batch"]))
        if "imgsz" in suggested:
            self.imgsz.setValue(int(suggested["imgsz"]))
        self.log(f"Applied suggested training settings: {suggested}", "ok")
        self.statusBar().showMessage("Suggested training settings applied")

    def show_last_train_report(self):
        if not self.project:
            QMessageBox.warning(self, "Report", "Create or open a project first")
            return
        summary = self.project.runs_dir / "train_summary.json"
        if not summary.exists() and not (self.project.runs_dir / "train").is_dir():
            QMessageBox.information(self, "Report", "No training results yet. Train a model first.")
            return
        applied = show_train_report(self, runs_dir=self.project.runs_dir)
        self._apply_train_suggestions(applied)


    def _on_train_failed(self, err: str):
        self.log(f"Training failed: {err}", "error")
        self.task_progress.setRange(0, 100)
        self.task_stage.setText("Failed / خطا")
        if self.btn_train:
            self.btn_train.setEnabled(True)
        chart = getattr(self, "train_chart", None)
        if chart is not None:
            chart.mark_done(False)
            chart.set_status(f"Failed: {err}")
        self._train_chart_armed = False
        self.statusBar().showMessage("Ready")
        self.status_box.setText(f"Training failed: {err}")
        QMessageBox.critical(self, "Train", err)


    def export_onnx(self):
        if not self.project:
            return
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "ONNX", "ultralytics is not installed. Train/export unavailable.")
            return
        weights = list(self.project.runs_dir.glob("**/weights/best.pt"))
        if not weights:
            QMessageBox.warning(self, "ONNX", "No best.pt found")
            return
        runner = TrainingRunner(self.project.datasets_dir / "x", self.project.runs_dir, imgsz=self.imgsz.value())
        try:
            out = runner.export_onnx(weights[-1], self.project.runs_dir / "model.onnx")
            self.status_box.setText(f"ONNX: {out}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "ONNX", str(exc))


    def build_package(self):
        if not self.project:
            return
        try:
            zip_path = PackageBuilder(self.project).build()
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            scp = f"scp {zip_path} user@JETSON_IP:/home/user/packages/"
            self.status_box.setText(f"Package {zip_path.name} ({size_mb:.1f} MB)\n{scp}")
            QMessageBox.information(self, "Package", f"{zip_path}\n{size_mb:.2f} MB\n\n{scp}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Package", str(exc))

    def transfer_package_to_usb(self):
        """Copy the latest Jetson package zip onto a connected USB flash drive."""
        if not self.project:
            QMessageBox.warning(self, "USB", "Create or open a project first")
            return
        zip_path = latest_package_zip(self.project.packages_dir)
        if zip_path is None:
            QMessageBox.warning(
                self,
                "USB",
                "No package zip found.\nBuild a Jetson package first, then transfer to USB.",
            )
            return

        drives = list_removable_drives()
        drive = pick_drive_for_copy(drives)
        if drive is None:
            if not drives:
                # Let the user pick any folder (useful if OS did not mark the drive removable)
                chosen = QFileDialog.getExistingDirectory(
                    self,
                    "Select USB flash drive",
                    "",
                )
                if not chosen:
                    return
                drive = Path(chosen)
            else:
                labels = [label for label, _ in drive_choices(drives)]
                label, ok = QInputDialog.getItem(
                    self,
                    "USB",
                    "Select USB flash drive:",
                    labels,
                    0,
                    False,
                )
                if not ok or not label:
                    return
                drive = dict(drive_choices(drives))[label]

        try:
            dest = copy_package_to_usb(zip_path, drive)
            size_mb = dest.stat().st_size / (1024 * 1024)
            msg = (
                f"Package copied to USB.\n\n"
                f"{dest}\n"
                f"{size_mb:.2f} MB\n\n"
                f"Folder on flash: {USB_PACKAGE_DIR}/\n"
                f"Plug this drive into the Jetson to auto-import."
            )
            self.status_box.setText(f"USB transfer OK: {dest}")
            self.statusBar().showMessage(f"Copied to USB: {dest}")
            QMessageBox.information(self, "USB", msg)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "USB", str(exc))

    def validate_pkg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Package ZIP", str(self.project.packages_dir if self.project else ""), "ZIP (*.zip)")
        if not path:
            return
        try:
            result = validate_package(path)
            ok = bool(result.get("ok"))
            mismatches = result.get("mismatches") or []
            size_mb = float(result.get("size_bytes") or 0) / (1024 * 1024)
            lines = [
                "OK" if ok else "FAILED",
                f"File: {path}",
                f"Size: {size_mb:.2f} MB",
                f"Root: {result.get('root', '')}",
                f"Mismatches: {len(mismatches)}",
            ]
            for item in mismatches[:12]:
                lines.append(f"  - {item.get('file')}: {item.get('error')}")
            if len(mismatches) > 12:
                lines.append(f"  … and {len(mismatches) - 12} more")
            msg = "\n".join(lines)
            if ok:
                QMessageBox.information(self, "Validate", msg)
            else:
                QMessageBox.warning(self, "Validate", msg)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Validate", str(exc))

