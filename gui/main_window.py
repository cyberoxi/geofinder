"""Main application window."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from gui.qt_compat import (
    Qt,
    QTimer,
    QAction,
    QKeySequence,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QSpinBox,
    qt_exec,
)

from core.config import deep_update, save_config
from core.logger import get_logger
from core.project_io import load_project, project_from_roi, roi_from_project, save_project
from core.types import ROI
from core.video_reader import VideoReader
from gui.controls_panel import ControlsPanel
from gui.settings_dialog import SettingsDialog
from gui.video_widget import VideoWidget
from gui.workers import ProcessingWorker, VideoLoadWorker

logger = get_logger("grt.gui")


class MainWindow(QMainWindow):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.setWindowTitle(
            f"{config.get('app', {}).get('name', 'Ground Region Tracker')} "
            f"v{config.get('app', {}).get('version', '1.0.0')}"
        )
        gui_cfg = config.get("gui", {})
        self.resize(int(gui_cfg.get("window_width", 1400)), int(gui_cfg.get("window_height", 900)))

        self.reader: Optional[VideoReader] = None
        self.roi: Optional[ROI] = None
        self._playing = False
        self._worker: Optional[ProcessingWorker] = None
        self._load_worker: Optional[VideoLoadWorker] = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_step)

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self.statusBar().showMessage("READY")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        self.video_widget = VideoWidget()
        self.video_widget.roi_completed.connect(self._on_roi_selected)
        splitter.addWidget(self.video_widget)

        self.controls = ControlsPanel(self.config)
        self.controls.settings_changed.connect(self._on_panel_settings)
        splitter.addWidget(self.controls)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # Timeline
        timeline = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_seek)
        timeline.addWidget(self.frame_slider, stretch=1)

        self.frame_spin = QSpinBox()
        self.frame_spin.setEnabled(False)
        self.frame_spin.valueChanged.connect(self.frame_slider.setValue)
        timeline.addWidget(QLabel("Frame"))
        timeline.addWidget(self.frame_spin)

        self.start_spin = QSpinBox()
        self.end_spin = QSpinBox()
        self.start_spin.setEnabled(False)
        self.end_spin.setEnabled(False)
        timeline.addWidget(QLabel("Start"))
        timeline.addWidget(self.start_spin)
        timeline.addWidget(QLabel("End"))
        timeline.addWidget(self.end_spin)

        self.meta_label = QLabel("No video")
        timeline.addWidget(self.meta_label)
        root.addLayout(timeline)

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self._act("Open Video…", self.open_video, QKeySequence.Open))
        file_menu.addAction(self._act("Open Project…", self.open_project))
        file_menu.addAction(self._act("Save Project…", self.save_project_dialog))
        file_menu.addAction(self._act("Export Results…", self.export_results_hint))
        file_menu.addSeparator()
        file_menu.addAction(self._act("Exit", self.close, QKeySequence.Quit))

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self._act("Settings…", self.open_settings))

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self._act("About", self.show_about))

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(self._act("Open", self.open_video))
        tb.addAction(self._act("Play", self.play))
        tb.addAction(self._act("Pause", self.pause))
        tb.addAction(self._act("Stop", self.stop_playback))
        tb.addAction(self._act("Prev", self.prev_frame))
        tb.addAction(self._act("Next", self.next_frame))
        tb.addSeparator()
        tb.addAction(self._act("Rect ROI", lambda: self.video_widget.set_selection_mode("rectangle")))
        tb.addAction(self._act("Poly ROI", lambda: self.video_widget.set_selection_mode("polygon")))
        tb.addAction(self._act("Confirm ROI", self.confirm_roi))
        tb.addAction(self._act("Reset Track", self.reset_tracking))
        tb.addSeparator()
        tb.addAction(self._act("Run", self.start_processing))
        tb.addAction(self._act("Stop Proc", self.stop_processing))
        tb.addAction(self._act("Export Video", self.export_results_hint))

    def _act(self, text, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    # --- Video I/O ---
    def open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            self.config.get("paths", {}).get("input_dir", ""),
            "Videos (*.mp4 *.avi *.mkv *.mov);;All (*)",
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path: str) -> None:
        self.pause()
        if self.reader:
            self.reader.release()
            self.reader = None
        self.controls.set_status("INITIALIZING", f"Loading {path}")
        prefer = bool(self.config.get("video", {}).get("prefer_gstreamer", True))
        self._load_worker = VideoLoadWorker(path, prefer_gstreamer=prefer)
        self._load_worker.loaded.connect(self._on_video_loaded)
        self._load_worker.failed.connect(self._on_video_failed)
        self._load_worker.frame_ready.connect(self._on_frame_ready)
        self._load_worker.start()

    def _on_video_loaded(self, reader: VideoReader) -> None:
        self.reader = reader
        info = reader.info
        assert info is not None
        n = max(info.frame_count - 1, 0)
        self.frame_slider.setEnabled(True)
        self.frame_spin.setEnabled(True)
        self.start_spin.setEnabled(True)
        self.end_spin.setEnabled(True)
        self.frame_slider.setRange(0, n)
        self.frame_spin.setRange(0, n)
        self.start_spin.setRange(0, n)
        self.end_spin.setRange(0, n)
        self.end_spin.setValue(n)
        self.meta_label.setText(
            f"{info.width}x{info.height} | {info.fps:.2f} FPS | {info.frame_count} frames | {info.backend}"
        )
        self.controls.set_status("READY", Path(info.path).name)
        self.statusBar().showMessage(f"Loaded: {info.path}")

    def _on_video_failed(self, message: str) -> None:
        self.controls.set_status("ERROR", message)
        QMessageBox.critical(self, "Video Error", message)

    def _on_frame_ready(self, frame, index: int) -> None:
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_spin.setValue(index)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.video_widget.show_frame(frame)
        self._update_time_label(index)

    def _update_time_label(self, index: int) -> None:
        if not self.reader:
            return
        t = self.reader.timestamp_of(index)
        self.statusBar().showMessage(f"Frame {index} | Time {t:.2f}s")

    def _on_seek(self, value: int) -> None:
        if not self.reader:
            return
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value)
        self.frame_spin.blockSignals(False)
        ok, frame = self.reader.seek(value)
        if ok and frame is not None:
            self.video_widget.show_frame(frame)
            self._update_time_label(value)

    def play(self) -> None:
        if not self.reader:
            return
        fps = max(self.reader.fps, 1.0)
        self._playing = True
        self._play_timer.start(int(1000 / fps))

    def pause(self) -> None:
        self._playing = False
        self._play_timer.stop()

    def stop_playback(self) -> None:
        self.pause()
        if self.reader:
            self._on_seek(self.start_spin.value())

    def _play_step(self) -> None:
        if not self.reader:
            return
        nxt = self.frame_slider.value() + 1
        if nxt > self.end_spin.value():
            self.pause()
            return
        self.frame_slider.setValue(nxt)

    def prev_frame(self) -> None:
        if self.reader:
            self.frame_slider.setValue(max(0, self.frame_slider.value() - 1))

    def next_frame(self) -> None:
        if self.reader:
            self.frame_slider.setValue(self.frame_slider.value() + 1)

    # --- ROI ---
    def _on_roi_selected(self, roi: ROI) -> None:
        roi.frame_index = self.frame_slider.value()
        self.roi = roi
        self.video_widget.set_roi(roi)
        cx, cy = roi.center
        self.controls.set_status(
            "READY",
            f"ROI {roi.roi_type.value} | center=({cx:.1f},{cy:.1f}) area={roi.area:.0f}",
        )

    def confirm_roi(self) -> None:
        if self.roi is None:
            QMessageBox.information(self, "ROI", "Select a Rectangle or Polygon ROI first.")
            return
        self.video_widget.set_selection_mode(None)
        QMessageBox.information(self, "ROI", "ROI confirmed. You can Run processing.")

    def reset_tracking(self) -> None:
        self.roi = None
        self.video_widget.clear_roi()
        self.controls.set_status("READY", "Tracking reset")

    # --- Processing ---
    def _merge_runtime_config(self) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.config)
        panel = self.controls.collect()
        cfg = deep_update(cfg, panel)
        cfg.setdefault("video", {})["start_frame"] = self.start_spin.value()
        cfg.setdefault("video", {})["end_frame"] = self.end_spin.value()
        return cfg

    def start_processing(self) -> None:
        if not self.reader:
            QMessageBox.warning(self, "Run", "Open a video first.")
            return
        cfg = self._merge_runtime_config()
        mode = cfg.get("tracking", {}).get("mode", "hybrid")
        if mode != "yolo" and self.roi is None:
            QMessageBox.warning(self, "Run", "Select and confirm an ROI first (or use YOLO mode).")
            return
        self.pause()
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Run", "Processing already running.")
            return

        roi = self.roi.copy() if self.roi else None
        if roi is not None:
            roi.frame_index = max(roi.frame_index, self.start_spin.value())

        self._worker = ProcessingWorker(self.reader.path, roi, cfg)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_updated.connect(self._on_proc_frame)
        self._worker.finished_ok.connect(self._on_proc_done)
        self._worker.failed.connect(self._on_proc_failed)
        self._worker.status.connect(lambda s: self.controls.set_status(s))
        self.controls.set_status("INITIALIZING")
        self._worker.start()

    def stop_processing(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self.statusBar().showMessage("Stopping…")

    def _on_progress(self, cur: int, total: int, info: dict) -> None:
        self.statusBar().showMessage(f"Processing {cur}/{total} | {info.get('status')} | conf={info.get('confidence', 0):.2f}")

    def _on_proc_frame(self, frame, info: dict) -> None:
        trail = None
        self.video_widget.show_frame(frame, overlay_info=info)

    def _on_proc_done(self, result: dict) -> None:
        self.controls.set_status("COMPLETED", str(result.get("paths", {})))
        QMessageBox.information(
            self,
            "Completed",
            "Processing finished.\n" + "\n".join(f"{k}: {v}" for k, v in result.get("paths", {}).items()),
        )

    def _on_proc_failed(self, message: str) -> None:
        self.controls.set_status("ERROR", message)
        QMessageBox.critical(self, "Processing Error", message)

    # --- Project / settings ---
    def _on_panel_settings(self, partial: dict) -> None:
        self.config = deep_update(self.config, partial)

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.config, self)
        if qt_exec(dlg):
            self.config = dlg.result_config()

    def save_project_dialog(self) -> None:
        if not self.reader or self.roi is None:
            QMessageBox.warning(self, "Save Project", "Need video + ROI.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            str(Path(self.config["paths"]["projects_dir"]) / "project.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        panel = self.controls.collect()
        data = project_from_roi(
            self.reader.path,
            self.roi,
            model_path=self.config.get("yolo", {}).get("model_path", ""),
            tracker_type=panel["tracking"]["tracker_type"],
            confidence_threshold=panel["tracking"]["confidence_threshold"],
            extras={"tracking_mode": panel["tracking"]["mode"]},
        )
        save_project(path, data)
        self.statusBar().showMessage(f"Project saved: {path}")

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            self.config.get("paths", {}).get("projects_dir", ""),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            data = load_project(path)
            self._load_video(data["video_path"])
            self.roi = roi_from_project(data)
            self.video_widget.set_roi(self.roi)
            # Seek after load — slight delay via timer
            QTimer.singleShot(500, lambda: self._apply_project(data))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Project Error", str(exc))

    def _apply_project(self, data: dict) -> None:
        idx = int(data.get("initial_frame", 0))
        self.frame_slider.setValue(idx)
        if self.roi:
            self.video_widget.set_roi(self.roi)

    def export_results_hint(self) -> None:
        QMessageBox.information(
            self,
            "Export",
            "Annotated video, CSV and JSON are written automatically to the output folder "
            f"when processing completes:\n{self.config.get('paths', {}).get('output_dir')}",
        )

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            f"{self.config.get('app', {}).get('name')}\n"
            f"Version {self.config.get('app', {}).get('version')}\n\n"
            "Hybrid ground-region tracking for NVIDIA Jetson Orin Nano.\n"
            "Modes: Manual ROI, YOLO Detection, Hybrid (Homography + Flow + YOLO).",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_processing()
        self.pause()
        if self.reader:
            self.reader.release()
        event.accept()
