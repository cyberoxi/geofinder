"""Desktop Dataset & Deployment Manager GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from shared.geometry import polygon_to_bbox
from shared.logging import setup_logging
from shared.icons import get_icon
from shared.theme import apply_theme, style_brand_label
from shared.qt_compat import (
    Qt,
    QAction,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QImage,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPixmap,
    QProgressBar,
    QPlainTextEdit,
    QScrollArea,
    QSettings,
    QSlider,
    QSplitter,
    QTabWidget,
    QTimer,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QSpinBox,
    QComboBox,
    QPushButton,
    QListWidget,
    QFormLayout,
    QGroupBox,
    QApplication,
    QSize,
    qt_exec,
)
from shared.schemas import Annotation, AnnotationSource, ROIType
from shared.video_reader import VideoReader
from desktop_app.annotation.interpolate import confirm_interpolated, interpolate_between_keyframes
from desktop_app.pipeline.auto import gpu_name, torch_cuda_ok
from desktop_app.project_manager.manager import ProjectManager
from desktop_app.gui.frame_canvas import FrameCanvas
from desktop_app.gui.train_chart import TrainLiveChart
from desktop_app.gui.satellite_dialog import SatelliteVideoDialog
from desktop_app.gui.workers import AutoPipelineWorker
from desktop_app.gui.pipeline_actions import PipelineActionsMixin
from desktop_app.gui.atlas_actions import AtlasActionsMixin
from desktop_app.gui.offline_actions import OfflineActionsMixin
from desktop_app.gui.serial_actions import SerialActionsMixin
from desktop_app.gui.training_help import make_setting_label
from desktop_app.training.advice import GRADE_COLOR, summarize_learning_status


class DesktopMainWindow(
    PipelineActionsMixin, AtlasActionsMixin, OfflineActionsMixin, SerialActionsMixin, QMainWindow
):
    APP_TITLE = "سامانه تنگسیری"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.APP_TITLE)
        self.setMinimumSize(960, 640)
        self.resize(1440, 920)
        self.project: Optional[ProjectManager] = None
        self.reader: Optional[VideoReader] = None
        self.current_video_id: Optional[str] = None
        self._train_worker = None
        self._dataset_worker = None
        self._pipeline_worker: Optional[AutoPipelineWorker] = None
        self._auto_labels_cache: Dict[str, Any] = {}
        self._build()
        self._restore_last_project()

    @staticmethod
    def _app_settings() -> QSettings:
        return QSettings()

    def _update_window_title(self) -> None:
        name = ""
        if self.project and self.project.meta:
            name = (self.project.meta.project_name or "").strip()
        self.setWindowTitle(f"{self.APP_TITLE} — {name}" if name else self.APP_TITLE)
        brand = f"تنگسیری — {name}" if name else "تنگسیری"
        if hasattr(self, "_tb_brand"):
            style_brand_label(self._tb_brand, brand)

    def _remember_project(self, path: str | Path) -> None:
        self._app_settings().setValue("last_project", str(Path(path).resolve()))

    def _restore_last_project(self) -> None:
        raw = self._app_settings().value("last_project", "")
        if not raw:
            return
        path = Path(str(raw))
        if not (path / "project.json").exists():
            return
        try:
            self.project = ProjectManager.open(path)
            self._refresh_videos()
            self._update_window_title()
            self._refresh_train_status()
            self.status_box.setText(f"Opened {path}")
            self.statusBar().showMessage(f"Restored project: {path}")
        except Exception:  # noqa: BLE001
            pass

    def edit_detection_filters(self) -> None:
        from desktop_app.gui.stabilizer_dialog import describe_stabilizer, edit_stabilizer_settings

        cfg = edit_stabilizer_settings(self)
        self.log(f"Detection {describe_stabilizer(cfg)} (applies to Live Test + next package)", "head")

    def _tb_action(
        self,
        text: str,
        icon_name: str,
        slot,
        tip: str = "",
        shortcut: str = "",
    ) -> QAction:
        act = QAction(get_icon(icon_name), text, self)
        act.triggered.connect(slot)
        if tip:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        if shortcut:
            act.setShortcut(shortcut)
        return act

    def _make_toolbar(self, name: str, object_name: str) -> QToolBar:
        tb = QToolBar(name)
        tb.setObjectName(object_name)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        return tb

    def _build_toolbar(self) -> None:
        self._tb_main = self._make_toolbar("Main", "mainToolBar")
        self._tb_pipeline = self._make_toolbar("Pipeline", "pipelineToolBar")
        self._tb_split = False
        self._tb_needed_width = 0
        self.addToolBar(self._tb_main)

        self._tb_brand = QLabel()
        style_brand_label(self._tb_brand)
        self._tb_main.addWidget(self._tb_brand)
        self._tb_main.addSeparator()

        self.act_new = self._tb_action("New", "new", self.new_project, "Create a new project", "Ctrl+N")
        self.act_open = self._tb_action("Open", "open", self.open_project, "Open project folder", "Ctrl+O")
        self.act_add = self._tb_action(
            "Add Video", "add_video", self.add_video, "Add a video to the project", "Ctrl+Shift+V"
        )
        self.act_play = self._tb_action("Play", "play", self.play, "Play / pause current video", "Space")
        self.act_pause = self._tb_action("Pause", "pause", self.pause, "Pause playback")
        self.act_stop = self._tb_action("Stop", "stop", self.stop, "Stop and go to start of video")
        self.act_prev = self._tb_action("Prev", "prev", self.prev_frame, "Previous frame", "Left")
        self.act_next = self._tb_action("Next", "next", self.next_frame, "Next frame", "Right")
        self.btn_play = None  # set in transport bar; toggles Play ↔ Pause
        self.act_sat = self._tb_action(
            "Satellite", "satellite", self.generate_satellite_video, "Generate satellite training videos"
        )
        self.act_landmarks = self._tb_action(
            "Landmarks", "landmarks", self.edit_landmarks, "Edit scene landmarks"
        )
        self.act_path = self._tb_action(
            "Cam Path", "path", self.design_camera_path, "Design camera path"
        )
        self.act_test_vids = self._tb_action(
            "Test Vids", "test", self.build_test_videos, "Build test videos (with / without target)"
        )
        self.act_pipeline = self._tb_action(
            "Pipeline",
            "pipeline",
            self.run_full_pipeline,
            "Dataset → Train → ONNX → Jetson package",
        )
        self.act_dataset = self._tb_action("Dataset", "dataset", self.build_dataset, "Build YOLO dataset")
        self.act_train = self._tb_action("Train", "train", self.train_yolo, "Train YOLO model")
        self.act_report = self._tb_action(
            "Report",
            "report",
            self.show_last_train_report,
            "Training results & improvement tips / گزارش آموزش",
        )
        self.act_test = self._tb_action(
            "Test", "test", self.test_model_on_video, "Live-play video and highlight detected target"
        )
        self.act_onnx = self._tb_action("ONNX", "onnx", self.export_onnx, "Export ONNX model")
        self.act_pkg = self._tb_action("Package", "package", self.build_package, "Build Jetson package")
        self.act_usb = self._tb_action(
            "To USB",
            "usb",
            self.transfer_package_to_usb,
            "Copy latest package to USB flash drive",
        )
        self.act_validate = self._tb_action(
            "Validate", "validate", self.validate_pkg, "Validate package ZIP"
        )

        # Play/Pause/Stop/Prev/Next live under the video canvas (not on the toolbar).
        for act in (self.act_play, self.act_pause, self.act_stop, self.act_prev, self.act_next):
            self.addAction(act)

        self._tb_primary = [
            self.act_new,
            self.act_open,
            self.act_add,
            None,
            self.act_sat,
            self.act_landmarks,
            self.act_path,
            self.act_test_vids,
        ]
        self._tb_secondary = [
            self.act_pipeline,
            self.act_dataset,
            self.act_train,
            self.act_report,
            self.act_test,
            None,
            self.act_onnx,
            self.act_pkg,
            None,
            self.act_usb,
            self.act_validate,
        ]
        self._fill_toolbar(self._tb_main, self._tb_primary + [None] + self._tb_secondary)
        QTimer.singleShot(0, self._adapt_toolbar_layout)

    @staticmethod
    def _fill_toolbar(tb: QToolBar, items: List[Optional[QAction]]) -> None:
        for item in items:
            if item is None:
                tb.addSeparator()
            else:
                tb.addAction(item)

    def _prune_trailing_separators(self, tb: QToolBar) -> None:
        actions = list(tb.actions())
        while actions and actions[-1].isSeparator():
            tb.removeAction(actions.pop())
        # Also remove consecutive separators left after moving secondary out.
        prev_sep = False
        for action in list(tb.actions()):
            if action.isSeparator():
                if prev_sep:
                    tb.removeAction(action)
                prev_sep = True
            else:
                prev_sep = False

    def _set_toolbar_split(self, split: bool) -> None:
        if split == self._tb_split:
            return
        self._tb_split = split

        for act in self._tb_secondary:
            if act is None:
                continue
            self._tb_main.removeAction(act)
            self._tb_pipeline.removeAction(act)

        self._tb_pipeline.clear()
        self._prune_trailing_separators(self._tb_main)

        if split:
            self._fill_toolbar(self._tb_pipeline, self._tb_secondary)
            self.addToolBar(self._tb_pipeline)
            self.insertToolBarBreak(self._tb_pipeline)
            self._tb_pipeline.show()
        else:
            try:
                self.removeToolBarBreak(self._tb_pipeline)
            except Exception:  # noqa: BLE001
                pass
            self.removeToolBar(self._tb_pipeline)
            self._tb_main.addSeparator()
            self._fill_toolbar(self._tb_main, self._tb_secondary)

    def _toolbar_content_width(self, tb: QToolBar) -> int:
        hint = tb.sizeHint().width()
        if hint > 0:
            return hint
        total = 0
        for action in tb.actions():
            widget = tb.widgetForAction(action)
            if widget is not None:
                total += max(widget.sizeHint().width(), 8)
            elif action.isSeparator():
                total += 12
            else:
                total += 56
        return total + 16

    def _adapt_toolbar_layout(self) -> None:
        if not hasattr(self, "_tb_main"):
            return
        if not self._tb_split:
            self._tb_needed_width = max(
                self._tb_needed_width, self._toolbar_content_width(self._tb_main)
            )
        needed = self._tb_needed_width
        if needed <= 0:
            return
        avail = max(self.width() - 24, 0)
        # Hysteresis avoids flicker at the decision boundary.
        if not self._tb_split and avail < needed:
            self._set_toolbar_split(True)
        elif self._tb_split and avail > needed + 80:
            self._set_toolbar_split(False)
            self._tb_needed_width = self._toolbar_content_width(self._tb_main)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._adapt_toolbar_layout()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._adapt_toolbar_layout)

    def _build(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self._tb_action("New Project…", "new", self.new_project, shortcut="Ctrl+N"))
        file_menu.addAction(self._tb_action("Open Project…", "open", self.open_project, shortcut="Ctrl+O"))
        file_menu.addAction(self._tb_action("Add Video…", "add_video", self.add_video))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, triggered=self.close))

        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(
            self._tb_action("Generate Satellite Video…", "satellite", self.generate_satellite_video)
        )
        tools_menu.addAction(self._tb_action("Edit Landmarks…", "landmarks", self.edit_landmarks))
        tools_menu.addAction(self._tb_action("Design Camera Path…", "path", self.design_camera_path))
        tools_menu.addAction(self._tb_action("Build Test Videos…", "test", self.build_test_videos))
        tools_menu.addAction(self._tb_action("Run Full Pipeline", "pipeline", self.run_full_pipeline))
        tools_menu.addAction(
            self._tb_action("Live Test Model on Video…", "test", self.test_model_on_video)
        )
        tools_menu.addAction(
            self._tb_action(
                "Detection Filters… / فیلتر تشخیص اشتباه", "settings", self.edit_detection_filters,
                tip="False-target filters for Live Test and the Jetson package",
            )
        )
        tools_menu.addAction(
            self._tb_action("Copy Package to USB…", "usb", self.transfer_package_to_usb)
        )

        atlas_menu = menubar.addMenu("Region &Atlas")
        atlas_menu.addAction(
            self._tb_action(
                "Build Region Base Model…", "satellite", self.build_region_atlas,
                tip="Select a large area and train one base model for all of it",
            )
        )
        atlas_menu.addAction(
            self._tb_action(
                "Fine-tune Sub-region from Atlas…", "train", self.finetune_from_atlas,
                tip="Pick a target inside an atlas — fast fine-tune from the base model",
            )
        )
        self._build_offline_menu()
        self._build_serial_menu()

        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(240)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        title_v = QLabel("VIDEOS")
        title_v.setObjectName("panelTitle")
        lv.addWidget(title_v)
        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self._on_video_selected)
        lv.addWidget(self.video_list, stretch=2)
        self.meta_label = QLabel("—")
        self.meta_label.setWordWrap(True)
        self.meta_label.setStyleSheet("color:#94a3b8;font-size:12px;")
        lv.addWidget(self.meta_label)
        title_l = QLabel("LOG")
        title_l.setObjectName("panelTitle")
        lv.addWidget(title_l)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setStyleSheet(
            "font-family:Consolas,'Cascadia Mono',monospace;font-size:11px;"
            "background:#0b1220;border:1px solid #243041;border-radius:6px;"
        )
        lv.addWidget(self.log_view, stretch=1)
        log_row = QHBoxLayout()
        b_log_clear = QPushButton("Clear")
        b_log_clear.clicked.connect(self.log_view.clear)
        b_log_save = QPushButton("Save log…")
        b_log_save.setIcon(get_icon("save"))
        b_log_save.clicked.connect(self._save_log)
        log_row.addWidget(b_log_clear)
        log_row.addWidget(b_log_save)
        lv.addLayout(log_row)
        splitter.addWidget(left)

        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(4, 4, 4, 4)
        self.mid_tabs = QTabWidget()
        self.mid_tabs.setDocumentMode(True)

        video_tab = QWidget()
        vv = QVBoxLayout(video_tab)
        vv.setContentsMargins(4, 4, 4, 4)
        vv.setSpacing(6)
        self.canvas = FrameCanvas()
        self.canvas.roi_done.connect(self._on_roi)
        vv.addWidget(self.canvas, stretch=1)

        transport = QHBoxLayout()
        transport.setSpacing(6)
        transport.addStretch(1)
        for act in (self.act_prev, self.act_play, self.act_stop, self.act_next):
            btn = QPushButton(act.icon(), act.text())
            btn.setToolTip(act.toolTip() or act.text())
            btn.setMinimumHeight(32)
            btn.setMinimumWidth(72)
            btn.clicked.connect(lambda _checked=False, a=act: a.trigger())
            if act is self.act_play:
                self.btn_play = btn
            transport.addWidget(btn)
        transport.addStretch(1)
        vv.addLayout(transport)

        tl = QHBoxLayout()
        tl.setSpacing(8)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._seek)
        self.frame_spin = QSpinBox()
        self.frame_spin.setFixedWidth(90)
        self.frame_spin.valueChanged.connect(self.slider.setValue)
        frame_lbl = QLabel("Frame")
        frame_lbl.setStyleSheet("color:#94a3b8;")
        tl.addWidget(self.slider, stretch=1)
        tl.addWidget(frame_lbl)
        tl.addWidget(self.frame_spin)
        vv.addLayout(tl)
        self.preview = QLabel("ROI preview")
        self.preview.setFixedHeight(110)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            "background:#0b1220;color:#64748b;border:1px solid #243041;border-radius:8px;"
        )
        vv.addWidget(self.preview)

        self.train_chart = TrainLiveChart()
        self.mid_tabs.addTab(video_tab, "Video / ویدیو")
        self.mid_tabs.addTab(self.train_chart, "Training / آموزش")
        self._tab_video = 0
        self._tab_train = 1
        mv.addWidget(self.mid_tabs, stretch=1)
        splitter.addWidget(mid)

        right = QWidget()
        right.setMinimumWidth(280)
        right.setMaximumWidth(360)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.setSpacing(8)
        box = QGroupBox("Training Settings")
        form = QFormLayout(box)
        form.setSpacing(8)
        self.model_type = QComboBox()
        self.model_type.addItems(["segmentation", "detection"])
        form.addRow(make_setting_label(box, "Model type", "model_type"), self.model_type)
        self.every_n = QSpinBox()
        self.every_n.setRange(1, 30)
        self.every_n.setValue(1)
        form.addRow(make_setting_label(box, "Every N", "every_n"), self.every_n)
        self.aug_n = QSpinBox()
        self.aug_n.setRange(0, 10)
        self.aug_n.setValue(1)
        form.addRow(make_setting_label(box, "Augment / image", "aug_n"), self.aug_n)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(40)
        form.addRow(make_setting_label(box, "Epochs", "epochs"), self.epochs)
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(4)
        form.addRow(make_setting_label(box, "Batch", "batch"), self.batch)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        form.addRow(make_setting_label(box, "Imgsz", "imgsz"), self.imgsz)
        self.device = QComboBox()
        self._gpu = gpu_name()
        self._cuda = torch_cuda_ok()
        self.device.addItem("Auto (GPU if available)", "auto")
        if self._gpu:
            self.device.addItem(f"GPU: {self._gpu}" + ("" if self._cuda else "  (needs CUDA PyTorch)"), "0")
        self.device.addItem("CPU", "cpu")
        if self._gpu and self._cuda:
            self.device.setCurrentIndex(self.device.findData("0"))
        self.device.setToolTip(
            "Training device. GPU is much faster; the weights are the same for the Jetson either way."
        )
        self.device.currentIndexChanged.connect(self._on_device_changed)
        form.addRow(make_setting_label(box, "Train on", "device"), self.device)
        self.train_status = QLabel("آموزش انجام نشده / Not trained yet")
        self.train_status.setWordWrap(True)
        self.train_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.train_status.setMinimumHeight(52)
        self.train_status.setStyleSheet(
            "color:#94a3b8;font-size:12px;padding:10px 10px;"
            "background:#151b24;border:1px solid #243041;border-radius:6px;"
        )
        form.addRow(self.train_status)
        rv.addWidget(box)

        self.btn_train = None
        self.btn_build = None
        self.btn_pipeline = QPushButton("  Run Full Pipeline")
        self.btn_pipeline.setObjectName("primaryButton")
        self.btn_pipeline.setIcon(get_icon("pipeline"))
        self.btn_pipeline.setIconSize(QSize(18, 18))
        self.btn_pipeline.setToolTip("Dataset → Train → ONNX → Jetson package → evaluation")
        self.btn_pipeline.clicked.connect(self.run_full_pipeline)
        rv.addWidget(self.btn_pipeline)

        actions_box = QGroupBox("Actions")
        actions_outer = QVBoxLayout(actions_box)
        actions_outer.setContentsMargins(6, 6, 6, 6)
        actions_outer.setSpacing(0)
        actions_inner = QWidget()
        actions_layout = QVBoxLayout(actions_inner)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)
        for text, icon_name, slot, key in [
            ("Region Atlas: Build Base Model", "satellite", self.build_region_atlas, None),
            ("Region Atlas: Fine-tune Sub-region", "train", self.finetune_from_atlas, None),
            ("Edit Landmarks", "landmarks", self.edit_landmarks, None),
            ("Camera Path", "path", self.design_camera_path, None),
            ("Build Test Videos", "test", self.build_test_videos, None),
            ("Live Test Model on Video", "test", self.test_model_on_video, None),
            ("Build Dataset", "dataset", self.build_dataset, "build"),
            ("Train YOLO", "train", self.train_yolo, "train"),
            ("Training Report", "report", self.show_last_train_report, None),
            ("Export ONNX", "onnx", self.export_onnx, None),
            ("Build Jetson Package", "package", self.build_package, None),
            ("Copy Package to USB", "usb", self.transfer_package_to_usb, None),
            ("Validate Package", "validate", self.validate_pkg, None),
        ]:
            b = QPushButton(f"  {text}")
            b.setIcon(get_icon(icon_name))
            b.setIconSize(QSize(16, 16))
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(slot)
            actions_layout.addWidget(b)
            if key == "train":
                self.btn_train = b
            if key == "build":
                self.btn_build = b
        actions_scroll = QScrollArea()
        actions_scroll.setWidgetResizable(True)
        actions_scroll.setWidget(actions_inner)
        no_frame = getattr(QFrame, "NoFrame", None) or QFrame.Shape.NoFrame
        actions_scroll.setFrameShape(no_frame)
        h_off = getattr(Qt, "ScrollBarAlwaysOff", None) or Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        v_on = getattr(Qt, "ScrollBarAsNeeded", None) or Qt.ScrollBarPolicy.ScrollBarAsNeeded
        actions_scroll.setHorizontalScrollBarPolicy(h_off)
        actions_scroll.setVerticalScrollBarPolicy(v_on)
        actions_scroll.setMinimumHeight(120)
        actions_outer.addWidget(actions_scroll)
        rv.addWidget(actions_box, stretch=1)

        self.task_progress = QProgressBar()
        self.task_progress.setRange(0, 100)
        self.task_progress.setValue(0)
        self.task_progress.setTextVisible(True)
        self.task_progress.setFormat("%p%")
        self.task_progress.setMinimumHeight(22)
        rv.addWidget(self.task_progress)
        self.task_stage = QLabel("")
        self.task_stage.setStyleSheet("font-weight:600;color:#14b8a6;")
        rv.addWidget(self.task_stage)
        self.status_box = QLabel("Ready")
        self.status_box.setWordWrap(True)
        self.status_box.setStyleSheet(
            "background:#151b24;border:1px solid #243041;border-radius:8px;"
            "padding:10px;color:#cbd5e1;"
        )
        rv.addWidget(self.status_box)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([280, 780, 320])

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._play_step)
        self._pending_roi = None
        self.statusBar().showMessage("Ready")

    def new_project(self):
        path = QFileDialog.getExistingDirectory(self, "Select empty folder for project")
        if not path:
            return
        name, ok = "project", True
        from shared.qt_compat import QInputDialog

        try:
            name, ok = QInputDialog.getText(self, "Project name", "Name:", text="target_region_project")
        except Exception:
            name = Path(path).name
        if not ok:
            return
        try:
            if self.project:
                self.project.close()
            self.project = None
            self._unload_video()
            self.project = ProjectManager.create(path, name or "project")
            self._remember_project(path)
            self._refresh_videos()
            self._update_window_title()
            self._refresh_train_status()
            self.status_box.setText(f"Created project at {path}")
        except Exception as exc:  # noqa: BLE001
            self.project = None
            self._update_window_title()
            self._refresh_train_status()
            QMessageBox.critical(self, "Error", str(exc))

    def open_project(self):
        path = QFileDialog.getExistingDirectory(self, "Open project folder")
        if not path:
            return
        self._open_project_path(path)

    def _open_project_path(self, path) -> bool:
        try:
            if self.project:
                self.project.close()
            self.project = None
            self._unload_video()
            self.project = ProjectManager.open(path)
            self._remember_project(path)
            self._refresh_videos()
            self._update_window_title()
            self._refresh_train_status()
            self.status_box.setText(f"Opened {path}")
            return True
        except Exception as exc:  # noqa: BLE001
            self.project = None
            self._update_window_title()
            self._refresh_train_status()
            QMessageBox.critical(self, "Error", str(exc))
            return False

    def add_video(self):
        if not self.project:
            QMessageBox.warning(self, "Project", "Create or open a project first")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Add video", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if not path:
            return
        try:
            asset = self.project.add_video(path)
            self._refresh_videos()
            self.status_box.setText(f"Added {asset.display_name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", str(exc))

    def generate_satellite_video(self):
        if not self.project:
            QMessageBox.warning(self, "Project", "Create or open a project first")
            return
        out_dir = self.project.exports_dir / "satellite"
        dlg = SatelliteVideoDialog(out_dir, self)
        if qt_exec(dlg) != QDialog.Accepted:
            return
        result = dlg.result or {}
        self._on_satellite_done(result)

    def _on_satellite_done(self, result: object):
        paths = result if isinstance(result, dict) else {}
        added = []
        first_row = None
        try:
            raw = paths.get("videos")
            if raw is None:
                raw = [
                    {"path": paths[k], "labels": None}
                    for k in ("zoom", "pan")
                    if paths.get(k) is not None
                ]
            items = list(raw)
            for item in items:
                vpath = Path(item["path"])
                labels = item.get("labels")
                if not labels:
                    sidecar = vpath.with_name(vpath.stem + ".labels.json")
                    if sidecar.exists():
                        labels = sidecar
                asset = self.project.add_video(vpath)
                if labels:
                    self.project.attach_auto_labels(asset.video_id, labels)
                added.append(asset.display_name)
                if first_row is None and self.project.meta:
                    first_row = len(self.project.meta.videos) - 1
            if paths.get("scene"):
                self.project.set_scene(
                    paths["scene"], paths.get("scene_preview"), paths.get("mosaic"), paths.get("candidates")
                )
            self._auto_labels_cache.clear()
            self._refresh_videos()
            classes = self.project.class_names()
            msg = f"Added {len(added)} video(s)" + (f" — classes: {', '.join(classes)}" if len(classes) > 1 else "")
            self.status_box.setText(msg)
            self.statusBar().showMessage("Satellite video ready")
            if first_row is not None:
                self.video_list.setCurrentRow(first_row)
            if not added:
                QMessageBox.warning(
                    self,
                    "Satellite Video",
                    "No videos were imported into the project.\n"
                    "Check exports/satellite — if MP4 files are there, use Add Video or Generate again.",
                )
                return
            if paths.get("run_pipeline"):
                self.run_full_pipeline()
            else:
                if first_row is not None:
                    self.play()
                QMessageBox.information(self, "Satellite Video", msg)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Satellite Video", str(exc))

    # ------------------------------------------------------------ automatic pipeline

    # ------------------------------------------------------------ log
    def log(self, msg: str, level: str = "info") -> None:
        from datetime import datetime

        prefix = {"info": "", "ok": "✔ ", "warn": "⚠ ", "error": "✖ ", "head": "■ "}.get(level, "")
        self.log_view.appendPlainText(f"[{datetime.now():%H:%M:%S}] {prefix}{msg}")
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _save_log(self) -> None:
        start = str(self.project.root / "pipeline_log.txt") if self.project else "pipeline_log.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Save log", start, "Text (*.txt)")
        if path:
            Path(path).write_text(self.log_view.toPlainText(), encoding="utf-8")

    @staticmethod
    def _fsize(path) -> str:
        try:
            n = Path(path).stat().st_size
        except OSError:
            return "missing"
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024.0
        return ""

    def _log_file(self, label: str, path) -> None:
        if path:
            self.log(f"   {label}: {path}  ({self._fsize(path)})")

    def _log_train_report(self, train: dict) -> None:
        """Final training metrics + saved files."""
        m = (train or {}).get("last_metrics", {}).get("metrics", {}) if train else {}
        pretty = {
            "metrics/precision(B)": "Precision (box)",
            "metrics/recall(B)": "Recall (box)",
            "metrics/mAP50(B)": "mAP50 (box)",
            "metrics/mAP50-95(B)": "mAP50-95 (box)",
            "metrics/precision(M)": "Precision (mask)",
            "metrics/recall(M)": "Recall (mask)",
            "metrics/mAP50(M)": "mAP50 (mask)",
            "metrics/mAP50-95(M)": "mAP50-95 (mask)",
        }
        self.log("Training finished / آموزش تمام شد", "head")
        self.log(f"   device: {train.get('device', '?')}   epochs run: {(train.get('last_metrics') or {}).get('epoch', '?')}")
        for k, name in pretty.items():
            if k in m:
                self.log(f"   {name:18s} {m[k]:.3f}")
        best = train.get("best_weights")
        self._log_file("best weights (best.pt)", best)
        if best:
            last = Path(best).with_name("last.pt")
            if last.exists():
                self._log_file("last weights", last)
        res_dir = train.get("results_dir")
        if res_dir:
            self.log(f"   training plots / results: {res_dir}")
            for f in ("results.csv", "results.png", "confusion_matrix.png", "BoxPR_curve.png", "MaskPR_curve.png"):
                if (Path(res_dir) / f).exists():
                    self._log_file(f, Path(res_dir) / f)

    def _device_value(self) -> str:
        return self.device.currentData() or "auto"

    def _on_device_changed(self, _i: int = 0) -> None:
        if self._device_value() == "0" and not self._cuda:
            QMessageBox.information(
                self,
                "GPU",
                "CUDA PyTorch is not available in this environment.\n"
                "Install a CUDA build of torch, or switch Train on → CPU / Auto.",
            )

    def _set_task_progress(self, stage: str, pct, msg: str = "") -> None:
        names = {
            "dataset": "Building dataset / ساخت دیتاست",
            "train": "Training / آموزش",
            "onnx": "Exporting ONNX",
            "package": "Jetson package / پکیج جتسون",
            "evaluate": "Evaluating / ارزیابی",
            "render": "Rendering videos / ساخت ویدیو",
            "atlas": "Region atlas / اطلس منطقه",
            "done": "Done / پایان",
        }
        if stage != getattr(self, "_last_stage", None):
            self.log(names.get(stage, stage), "head")
            self._last_stage = stage
        if msg and " batch " not in msg and msg != getattr(self, "_last_log_msg", None):
            self.log(msg)
            self._last_log_msg = msg
        self.task_stage.setText(names.get(stage, stage))
        if isinstance(pct, dict):
            pct = pct.get("pct")
        if pct is None:
            self.task_progress.setRange(0, 0)  # busy indicator
        else:
            self.task_progress.setRange(0, 100)
            self.task_progress.setValue(int(round(float(pct))))
        if msg:
            self.status_box.setText(msg)

    def _unload_video(self) -> None:
        """Drop the open reader / canvas so a previous project's video is not shown."""
        self.pause()
        if self.reader:
            self.reader.release()
            self.reader = None
        self.current_video_id = None
        self._auto_labels_cache.clear()
        self.canvas.clear()
        self.preview.clear()
        self.preview.setText("ROI preview")
        self.meta_label.setText("—")
        self.slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setValue(0)
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setValue(0)
        self.slider.blockSignals(False)
        self.frame_spin.blockSignals(False)

    def _refresh_videos(self):
        self.video_list.blockSignals(True)
        self.video_list.clear()
        if self.project and self.project.meta:
            for v in self.project.meta.videos:
                tag = " [auto-labeled]" if self.project.auto_labels_path(v.video_id) else ""
                self.video_list.addItem(
                    f"{v.display_name}{tag} [{v.video_id}] {v.width}x{v.height} {v.fps:.1f}fps {v.frame_count}f"
                )
        self.video_list.blockSignals(False)
        if self.video_list.count() == 0:
            self._unload_video()

    def _refresh_train_status(self) -> None:
        """Show whether this target is trained; next train replaces the previous model."""
        trained = bool(self.project) and self._has_trained_model()
        learning = None
        if trained and self.project:
            try:
                learning = summarize_learning_status(runs_dir=self.project.runs_dir)
            except Exception:  # noqa: BLE001
                learning = None

        chart = getattr(self, "train_chart", None)
        if chart is not None and not getattr(self, "_train_chart_armed", False):
            chart.set_learning_status(learning)

        lbl = getattr(self, "train_status", None)
        if lbl is not None:
            style_idle = (
                "color:#94a3b8;font-size:12px;padding:10px 10px;"
                "background:#151b24;border:1px solid #243041;border-radius:6px;"
            )
            style_ok = (
                "color:#5eead4;font-size:12px;padding:10px 10px;"
                "background:#0f2a28;border:1px solid #134e4a;border-radius:6px;"
            )
            if not self.project:
                lbl.setText("آموزش انجام نشده / Not trained yet")
                lbl.setStyleSheet(style_idle)
                lbl.setMinimumHeight(52)
                lbl.setToolTip("")
            elif trained:
                name = ""
                if self.project.meta:
                    name = (self.project.meta.project_name or "").strip()
                who = f" برای «{name}»" if name else ""
                grade_line = ""
                if learning is not None:
                    grade_line = f"{learning.grade_fa} · {learning.grade.upper()}"
                    if learning.map50 is not None:
                        grade_line += f" · mAP50={learning.map50:.3f}"
                    grade_line += "\n"
                    color = GRADE_COLOR.get(learning.grade, "#5eead4")
                    style_ok = (
                        f"color:{color};font-size:12px;padding:10px 10px;"
                        "background:#0f2a28;border:1px solid #134e4a;border-radius:6px;"
                    )
                lbl.setText(
                    f"✓ آموزش انجام شده{who}\n"
                    f"{grade_line}"
                    "ترین بعدی جایگزین قبلی می‌شود · replaces previous"
                )
                lbl.setStyleSheet(style_ok)
                lbl.setMinimumHeight(88 if learning is not None else 72)
                tip = (
                    "A model already exists for this target. "
                    "Train or Run Full Pipeline will overwrite the previous weights."
                )
                if learning is not None:
                    tip = f"{learning.headline}\n\n{tip}"
                lbl.setToolTip(tip)
            else:
                lbl.setText("آموزش انجام نشده / Not trained yet")
                lbl.setStyleSheet(style_idle)
                lbl.setMinimumHeight(52)
                lbl.setToolTip("Train or Run Full Pipeline to create the first model for this target.")
            lbl.adjustSize()
            lbl.updateGeometry()
            box = lbl.parentWidget()
            if box is not None:
                box.updateGeometry()

        pipe_running = (
            getattr(self, "_pipeline_worker", None) is not None
            and self._pipeline_worker.isRunning()
        )
        if getattr(self, "btn_pipeline", None) is not None and not pipe_running:
            if trained:
                self.btn_pipeline.setText("  Run Full Pipeline (retrain)")
                self.btn_pipeline.setToolTip(
                    "Dataset → Train → ONNX → Jetson package → evaluation\n"
                    "Already trained — this run replaces the previous model / "
                    "آموزش قبلی جایگزین می‌شود"
                )
            else:
                self.btn_pipeline.setText("  Run Full Pipeline")
                self.btn_pipeline.setToolTip("Dataset → Train → ONNX → Jetson package → evaluation")
            self.btn_pipeline.setIcon(get_icon("pipeline"))

        if getattr(self, "btn_train", None) is not None:
            if trained:
                self.btn_train.setText("  Retrain YOLO (replaces previous)")
                self.btn_train.setToolTip(
                    "Train again — overwrites the previous weights for this target / "
                    "آموزش مجدد؛ مدل قبلی جایگزین می‌شود"
                )
            else:
                self.btn_train.setText("  Train YOLO")
                self.btn_train.setToolTip("Train YOLO model")

        act = getattr(self, "act_train", None)
        if act is not None:
            if trained:
                act.setText("Retrain")
                act.setToolTip(
                    "Retrain — replaces previous model / آموزش مجدد جایگزین مدل قبلی"
                )
            else:
                act.setText("Train")
                act.setToolTip("Train YOLO model")

        act_pipe = getattr(self, "act_pipeline", None)
        if act_pipe is not None and not pipe_running:
            if trained:
                act_pipe.setToolTip(
                    "Full pipeline — retrain replaces previous model / "
                    "ترین بعدی جایگزین قبلی"
                )
            else:
                act_pipe.setToolTip("Dataset → Train → ONNX → Jetson package")

    def _on_video_selected(self, row: int):
        if row < 0 or not self.project or not self.project.meta:
            self._unload_video()
            return
        asset = self.project.meta.videos[row]
        self.current_video_id = asset.video_id
        if self.reader:
            self.reader.release()
        self.reader = VideoReader(str(self.project.resolve_video(asset.video_id)), prefer_gstreamer=False)
        n = max(asset.frame_count - 1, 0)
        self.slider.setRange(0, n)
        self.frame_spin.setRange(0, n)
        self.meta_label.setText(
            f"{asset.relative_path}\n{asset.width}x{asset.height} | {asset.fps:.2f} FPS | {asset.frame_count} frames | {asset.duration:.1f}s"
        )
        self._show_video_canvas()
        self._seek(0)

    def _seek(self, value: int):
        if not self.reader:
            return
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value)
        self.frame_spin.blockSignals(False)
        ok, frame = self.reader.seek(value)
        if not ok or frame is None:
            return
        ann = None
        if self.project and self.current_video_id:
            for a in self.project.list_annotations(self.current_video_id):
                if a.frame_id == value:
                    ann = a
                    break
        self.canvas.show_frame(frame, ann, self._auto_objects(value))
        if ann:
            self._show_preview(frame, ann.polygon_points)

    def _auto_objects(self, frame_id: int):
        if not self.project or not self.current_video_id:
            return []
        vid = self.current_video_id
        if vid not in self._auto_labels_cache:
            self._auto_labels_cache[vid] = self.project.load_auto_labels(vid)
        data = self._auto_labels_cache[vid]
        if not data or frame_id >= len(data.get("frames", [])):
            return []
        names = data.get("classes") or []
        return [
            (int(c), poly, names[int(c)] if int(c) < len(names) else str(c)) for c, poly in data["frames"][frame_id]
        ]

    def _show_preview(self, frame, points):
        try:
            from shared.geometry import extract_roi_crop

            crop, _ = extract_roi_crop(frame, np.asarray(points, dtype=np.float32))
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.preview.setPixmap(QPixmap.fromImage(qimg.copy()).scaled(200, 120, Qt.KeepAspectRatio))
        except Exception:
            self.preview.setText("Preview unavailable")

    def play(self):
        """Toggle: start playback, or pause if already playing."""
        if self._timer.isActive():
            self.pause()
            return
        if self.reader:
            self._show_video_canvas()
            self._timer.start(int(1000 / max(self.reader.fps, 1)))
            self._set_play_button(playing=True)

    def pause(self):
        self._timer.stop()
        self._set_play_button(playing=False)

    def stop(self):
        """Stop playback and jump to the first frame."""
        self.pause()
        if self.reader:
            self.slider.setValue(0)

    def _set_play_button(self, playing: bool):
        if playing:
            tip = "Pause playback"
            text, icon = "Pause", get_icon("pause")
        else:
            tip = "Play / pause current video"
            text, icon = "Play", get_icon("play")
        self.act_play.setText(text)
        self.act_play.setIcon(icon)
        self.act_play.setToolTip(tip)
        self.act_play.setStatusTip(tip)
        if self.btn_play is not None:
            self.btn_play.setText(text)
            self.btn_play.setIcon(icon)
            self.btn_play.setToolTip(tip)

    def prev_frame(self):
        if self.reader:
            self.slider.setValue(max(0, self.slider.value() - 1))

    def next_frame(self):
        if self.reader:
            self.slider.setValue(min(self.slider.maximum(), self.slider.value() + 1))

    def _play_step(self):
        v = self.slider.value() + 1
        if v > self.slider.maximum():
            self.pause()
            return
        self.slider.setValue(v)

    def _on_roi(self, roi):
        self._pending_roi = roi
        if self.canvas._frame is not None:
            self._show_preview(self.canvas._frame, roi.to_list())
        self.save_annotation(roi)

    def save_current_hint(self):
        if self._pending_roi is None:
            QMessageBox.information(self, "ROI", "Draw a ROI first")
            return
        self.save_annotation(self._pending_roi)

    def save_annotation(self, roi):
        if not self.project or not self.current_video_id or not self.reader:
            return
        asset = self.project.get_video(self.current_video_id)
        frame_id = self.slider.value()
        pts = roi.to_list()
        bbox = list(polygon_to_bbox(np.asarray(pts, dtype=np.float32)))
        # zoom relative to first manual of this video
        manuals = [a for a in self.project.list_annotations(self.current_video_id) if a.annotation_source == "manual"]
        zoom = 1.0
        if manuals:
            import math
            from shared.geometry import polygon_area

            a0 = max(polygon_area(np.asarray(manuals[0].polygon_points, dtype=np.float32)), 1.0)
            zoom = math.sqrt(polygon_area(np.asarray(pts, dtype=np.float32)) / a0)
        ann = Annotation(
            video_id=self.current_video_id,
            frame_id=frame_id,
            timestamp=self.reader.timestamp_of(frame_id),
            image_width=asset.width,
            image_height=asset.height,
            roi_type=roi.roi_type.value,
            polygon_points=pts,
            bounding_box=bbox,
            annotation_source=AnnotationSource.MANUAL.value,
            zoom_level=zoom,
            quality_score=1.0,
            class_name=self.project.meta.target_class if self.project.meta else "target_region",
        )
        self.project.upsert_annotation(ann)
        if self.project.meta:
            self.project.meta.model_type = self.model_type.currentText()
            self.project.save_meta()
        self._seek(frame_id)
        self.status_box.setText(f"Saved annotation at frame {frame_id}")

    def run_interpolate(self):
        if not self.project or not self.current_video_id:
            return
        manuals = self.project.list_annotations(self.current_video_id, include_interpolated=False)
        generated = interpolate_between_keyframes(manuals, class_name=self.project.meta.target_class if self.project.meta else "target_region")
        # clear old interpolated
        for a in self.project.list_annotations(self.current_video_id):
            if a.annotation_source == AnnotationSource.INTERPOLATED.value and a.annotation_id:
                self.project.delete_annotation(a.annotation_id)
        for g in generated:
            self.project.upsert_annotation(g)
        self.status_box.setText(f"Interpolated {len(generated)} frames")

    def confirm_interp(self):
        if not self.project or not self.current_video_id:
            return
        frame_id = self.slider.value()
        for a in self.project.list_annotations(self.current_video_id):
            if a.frame_id == frame_id and a.annotation_source == AnnotationSource.INTERPOLATED.value:
                c = confirm_interpolated(a)
                if a.annotation_id:
                    self.project.delete_annotation(a.annotation_id)
                self.project.upsert_annotation(c)
                self.status_box.setText(f"Confirmed frame {frame_id}")
                return

    def closeEvent(self, e):  # noqa: N802
        self.pause()
        if self.reader:
            self.reader.release()
        if self.project:
            self.project.close()
        e.accept()


# Ensure QInputDialog available
try:
    from shared.qt_compat import QInputDialog  # type: ignore
except Exception:
    try:
        from PySide6.QtWidgets import QInputDialog
        import shared.qt_compat as qc
        qc.QInputDialog = QInputDialog
    except Exception:
        pass


def main():
    setup_logging()
    # Add QInputDialog to qt_compat dynamically if missing
    import shared.qt_compat as qc
    if not hasattr(qc, "QInputDialog"):
        try:
            from PySide6.QtWidgets import QInputDialog as _QID
            qc.QInputDialog = _QID
        except Exception:
            from PyQt5.QtWidgets import QInputDialog as _QID
            qc.QInputDialog = _QID
    from shared.app_icon import apply_app_icon, ensure_windows_app_id

    ensure_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("سامانه تنگسیری")
    app.setOrganizationName("GeoFinder")
    apply_theme(app)
    win = DesktopMainWindow()
    apply_app_icon(app, win)
    win.show()
    return qt_exec(app)


if __name__ == "__main__":
    raise SystemExit(main())
