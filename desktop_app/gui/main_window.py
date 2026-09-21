"""Desktop Dataset & Deployment Manager GUI."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from shared.geometry import ROISelector, polygon_to_bbox
from shared.logging import setup_logging
from shared.qt_compat import (
    Qt,
    QAction,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSlider,
    QSplitter,
    QThread,
    QTimer,
    QVBoxLayout,
    QWidget,
    QSpinBox,
    QComboBox,
    QPushButton,
    QListWidget,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QDoubleSpinBox,
    QCheckBox,
    QApplication,
    QImage,
    QPixmap,
    Signal,
    qt_exec,
    mouse_xy,
)
from shared.schemas import Annotation, AnnotationSource, ROIType
from shared.video_reader import VideoReader
from desktop_app.annotation.interpolate import confirm_interpolated, interpolate_between_keyframes
from desktop_app.dataset.builder import DatasetBuilder
from desktop_app.packaging.builder import PackageBuilder, validate_package
from desktop_app.pipeline.auto import AutoPipeline, AutoPipelineConfig
from desktop_app.project_manager.manager import ProjectManager
from desktop_app.satellite import generate_satellite_videos
from desktop_app.satellite.map_widget import SatelliteMapPanel
from desktop_app.training.trainer import TrainingRunner


class SatelliteVideoDialog(QDialog):
    """Interactive satellite map: select region, then generate zoom/pan videos."""

    def __init__(self, out_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Satellite Video")
        self.resize(1100, 720)
        self.out_dir = Path(out_dir)
        self.result: Optional[Dict[str, Any]] = None
        self._worker: Optional[SatelliteWorker] = None
        self._busy = False

        root = QHBoxLayout(self)
        self.map_panel = SatelliteMapPanel()
        root.addWidget(self.map_panel, stretch=3)

        side = QWidget()
        form = QFormLayout(side)
        form.addRow(
            QLabel(
                "1. Search a place or pan/zoom\n"
                "2. Toggle Satellite / Map view\n"
                "3. Select Region → drag a box (the target)\n"
                "4. Generate — videos are labeled automatically\n"
                "   (target + landmarks around it), then the\n"
                "   dataset/train/ONNX/package pipeline can run by itself"
            )
        )

        jump_row = QHBoxLayout()
        self.jump_lat = QDoubleSpinBox()
        self.jump_lat.setDecimals(5)
        self.jump_lat.setRange(-85.0, 85.0)
        self.jump_lat.setValue(35.68920)
        self.jump_lon = QDoubleSpinBox()
        self.jump_lon.setDecimals(5)
        self.jump_lon.setRange(-180.0, 180.0)
        self.jump_lon.setValue(51.38900)
        btn_go = QPushButton("Go")
        btn_go.clicked.connect(self._jump)
        jump_row.addWidget(self.jump_lat)
        jump_row.addWidget(self.jump_lon)
        jump_row.addWidget(btn_go)
        form.addRow("Jump lat/lon", jump_row)

        self.tile_zoom = QSpinBox()
        self.tile_zoom.setRange(12, 19)
        self.tile_zoom.setValue(17)
        self.tile_zoom.setToolTip("Mosaic download zoom (detail). Map zoom is independent.")
        form.addRow("Download zoom", self.tile_zoom)

        self.zoom_out = QSpinBox()
        self.zoom_out.setRange(4, 24)
        self.zoom_out.setValue(12)
        self.zoom_out.setToolTip(
            "How many times wider the video starts vs your selection. "
            "Higher = starts more zoomed out, then flies into the box."
        )
        form.addRow("Zoom depth (×)", self.zoom_out)

        self.duration = QDoubleSpinBox()
        self.duration.setRange(2.0, 60.0)
        self.duration.setValue(8.0)
        form.addRow("Duration (s)", self.duration)

        self.fps = QSpinBox()
        self.fps.setRange(10, 60)
        self.fps.setValue(24)
        form.addRow("FPS", self.fps)

        self.mode_zoom = QCheckBox("Zoom flyover")
        self.mode_zoom.setChecked(True)
        self.mode_pan = QCheckBox("Horizontal pan")
        self.mode_pan.setChecked(True)
        modes = QVBoxLayout()
        modes.addWidget(self.mode_zoom)
        modes.addWidget(self.mode_pan)
        form.addRow("Modes", modes)

        self.auto_label = QCheckBox("Auto-label target + landmarks")
        self.auto_label.setChecked(True)
        form.addRow(self.auto_label)

        self.n_landmarks = QSpinBox()
        self.n_landmarks.setRange(0, 8)
        self.n_landmarks.setValue(4)
        self.n_landmarks.setToolTip(
            "Distinctive objects around the target found automatically. The model learns them too, "
            "so the target can be located from its surroundings when it is small or out of view."
        )
        form.addRow("Landmarks", self.n_landmarks)

        self.n_videos = QSpinBox()
        self.n_videos.setRange(1, 16)
        self.n_videos.setValue(6)
        self.n_videos.setToolTip("Varied clips: rotated/off-centre zooms, low passes, lighting changes")
        form.addRow("Training videos", self.n_videos)

        self.run_pipeline = QCheckBox("Then run full pipeline (dataset → train → ONNX → package)")
        self.run_pipeline.setChecked(True)
        form.addRow(self.run_pipeline)
        self.auto_label.toggled.connect(self.n_landmarks.setEnabled)
        self.auto_label.toggled.connect(self.n_videos.setEnabled)
        self.auto_label.toggled.connect(self.run_pipeline.setEnabled)

        self.sel_label = QLabel("No region selected")
        self.sel_label.setWordWrap(True)
        form.addRow(self.sel_label)
        self.map_panel.map.selection_changed.connect(self._sync_selection_label)
        self.map_panel.map.view_changed.connect(self._sync_selection_label)

        self.preview = QLabel("Video preview")
        self.preview.setMinimumHeight(200)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#1a1a1a;color:#aaa;border:1px solid #333;")
        self.preview.setScaledContents(False)
        form.addRow(self.preview)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        form.addRow(self.progress)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        form.addRow(self.progress_label)

        buttons = QDialogButtonBox()
        self.btn_generate = buttons.addButton("Generate Video", QDialogButtonBox.AcceptRole)
        self.btn_cancel = buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self._on_cancel)
        form.addRow(buttons)

        root.addWidget(side, stretch=1)

    def _jump(self) -> None:
        self.map_panel.map.go_to(self.jump_lat.value(), self.jump_lon.value())

    def _sync_selection_label(self) -> None:
        bbox = self.map_panel.map.selection_bbox()
        if not bbox:
            self.sel_label.setText("No region selected — use Select Region and drag a box")
            return
        s, w, n, e = bbox
        self.sel_label.setText(f"S={s:.6f}\nW={w:.6f}\nN={n:.6f}\nE={e:.6f}")

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (
            self.jump_lat,
            self.jump_lon,
            self.tile_zoom,
            self.zoom_out,
            self.duration,
            self.fps,
            self.mode_zoom,
            self.mode_pan,
            self.auto_label,
            self.n_landmarks,
            self.n_videos,
            self.run_pipeline,
            self.btn_generate,
            self.map_panel,
        ):
            w.setEnabled(enabled)

    def _try_accept(self) -> None:
        if self._busy:
            return
        if not self.map_panel.map.selection_bbox():
            QMessageBox.warning(
                self,
                "Select Region",
                "Select a region on the satellite map first.\n"
                "Click “Select Region”, then drag a rectangle.",
            )
            return
        modes: List[str] = []
        if self.mode_zoom.isChecked():
            modes.append("zoom")
        if self.mode_pan.isChecked():
            modes.append("pan")
        if not modes:
            QMessageBox.warning(self, "Modes", "Select at least one mode (zoom or pan)")
            return

        self._busy = True
        self._set_controls_enabled(False)
        self.progress.setValue(0)
        self.progress_label.setText("Starting…")
        self.preview.setText("Downloading…")
        self.btn_cancel.setText("Cancel")

        self._worker = SatelliteWorker(self.params(), self.out_dir)
        self._worker.progress.connect(self._on_progress)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished_ok.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.progress.setValue(max(0, min(100, int(100 * done / total))))
        self.progress_label.setText(message)

    def _on_frame(self, frame: object) -> None:
        if frame is None or not isinstance(frame, np.ndarray):
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy()).scaled(
            self.preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview.setPixmap(pix)

    def _on_worker_done(self, result: object) -> None:
        self._busy = False
        self.result = result if isinstance(result, dict) else {}
        self.progress.setValue(100)
        self.progress_label.setText("Done")
        self.accept()

    def _on_worker_failed(self, err: str) -> None:
        self._busy = False
        self._set_controls_enabled(True)
        self.progress_label.setText(f"Failed: {err}")
        QMessageBox.critical(self, "Satellite Video", err)

    def _on_cancel(self) -> None:
        if self._busy and self._worker and self._worker.isRunning():
            # Soft cancel: disable further UI updates; thread finishes in background
            self._busy = False
            try:
                self._worker.progress.disconnect(self._on_progress)
                self._worker.frame_ready.disconnect(self._on_frame)
                self._worker.finished_ok.disconnect(self._on_worker_done)
                self._worker.failed.disconnect(self._on_worker_failed)
            except (TypeError, RuntimeError):
                pass
        self.reject()

    def closeEvent(self, event):  # noqa: N802
        if self._busy and self._worker and self._worker.isRunning():
            self._on_cancel()
        super().closeEvent(event)

    def params(self) -> Dict[str, Any]:
        bbox = self.map_panel.map.selection_bbox()
        assert bbox is not None
        south, west, north, east = bbox
        modes: List[str] = []
        if self.mode_zoom.isChecked():
            modes.append("zoom")
        if self.mode_pan.isChecked():
            modes.append("pan")
        lat = (south + north) / 2.0
        lon = (west + east) / 2.0
        span = max(north - south, east - west)
        return {
            "lat": lat,
            "lon": lon,
            "span": span,
            "south": south,
            "west": west,
            "north": north,
            "east": east,
            "zoom": int(self.tile_zoom.value()),
            "zoom_out_factor": float(self.zoom_out.value()),
            "duration_s": float(self.duration.value()),
            "fps": float(self.fps.value()),
            "modes": modes,
            "auto_label": self.auto_label.isChecked(),
            "num_landmarks": int(self.n_landmarks.value()),
            "n_videos": int(self.n_videos.value()),
            "run_pipeline": self.auto_label.isChecked() and self.run_pipeline.isChecked(),
        }


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


class AutoPipelineWorker(QThread):
    """Dataset → train → ONNX → package → evaluation, off the UI thread."""

    progress = Signal(str, str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, project_root: Path, cfg: AutoPipelineConfig):
        super().__init__()
        self.pipeline = AutoPipeline(project_root, cfg, progress_cb=lambda st, m: self.progress.emit(st, m))

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


class FrameCanvas(QLabel):
    roi_done = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#1a1a1a;color:#aaa;")
        self.setText("Open a project and select a video")
        self._frame = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._mode = None
        self._points: List = []
        self._rect_start = None
        self._cursor = None
        self._overlay_ann = None
        self._overlay_objects: List = []
        self.setMouseTracking(True)

    def set_mode(self, mode: Optional[str]) -> None:
        self._mode = mode
        self._points = []
        self._rect_start = None
        self.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)

    def show_frame(self, frame, annotation=None, objects=None) -> None:
        """``objects``: auto labels [(class_id, polygon, name), ...]."""
        self._frame = frame
        self._overlay_ann = annotation
        self._overlay_objects = objects or []
        self._paint()

    def _paint(self) -> None:
        if self._frame is None:
            return
        img = self._frame.copy()
        if self._overlay_ann is not None:
            pts = np.array(self._overlay_ann.polygon_points, dtype=np.int32)
            color = (0, 255, 255) if self._overlay_ann.annotation_source == "interpolated" else (0, 220, 0)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
        for cid, poly, name in self._overlay_objects:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            color = (0, 220, 0) if cid == 0 else (0, 160, 255)
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
            x, y = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
            cv2.putText(img, name, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if self._mode == "polygon" and self._points:
            pts = np.array(self._points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)
        if self._mode == "rectangle" and self._rect_start and self._cursor:
            cv2.rectangle(img, (int(self._rect_start[0]), int(self._rect_start[1])), (int(self._cursor[0]), int(self._cursor[1])), (0, 255, 255), 2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy()).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scale = pix.width() / w
        self._offset = ((self.width() - pix.width()) // 2, (self.height() - pix.height()) // 2)
        self.setPixmap(pix)

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._paint()

    def _to_image(self, x, y):
        if self._frame is None or self._scale <= 0:
            return None
        ix = (x - self._offset[0]) / self._scale
        iy = (y - self._offset[1]) / self._scale
        h, w = self._frame.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            return float(ix), float(iy)
        return None

    def mousePressEvent(self, e):  # noqa: N802
        pt = self._to_image(*mouse_xy(e))
        if not pt or not self._mode:
            return
        if self._mode == "rectangle" and e.button() == Qt.LeftButton:
            self._rect_start = pt
            self._cursor = pt
        elif self._mode == "polygon":
            if e.button() == Qt.LeftButton:
                self._points.append(pt)
            elif e.button() == Qt.RightButton and len(self._points) >= 3:
                self._finish_poly()
        self._paint()

    def mouseMoveEvent(self, e):  # noqa: N802
        pt = self._to_image(*mouse_xy(e))
        if pt:
            self._cursor = pt
            self._paint()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self._mode == "rectangle" and self._rect_start and e.button() == Qt.LeftButton:
            pt = self._to_image(*mouse_xy(e))
            if not pt:
                return
            try:
                roi = ROISelector.from_rectangle(self._rect_start[0], self._rect_start[1], pt[0], pt[1])
                self.roi_done.emit(roi)
                self._mode = None
                self.setCursor(Qt.ArrowCursor)
            except ValueError:
                pass
            self._rect_start = None
            self._paint()

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if self._mode == "polygon" and len(self._points) >= 3:
            self._finish_poly()

    def _finish_poly(self):
        try:
            roi = ROISelector.from_polygon([[p[0], p[1]] for p in self._points])
            self.roi_done.emit(roi)
            self._mode = None
            self.setCursor(Qt.ArrowCursor)
        except ValueError:
            pass
        self._paint()


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


class DesktopMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desktop Dataset & Deployment Manager")
        self.resize(1400, 900)
        self.project: Optional[ProjectManager] = None
        self.reader: Optional[VideoReader] = None
        self.current_video_id: Optional[str] = None
        self._train_worker = None
        self._dataset_worker = None
        self._pipeline_worker: Optional[AutoPipelineWorker] = None
        self._infer_worker: Optional[InferenceWorker] = None
        self._auto_labels_cache: Dict[str, Any] = {}
        self._build()

    def _build(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(QAction("New Project…", self, triggered=self.new_project))
        file_menu.addAction(QAction("Open Project…", self, triggered=self.open_project))
        file_menu.addAction(QAction("Add Video…", self, triggered=self.add_video))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, triggered=self.close))

        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction(
            QAction("Generate Satellite Video…", self, triggered=self.generate_satellite_video)
        )
        tools_menu.addAction(QAction("Run Full Pipeline", self, triggered=self.run_full_pipeline))
        tools_menu.addAction(QAction("Test Model on Video…", self, triggered=self.test_model_on_video))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        lv = QVBoxLayout(left)
        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self._on_video_selected)
        lv.addWidget(QLabel("Videos"))
        lv.addWidget(self.video_list)
        self.meta_label = QLabel("—")
        self.meta_label.setWordWrap(True)
        lv.addWidget(self.meta_label)
        self.ann_list = QListWidget()
        lv.addWidget(QLabel("Annotations (timeline)"))
        lv.addWidget(self.ann_list)
        self.ann_list.itemClicked.connect(self._on_ann_clicked)
        splitter.addWidget(left)

        mid = QWidget()
        mv = QVBoxLayout(mid)
        self.canvas = FrameCanvas()
        self.canvas.roi_done.connect(self._on_roi)
        mv.addWidget(self.canvas, stretch=1)
        tl = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._seek)
        self.frame_spin = QSpinBox()
        self.frame_spin.valueChanged.connect(self.slider.setValue)
        tl.addWidget(self.slider)
        tl.addWidget(self.frame_spin)
        mv.addLayout(tl)
        btns = QHBoxLayout()
        for text, slot in [
            ("Play", self.play),
            ("Pause", self.pause),
            ("Rect ROI", lambda: self.canvas.set_mode("rectangle")),
            ("Poly ROI", lambda: self.canvas.set_mode("polygon")),
            ("Save Ann", self.save_current_hint),
            ("Interpolate", self.run_interpolate),
            ("Confirm Interp", self.confirm_interp),
        ]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            btns.addWidget(b)
        mv.addLayout(btns)
        self.preview = QLabel("ROI preview")
        self.preview.setFixedHeight(120)
        self.preview.setAlignment(Qt.AlignCenter)
        mv.addWidget(self.preview)
        splitter.addWidget(mid)

        right = QWidget()
        rv = QVBoxLayout(right)
        box = QGroupBox("Dataset / Train / Package")
        form = QFormLayout(box)
        self.model_type = QComboBox()
        self.model_type.addItems(["segmentation", "detection"])
        form.addRow("Model type", self.model_type)
        self.every_n = QSpinBox()
        self.every_n.setRange(1, 30)
        self.every_n.setValue(1)
        form.addRow("Every N", self.every_n)
        self.aug_n = QSpinBox()
        self.aug_n.setRange(0, 10)
        self.aug_n.setValue(1)
        form.addRow("Augment / image", self.aug_n)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(40)
        form.addRow("Epochs", self.epochs)
        self.batch = QSpinBox()
        self.batch.setRange(1, 64)
        self.batch.setValue(4)
        form.addRow("Batch", self.batch)
        self.imgsz = QSpinBox()
        self.imgsz.setRange(320, 1280)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(640)
        form.addRow("Imgsz", self.imgsz)
        self.device = QLineEdit("auto")
        self.device.setToolTip("auto = CUDA GPU if available, else CPU; or 0 / cpu")
        form.addRow("Device", self.device)
        rv.addWidget(box)
        self.btn_train = None
        self.btn_build = None
        self.btn_pipeline = QPushButton("▶ Run Full Pipeline")
        self.btn_pipeline.setToolTip("Dataset → Train → ONNX → Jetson package → evaluation, automatically")
        self.btn_pipeline.setStyleSheet("font-weight:bold;padding:6px;")
        self.btn_pipeline.clicked.connect(self.run_full_pipeline)
        rv.addWidget(self.btn_pipeline)
        for text, slot in [
            ("Test Model on Video…", self.test_model_on_video),
            ("Build Dataset", self.build_dataset),
            ("Train YOLO", self.train_yolo),
            ("Export ONNX", self.export_onnx),
            ("Build Jetson Package", self.build_package),
            ("Validate Package", self.validate_pkg),
        ]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            rv.addWidget(b)
            if text == "Train YOLO":
                self.btn_train = b
            if text == "Build Dataset":
                self.btn_build = b
        self.status_box = QLabel("Ready")
        self.status_box.setWordWrap(True)
        rv.addWidget(self.status_box)
        rv.addStretch(1)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)

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
        self.project = ProjectManager.create(path, name or "project")
        self._refresh_videos()
        self.status_box.setText(f"Created project at {path}")

    def open_project(self):
        path = QFileDialog.getExistingDirectory(self, "Open project folder")
        if not path:
            return
        try:
            self.project = ProjectManager.open(path)
            self._refresh_videos()
            self.status_box.setText(f"Opened {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", str(exc))

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
            items = paths.get("videos") or [
                {"path": paths[k], "labels": None} for k in ("zoom", "pan") if paths.get(k) is not None
            ]
            for item in items:
                asset = self.project.add_video(item["path"])
                if item.get("labels"):
                    self.project.attach_auto_labels(asset.video_id, item["labels"])
                added.append(asset.display_name)
                if first_row is None and self.project.meta:
                    first_row = len(self.project.meta.videos) - 1
            if paths.get("scene"):
                self.project.set_scene(paths["scene"], paths.get("scene_preview"), paths.get("mosaic"))
            self._auto_labels_cache.clear()
            self._refresh_videos()
            classes = self.project.class_names()
            msg = f"Added {len(added)} video(s)" + (f" — classes: {', '.join(classes)}" if len(classes) > 1 else "")
            self.status_box.setText(msg)
            self.statusBar().showMessage("Satellite video ready")
            if first_row is not None:
                self.video_list.setCurrentRow(first_row)
            if paths.get("run_pipeline"):
                self.run_full_pipeline()
            else:
                if first_row is not None:
                    self.play()
                QMessageBox.information(self, "Satellite Video", msg)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Satellite Video", str(exc))

    # ------------------------------------------------------------ automatic pipeline

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
        cfg = AutoPipelineConfig(
            model_type=self.model_type.currentText(),
            epochs=self.epochs.value(),
            batch=self.batch.value(),
            imgsz=self.imgsz.value(),
            device=self.device.text().strip() or "auto",
            augment_per_image=self.aug_n.value(),
        )
        self.pause()
        self._pipeline_worker = AutoPipelineWorker(self.project.root, cfg)
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.finished_ok.connect(self._on_pipeline_ok)
        self._pipeline_worker.failed.connect(self._on_pipeline_failed)
        self.btn_pipeline.setText("■ Stop Pipeline")
        self._pipeline_worker.start()
        self.statusBar().showMessage("Automatic pipeline running…")

    def _on_pipeline_progress(self, stage: str, msg: str):
        self.status_box.setText(f"[{stage}] {msg}")
        self.statusBar().showMessage(f"Pipeline: {stage}")

    def _on_pipeline_ok(self, summary: dict):
        self.btn_pipeline.setText("▶ Run Full Pipeline")
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
        QMessageBox.information(self, "Pipeline finished", "\n\n".join(lines))

    def _on_pipeline_failed(self, err: str):
        self.btn_pipeline.setText("▶ Run Full Pipeline")
        self.statusBar().showMessage("Pipeline failed")
        self.status_box.setText(f"Pipeline failed: {err}")
        QMessageBox.critical(self, "Pipeline", err)

    def test_model_on_video(self):
        """Run the trained model on any video: target (green), from landmarks (magenta), landmarks (orange)."""
        if not self.project:
            QMessageBox.warning(self, "Test", "Create or open a project first")
            return
        weights = sorted(self.project.runs_dir.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
        if not weights:
            QMessageBox.warning(self, "Test", "Train a model first (Run Full Pipeline)")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Video to test", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if not path:
            return
        out_video = self.project.exports_dir / "evaluation" / f"{Path(path).stem}_localized.mp4"
        scene = self.project.scene_path if self.project.scene_path.exists() else None
        self._infer_worker = InferenceWorker(weights[-1], Path(path), scene, out_video, self.imgsz.value(), self.device.text().strip())
        self._infer_worker.progress.connect(self.status_box.setText)
        self._infer_worker.finished_ok.connect(self._on_infer_ok)
        self._infer_worker.failed.connect(lambda e: QMessageBox.critical(self, "Test", e))
        self._infer_worker.start()
        self.status_box.setText("Running model on video…")

    def _on_infer_ok(self, m: dict):
        msg = (
            f"Frames: {m['frames']}\n"
            f"Target detected directly: {m['target_detected_rate']:.0%}\n"
            f"Target located from landmarks: {m['inferred_from_landmarks_rate']:.0%}\n"
            f"Located overall: {m['located_rate']:.0%}\n\n"
            f"Output video:\n{m['output_video']}"
        )
        self.status_box.setText(msg)
        try:
            asset = self.project.add_video(m["output_video"])
            self._refresh_videos()
            msg += f"\n\n(added to video list as {asset.display_name})"
        except Exception:  # noqa: BLE001
            pass
        QMessageBox.information(self, "Test result", msg)

    def _refresh_videos(self):
        self.video_list.clear()
        if not self.project or not self.project.meta:
            return
        for v in self.project.meta.videos:
            tag = " [auto-labeled]" if self.project.auto_labels_path(v.video_id) else ""
            self.video_list.addItem(
                f"{v.display_name}{tag} [{v.video_id}] {v.width}x{v.height} {v.fps:.1f}fps {v.frame_count}f"
            )

    def _on_video_selected(self, row: int):
        if row < 0 or not self.project or not self.project.meta:
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
        self._seek(0)
        self._refresh_anns()

    def _refresh_anns(self):
        self.ann_list.clear()
        if not self.project or not self.current_video_id:
            return
        for a in self.project.list_annotations(self.current_video_id):
            self.ann_list.addItem(f"f{a.frame_id} {a.annotation_source} zoom={a.zoom_level:.2f}")

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
        if self.reader:
            self._timer.start(int(1000 / max(self.reader.fps, 1)))

    def pause(self):
        self._timer.stop()

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
        self._refresh_anns()
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
        self._refresh_anns()
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
                self._refresh_anns()
                self.status_box.setText(f"Confirmed frame {frame_id}")
                return

    def _on_ann_clicked(self, item):
        text = item.text()
        if text.startswith("f"):
            try:
                fid = int(text.split()[0][1:])
                self.slider.setValue(fid)
            except ValueError:
                pass

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
                "No annotations found.\n\n"
                "Open a video, draw Rect/Poly ROI, click Save Ann "
                "(optionally Interpolate + Confirm), then Build Dataset.",
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
        self._dataset_worker.progress.connect(self.status_box.setText)
        self._dataset_worker.finished_ok.connect(self._on_dataset_ok)
        self._dataset_worker.failed.connect(self._on_dataset_failed)
        self._dataset_worker.start()

    def _on_dataset_ok(self, out: str):
        if self.btn_build:
            self.btn_build.setEnabled(True)
        self.statusBar().showMessage("Ready")
        self.status_box.setText(f"Dataset built: {out}")
        QMessageBox.information(self, "Dataset", f"Created:\n{out}")

    def _on_dataset_failed(self, err: str):
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

        device = self.device.text().strip() or "cpu"
        if device not in ("cpu", "mps"):
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
                    self.device.setText("cpu")
                    self.status_box.setText("CUDA not available — using CPU")
            except Exception:
                device = "cpu"
                self.device.setText("cpu")

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
        self._train_worker.progress.connect(lambda d: self.status_box.setText(json.dumps(d)[:300]))
        self._train_worker.finished_ok.connect(self._on_train_ok)
        self._train_worker.failed.connect(self._on_train_failed)
        if self.btn_train:
            self.btn_train.setEnabled(False)
        self._train_worker.start()
        self.status_box.setText(f"Training started on {device}… (first run may download weights)")
        self.statusBar().showMessage("Training…")

    def _on_train_ok(self, result: dict):
        if self.btn_train:
            self.btn_train.setEnabled(True)
        self.statusBar().showMessage("Ready")
        self.status_box.setText(str(result)[:500])
        QMessageBox.information(self, "Train", str(result))

    def _on_train_failed(self, err: str):
        if self.btn_train:
            self.btn_train.setEnabled(True)
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

    def validate_pkg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Package ZIP", str(self.project.packages_dir if self.project else ""), "ZIP (*.zip)")
        if not path:
            return
        try:
            result = validate_package(path)
            QMessageBox.information(self, "Validate", json.dumps(result, indent=2)[:2000])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Validate", str(exc))

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
    app = QApplication(sys.argv)
    win = DesktopMainWindow()
    win.show()
    return qt_exec(app)


if __name__ == "__main__":
    raise SystemExit(main())
