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
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
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
from desktop_app.project_manager.manager import ProjectManager
from desktop_app.training.trainer import TrainingRunner


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
        self.setMouseTracking(True)

    def set_mode(self, mode: Optional[str]) -> None:
        self._mode = mode
        self._points = []
        self._rect_start = None
        self.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)

    def show_frame(self, frame, annotation=None) -> None:
        self._frame = frame
        self._overlay_ann = annotation
        self._paint()

    def _paint(self) -> None:
        if self._frame is None:
            return
        img = self._frame.copy()
        if self._overlay_ann is not None:
            pts = np.array(self._overlay_ann.polygon_points, dtype=np.int32)
            color = (0, 255, 255) if self._overlay_ann.annotation_source == "interpolated" else (0, 220, 0)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
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


class DesktopMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Desktop Dataset & Deployment Manager")
        self.resize(1400, 900)
        self.project: Optional[ProjectManager] = None
        self.reader: Optional[VideoReader] = None
        self.current_video_id: Optional[str] = None
        self._train_worker = None
        self._build()

    def _build(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(QAction("New Project…", self, triggered=self.new_project))
        file_menu.addAction(QAction("Open Project…", self, triggered=self.open_project))
        file_menu.addAction(QAction("Add Video…", self, triggered=self.add_video))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exit", self, triggered=self.close))

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
        form.addRow("Augment / image", self.aug_n)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(30)
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
        self.device = QLineEdit("0")
        form.addRow("Device", self.device)
        rv.addWidget(box)
        for text, slot in [
            ("Build Dataset", self.build_dataset),
            ("Train YOLO", self.train_yolo),
            ("Export ONNX", self.export_onnx),
            ("Build Jetson Package", self.build_package),
            ("Validate Package", self.validate_pkg),
        ]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            rv.addWidget(b)
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

    def _refresh_videos(self):
        self.video_list.clear()
        if not self.project or not self.project.meta:
            return
        for v in self.project.meta.videos:
            self.video_list.addItem(f"{v.display_name} [{v.video_id}] {v.width}x{v.height} {v.fps:.1f}fps {v.frame_count}f")

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
        self.canvas.show_frame(frame, ann)
        if ann:
            self._show_preview(frame, ann.polygon_points)

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
            return
        try:
            if self.project.meta:
                self.project.meta.model_type = self.model_type.currentText()
                self.project.save_meta()
            builder = DatasetBuilder(
                self.project,
                model_type=self.model_type.currentText(),
                every_n=self.every_n.value(),
                augment_per_image=self.aug_n.value(),
                class_name=self.project.meta.target_class if self.project.meta else "target_region",
            )
            out = builder.build()
            self.status_box.setText(f"Dataset built: {out}")
            QMessageBox.information(self, "Dataset", f"Created:\n{out}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Dataset", str(exc))

    def train_yolo(self):
        if not self.project:
            return
        datasets = sorted(self.project.datasets_dir.glob("*/data.yaml"))
        if not datasets:
            QMessageBox.warning(self, "Train", "Build a dataset first")
            return
        runner = TrainingRunner(
            datasets[-1],
            self.project.runs_dir,
            model_type=self.model_type.currentText(),
            epochs=self.epochs.value(),
            batch=self.batch.value(),
            imgsz=self.imgsz.value(),
            device=self.device.text().strip() or "cpu",
        )
        self._train_worker = TrainWorker(runner)
        self._train_worker.progress.connect(lambda d: self.status_box.setText(json.dumps(d)[:300]))
        self._train_worker.finished_ok.connect(lambda d: QMessageBox.information(self, "Train", str(d)))
        self._train_worker.failed.connect(lambda e: QMessageBox.critical(self, "Train", e))
        self._train_worker.start()
        self.status_box.setText("Training started…")

    def export_onnx(self):
        if not self.project:
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
