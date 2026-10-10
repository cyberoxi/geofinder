"""Graphical camera-path designer: keyframes on the map, live preview, labeled video output."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from desktop_app.gui.landmark_editor import MosaicView
from desktop_app.satellite.synth import (
    CameraKey,
    MosaicCamera,
    Pose,
    frame_corners,
    frame_labels,
    path_from_keyframes,
    preset_keys,
)
from shared.landmarks import SceneLayout
from shared.qt_compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QImage,
    QLabel,
    QLineEdit,
    QListWidget,
    QPixmap,
    QPushButton,
    QSlider,
    QSpinBox,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
)

PRESETS = [
    ("zoom_in", "Zoom in to target / زوم به هدف"),
    ("zoom_out", "Zoom out from target / دور شدن از هدف"),
    ("orbit", "Orbit around target / چرخش دور هدف"),
    ("flyover", "Fly-over pass / عبور از روی هدف"),
    ("spiral", "Spiral in / نزدیک شدن مارپیچ"),
]
PREVIEW_SIZE = (480, 270)


class PathDesignerDialog(QDialog):
    """
    Design how the camera moves over the scene:
      • click the map to add keyframes (each shows its frame footprint)
      • per keyframe: view width (zoom), rotation, travel time
      • presets, smooth spline path, live preview with the training labels
    """

    def __init__(
        self,
        mosaic: np.ndarray,
        layout: SceneLayout,
        parent=None,
        sources: Optional[Sequence[str]] = None,
        fps: float = 24.0,
    ):
        super().__init__(parent)
        self.setWindowTitle("Camera path designer / طراحی مسیر دوربین")
        self.resize(1500, 900)
        self.mosaic = mosaic
        self.layout_ = layout
        self.mpp = layout.meters_per_px or 1.0
        self.aspect = 16 / 9
        tp = layout.polygon(0)
        (x0, y0), (x1, y1) = tp.min(axis=0), tp.max(axis=0)
        self.t_center = (float((x0 + x1) / 2), float((y0 + y1) / 2))
        self.t_size = float(max(x1 - x0, y1 - y0))
        self.cam = MosaicCamera(mosaic, PREVIEW_SIZE)
        self.max_view = self.cam.max_view_h(*self.t_center, 0.0)
        self.keys: List[CameraKey] = preset_keys("zoom_in", self.t_center, self.t_size, self.max_view)
        self.poses: List[Pose] = []
        self.click_mode: Optional[str] = None
        self._loading = False

        root = QHBoxLayout(self)
        # ---------------- map
        left = QVBoxLayout()
        self.map = MosaicView(mosaic)
        self.map.clicked.connect(self._on_map_click)
        left.addWidget(self.map, stretch=1)
        self.map_hint = QLabel("Wheel = zoom · drag = pan · use “Add” / “Move” then click the map")
        left.addWidget(self.map_hint)
        root.addLayout(left, stretch=3)

        # ---------------- controls
        side = QWidget()
        sv = QVBoxLayout(side)

        pre_box = QGroupBox("Presets / الگوهای آماده")
        pv = QHBoxLayout(pre_box)
        self.preset = QComboBox()
        for key, title in PRESETS:
            self.preset.addItem(title, key)
        b_preset = QPushButton("Replace / جایگزینی")
        b_preset.setToolTip("Replace the whole path with this pattern")
        b_preset.clicked.connect(self._apply_preset)
        b_append = QPushButton("+ Append / افزودن به ادامه")
        b_append.setToolTip(
            "Append this pattern after the current path — combine patterns, e.g. fly-over → orbit → zoom in"
        )
        b_append.clicked.connect(self._append_preset)
        pv.addWidget(self.preset, stretch=1)
        pv.addWidget(b_preset)
        pv.addWidget(b_append)
        sv.addWidget(pre_box)

        kf_box = QGroupBox("Keyframes / فریم‌های کلیدی")
        kv = QVBoxLayout(kf_box)
        self.klist = QListWidget()
        self.klist.setMaximumHeight(170)
        self.klist.currentRowChanged.connect(self._on_select)
        kv.addWidget(self.klist)
        row = QHBoxLayout()
        self.b_add = QPushButton("+ Add (click map)")
        self.b_add.setCheckable(True)
        self.b_add.clicked.connect(lambda: self._set_click_mode("add"))
        self.b_move = QPushButton("✥ Move (click map)")
        self.b_move.setCheckable(True)
        self.b_move.clicked.connect(lambda: self._set_click_mode("move"))
        b_dup = QPushButton("⏸ Hold")
        b_dup.setToolTip("Duplicate the selected keyframe = the camera stays still for its travel time")
        b_dup.clicked.connect(self._hold)
        b_del = QPushButton("✕ Delete")
        b_del.clicked.connect(self._delete)
        for b in (self.b_add, self.b_move, b_dup, b_del):
            row.addWidget(b)
        kv.addLayout(row)

        form = QFormLayout()
        self.e_width = QDoubleSpinBox()
        self.e_width.setRange(10.0, 50000.0)
        self.e_width.setDecimals(0)
        self.e_width.setSuffix(" m")
        self.e_width.setToolTip("Ground width seen by the camera (smaller = zoomed in)")
        self.e_zoom = QSlider(Qt.Horizontal)
        self.e_zoom.setRange(0, 1000)
        self.e_angle = QSpinBox()
        self.e_angle.setRange(-180, 180)
        self.e_angle.setSuffix(" °")
        self.e_angle.setWrapping(True)
        self.e_time = QDoubleSpinBox()
        self.e_time.setRange(0.1, 60.0)
        self.e_time.setSingleStep(0.5)
        self.e_time.setSuffix(" s")
        self.e_time.setToolTip("Travel time from the previous keyframe")
        form.addRow("View width / عرض دید", self.e_width)
        form.addRow("Zoom", self.e_zoom)
        form.addRow("Rotation / چرخش", self.e_angle)
        form.addRow("Time from previous / زمان", self.e_time)
        kv.addLayout(form)
        self.e_width.valueChanged.connect(self._on_edit_width)
        self.e_zoom.valueChanged.connect(self._on_edit_zoom)
        self.e_angle.valueChanged.connect(self._on_edit)
        self.e_time.valueChanged.connect(self._on_edit)
        opt = QHBoxLayout()
        self.smooth = QCheckBox("Smooth curve")
        self.smooth.setChecked(True)
        self.ease = QCheckBox("Ease in/out")
        self.ease.setChecked(True)
        self.smooth.toggled.connect(lambda _v: self._rebuild())
        self.ease.toggled.connect(lambda _v: self._rebuild())
        opt.addWidget(self.smooth)
        opt.addWidget(self.ease)
        kv.addLayout(opt)
        sv.addWidget(kf_box)

        prev_box = QGroupBox("Preview / پیش‌نمایش")
        pvv = QVBoxLayout(prev_box)
        self.preview = QLabel()
        self.preview.setFixedSize(*PREVIEW_SIZE)
        self.preview.setStyleSheet("background:#000;")
        pvv.addWidget(self.preview, alignment=Qt.AlignCenter)
        trow = QHBoxLayout()
        self.b_play = QPushButton("▶")
        self.b_play.setFixedWidth(40)
        self.b_play.clicked.connect(self._toggle_play)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.valueChanged.connect(self._show_frame)
        self.t_label = QLabel("0.0 / 0.0 s")
        trow.addWidget(self.b_play)
        trow.addWidget(self.timeline, stretch=1)
        trow.addWidget(self.t_label)
        pvv.addLayout(trow)
        self.show_labels = QCheckBox("Show training labels (target + landmarks)")
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(lambda _v: self._show_frame(self.timeline.value()))
        pvv.addWidget(self.show_labels)
        sv.addWidget(prev_box)

        out_box = QGroupBox("Output videos / خروجی")
        of = QFormLayout(out_box)
        lrow = QHBoxLayout()
        self.o_day = QCheckBox("Day")
        self.o_day.setChecked(True)
        self.o_night = QCheckBox("Night")
        self.o_ir = QCheckBox("Thermal/IR")
        for c in (self.o_day, self.o_night, self.o_ir):
            lrow.addWidget(c)
        of.addRow("Lighting", lrow)
        self.o_sources = QCheckBox(f"All imagery sources ({len(sources or [])})")
        self.o_sources.setEnabled(bool(sources) and len(sources) > 1)
        of.addRow(self.o_sources)
        self.o_fps = QSpinBox()
        self.o_fps.setRange(5, 60)
        self.o_fps.setValue(int(fps))
        self.o_fps.valueChanged.connect(lambda _v: self._rebuild())
        of.addRow("FPS", self.o_fps)
        self.o_name = QLineEdit("my_path")
        of.addRow("Name", self.o_name)
        srow = QHBoxLayout()
        b_save = QPushButton("Save path…")
        b_save.clicked.connect(self._save)
        b_load = QPushButton("Load path…")
        b_load.clicked.connect(self._load)
        srow.addWidget(b_save)
        srow.addWidget(b_load)
        of.addRow(srow)
        sv.addWidget(out_box)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        sv.addWidget(self.info)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Render videos & add to project / ساخت ویدیو")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        sv.addWidget(buttons)
        root.addWidget(side, stretch=2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.map.set_painter(self._paint_map)
        self._refresh_list(select=0)
        self._rebuild()

    # ------------------------------------------------------------ zoom <-> width helpers
    def _width_m(self, view_h: float) -> float:
        return view_h * self.aspect * self.mpp

    def _view_h(self, width_m: float) -> float:
        return width_m / (self.aspect * self.mpp)

    def _zoom_to_slider(self, view_h: float) -> int:
        lo, hi = math.log(max(8.0, self.t_size * 0.5)), math.log(max(self.max_view, 16.0))
        return int(round(1000 * (hi - math.log(max(view_h, 1.0))) / max(1e-6, hi - lo)))

    def _slider_to_view(self, v: int) -> float:
        lo, hi = math.log(max(8.0, self.t_size * 0.5)), math.log(max(self.max_view, 16.0))
        return math.exp(hi - (hi - lo) * v / 1000.0)

    # ------------------------------------------------------------ list / editing
    def _refresh_list(self, select: Optional[int] = None) -> None:
        cur = self.klist.currentRow() if select is None else select
        self.klist.blockSignals(True)
        self.klist.clear()
        t = 0.0
        for i, k in enumerate(self.keys):
            t += 0.0 if i == 0 else k.duration
            self.klist.addItem(f"{i + 1}.  t={t:5.1f}s   width {self._width_m(k.view_h):6.0f} m   rot {k.angle:+.0f}°")
        self.klist.blockSignals(False)
        if self.keys:
            self.klist.setCurrentRow(max(0, min(cur, len(self.keys) - 1)))
            self._on_select(self.klist.currentRow())

    def _on_select(self, row: int) -> None:
        if not (0 <= row < len(self.keys)):
            return
        k = self.keys[row]
        self._loading = True
        self.e_width.setValue(self._width_m(k.view_h))
        self.e_zoom.setValue(self._zoom_to_slider(k.view_h))
        self.e_angle.setValue(int(round(((k.angle + 180) % 360) - 180)))
        self.e_time.setValue(max(0.1, k.duration))
        self.e_time.setEnabled(row > 0)
        self._loading = False
        self.map.refresh()

    def _cur(self) -> Optional[CameraKey]:
        r = self.klist.currentRow()
        return self.keys[r] if 0 <= r < len(self.keys) else None

    def _on_edit_width(self, _v=None) -> None:
        if self._loading or self._cur() is None:
            return
        self._cur().view_h = self._view_h(self.e_width.value())
        self._loading = True
        self.e_zoom.setValue(self._zoom_to_slider(self._cur().view_h))
        self._loading = False
        self._after_edit()

    def _on_edit_zoom(self, v: int) -> None:
        if self._loading or self._cur() is None:
            return
        self._cur().view_h = self._slider_to_view(v)
        self._loading = True
        self.e_width.setValue(self._width_m(self._cur().view_h))
        self._loading = False
        self._after_edit()

    def _on_edit(self, _v=None) -> None:
        if self._loading or self._cur() is None:
            return
        self._cur().angle = float(self.e_angle.value())
        self._cur().duration = float(self.e_time.value())
        self._after_edit()

    def _after_edit(self) -> None:
        r = self.klist.currentRow()
        self._refresh_list(select=r)
        self._rebuild()

    def _set_click_mode(self, mode: str) -> None:
        self.click_mode = None if self.click_mode == mode else mode
        self.b_add.setChecked(self.click_mode == "add")
        self.b_move.setChecked(self.click_mode == "move")
        self.map_hint.setText(
            {"add": "Click on the map to ADD a keyframe after the selected one / برای افزودن روی نقشه کلیک کنید",
             "move": "Click on the map to MOVE the selected keyframe / برای جابجایی روی نقشه کلیک کنید"}.get(
                self.click_mode, "Wheel = zoom · drag = pan · use “Add” / “Move” then click the map"
            )
        )

    def _on_map_click(self, x: float, y: float) -> None:
        r = self.klist.currentRow()
        if self.click_mode == "add":
            ref = self.keys[r] if 0 <= r < len(self.keys) else CameraKey(x, y, self.t_size * 3, 0.0, 0.0)
            self.keys.insert(r + 1, CameraKey(x, y, ref.view_h, ref.angle, 2.0))
            self._refresh_list(select=r + 1)
            self._rebuild()
        elif self.click_mode == "move" and 0 <= r < len(self.keys):
            self.keys[r].cx, self.keys[r].cy = x, y
            self._refresh_list(select=r)
            self._rebuild()

    def _hold(self) -> None:
        r = self.klist.currentRow()
        if 0 <= r < len(self.keys):
            k = self.keys[r]
            self.keys.insert(r + 1, CameraKey(k.cx, k.cy, k.view_h, k.angle, 1.5))
            self._refresh_list(select=r + 1)
            self._rebuild()

    def _delete(self) -> None:
        r = self.klist.currentRow()
        if 0 <= r < len(self.keys) and len(self.keys) > 1:
            self.keys.pop(r)
            self.keys[0].duration = 0.0
            self._refresh_list(select=max(0, r - 1))
            self._rebuild()

    def _apply_preset(self) -> None:
        self.keys = preset_keys(self.preset.currentData(), self.t_center, self.t_size, self.max_view)
        self._refresh_list(select=0)
        self._rebuild()

    def _append_preset(self) -> None:
        """Chain patterns: the new one continues from where the current path ends."""
        new = preset_keys(self.preset.currentData(), self.t_center, self.t_size, self.max_view)
        if not new:
            return
        if not self.keys:
            self.keys = new
            first = 0
        else:
            # Travel time from the end of the current path into the first key of the pattern
            new[0].duration = 2.0
            first = len(self.keys)
            self.keys.extend(new)
        self._refresh_list(select=first)
        self._rebuild()

    # ------------------------------------------------------------ path + preview
    def _rebuild(self) -> None:
        fps = float(self.o_fps.value()) if hasattr(self, "o_fps") else 24.0
        raw = path_from_keyframes(self.keys, fps, smooth=self.smooth.isChecked(), ease=self.ease.isChecked())
        self.poses = [self.cam.fit(p) for p in raw]
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, max(0, len(self.poses) - 1))
        self.timeline.blockSignals(False)
        dur = len(self.poses) / fps
        clipped = sum(1 for a, b in zip(raw, self.poses) if b.view_h < a.view_h - 1e-3)
        warn = f"  ⚠ {clipped} frames limited by the map edge (zoomed in automatically)" if clipped else ""
        self.info.setText(f"{len(self.keys)} keyframes · {len(self.poses)} frames · {dur:.1f} s{warn}")
        self._show_frame(min(self.timeline.value(), max(0, len(self.poses) - 1)))
        self.map.refresh()

    def _show_frame(self, i: int) -> None:
        if not self.poses:
            return
        i = max(0, min(i, len(self.poses) - 1))
        img, M = self.cam.render(self.poses[i])
        if self.show_labels.isChecked():
            for cid, poly in frame_labels(self.layout_, M, *PREVIEW_SIZE):
                color = (0, 220, 0) if cid == 0 else (0, 160, 255)
                cv2.polylines(img, [np.int32(poly).reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.preview.setPixmap(QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888).copy()))
        fps = float(self.o_fps.value())
        self.t_label.setText(f"{i / fps:4.1f} / {len(self.poses) / fps:4.1f} s   width {self._width_m(self.poses[i].view_h):.0f} m")
        self._cur_frame = i
        if not self.timer.isActive() or i % 3 == 0:
            self.map.refresh()

    def _toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.b_play.setText("▶")
        else:
            if self.timeline.value() >= self.timeline.maximum():
                self.timeline.setValue(0)
            self.timer.start(int(1000 / min(30, self.o_fps.value())))
            self.b_play.setText("⏸")

    def _tick(self) -> None:
        v = self.timeline.value() + max(1, int(round(self.o_fps.value() / 30)))
        if v > self.timeline.maximum():
            self.timer.stop()
            self.b_play.setText("▶")
            return
        self.timeline.setValue(v)

    def _paint_map(self, img: np.ndarray, s: float) -> None:
        th = max(2, int(img.shape[1] / 700))
        for cid in range(len(self.layout_.classes)):
            color = (0, 220, 0) if cid == 0 else (0, 160, 255)
            cv2.polylines(img, [np.int32(self.layout_.polygon(cid) * s).reshape(-1, 1, 2)], True, color, th, cv2.LINE_AA)
        if self.poses:
            pts = np.int32([[p.cx * s, p.cy * s] for p in self.poses[:: max(1, len(self.poses) // 300)]])
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (255, 255, 0), th, cv2.LINE_AA)
        sel = self.klist.currentRow()
        for i, k in enumerate(self.keys):
            corners = frame_corners(Pose(k.cx, k.cy, k.view_h, k.angle), self.aspect) * s
            color = (0, 0, 255) if i == sel else (255, 255, 255)
            cv2.polylines(img, [np.int32(corners).reshape(-1, 1, 2)], True, color, th + (1 if i == sel else 0), cv2.LINE_AA)
            # "up" marker on the frame's top edge shows the camera heading
            top = (corners[0] + corners[1]) / 2
            cv2.circle(img, tuple(np.int32(top)), th * 2, color, -1)
            c = (int(k.cx * s), int(k.cy * s))
            cv2.circle(img, c, th * 3, color, -1)
            cv2.putText(img, str(i + 1), (c[0] + th * 4, c[1] - th * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5 + th * 0.15, color, th, cv2.LINE_AA)
        i = getattr(self, "_cur_frame", None)
        if i is not None and self.poses:
            corners = frame_corners(self.poses[i], self.aspect) * s
            cv2.polylines(img, [np.int32(corners).reshape(-1, 1, 2)], True, (255, 0, 255), th + 1, cv2.LINE_AA)

    # ------------------------------------------------------------ save / load / result
    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save camera path", f"{self.o_name.text()}.path.json", "Camera path (*.json)")
        if path:
            Path(path).write_text(json.dumps({"keys": [k.to_dict() for k in self.keys]}, indent=2), encoding="utf-8")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load camera path", "", "Camera path (*.json)")
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.keys = [CameraKey.from_dict(d) for d in data.get("keys", [])] or self.keys
            self._refresh_list(select=0)
            self._rebuild()

    def _accept(self) -> None:
        if len(self.poses) < 2:
            self.info.setText("⚠ The path needs at least 2 keyframes")
            return
        if not (self.o_day.isChecked() or self.o_night.isChecked() or self.o_ir.isChecked()):
            self.info.setText("⚠ Choose at least one lighting option")
            return
        self.timer.stop()
        self.accept()

    def result_spec(self) -> dict:
        lights = [n for n, c in (("day", self.o_day), ("night", self.o_night), ("ir", self.o_ir)) if c.isChecked()]
        return {
            "keys": [CameraKey(k.cx, k.cy, k.view_h, k.angle, k.duration) for k in self.keys],
            "lights": lights,
            "all_sources": self.o_sources.isChecked(),
            "fps": float(self.o_fps.value()),
            "name": self.o_name.text().strip() or "custom",
        }
