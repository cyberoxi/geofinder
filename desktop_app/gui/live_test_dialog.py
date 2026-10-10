"""Live model test: configure, then play; stop and replay from the start."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from shared.qt_compat import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QImage,
    QLabel,
    QLineEdit,
    QPixmap,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSettings,
    QSizePolicy,
    QSpinBox,
    Qt,
    QThread,
    QVBoxLayout,
    Signal,
)

# Path to a video file, or int camera device index.
Source = Union[Path, int]

_SETTINGS_ORG = "GeoFinder"
_SETTINGS_APP = "Desktop"
_CONF_KEY = "live_test/min_confidence_pct"
_DEFAULT_CONF_PCT = 35


def _load_conf_pct() -> int:
    raw = QSettings(_SETTINGS_ORG, _SETTINGS_APP).value(_CONF_KEY, _DEFAULT_CONF_PCT)
    try:
        return max(1, min(99, int(raw)))
    except (TypeError, ValueError):
        return _DEFAULT_CONF_PCT


def _save_conf_pct(pct: int) -> None:
    QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(_CONF_KEY, int(pct))


def _source_label(source: Source) -> str:
    if isinstance(source, int):
        return f"camera:{source}"
    return Path(source).name


class LiveInferenceWorker(QThread):
    """Load the model once; play / stop / replay video or camera on demand."""

    frame_ready = Signal(object, dict)  # BGR image, localization dict
    finished_ok = Signal(dict)
    failed = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        weights: Path,
        source: Optional[Source],
        scene: Optional[Path],
        imgsz: int,
        device: str,
        conf: float = 0.35,
        realtime: bool = True,
    ):
        super().__init__()
        self.weights = Path(weights)
        self.source: Optional[Source] = (
            Path(source) if isinstance(source, (str, Path)) else source
        )
        self.scene = Path(scene) if scene else None
        self.imgsz = imgsz
        self.device = device
        self.conf = float(conf)
        self.realtime = realtime
        self._shutdown = False
        self._stop_play = False
        self._paused = False
        self._play_requested = False

    def set_source(self, source: Source):
        """Switch input (file path or camera index). Call while idle / before Play."""
        if isinstance(source, int):
            self.source = int(source)
        else:
            self.source = Path(source)

    def request_play(self):
        self._stop_play = False
        self._paused = False
        self._play_requested = True

    def request_stop_play(self):
        self._stop_play = True
        self._paused = False

    def request_shutdown(self):
        self._shutdown = True
        self._stop_play = True
        self._play_requested = False
        self._paused = False

    def set_paused(self, paused: bool):
        self._paused = paused

    def set_confidence(self, conf: float):
        self.conf = max(0.01, min(0.99, float(conf)))

    def run(self):
        try:
            from ultralytics import YOLO

            from desktop_app.inference.localize import (
                _detections_from_result,
                draw_localization,
                localize_frame,
            )
            from desktop_app.pipeline.auto import resolve_device
            from shared.landmarks import SceneLayout, TargetLocalizer
            from shared.video_reader import VideoReader

            device = resolve_device(self.device)
            self.status.emit(f"Loading model… ({self.weights.name}) on {device}")
            model = YOLO(str(self.weights))
            names = (
                [model.names[i] for i in sorted(model.names)]
                if isinstance(model.names, dict)
                else list(model.names)
            )
            self.status.emit(f"Model loaded — {len(names)} class(es)")
            localizer = None
            if self.scene and self.scene.exists():
                layout = SceneLayout.load(self.scene)
                if len(layout.classes) == len(names):
                    localizer = TargetLocalizer(layout, min_confidence=self.conf)
                    self.status.emit("Ready — set confidence, then Play (landmarks enabled)")
                else:
                    self.status.emit(
                        f"Ready — landmarks off (scene {len(layout.classes)} ≠ model {len(names)}). Press Play"
                    )
            else:
                self.status.emit("Ready — set confidence, then Play (no scene.json for landmarks)")

            while not self._shutdown:
                while not self._play_requested and not self._shutdown:
                    time.sleep(0.05)
                if self._shutdown:
                    break
                self._play_requested = False
                self._stop_play = False
                self._paused = False

                src = self.source
                if src is None:
                    self.status.emit("No source — pick a video file or camera, then Play")
                    self.finished_ok.emit(
                        {
                            "video": "",
                            "frames": 0,
                            "target_detected_rate": 0.0,
                            "inferred_from_landmarks_rate": 0.0,
                            "located_rate": 0.0,
                            "sources": {},
                            "stopped": True,
                        }
                    )
                    continue

                try:
                    if isinstance(src, int):
                        self.status.emit(f"Opening camera… index {src}")
                        reader = VideoReader.from_camera(src)
                    else:
                        self.status.emit(f"Opening video… {Path(src).name}")
                        reader = VideoReader(str(src), prefer_gstreamer=False)
                except Exception as open_exc:  # noqa: BLE001
                    self.status.emit(f"Open failed: {open_exc}")
                    self.finished_ok.emit(
                        {
                            "video": _source_label(src),
                            "frames": 0,
                            "target_detected_rate": 0.0,
                            "inferred_from_landmarks_rate": 0.0,
                            "located_rate": 0.0,
                            "sources": {},
                            "stopped": True,
                            "error": str(open_exc),
                        }
                    )
                    continue

                info = reader.info
                assert info is not None
                is_cam = isinstance(src, int)
                fps = max(1.0, float(info.fps) or 25.0)
                frame_budget = 1.0 / fps
                total = 0 if is_cam else (info.frame_count if info.frame_count > 0 else 0)
                stats = {"frames": 0, "yolo": 0, "landmarks": 0, "none": 0, "yolo_low": 0}
                fid = -1
                self.status.emit(
                    f"First inference… "
                    f"({'live' if is_cam else (total or '?')} frames @ {fps:.1f} fps, "
                    f"conf≥{self.conf:.0%})"
                )
                serial = None
                try:
                    from desktop_app.gui.serial_dialog import load_serial_settings
                    from shared.serial_out import SerialOffsetSender

                    serial = SerialOffsetSender(load_serial_settings())
                    if serial.enabled:
                        if serial.open():
                            self.status.emit(
                                f"Serial ON — {serial.config.get('port')} @ "
                                f"{serial.config.get('baudrate')} (dx,dy)"
                            )
                        else:
                            self.status.emit("Serial enabled but open failed — continuing without UART")
                    while not self._shutdown and not self._stop_play:
                        while self._paused and not self._shutdown and not self._stop_play:
                            time.sleep(0.05)
                        if self._shutdown or self._stop_play:
                            break
                        t0 = time.perf_counter()
                        ok, frame = reader.read()
                        if not ok or frame is None:
                            break
                        fid += 1
                        min_conf = self.conf
                        if localizer is not None:
                            localizer.min_confidence = min_conf
                        r = model.predict(
                            frame, imgsz=self.imgsz, conf=min_conf, device=device, verbose=False
                        )
                        dets = _detections_from_result(r[0], frame.shape) if r else []
                        loc = localize_frame(dets, localizer, frame.shape, target_conf=min_conf)
                        stats["frames"] += 1
                        stats[loc["source"]] += 1
                        annotated = draw_localization(frame, dets, loc, names)
                        off = loc.get("offset") or {}
                        dx = off.get("dx")
                        dy = off.get("dy")
                        dist = off.get("distance_px")
                        tracking = dx is not None and dy is not None and loc["source"] != "none"
                        if serial is not None:
                            if tracking:
                                serial.send_offset(
                                    float(dx),
                                    float(dy),
                                    distance=float(dist or 0.0),
                                    status=str(loc["source"]),
                                    tracking=True,
                                )
                            else:
                                serial.send_offset(0.0, 0.0, status="none", tracking=False)
                        if fid == 0:
                            self.status.emit("Playing…" + (" (camera live)" if is_cam else ""))
                        self.frame_ready.emit(
                            annotated,
                            {
                                "frame": fid,
                                "total": total,
                                "source": loc["source"],
                                "confidence": float(loc["confidence"]),
                                "fps": fps,
                                "min_conf": min_conf,
                                "distance_px": dist,
                                "dx": dx,
                                "dy": dy,
                                "landmarks_used": loc.get("landmarks_used") or [],
                                "has_localizer": localizer is not None,
                            },
                        )
                        if self.realtime and not is_cam:
                            elapsed = time.perf_counter() - t0
                            delay = frame_budget - elapsed
                            if delay > 0:
                                time.sleep(delay)
                finally:
                    if serial is not None:
                        serial.close()
                    reader.release()

                if self._shutdown:
                    break
                n = max(1, stats["frames"])
                self.finished_ok.emit(
                    {
                        "video": _source_label(src),
                        "frames": stats["frames"],
                        "target_detected_rate": stats["yolo"] / n,
                        "inferred_from_landmarks_rate": stats["landmarks"] / n,
                        "located_rate": (stats["yolo"] + stats["landmarks"] + stats["yolo_low"]) / n,
                        "sources": stats,
                        "stopped": self._stop_play,
                    }
                )
                self.status.emit("Stopped — adjust settings and press Play again")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class LiveTestDialog(QDialog):
    """
    Configure confidence and input (video file or camera), then Play.
    Stop returns to idle so Play can restart (model stays loaded).
    """

    def __init__(
        self,
        weights: Path,
        video: Optional[Path],
        scene: Optional[Path],
        imgsz: int,
        device: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Live Test / تست زنده مدل")
        self.setMinimumSize(960, 640)
        self.resize(1180, 760)
        self._metrics: Optional[Dict[str, Any]] = None
        self._playing = False
        self._awaiting_first_frame = False
        self._model_ready = False
        initial: Optional[Source] = Path(video) if video else None

        root = QVBoxLayout(self)
        hint = QLabel(
            "۱) منبع را انتخاب کنید (ویدیو یا وب‌کم)  ۲) حداقل دقت  ۳) Play  ۴) Stop برای تعویض منبع\n"
            "سبز = هدف مستقیم · ارغوانی = از اشیا کمکی · نارنجی = لندمارک · فاصله تا مرکز روی فریم"
        )
        hint.setStyleSheet("color:#9ca3af;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.view = QLabel("Loading model…")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setStyleSheet("background:#0b1220; color:#9ca3af; font-size:14px;")
        root.addWidget(self.view, stretch=1)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)  # indeterminate
        self.busy.setTextVisible(False)
        self.busy.setFixedHeight(6)
        self.busy.setStyleSheet(
            "QProgressBar{border:none;background:#1e293b;}"
            "QProgressBar::chunk{background:#38bdf8;}"
        )
        root.addWidget(self.busy)

        self.info = QLabel("Loading model…")
        self.info.setWordWrap(True)
        root.addWidget(self.info)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        self.log.setFixedHeight(96)
        self.log.setPlaceholderText("Log / لاگ")
        self.log.setStyleSheet(
            "QPlainTextEdit{background:#0f172a;color:#cbd5e1;border:1px solid #1e293b;font-size:11px;}"
        )
        root.addWidget(self.log)
        src_note = _source_label(initial) if initial is not None else "(none — pick below)"
        self._append_log(f"Open live test — weights={Path(weights).name}, source={src_note}")

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source / منبع:"))
        self.source_mode = QComboBox()
        self.source_mode.addItem("Video file / فایل ویدیو", "file")
        self.source_mode.addItem("Camera / وب‌کم", "camera")
        self.source_mode.currentIndexChanged.connect(self._on_source_mode)
        src_row.addWidget(self.source_mode)

        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Path to video…")
        self.video_path.setMinimumWidth(280)
        if isinstance(initial, Path):
            self.video_path.setText(str(initial))
        src_row.addWidget(self.video_path, stretch=1)

        self.btn_browse = QPushButton("Browse… / انتخاب")
        self.btn_browse.clicked.connect(self._browse_video)
        src_row.addWidget(self.btn_browse)

        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 9)
        self.cam_index.setValue(0)
        self.cam_index.setPrefix("Cam ")
        self.cam_index.setToolTip("Device index: 0 = default webcam, 1 = next camera, …")
        self.cam_index.setVisible(False)
        src_row.addWidget(self.cam_index)
        root.addLayout(src_row)

        row = QHBoxLayout()
        conf_lab = QLabel("Min confidence / حداقل دقت:")
        conf_lab.setToolTip("Only detections ≥ this % count. Set before Play; can also change while playing.")
        row.addWidget(conf_lab)
        self.conf_pct = QSpinBox()
        self.conf_pct.setRange(1, 99)
        self.conf_pct.setValue(_load_conf_pct())
        self.conf_pct.setSuffix(" %")
        self.conf_pct.setToolTip("Detection confidence must be ≥ this percent (saved for next time)")
        self.conf_pct.valueChanged.connect(self._on_conf_changed)
        row.addWidget(self.conf_pct)

        self.btn_play = QPushButton("Play / پخش")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._on_play)
        row.addWidget(self.btn_play)

        self.btn_pause = QPushButton("Pause / توقف موقت")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.toggled.connect(self._on_pause)
        row.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("Stop / توقف")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        root.addLayout(row)

        self._worker = LiveInferenceWorker(
            weights, initial, scene, imgsz, device, conf=self.conf_pct.value() / 100.0
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.status.connect(self._on_status)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._set_busy(True)
        self._worker.start()
        self._on_source_mode()

    def _append_log(self, text: str):
        stamp = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {text}")

    def _set_busy(self, busy: bool):
        self.busy.setVisible(busy)
        if busy:
            self.busy.setRange(0, 0)

    def _set_source_enabled(self, enabled: bool):
        self.source_mode.setEnabled(enabled)
        self.video_path.setEnabled(enabled)
        self.btn_browse.setEnabled(enabled)
        self.cam_index.setEnabled(enabled)

    def _set_idle_controls(self):
        self._playing = False
        self._awaiting_first_frame = False
        self._set_busy(False)
        self.btn_play.setEnabled(self._model_ready)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.blockSignals(True)
        self.btn_pause.setChecked(False)
        self.btn_pause.blockSignals(False)
        self.btn_pause.setText("Pause / توقف موقت")
        self.conf_pct.setEnabled(True)
        self._set_source_enabled(True)

    def _set_playing_controls(self):
        self._playing = True
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.conf_pct.setEnabled(True)
        self._set_source_enabled(False)

    def _on_source_mode(self, *_args):
        is_cam = self.source_mode.currentData() == "camera"
        self.video_path.setVisible(not is_cam)
        self.btn_browse.setVisible(not is_cam)
        self.cam_index.setVisible(is_cam)

    def _browse_video(self):
        start = self.video_path.text().strip() or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Video to test (live)",
            start,
            "Videos (*.mp4 *.avi *.mkv *.mov);;All files (*.*)",
        )
        if path:
            self.video_path.setText(path)
            self._append_log(f"Video selected — {Path(path).name}")

    def _resolve_source(self) -> Optional[Source]:
        if self.source_mode.currentData() == "camera":
            return int(self.cam_index.value())
        text = self.video_path.text().strip()
        if not text:
            return None
        return Path(text)

    def _on_status(self, text: str):
        self.info.setText(text)
        self._append_log(text)
        low = text.lower()
        if low.startswith("ready"):
            self._model_ready = True
            if not self._playing:
                self._set_busy(False)
                self.btn_play.setEnabled(True)
                pix = self.view.pixmap()
                if pix is None or pix.isNull():
                    self.view.setText("Ready — set confidence, then press Play")
        elif self._awaiting_first_frame:
            pix = self.view.pixmap()
            if pix is None or pix.isNull():
                self.view.setText(text)

    def _on_frame(self, bgr: np.ndarray, meta: dict):
        if self._awaiting_first_frame:
            self._awaiting_first_frame = False
            self._set_busy(False)
            self._append_log(f"First frame ready (#{meta.get('frame', 0)})")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self.view.setPixmap(
            pix.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        src = meta.get("source", "none")
        tag = {
            "yolo": "TARGET found",
            "yolo_low": "target? (low conf)",
            "landmarks": "TARGET via landmarks",
            "none": "searching…",
        }.get(src, src)
        total = meta.get("total") or "?"
        parts = [
            f"Frame {meta.get('frame', 0)}/{total}",
            tag,
            f"conf {meta.get('confidence', 0):.2f}",
            f"min ≥{meta.get('min_conf', 0) * 100:.0f}%",
        ]
        dist = meta.get("distance_px")
        if dist is not None:
            parts.append(
                f"center dist {dist:.0f}px (dx={meta.get('dx', 0):+.0f}, dy={meta.get('dy', 0):+.0f})"
            )
        used = meta.get("landmarks_used") or []
        if used:
            parts.append(f"aux {used}")
        self.info.setText("  ·  ".join(parts))

    def _on_conf_changed(self, pct: int):
        self._worker.set_confidence(pct / 100.0)
        _save_conf_pct(pct)

    def _on_play(self):
        source = self._resolve_source()
        if source is None:
            self.info.setText("Pick a video file or switch to Camera, then Play")
            self._append_log("Play blocked — no source selected")
            return
        if isinstance(source, Path) and not source.exists():
            self.info.setText(f"Video not found: {source}")
            self._append_log(f"Play blocked — missing file {source}")
            return
        conf = self.conf_pct.value()
        self._worker.set_source(source)
        self._worker.set_confidence(conf / 100.0)
        self._awaiting_first_frame = True
        self._set_busy(True)
        self._set_playing_controls()
        self.view.clear()
        kind = "camera" if isinstance(source, int) else "video"
        self.view.setText(f"Starting playback…\nOpening {kind} & first inference")
        msg = f"Play requested — {_source_label(source)}, min confidence {conf}%"
        self.info.setText(msg)
        self._append_log(msg)
        self._worker.request_play()

    def _on_pause(self, paused: bool):
        if not self._playing:
            return
        self._worker.set_paused(paused)
        self.btn_pause.setText("Resume / ادامه" if paused else "Pause / توقف موقت")
        self._append_log("Paused" if paused else "Resumed")

    def _on_stop(self):
        self._worker.request_stop_play()
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self._append_log("Stop requested")

    def _on_done(self, metrics: dict):
        self._metrics = metrics
        self._set_idle_controls()
        if metrics.get("error"):
            err = metrics["error"]
            self.info.setText(f"Open failed: {err}")
            self._append_log(f"Open failed: {err}")
            self.view.setText(f"Open failed:\n{err}")
            return
        if metrics.get("frames", 0) == 0 and not metrics.get("video"):
            return
        stopped = metrics.get("stopped")
        summary = (
            f"{'Stopped' if stopped else 'Finished'} — frames {metrics['frames']} · "
            f"direct {metrics['target_detected_rate']:.0%} · "
            f"landmarks {metrics['inferred_from_landmarks_rate']:.0%} · "
            f"located {metrics['located_rate']:.0%}  ·  Press Play to run again"
        )
        self.info.setText(summary)
        self._append_log(summary)

    def _on_fail(self, err: str):
        self._awaiting_first_frame = False
        self._set_busy(False)
        self.info.setText(f"Error: {err}")
        self._append_log(f"Error: {err}")
        self.view.setText(f"Error:\n{err}")
        self._set_idle_controls()
        self.btn_play.setEnabled(False)

    def closeEvent(self, event):
        _save_conf_pct(self.conf_pct.value())
        self._worker.request_shutdown()
        if not self._worker.wait(5000):
            self._worker.terminate()
            self._worker.wait(1000)
        super().closeEvent(event)

    @property
    def metrics(self) -> Optional[Dict[str, Any]]:
        return self._metrics
