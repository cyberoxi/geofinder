"""Interactive landmark editor: see the target + landmark objects on the satellite image and edit them."""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from desktop_app.satellite.objects import ObjectCandidate, select_landmarks
from desktop_app.satellite.scene import layout_from_selection
from shared.landmarks import SceneLayout
from shared.qt_compat import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QImage,
    QLabel,
    QListWidget,
    QPixmap,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
    mouse_xy,
)

TARGET = (0, 220, 0)
SELECTED = (0, 150, 255)
CANDIDATE = (230, 230, 120)
DRAW = (255, 0, 255)

KIND_FA = {
    "sports_field": "زمین ورزشی",
    "building": "ساختمان",
    "roundabout": "میدان",
    "green_area": "فضای سبز",
    "water": "آب",
    "open_area": "محوطه باز",
    "intersection": "چهارراه",
    "patch": "بافت",
    "manual": "دستی",
}


class MosaicView(QLabel):
    """Zoom (wheel) / pan (drag) view of a large image; clicks and drawings in mosaic coordinates."""

    clicked = Signal(float, float)
    shape_drawn = Signal(object)  # np.ndarray (N, 2) mosaic px

    def __init__(self, mosaic: np.ndarray, max_base: int = 2600):
        super().__init__()
        self.setMinimumSize(700, 520)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111;")
        self.setMouseTracking(True)
        H, W = mosaic.shape[:2]
        self.base_scale = min(1.0, max_base / float(max(W, H)))
        self.base = (
            cv2.resize(mosaic, (int(W * self.base_scale), int(H * self.base_scale)), interpolation=cv2.INTER_AREA)
            if self.base_scale < 1
            else mosaic.copy()
        )
        self.overlay = self.base.copy()
        self.zoom = 0.0  # display px per base px (0 = fit on first paint)
        self._user_zoomed = False
        self.center = np.array([self.base.shape[1] / 2.0, self.base.shape[0] / 2.0])
        self.mode: Optional[str] = None  # None | "rect" | "poly"
        self._press = None
        self._dragging = False
        self._points: List[np.ndarray] = []
        self._cursor: Optional[np.ndarray] = None
        self._painter: Optional[Callable[[np.ndarray, float], None]] = None

    # ------------------------------------------------------------ coordinates
    def _fit_zoom(self) -> float:
        return min(self.width() / self.base.shape[1], self.height() / self.base.shape[0])

    def _to_base(self, x: float, y: float) -> np.ndarray:
        return self.center + (np.array([x, y]) - np.array([self.width(), self.height()]) / 2.0) / self.zoom

    def to_mosaic(self, x: float, y: float) -> np.ndarray:
        return self._to_base(x, y) / self.base_scale

    # ------------------------------------------------------------ drawing
    def set_painter(self, fn: Callable[[np.ndarray, float], None]) -> None:
        """fn(image_at_base_res, base_scale) draws overlays in place."""
        self._painter = fn
        self.refresh()

    def refresh(self) -> None:
        self.overlay = self.base.copy()
        if self._painter:
            self._painter(self.overlay, self.base_scale)
        self._paint()

    def fit(self) -> None:
        self._user_zoomed = False
        self.zoom = self._fit_zoom()
        self.center = np.array([self.base.shape[1] / 2.0, self.base.shape[0] / 2.0])
        self._paint()

    def _paint(self) -> None:
        if self.width() < 10 or self.height() < 10:
            return
        if self.zoom <= 0:
            self.zoom = self._fit_zoom()
        w, h = self.width(), self.height()
        # Affine: base → widget
        M = np.array(
            [[self.zoom, 0, w / 2.0 - self.zoom * self.center[0]], [0, self.zoom, h / 2.0 - self.zoom * self.center[1]]],
            np.float64,
        )
        interp = cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_LINEAR
        view = cv2.warpAffine(self.overlay, M, (w, h), flags=interp, borderValue=(17, 17, 17))
        if self.mode and self._points:
            pts = [(M[:, :2] @ (p * self.base_scale) + M[:, 2]) for p in self._points]
            if self.mode == "rect" and self._cursor is not None:
                c = M[:, :2] @ (self._cursor * self.base_scale) + M[:, 2]
                cv2.rectangle(view, tuple(map(int, pts[0])), tuple(map(int, c)), DRAW, 2)
            elif self.mode == "poly":
                seq = pts + ([M[:, :2] @ (self._cursor * self.base_scale) + M[:, 2]] if self._cursor is not None else [])
                cv2.polylines(view, [np.int32(seq).reshape(-1, 1, 2)], False, DRAW, 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg.copy()))

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        if self.zoom <= 0 or not self._user_zoomed:
            self.zoom = self._fit_zoom()
        self._paint()

    # ------------------------------------------------------------ interaction
    def set_mode(self, mode: Optional[str]) -> None:
        self.mode = mode
        self._points = []
        self.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)
        self._paint()

    def wheelEvent(self, e):  # noqa: N802
        delta = e.angleDelta().y()
        if not delta:
            return
        try:
            p = e.position()
            x, y = p.x(), p.y()
        except AttributeError:  # PyQt5
            x, y = e.x(), e.y()
        self._user_zoomed = True
        before = self._to_base(x, y)
        factor = 1.25 if delta > 0 else 0.8
        self.zoom = float(np.clip(self.zoom * factor, self._fit_zoom() * 0.5, 8.0))
        after = self._to_base(x, y)
        self.center += before - after
        self._paint()

    def mousePressEvent(self, e):  # noqa: N802
        x, y = mouse_xy(e)
        if e.button() == Qt.RightButton and self.mode == "poly" and len(self._points) >= 3:
            self._finish_poly()
            return
        if e.button() != Qt.LeftButton:
            return
        if self.mode == "rect":
            self._points = [self.to_mosaic(x, y)]
            return
        if self.mode == "poly":
            self._points.append(self.to_mosaic(x, y))
            self._paint()
            return
        self._press = (x, y, self.center.copy())
        self._dragging = False

    def mouseMoveEvent(self, e):  # noqa: N802
        x, y = mouse_xy(e)
        if self.mode:
            self._cursor = self.to_mosaic(x, y)
            if self._points:
                self._paint()
            return
        if self._press is not None:
            dx, dy = x - self._press[0], y - self._press[1]
            if self._dragging or math.hypot(dx, dy) > 4:
                self._dragging = True
                self.center = self._press[2] - np.array([dx, dy]) / self.zoom
                self._paint()

    def mouseReleaseEvent(self, e):  # noqa: N802
        x, y = mouse_xy(e)
        if self.mode == "rect" and self._points and e.button() == Qt.LeftButton:
            a, b = self._points[0], self.to_mosaic(x, y)
            if abs(a[0] - b[0]) > 3 and abs(a[1] - b[1]) > 3:
                x0, y0 = np.minimum(a, b)
                x1, y1 = np.maximum(a, b)
                self.shape_drawn.emit(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32))
            self.set_mode(None)
            return
        if self._press is not None and not self._dragging and e.button() == Qt.LeftButton:
            m = self.to_mosaic(x, y)
            self.clicked.emit(float(m[0]), float(m[1]))
        self._press = None
        self._dragging = False

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if self.mode == "poly" and len(self._points) >= 3:
            self._finish_poly()

    def _finish_poly(self) -> None:
        pts = np.asarray(self._points, np.float32)
        self.set_mode(None)
        if len(pts) >= 3 and abs(cv2.contourArea(pts)) > 9:
            self.shape_drawn.emit(pts)


class LandmarkEditorDialog(QDialog):
    """
    Shows the target (green), selected landmarks (orange, numbered) and all detected
    candidate objects (light).  Click a candidate to add it, click a landmark to
    remove it, or draw your own rectangle / polygon.
    """

    def __init__(
        self,
        mosaic: np.ndarray,
        layout: SceneLayout,
        candidates: Sequence[ObjectCandidate],
        parent=None,
        title: str = "Landmarks — review & edit",
        extra_mosaics: Optional[Sequence[tuple]] = None,
        extra_masks: Optional[dict] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1300, 820)
        self.mosaic = mosaic
        self.layout_in = layout
        self.mpp = layout.meters_per_px or 0.0
        self.target_poly = layout.polygon(0)
        self.candidates: List[ObjectCandidate] = list(candidates)
        self.extras: List[tuple] = list(extra_mosaics or [])
        self.extra_masks: dict = dict(extra_masks or {})
        tc = self.target_poly.mean(axis=0)
        # Selected = current layout landmarks (reuse candidate objects when they match)
        self.selected: List[ObjectCandidate] = []
        infos = layout.meta.get("landmarks") or []
        for i in range(1, len(layout.classes)):
            poly = layout.polygon(i)
            match = next((c for c in self.candidates if c.polygon.shape == poly.shape and np.allclose(c.polygon, poly, atol=0.6)), None)
            if match is None:
                kind = infos[i - 1].get("kind", "manual") if i - 1 < len(infos) else "manual"
                match = ObjectCandidate(polygon=poly, kind=kind, score=1.0)
                match.distance = float(np.linalg.norm(match.center - tc))
                self.candidates.append(match)
            self.selected.append(match)

        root = QHBoxLayout(self)
        self.view = MosaicView(mosaic)
        self.view.clicked.connect(self._on_click)
        self.view.shape_drawn.connect(self._on_drawn)
        root.addWidget(self.view, stretch=4)

        side = QWidget()
        sv = QVBoxLayout(side)
        help_lbl = QLabel(
            "سبز = منطقه اصلی (هدف)\n"
            "نارنجی = شاخص‌های انتخاب‌شده برای آموزش\n"
            "زرد کم‌رنگ = اجسام پیشنهادی\n\n"
            "• کلیک روی جسم پیشنهادی ← اضافه شود\n"
            "• کلیک روی شاخص نارنجی ← حذف شود\n"
            "• چرخ ماوس = زوم، کشیدن = جابجایی\n"
            "• «مستطیل» یا «چندضلعی» برای رسم دستی\n"
            "  (چندضلعی: کلیک نقاط، دابل‌کلیک پایان)"
        )
        help_lbl.setWordWrap(True)
        help_lbl.setLayoutDirection(Qt.RightToLeft)
        sv.addWidget(help_lbl)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(lambda _r: self._redraw())
        sv.addWidget(QLabel("Landmarks / شاخص‌ها"))
        sv.addWidget(self.list, stretch=1)

        row = QHBoxLayout()
        b_rm = QPushButton("Remove / حذف")
        b_rm.clicked.connect(self._remove_current)
        b_clear = QPushButton("Clear / پاک‌کردن همه")
        b_clear.clicked.connect(self._clear)
        row.addWidget(b_rm)
        row.addWidget(b_clear)
        sv.addLayout(row)

        row2 = QHBoxLayout()
        b_rect = QPushButton("▭ Rectangle / مستطیل")
        b_rect.clicked.connect(lambda: self.view.set_mode("rect"))
        b_poly = QPushButton("⬠ Polygon / چندضلعی")
        b_poly.clicked.connect(lambda: self.view.set_mode("poly"))
        row2.addWidget(b_rect)
        row2.addWidget(b_poly)
        sv.addLayout(row2)

        row3 = QHBoxLayout()
        self.auto_k = QSpinBox()
        self.auto_k.setRange(1, 24)
        self.auto_k.setValue(max(1, len(self.selected) or 8))
        b_auto = QPushButton("Auto-select / انتخاب خودکار")
        b_auto.clicked.connect(self._auto_select)
        row3.addWidget(self.auto_k)
        row3.addWidget(b_auto)
        sv.addLayout(row3)

        self.show_cands = QCheckBox("Show candidates / نمایش پیشنهادها")
        self.show_cands.setChecked(True)
        self.show_cands.toggled.connect(lambda _v: self._redraw())
        sv.addWidget(self.show_cands)
        b_fit = QPushButton("Fit view / نمای کامل")
        b_fit.clicked.connect(self.view.fit)
        sv.addWidget(b_fit)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        sv.addWidget(self.info)

        sv.addWidget(QLabel("Imagery sources / منابع تصویر (other satellites, dates, night)"))
        self.src_list = QListWidget()
        self.src_list.setMaximumHeight(110)
        sv.addWidget(self.src_list)
        srow = QHBoxLayout()
        b_add_src = QPushButton("+ Add satellite image… / افزودن تصویر")
        b_add_src.setToolTip(
            "Image of the same area from another satellite or sensor — e.g. a real NIGHT image "
            "(SDGSAT-1, Jilin-1, ISS photo). It is aligned to this map and trained with the same labels."
        )
        b_add_src.clicked.connect(self._add_source)
        b_rm_src = QPushButton("− Remove")
        b_rm_src.clicked.connect(self._remove_source)
        srow.addWidget(b_add_src)
        srow.addWidget(b_rm_src)
        sv.addLayout(srow)
        self._refresh_sources()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("OK — build videos / تأیید")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        sv.addWidget(buttons)
        root.addWidget(side, stretch=1)

        self.view.set_painter(self._paint_overlays)
        self._refresh_list()

    # ------------------------------------------------------------ overlays
    def _paint_overlays(self, img: np.ndarray, s: float) -> None:
        thick = max(2, int(round(3 * max(1.0, img.shape[1] / 1600))))
        font = max(0.5, 0.6 * img.shape[1] / 1600)
        if self.show_cands.isChecked():
            for c in self.candidates:
                if c in self.selected:
                    continue
                cv2.polylines(img, [np.int32(c.polygon * s).reshape(-1, 1, 2)], True, CANDIDATE, max(1, thick // 2), cv2.LINE_AA)
        cur = self.list.currentRow()
        for i, c in enumerate(self.selected):
            pts = np.int32(c.polygon * s).reshape(-1, 1, 2)
            color = (0, 0, 255) if i == cur else SELECTED
            cv2.polylines(img, [pts], True, color, thick + (2 if i == cur else 0), cv2.LINE_AA)
            x, y = pts[:, 0, 0].min(), pts[:, 0, 1].min()
            label = f"{i + 1}"
            cv2.putText(img, label, (int(x), max(14, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(img, label, (int(x), max(14, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
        tp = np.int32(self.target_poly * s).reshape(-1, 1, 2)
        cv2.polylines(img, [tp], True, TARGET, thick + 1, cv2.LINE_AA)
        cv2.putText(img, "TARGET", (int(tp[:, 0, 0].min()), max(14, int(tp[:, 0, 1].min()) - 6)), cv2.FONT_HERSHEY_SIMPLEX, font, TARGET, thick, cv2.LINE_AA)

    def _redraw(self) -> None:
        self.view.refresh()

    def _refresh_list(self) -> None:
        row = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for i, c in enumerate(self.selected):
            dist = f"{c.distance * self.mpp:.0f} m" if self.mpp else f"{c.distance:.0f} px"
            size = f"{c.size * self.mpp:.0f} m" if self.mpp else f"{c.size:.0f} px"
            self.list.addItem(f"{i + 1}. {c.kind} ({KIND_FA.get(c.kind, c.kind)}) — {dist}, size {size}")
        self.list.blockSignals(False)
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)
        kinds = {}
        for c in self.selected:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        self.info.setText(
            f"{len(self.selected)} landmarks selected, {len(self.candidates)} candidates.\n"
            + ", ".join(f"{k}: {v}" for k, v in kinds.items())
            + ("\n⚠ at least 2 landmarks are recommended" if len(self.selected) < 2 else "")
        )
        self._redraw()

    # ------------------------------------------------------------ actions
    def _hit(self, objs: Sequence[ObjectCandidate], x: float, y: float) -> Optional[ObjectCandidate]:
        hits = [c for c in objs if cv2.pointPolygonTest(c.polygon.reshape(-1, 1, 2).astype(np.float32), (x, y), False) >= 0]
        return min(hits, key=lambda c: c.size) if hits else None

    def _on_click(self, x: float, y: float) -> None:
        hit = self._hit(self.selected, x, y)
        if hit is not None:
            self.selected.remove(hit)
        else:
            cand = self._hit([c for c in self.candidates if c not in self.selected] if self.show_cands.isChecked() else [], x, y)
            if cand is None:
                return
            if cv2.pointPolygonTest(self.target_poly.reshape(-1, 1, 2), (x, y), False) >= 0:
                return
            self.selected.append(cand)
            self.list.setCurrentRow(len(self.selected) - 1)
        self._refresh_list()

    def _on_drawn(self, poly: np.ndarray) -> None:
        tc = self.target_poly.mean(axis=0)
        c = ObjectCandidate(polygon=np.asarray(poly, np.float32), kind="manual", score=1.0)
        c.distance = float(np.linalg.norm(c.center - tc))
        self.candidates.append(c)
        self.selected.append(c)
        self._refresh_list()
        self.list.setCurrentRow(len(self.selected) - 1)

    def _remove_current(self) -> None:
        r = self.list.currentRow()
        if 0 <= r < len(self.selected):
            self.selected.pop(r)
            self._refresh_list()

    def _clear(self) -> None:
        self.selected = []
        self._refresh_list()

    def _auto_select(self) -> None:
        (x0, y0), (x1, y1) = self.target_poly.min(axis=0), self.target_poly.max(axis=0)
        H, W = self.mosaic.shape[:2]
        pool = [c for c in self.candidates if c.kind != "manual"]
        manual = [c for c in self.selected if c.kind == "manual"]
        k = max(0, self.auto_k.value() - len(manual))
        self.selected = manual + select_landmarks(pool, (x0, y0, x1, y1), (W, H), k=k)
        self._refresh_list()

    # ------------------------------------------------------------ imagery sources
    def _refresh_sources(self) -> None:
        self.src_list.clear()
        self.src_list.addItem("• base map (primary)")
        for name, _img in self.extras:
            tag = "🌙 night" if name.startswith("night:") else "day"
            cov = self.extra_masks.get(name)
            cov_s = f", covers {100 * float((cov > 0).mean()):.0f}%" if cov is not None else ""
            self.src_list.addItem(f"• {name.replace('night:', '')} ({tag}{cov_s})")

    def _add_source(self) -> None:
        from shared.qt_compat import QFileDialog
        from shared.qt_compat import qt_exec as _exec

        path, _ = QFileDialog.getOpenFileName(
            self, "Image from another satellite", "", "Images (*.tif *.tiff *.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.info.setText(f"Cannot read {path}")
            return
        from pathlib import Path as _P

        dlg = PointPairDialog(img, self.mosaic, name=_P(path).stem, parent=self)
        if "night" in _P(path).stem.lower():
            dlg.is_night.setChecked(True)
        if _exec(dlg) != QDialog.Accepted:
            return
        name, warped, mask = dlg.aligned()
        existing = {n for n, _ in self.extras}
        base, k = name, 2
        while name in existing:
            name = f"{base}_{k}"
            k += 1
        self.extras.append((name, warped))
        self.extra_masks[name] = mask
        self._refresh_sources()

    def _remove_source(self) -> None:
        r = self.src_list.currentRow()
        if r <= 0 or r - 1 >= len(self.extras):
            return
        name, _ = self.extras.pop(r - 1)
        self.extra_masks.pop(name, None)
        self._refresh_sources()

    def result_extras(self):
        return list(self.extras), dict(self.extra_masks)

    # ------------------------------------------------------------ result
    def result_layout(self) -> SceneLayout:
        meta = {k: v for k, v in self.layout_in.meta.items() if k != "landmarks"}
        meta["edited"] = True
        H, W = self.mosaic.shape[:2]
        return layout_from_selection(
            self.target_poly.tolist(),
            self.selected,
            (W, H),
            target_class=self.layout_in.classes[0],
            meters_per_px=self.mpp,
            meta=meta,
        )

    def result_candidates(self) -> List[ObjectCandidate]:
        return self.candidates


class TargetPickerDialog(QDialog):
    """Draw the target region (rectangle) on your own image."""

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select target region / انتخاب منطقه اصلی")
        self.resize(1200, 800)
        self.image = image
        self.rect: Optional[tuple] = None
        root = QVBoxLayout(self)
        hint = QLabel(
            "روی تصویر یک مستطیل دور منطقه اصلی بکشید (چرخ ماوس = زوم، کشیدن با دکمه «جابجایی» فعال است).\n"
            "Drag a rectangle around the target region. Wheel = zoom."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)
        self.view = MosaicView(image)
        self.view.shape_drawn.connect(self._on_rect)
        root.addWidget(self.view, stretch=1)
        row = QHBoxLayout()
        b_draw = QPushButton("▭ Draw target / رسم منطقه")
        b_draw.clicked.connect(lambda: self.view.set_mode("rect"))
        b_pan = QPushButton("✋ Pan / جابجایی")
        b_pan.clicked.connect(lambda: self.view.set_mode(None))
        self.info = QLabel("No region yet")
        row.addWidget(b_draw)
        row.addWidget(b_pan)
        row.addWidget(self.info, stretch=1)
        root.addLayout(row)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self.view.set_painter(self._paint)
        self.view.set_mode("rect")

    def _paint(self, img: np.ndarray, s: float) -> None:
        if self.rect:
            x0, y0, x1, y1 = (int(v * s) for v in self.rect)
            cv2.rectangle(img, (x0, y0), (x1, y1), TARGET, max(2, int(img.shape[1] / 600)))

    def _on_rect(self, poly: np.ndarray) -> None:
        (x0, y0), (x1, y1) = poly.min(axis=0), poly.max(axis=0)
        H, W = self.image.shape[:2]
        self.rect = (max(0.0, float(x0)), max(0.0, float(y0)), min(float(W), float(x1)), min(float(H), float(y1)))
        self.info.setText(f"Target: {int(self.rect[2] - self.rect[0])} × {int(self.rect[3] - self.rect[1])} px")
        self.view.refresh()
        self.view.set_mode("rect")

    def _accept(self) -> None:
        if not self.rect or self.rect[2] - self.rect[0] < 8 or self.rect[3] - self.rect[1] < 8:
            self.info.setText("⚠ Draw the target rectangle first / ابتدا منطقه را رسم کنید")
            return
        self.accept()


class PointPairDialog(QDialog):
    """
    Align an image from another satellite (e.g. a real night image) to the scene mosaic:
    try automatic alignment, or click matching points — left image, then the same spot on the right.
    """

    def __init__(self, image: np.ndarray, mosaic: np.ndarray, name: str = "extra", parent=None):
        super().__init__(parent)
        from shared.qt_compat import QLineEdit

        self.setWindowTitle("Align image from another satellite / هم‌ترازی تصویر ماهواره دیگر")
        self.resize(1500, 850)
        self.image = image
        self.mosaic = mosaic
        self.pairs: List[tuple] = []  # ((ix, iy), (mx, my))
        self._pending: Optional[tuple] = None
        self.alignment = None
        self.warped: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None

        root = QVBoxLayout(self)
        hint = QLabel(
            "۱) «هم‌ترازی خودکار» را امتحان کنید (برای تصاویر روز معمولاً کار می‌کند).\n"
            "۲) اگر نشد: روی یک نقطه مشخص در تصویر چپ کلیک کنید، سپس همان نقطه را در تصویر راست — حداقل ۴ جفت\n"
            "   (تقاطع خیابان‌ها، گوشه ساختمان‌ها، میدان‌ها). برای تصویر شب، نقاط روشن مثل تقاطع‌های پرنور مناسب‌اند.\n"
            "Left: new image · Right: scene mosaic. Try Auto-align, or click ≥4 matching point pairs (left then right)."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)
        views = QHBoxLayout()
        self.left = MosaicView(image, max_base=2000)
        self.right = MosaicView(mosaic, max_base=2000)
        self.left.setMinimumSize(560, 480)
        self.right.setMinimumSize(560, 480)
        self.left.clicked.connect(self._click_left)
        self.right.clicked.connect(self._click_right)
        views.addWidget(self.left)
        views.addWidget(self.right)
        root.addLayout(views, stretch=1)

        row = QHBoxLayout()
        b_auto = QPushButton("⚙ Auto-align / هم‌ترازی خودکار")
        b_auto.clicked.connect(self._auto)
        b_undo = QPushButton("↶ Undo pair")
        b_undo.clicked.connect(self._undo)
        b_clear = QPushButton("Clear points")
        b_clear.clicked.connect(self._clear)
        self.overlay = QCheckBox("Overlay preview / پیش‌نمایش روی هم")
        self.overlay.setChecked(True)
        self.overlay.toggled.connect(lambda _v: self.right.refresh())
        self.is_night = QCheckBox("This is a NIGHT image / تصویر شب است")
        self.name = QLineEdit(name)
        self.name.setMaximumWidth(260)
        for w_ in (b_auto, b_undo, b_clear, self.overlay, self.is_night, QLabel("Name:"), self.name):
            row.addWidget(w_)
        root.addLayout(row)
        self.info = QLabel("0 pairs")
        root.addWidget(self.info)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.left.set_painter(lambda img, s: self._paint_pts(img, s, 0))
        self.right.set_painter(self._paint_right)

    # ---- drawing
    def _paint_pts(self, img: np.ndarray, s: float, side: int) -> None:
        r = max(4, int(img.shape[1] / 250))
        pts = [p[side] for p in self.pairs] + ([self._pending] if (self._pending and side == 0) else [])
        for i, (x, y) in enumerate(pts):
            c = (int(x * s), int(y * s))
            cv2.circle(img, c, r, (0, 0, 255), -1)
            cv2.circle(img, c, r + 2, (255, 255, 255), 2)
            cv2.putText(img, str(i + 1), (c[0] + r + 3, c[1] - r), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, r / 8), (0, 255, 255), 2)

    def _paint_right(self, img: np.ndarray, s: float) -> None:
        if self.overlay.isChecked() and self.warped is not None:
            w_small = cv2.resize(self.warped, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            m_small = cv2.resize(self.mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            img[m_small] = (0.45 * img[m_small] + 0.55 * w_small[m_small]).astype(np.uint8)
        self._paint_pts(img, s, 1)

    # ---- interaction
    def _click_left(self, x: float, y: float) -> None:
        self._pending = (x, y)
        self.left.refresh()
        self.info.setText(f"{len(self.pairs)} pairs — now click the SAME point on the right image")

    def _click_right(self, x: float, y: float) -> None:
        if self._pending is None:
            self.info.setText("Click on the LEFT image first / ابتدا روی تصویر چپ کلیک کنید")
            return
        self.pairs.append((self._pending, (x, y)))
        self._pending = None
        self._update_from_pairs()

    def _update_from_pairs(self) -> None:
        from desktop_app.satellite.align import align_from_points, warp_to_mosaic

        msg = f"{len(self.pairs)} pairs"
        if len(self.pairs) >= 3:
            try:
                self.alignment = align_from_points([p[0] for p in self.pairs], [p[1] for p in self.pairs])
                self.warped, self.mask = warp_to_mosaic(self.image, self.alignment, self.mosaic.shape)
                msg += f" — aligned, mean error {self.alignment.reproj_error:.1f} px"
            except ValueError as exc:
                msg += f" — {exc}"
        else:
            msg += " — need at least 4"
        self.info.setText(msg)
        self.left.refresh()
        self.right.refresh()

    def _undo(self) -> None:
        if self._pending is not None:
            self._pending = None
        elif self.pairs:
            self.pairs.pop()
        self.alignment = self.warped = self.mask = None
        self._update_from_pairs()

    def _clear(self) -> None:
        self.pairs, self._pending = [], None
        self.alignment = self.warped = self.mask = None
        self._update_from_pairs()

    def _auto(self) -> None:
        from desktop_app.satellite.align import auto_align, warp_to_mosaic
        from shared.qt_compat import QApplication

        self.info.setText("Auto-aligning…")
        QApplication.processEvents()
        al = auto_align(self.image, self.mosaic)
        if al is None:
            self.info.setText(
                "Auto-align failed (common for night↔day). Click ≥4 matching point pairs instead.\n"
                "هم‌ترازی خودکار نشد؛ لطفاً حداقل ۴ جفت نقطه متناظر را کلیک کنید."
            )
            return
        self.alignment = al
        self.warped, self.mask = warp_to_mosaic(self.image, al, self.mosaic.shape)
        self.info.setText(f"Auto-aligned: {al.inliers} matches, error {al.reproj_error:.1f} px — check the overlay")
        self.right.refresh()

    def _accept(self) -> None:
        if self.warped is None:
            self.info.setText("⚠ Align first (auto or ≥4 point pairs) / ابتدا هم‌ترازی کنید")
            return
        if float((self.mask > 0).mean()) < 0.05:
            self.info.setText("⚠ The aligned image barely overlaps the scene — check the points")
            return
        self.accept()

    def aligned(self):
        """(name, warped image in mosaic frame, coverage mask)."""
        name = self.name.text().strip() or "extra"
        if self.is_night.isChecked() and not name.startswith("night:"):
            name = "night:" + name
        return name, self.warped, self.mask
