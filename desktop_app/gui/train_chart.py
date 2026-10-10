"""Live training metrics chart shown in the mid panel during YOLO training."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from shared.qt_compat import (
    QColor,
    QFont,
    QHBoxLayout,
    QLabel,
    QPainter,
    QPen,
    QPointF,
    QPolygonF,
    QPushButton,
    Qt,
    QVBoxLayout,
    QWidget,
)

# (legend label, metric key candidates, stroke color)
_SCORE_SERIES: Sequence[Tuple[str, Tuple[str, ...], str]] = (
    ("mAP50", ("metrics/mAP50(M)", "metrics/mAP50(B)"), "#34d399"),
    ("Precision", ("metrics/precision(M)", "metrics/precision(B)"), "#38bdf8"),
    ("Recall", ("metrics/recall(M)", "metrics/recall(B)"), "#fbbf24"),
)
_LOSS_KEYS: Tuple[str, ...] = ("train/box_loss", "train/seg_loss", "train/cls_loss")


def _pick(metrics: Dict[str, float], keys: Tuple[str, ...]) -> Optional[float]:
    for k in keys:
        if k in metrics:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                continue
    return None


class _PlotCanvas(QWidget):
    """Paints score curves (0–1) and optional loss on a second axis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.epochs: List[int] = []
        self.scores: Dict[str, List[Optional[float]]] = {name: [] for name, _, _ in _SCORE_SERIES}
        self.losses: List[Optional[float]] = []
        self.total_epochs: int = 0
        self.batch_frac: float = 0.0  # 0..1 within current unfinished epoch
        self.current_epoch: int = 0

    def reset(self, total_epochs: int = 0) -> None:
        self.epochs.clear()
        for k in self.scores:
            self.scores[k].clear()
        self.losses.clear()
        self.total_epochs = max(0, int(total_epochs or 0))
        self.batch_frac = 0.0
        self.current_epoch = 0
        self.update()

    def set_batch(self, epoch: int, epochs: int, fraction: float) -> None:
        self.current_epoch = int(epoch or 0)
        if epochs:
            self.total_epochs = max(self.total_epochs, int(epochs))
        self.batch_frac = max(0.0, min(1.0, float(fraction or 0.0)))
        self.update()

    def add_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        ep = int(epoch or 0)
        if ep <= 0:
            return
        if self.epochs and ep == self.epochs[-1]:
            idx = -1
        else:
            self.epochs.append(ep)
            for name in self.scores:
                self.scores[name].append(None)
            self.losses.append(None)
            idx = -1
        for name, keys, _ in _SCORE_SERIES:
            val = _pick(metrics, keys)
            if val is not None:
                self.scores[name][idx] = val
        loss_vals = [metrics[k] for k in _LOSS_KEYS if k in metrics]
        if loss_vals:
            self.losses[idx] = float(sum(loss_vals) / len(loss_vals))
        if self.total_epochs:
            self.total_epochs = max(self.total_epochs, ep)
        self.batch_frac = 0.0
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0b1220"))

        left, right, top, bottom = 52, 52, 28, 36
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)
        x0, y0 = left, top

        p.setPen(QPen(QColor("#243041"), 1))
        p.setBrush(QColor("#0f172a"))
        p.drawRoundedRect(x0, y0, plot_w, plot_h, 8, 8)

        # Grid
        p.setFont(QFont("Segoe UI", 8))
        for i in range(5):
            y = y0 + int(plot_h * i / 4)
            p.setPen(QPen(QColor("#1e293b"), 1, Qt.DashLine))
            p.drawLine(x0, y, x0 + plot_w, y)
            score = 1.0 - i / 4
            p.setPen(QColor("#64748b"))
            p.drawText(4, y + 4, f"{score:.2f}")

        xmax = max(self.total_epochs, self.epochs[-1] if self.epochs else 1, 1)
        # X ticks
        p.setPen(QColor("#64748b"))
        step = max(1, xmax // 6)
        for ep in range(0, xmax + 1, step):
            x = x0 + int(plot_w * ep / xmax)
            p.setPen(QPen(QColor("#1e293b"), 1, Qt.DotLine))
            p.drawLine(x, y0, x, y0 + plot_h)
            p.setPen(QColor("#64748b"))
            p.drawText(x - 8, y0 + plot_h + 16, str(ep))
        p.drawText(x0 + plot_w // 2 - 20, h - 6, "Epoch")

        # Loss axis (right) if we have values
        loss_max = max((v for v in self.losses if v is not None), default=0.0)
        if loss_max > 0:
            p.setPen(QColor("#f87171"))
            for i in range(5):
                y = y0 + int(plot_h * i / 4)
                lv = loss_max * (1.0 - i / 4)
                p.drawText(x0 + plot_w + 6, y + 4, f"{lv:.2f}")

        def _xy_score(ep: float, val: float) -> QPointF:
            x = x0 + plot_w * (ep / xmax)
            y = y0 + plot_h * (1.0 - max(0.0, min(1.0, val)))
            return QPointF(x, y)

        def _xy_loss(ep: float, val: float) -> QPointF:
            x = x0 + plot_w * (ep / xmax)
            y = y0 + plot_h * (1.0 - max(0.0, min(1.0, val / loss_max))) if loss_max > 0 else y0 + plot_h
            return QPointF(x, y)

        # Loss curve (behind scores)
        if any(v is not None for v in self.losses) and loss_max > 0:
            pts = QPolygonF()
            for ep, lv in zip(self.epochs, self.losses):
                if lv is None:
                    continue
                pts.append(_xy_loss(float(ep), float(lv)))
            n = len(pts)
            if n >= 2:
                p.setPen(QPen(QColor("#f8717188"), 2))
                p.drawPolyline(pts)
            elif n == 1:
                p.setBrush(QColor("#f87171"))
                p.setPen(QPen(Qt.NoPen))
                p.drawEllipse(pts[0], 3, 3)

        # Score curves
        for name, _keys, color in _SCORE_SERIES:
            series = self.scores[name]
            pts = QPolygonF()
            for ep, val in zip(self.epochs, series):
                if val is None:
                    continue
                pts.append(_xy_score(float(ep), float(val)))
            if len(pts) >= 2:
                p.setPen(QPen(QColor(color), 2.5))
                p.drawPolyline(pts)
            for i in range(len(pts)):
                p.setBrush(QColor(color))
                p.setPen(QPen(Qt.NoPen))
                p.drawEllipse(pts[i], 3.5, 3.5)

        # In-progress cursor (batch fraction across whole run)
        if self.total_epochs and self.batch_frac > 0:
            x = x0 + int(plot_w * self.batch_frac)
            p.setPen(QPen(QColor("#94a3b8"), 1, Qt.DashLine))
            p.drawLine(x, y0, x, y0 + plot_h)

        # Legend
        lx = x0 + 10
        ly = y0 + 14
        p.setFont(QFont("Segoe UI", 9))
        for name, _keys, color in _SCORE_SERIES:
            p.setBrush(QColor(color))
            p.setPen(QPen(Qt.NoPen))
            p.drawRect(lx, ly - 7, 12, 4)
            p.setPen(QColor("#cbd5e1"))
            p.drawText(lx + 16, ly, name)
            lx += 88
        if loss_max > 0:
            p.setBrush(QColor("#f87171"))
            p.setPen(QPen(Qt.NoPen))
            p.drawRect(lx, ly - 7, 12, 4)
            p.setPen(QColor("#cbd5e1"))
            p.drawText(lx + 16, ly, "Loss")

        if not self.epochs:
            p.setPen(QColor("#64748b"))
            p.setFont(QFont("Segoe UI", 11))
            p.drawText(self.rect(), Qt.AlignCenter, "Waiting for first epoch…\nدر انتظار اولین epoch")


class TrainLiveChart(QWidget):
    """Mid-panel live learning curves + headline metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "TrainLiveChart { background:#0b1220; border:1px solid #243041; border-radius:8px; }"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        head = QHBoxLayout()
        self.title = QLabel("Learning progress / پیشرفت یادگیری")
        self.title.setStyleSheet("color:#e2e8f0;font-weight:600;font-size:14px;")
        head.addWidget(self.title, stretch=1)
        self.btn_back = QPushButton("Back to video")
        self.btn_back.setToolTip("بازگشت به نمایش ویدیو")
        self.btn_back.setFixedHeight(28)
        head.addWidget(self.btn_back)
        root.addLayout(head)

        self.subtitle = QLabel("Training…")
        self.subtitle.setStyleSheet("color:#94a3b8;")
        self.subtitle.setWordWrap(True)
        root.addWidget(self.subtitle)

        self.stats = QLabel("—")
        self.stats.setStyleSheet(
            "color:#cbd5e1;background:#111827;border:1px solid #243041;"
            "border-radius:6px;padding:8px 10px;"
        )
        self.stats.setWordWrap(True)
        root.addWidget(self.stats)

        self.plot = _PlotCanvas()
        root.addWidget(self.plot, stretch=1)

        tip = QLabel(
            "mAP50 ↑ بهتر · Precision/Recall تعادل تشخیص · Loss ↓ باید کم شود"
        )
        tip.setStyleSheet("color:#64748b;font-size:11px;")
        tip.setWordWrap(True)
        root.addWidget(tip)

        self._last_metrics: Dict[str, float] = {}

    def reset(self, total_epochs: int = 0) -> None:
        self.plot.reset(total_epochs)
        self._last_metrics = {}
        self.subtitle.setText("Training…")
        self.stats.setText("—")

    def set_status(self, text: str) -> None:
        if text:
            self.subtitle.setText(text)

    def set_batch(
        self,
        epoch: int,
        epochs: int,
        batch: int,
        nb: int,
        fraction: float,
        eta_sec: Optional[float] = None,
    ) -> None:
        self.plot.set_batch(epoch, epochs, fraction)
        eta_s = ""
        if eta_sec:
            eta_s = f"  ·  ETA {int(eta_sec // 60)}m{int(eta_sec % 60):02d}s"
        self.subtitle.setText(f"Epoch {epoch}/{epochs}  ·  batch {batch}/{nb}{eta_s}")

    def add_epoch(self, epoch: int, metrics: Dict[str, float], epochs: int = 0) -> None:
        self._last_metrics = dict(metrics or {})
        if epochs:
            self.plot.total_epochs = max(self.plot.total_epochs, int(epochs))
        self.plot.add_epoch(epoch, self._last_metrics)
        parts = []
        for name, keys, _ in _SCORE_SERIES:
            v = _pick(self._last_metrics, keys)
            if v is not None:
                parts.append(f"{name} {v:.3f}")
        loss_vals = [self._last_metrics[k] for k in _LOSS_KEYS if k in self._last_metrics]
        if loss_vals:
            parts.append(f"Loss {sum(loss_vals) / len(loss_vals):.3f}")
        self.stats.setText(
            f"Epoch {epoch}" + (f" / {epochs}" if epochs else "") + ("  ·  " + "  ·  ".join(parts) if parts else "")
        )
        self.subtitle.setText(f"Epoch {epoch} finished")

    def mark_done(self, ok: bool = True) -> None:
        self.subtitle.setText("Training finished" if ok else "Training failed / آموزش ناموفق")
