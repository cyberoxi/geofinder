"""Synced mini previews of secondary satellite layers (same viewport as main map)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from shared.qt_compat import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPixmap,
    QScrollArea,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    Signal,
)
from desktop_app.gui.archive_cards import numpy_to_pixmap
from desktop_app.gui.workers import StageWorker
from desktop_app.satellite.preview_fetch import fetch_basemap_preview

_THUMB_W, _THUMB_H = 160, 110


class _MiniCard(QFrame):
    clicked = Signal(str)  # layer key

    def __init__(self, layer: str, title: str, parent=None):
        super().__init__(parent)
        self.layer = layer
        shape = getattr(QFrame, "StyledPanel", None) or QFrame.Shape.StyledPanel
        self.setFrameShape(shape)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(_THUMB_W + 12, _THUMB_H + 36)
        self.setStyleSheet(
            "QFrame { background:#222; border:1px solid #444; border-radius:4px; }"
            "QFrame:hover { border-color:#4ea1ff; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(2)
        self.caption = QLabel(title)
        self.caption.setStyleSheet("color:#ddd;font-size:11px;")
        self.caption.setWordWrap(True)
        lay.addWidget(self.caption)
        self.img = QLabel("…")
        self.img.setFixedSize(_THUMB_W, _THUMB_H)
        self.img.setAlignment(Qt.AlignCenter)
        self.img.setStyleSheet("background:#111;color:#666;")
        lay.addWidget(self.img)

    def set_image(self, arr: Optional[np.ndarray]) -> None:
        if arr is None:
            self.img.setText("—")
            self.img.setPixmap(QPixmap())
            return
        self.img.setPixmap(numpy_to_pixmap(arr, _THUMB_W, _THUMB_H))

    def mousePressEvent(self, event):  # noqa: N802
        super().mousePressEvent(event)
        self.clicked.emit(self.layer)


class SatelliteMiniStrip(QWidget):
    """
    Horizontal strip under the main map: other selected satellites show the
    same viewport. Click a mini to promote it to the main map.
    """

    promote_layer = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(0)
        self.setMaximumHeight(168)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 4, 0, 0)
        root.setSpacing(2)
        self.hint = QLabel(
            "Other satellites (same view) — click a thumbnail to make it the main map"
        )
        self.hint.setStyleSheet("color:#999;font-size:11px;")
        root.addWidget(self.hint)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        h_bar = getattr(Qt, "ScrollBarAsNeeded", None) or Qt.ScrollBarPolicy.ScrollBarAsNeeded
        v_off = getattr(Qt, "ScrollBarAlwaysOff", None) or Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        self.scroll.setHorizontalScrollBarPolicy(h_bar)
        self.scroll.setVerticalScrollBarPolicy(v_off)
        self.scroll.setFixedHeight(140)
        self.row_host = QWidget()
        self.row = QHBoxLayout(self.row_host)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(8)
        self.row.addStretch(1)
        self.scroll.setWidget(self.row_host)
        root.addWidget(self.scroll)

        self._cards: Dict[str, _MiniCard] = {}
        self._layers: List[Tuple[str, str]] = []
        self._bbox: Optional[Tuple[float, float, float, float]] = None
        self._worker: Optional[StageWorker] = None
        self._pending = False
        self._alive = True
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(450)
        self._debounce.timeout.connect(self._refresh_now)
        self.setVisible(False)

    def set_secondary_layers(self, layers: List[Tuple[str, str]]) -> None:
        """layers = [(key, title), ...] excluding the main map basemap."""
        self._layers = list(layers)
        keys = [k for k, _ in self._layers]
        if set(keys) != set(self._cards.keys()):
            while self.row.count():
                item = self.row.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
            self._cards.clear()
            for key, title in self._layers:
                card = _MiniCard(key, title)
                card.clicked.connect(self.promote_layer.emit)
                self.row.addWidget(card)
                self._cards[key] = card
            self.row.addStretch(1)
        else:
            for key, title in self._layers:
                if key in self._cards:
                    self._cards[key].caption.setText(title)
        self.setVisible(bool(self._layers))
        if self._layers and self._bbox is not None:
            self.schedule_refresh(self._bbox)

    def schedule_refresh(self, bbox: Tuple[float, float, float, float]) -> None:
        self._bbox = bbox
        if not self._layers or not self._alive:
            return
        self._debounce.start()

    def shutdown(self) -> None:
        """Stop timers/workers before the parent dialog is destroyed."""
        self._alive = False
        self._debounce.stop()
        self._pending = False
        w = self._worker
        self._worker = None
        if w is None:
            return
        try:
            w.finished_ok.disconnect(self._on_ready)
        except (TypeError, RuntimeError):
            pass
        if w.isRunning():
            # Let the job finish; do not GC the QThread while running
            w.finished.connect(w.deleteLater)
            if not w.wait(3000):
                w.terminate()
                w.wait(500)
        else:
            w.deleteLater()

    def _refresh_now(self) -> None:
        if not self._alive or not self._layers or self._bbox is None:
            return
        if self._worker is not None and self._worker.isRunning():
            # Coalesce: run again when the current fetch finishes
            self._pending = True
            return

        south, west, north, east = self._bbox
        layers = list(self._layers)

        def job(progress, _frame):
            out: Dict[str, np.ndarray] = {}
            n = max(1, len(layers))
            for i, (key, _title) in enumerate(layers):
                if progress:
                    progress(i, n, f"Mini preview {i + 1}/{n}")
                img = fetch_basemap_preview(
                    south, west, north, east, key, max_side=max(_THUMB_W, _THUMB_H), max_tiles=20
                )
                if img is not None:
                    out[key] = img
            if progress:
                progress(n, n, "Mini previews ready")
            return out

        self._worker = StageWorker(job)
        self._worker.finished_ok.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, _err: str) -> None:
        if not self._alive:
            return
        if self._pending:
            self._pending = False
            self._refresh_now()

    def _on_ready(self, mapping: object) -> None:
        if not self._alive:
            return
        if isinstance(mapping, dict):
            for key, card in self._cards.items():
                card.set_image(mapping.get(key))
        if self._pending:
            self._pending = False
            self._refresh_now()
