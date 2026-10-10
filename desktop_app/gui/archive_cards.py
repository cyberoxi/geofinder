"""Thumbnail cards for the satellite compare gallery."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from shared.qt_compat import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QImage,
    QLabel,
    QPixmap,
    QSizePolicy,
    QVBoxLayout,
    Qt,
)
from desktop_app.satellite.archives import ArchiveScene

CARD_W, CARD_H = 220, 200


def numpy_to_pixmap(img: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy()).scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class PreviewCard(QFrame):
    """One satellite / scene tile: thumbnail + checkbox + caption."""

    def __init__(
        self,
        key: str,
        title: str,
        subtitle: str,
        *,
        checkable: bool = True,
        checked: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.key = key
        self.scene: Optional[ArchiveScene] = None
        self._click_cb = None
        shape = getattr(QFrame, "StyledPanel", None) or QFrame.Shape.StyledPanel
        self.setFrameShape(shape)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(CARD_W, CARD_H)
        self.setStyleSheet(
            "QFrame { background:#2a2a2a; border:1px solid #444; border-radius:6px; }"
            "QFrame[selected='true'] { border:2px solid #4ea1ff; }"
        )
        self.setProperty("selected", False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        top = QHBoxLayout()
        self.check = QCheckBox()
        self.check.setVisible(checkable)
        self.check.setChecked(checked and checkable)
        if not checkable:
            self.check.setEnabled(False)
        top.addWidget(self.check)
        self.caption = QLabel(title)
        self.caption.setStyleSheet("color:#eee;font-weight:600;")
        self.caption.setWordWrap(True)
        top.addWidget(self.caption, stretch=1)
        lay.addLayout(top)

        self.thumb = QLabel("…")
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setMinimumHeight(120)
        self.thumb.setStyleSheet("background:#111;color:#888;border-radius:4px;")
        expand = getattr(QSizePolicy, "Expanding", None) or QSizePolicy.Policy.Expanding
        self.thumb.setSizePolicy(expand, expand)
        lay.addWidget(self.thumb, stretch=1)

        self.sub = QLabel(subtitle)
        self.sub.setStyleSheet("color:#aaa;font-size:11px;")
        self.sub.setWordWrap(True)
        lay.addWidget(self.sub)

    def set_image(self, img: Optional[np.ndarray]) -> None:
        if img is None:
            self.thumb.setText("No preview")
            self.thumb.setPixmap(QPixmap())
            return
        self.thumb.setPixmap(numpy_to_pixmap(img, CARD_W - 20, 130))

    def set_selected_style(self, on: bool) -> None:
        self.setProperty("selected", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):  # noqa: N802
        super().mousePressEvent(event)
        if self._click_cb:
            self._click_cb(self)
