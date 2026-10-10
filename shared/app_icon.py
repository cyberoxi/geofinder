"""Application brand icon for window chrome and Windows taskbar."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from shared.qt_compat import QApplication, QColor, QIcon, QPainter, QPen, QPixmap, QPointF, Qt

try:
    from shared.qt_compat import QPolygonF
except ImportError:  # pragma: no cover
    from PySide6.QtGui import QPolygonF  # type: ignore

_ICON_PATH = Path(__file__).resolve().parent / "assets" / "app.ico"
_APP_USER_MODEL_ID = "GeoFinder.Tangasiri.Desktop.1"

_CACHE: Optional[QIcon] = None


def icon_path() -> Path:
    return _ICON_PATH


def ensure_windows_app_id(app_id: str = _APP_USER_MODEL_ID) -> None:
    """Group the process under our AppUserModelID so the taskbar uses our icon."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)  # type: ignore[attr-defined]
    except Exception:
        pass


def _paint_fallback(size: int) -> QPixmap:
    """Vector fallback if app.ico is missing (matches brand teal / crosshair look)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    s = float(size)
    m = s * 0.06
    bg = QColor("#0f172a")
    p.setPen(Qt.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(m, m, s - 2 * m, s - 2 * m, s * 0.22, s * 0.22)

    cx = cy = s * 0.5
    r = s * 0.28
    p.setBrush(QColor("#38bdf8"))
    p.drawEllipse(QPointF(cx, cy), r, r)
    p.setBrush(QColor("#14b8a6"))
    p.drawEllipse(QPointF(cx - r * 0.15, cy - r * 0.1), r * 0.55, r * 0.45)
    p.drawEllipse(QPointF(cx + r * 0.25, cy + r * 0.2), r * 0.35, r * 0.3)

    pen = QPen(QColor("#0f172a"))
    pen.setWidthF(max(1.0, s * 0.018))
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QPointF(cx, cy), r, r)
    p.drawEllipse(QPointF(cx, cy), r * 0.55, r)
    p.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))

    accent = QColor("#5eead4")
    white = QColor("#f8fafc")
    arm = s * 0.12
    gap = r * 0.55
    for color, width in ((accent, s * 0.035), (white, s * 0.02)):
        pen = QPen(color)
        pen.setWidthF(max(1.0, width))
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(cx, cy - r - arm), QPointF(cx, cy - gap))
        p.drawLine(QPointF(cx, cy + gap), QPointF(cx, cy + r + arm))
        p.drawLine(QPointF(cx - r - arm, cy), QPointF(cx - gap, cy))
        p.drawLine(QPointF(cx + gap, cy), QPointF(cx + r + arm, cy))

    bracket = s * 0.08
    inset = s * 0.18
    pen = QPen(white)
    pen.setWidthF(max(1.2, s * 0.03))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    # four corner brackets
    for ox, oy, sx, sy in (
        (inset, inset, 1, 1),
        (s - inset, inset, -1, 1),
        (inset, s - inset, 1, -1),
        (s - inset, s - inset, -1, -1),
    ):
        p.drawLine(QPointF(ox, oy), QPointF(ox + sx * bracket, oy))
        p.drawLine(QPointF(ox, oy), QPointF(ox, oy + sy * bracket))

    # location pin
    pin_x, pin_y = cx + r * 0.15, cy - r * 0.05
    pin_r = s * 0.055
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#f59e0b"))
    p.drawEllipse(QPointF(pin_x, pin_y - pin_r * 0.35), pin_r, pin_r)
    p.drawPolygon(
        QPolygonF(
            [
                QPointF(pin_x - pin_r * 0.7, pin_y),
                QPointF(pin_x + pin_r * 0.7, pin_y),
                QPointF(pin_x, pin_y + pin_r * 1.35),
            ]
        )
    )
    p.setBrush(QColor("#0f172a"))
    p.drawEllipse(QPointF(pin_x, pin_y - pin_r * 0.35), pin_r * 0.35, pin_r * 0.35)

    p.end()
    return pm


def get_app_icon() -> QIcon:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    icon = QIcon()
    if _ICON_PATH.is_file():
        icon = QIcon(str(_ICON_PATH))
    if icon.isNull():
        icon = QIcon()
        for size in (16, 24, 32, 48, 64, 128, 256):
            icon.addPixmap(_paint_fallback(size))

    _CACHE = icon
    return icon


def apply_app_icon(app: QApplication, window=None) -> QIcon:
    """Set application (+ optional window) icon for the taskbar / title bar."""
    icon = get_app_icon()
    app.setWindowIcon(icon)
    if window is not None:
        window.setWindowIcon(icon)
    return icon
