"""Painted flat toolbar icons — no external asset dependency."""

from __future__ import annotations

from typing import Dict, List, Tuple

from shared.qt_compat import QColor, QPainter, QPen, QPixmap, QPointF, Qt

try:
    from shared.qt_compat import QIcon, QPolygonF
except ImportError:  # pragma: no cover
    from PySide6.QtGui import QIcon, QPolygonF  # type: ignore

_ACCENT = QColor("#14b8a6")
_FG = QColor("#e2e8f0")
_MUTED = QColor("#94a3b8")
_WARN = QColor("#f59e0b")
_DANGER = QColor("#ef4444")
_OK = QColor("#22c55e")
_BLUE = QColor("#38bdf8")


def _pix(size: int = 22) -> Tuple[QPixmap, QPainter]:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    return pm, p


def _pen(color: QColor, width: float = 1.8) -> QPen:
    pen = QPen(color)
    pen.setWidthF(width)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


def _poly(pts: List[Tuple[float, float]]) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in pts])


def icon_new(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_FG, 1.6))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(4, 3, 12, 15, 1.5, 1.5)
    p.setPen(_pen(_ACCENT, 2.0))
    p.drawLine(11, 8, 11, 14)
    p.drawLine(8, 11, 14, 11)
    p.end()
    return QIcon(pm)


def icon_open(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_FG, 1.5))
    p.setBrush(QColor("#334155"))
    p.drawRoundedRect(3, 8, 16, 10, 1.5, 1.5)
    p.setBrush(_ACCENT)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(3, 5, 8, 4, 1, 1)
    p.end()
    return QIcon(pm)


def icon_add_video(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_FG, 1.5))
    p.setBrush(QColor("#1e293b"))
    p.drawRoundedRect(2, 5, 12, 12, 2, 2)
    p.setBrush(_ACCENT)
    p.setPen(Qt.NoPen)
    p.drawPolygon(_poly([(7, 8), (7, 14), (13, 11)]))
    p.setPen(_pen(_ACCENT, 2.0))
    p.drawLine(17, 8, 17, 14)
    p.drawLine(14, 11, 20, 11)
    p.end()
    return QIcon(pm)


def icon_play(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_OK)
    p.drawPolygon(_poly([(7, 4), (7, 18), (18, 11)]))
    p.end()
    return QIcon(pm)


def icon_pause(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_WARN)
    p.drawRoundedRect(6, 4, 4, 14, 1, 1)
    p.drawRoundedRect(12, 4, 4, 14, 1, 1)
    p.end()
    return QIcon(pm)


def icon_stop(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_DANGER)
    p.drawRoundedRect(5, 5, 12, 12, 2, 2)
    p.end()
    return QIcon(pm)


def icon_prev(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_FG)
    p.drawRoundedRect(5, 5, 3, 12, 0.5, 0.5)
    p.drawPolygon(_poly([(17, 5), (17, 17), (8, 11)]))
    p.end()
    return QIcon(pm)


def icon_next(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_FG)
    p.drawPolygon(_poly([(5, 5), (5, 17), (14, 11)]))
    p.drawRoundedRect(14, 5, 3, 12, 0.5, 0.5)
    p.end()
    return QIcon(pm)


def icon_rect(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_ACCENT, 1.8))
    p.setBrush(QColor(20, 184, 166, 40))
    p.drawRect(4, 5, 14, 12)
    p.end()
    return QIcon(pm)


def icon_poly(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_BLUE, 1.8))
    p.setBrush(QColor(56, 189, 248, 40))
    p.drawPolygon(_poly([(4, 16), (7, 5), (16, 4), (18, 14), (11, 18)]))
    p.end()
    return QIcon(pm)


def icon_confirm(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_OK, 2.2))
    p.drawLine(5, 11, 9, 16)
    p.drawLine(9, 16, 17, 6)
    p.end()
    return QIcon(pm)


def icon_reset(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_MUTED, 1.8))
    p.setBrush(Qt.NoBrush)
    p.drawArc(4, 4, 14, 14, 40 * 16, 280 * 16)
    p.setPen(Qt.NoPen)
    p.setBrush(_MUTED)
    p.drawPolygon(_poly([(15, 3), (19, 8), (13, 9)]))
    p.end()
    return QIcon(pm)


def icon_run(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_ACCENT)
    p.drawEllipse(3, 3, 16, 16)
    p.setBrush(QColor("#0f172a"))
    p.drawPolygon(_poly([(9, 7), (9, 15), (16, 11)]))
    p.end()
    return QIcon(pm)


def icon_stop_proc(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_DANGER)
    p.drawEllipse(3, 3, 16, 16)
    p.setBrush(QColor("#0f172a"))
    p.drawRoundedRect(7, 7, 8, 8, 1.5, 1.5)
    p.end()
    return QIcon(pm)


def icon_export(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_FG, 1.6))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(4, 10, 14, 8, 1.5, 1.5)
    p.setPen(_pen(_ACCENT, 2.0))
    p.drawLine(11, 3, 11, 12)
    p.drawLine(7, 7, 11, 3)
    p.drawLine(15, 7, 11, 3)
    p.end()
    return QIcon(pm)


def icon_satellite(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_ACCENT)
    p.drawEllipse(8, 8, 6, 6)
    p.setPen(_pen(_BLUE, 1.5))
    p.setBrush(QColor(56, 189, 248, 50))
    p.drawEllipse(4, 4, 14, 14)
    p.setPen(_pen(_MUTED, 1.4))
    p.drawLine(3, 6, 7, 9)
    p.drawLine(19, 6, 15, 9)
    p.drawLine(3, 16, 7, 13)
    p.drawLine(19, 16, 15, 13)
    p.end()
    return QIcon(pm)


def icon_landmarks(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_WARN)
    p.drawPolygon(_poly([(11, 3), (18, 18), (4, 18)]))
    p.setBrush(QColor("#0f172a"))
    p.drawEllipse(9, 12, 4, 4)
    p.end()
    return QIcon(pm)


def icon_path(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_ACCENT, 2.0))
    p.setBrush(Qt.NoBrush)
    p.drawPolyline(_poly([(3, 16), (7, 8), (12, 12), (16, 5), (19, 9)]))
    p.setPen(Qt.NoPen)
    p.setBrush(_BLUE)
    for x, y in ((3, 16), (12, 12), (19, 9)):
        p.drawEllipse(QPointF(x, y), 2.2, 2.2)
    p.end()
    return QIcon(pm)


def icon_pipeline(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    for i, c in enumerate((_BLUE, _ACCENT, _WARN, _OK)):
        x = 3 + i * 4.5
        p.setBrush(c)
        p.drawRoundedRect(int(x), 7, 3, 8, 1, 1)
        if i < 3:
            p.setBrush(_MUTED)
            p.drawRect(int(x) + 3, 10, 2, 2)
    p.end()
    return QIcon(pm)


def icon_dataset(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_BLUE)
    p.drawRoundedRect(4, 4, 14, 4, 1, 1)
    p.setBrush(_ACCENT)
    p.drawRoundedRect(4, 9, 14, 4, 1, 1)
    p.setBrush(_MUTED)
    p.drawRoundedRect(4, 14, 14, 4, 1, 1)
    p.end()
    return QIcon(pm)


def icon_train(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_ACCENT, 1.6))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(4, 4, 14, 14)
    p.setPen(_pen(_FG, 1.4))
    p.drawArc(7, 7, 8, 8, 30 * 16, 200 * 16)
    p.setPen(Qt.NoPen)
    p.setBrush(_WARN)
    p.drawEllipse(9, 9, 4, 4)
    p.end()
    return QIcon(pm)


def icon_onnx(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_FG, 1.5))
    p.setBrush(QColor("#1e293b"))
    p.drawRoundedRect(3, 4, 16, 14, 2, 2)
    p.setPen(_pen(_ACCENT, 1.8))
    p.drawLine(7, 8, 7, 14)
    p.drawLine(7, 8, 11, 14)
    p.drawLine(11, 14, 11, 8)
    p.drawLine(14, 8, 14, 14)
    p.end()
    return QIcon(pm)


def icon_package(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_FG, 1.5))
    p.setBrush(QColor("#334155"))
    p.drawPolygon(_poly([(11, 3), (19, 7), (19, 15), (11, 19), (3, 15), (3, 7)]))
    p.setPen(_pen(_ACCENT, 1.4))
    p.drawLine(11, 3, 11, 11)
    p.drawLine(3, 7, 11, 11)
    p.drawLine(19, 7, 11, 11)
    p.end()
    return QIcon(pm)


def icon_validate(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_MUTED, 1.5))
    p.setBrush(QColor("#1e293b"))
    p.drawRoundedRect(5, 3, 12, 16, 1.5, 1.5)
    p.setPen(_pen(_OK, 2.0))
    p.drawLine(8, 11, 10, 14)
    p.drawLine(10, 14, 15, 8)
    p.end()
    return QIcon(pm)


def icon_report(size: int = 22) -> QIcon:
    """Bar chart — training report."""
    pm, p = _pix(size)
    p.setPen(_pen(_MUTED, 1.4))
    p.drawLine(4, 18, 18, 18)
    p.drawLine(4, 4, 4, 18)
    p.setPen(Qt.NoPen)
    p.setBrush(_ACCENT)
    p.drawRect(6, 12, 3, 6)
    p.setBrush(_OK)
    p.drawRect(11, 7, 3, 11)
    p.setBrush(_WARN)
    p.drawRect(16, 9, 3, 9)
    p.end()
    return QIcon(pm)


def icon_test(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_BLUE, 1.6))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(4, 4, 10, 10)
    p.setPen(_pen(_ACCENT, 2.0))
    p.drawLine(12, 12, 18, 18)
    p.end()
    return QIcon(pm)


def icon_settings(size: int = 22) -> QIcon:
    import math

    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    p.setBrush(_MUTED)
    cx, cy, r = 11.0, 11.0, 7.0
    for i in range(8):
        a = i * math.pi / 4
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        p.drawEllipse(QPointF(x, y), 2.0, 2.0)
    p.setBrush(QColor("#0f172a"))
    p.drawEllipse(7, 7, 8, 8)
    p.setBrush(_ACCENT)
    p.drawEllipse(9, 9, 4, 4)
    p.end()
    return QIcon(pm)


def icon_save(size: int = 22) -> QIcon:
    pm, p = _pix(size)
    p.setPen(_pen(_FG, 1.5))
    p.setBrush(QColor("#334155"))
    p.drawRoundedRect(3, 3, 16, 16, 2, 2)
    p.setBrush(_ACCENT)
    p.setPen(Qt.NoPen)
    p.drawRect(6, 3, 10, 7)
    p.setBrush(QColor("#0f172a"))
    p.drawRect(7, 12, 8, 5)
    p.end()
    return QIcon(pm)


def icon_usb(size: int = 22) -> QIcon:
    """USB flash stick — bright so it reads on the dark toolbar."""
    pm, p = _pix(size)
    p.setPen(Qt.NoPen)
    # metal connector
    p.setBrush(_MUTED)
    p.drawRoundedRect(8, 2, 6, 7, 1.0, 1.0)
    p.setBrush(QColor("#0f172a"))
    p.drawRect(9, 3, 1.5, 4)
    p.drawRect(12.5, 3, 1.5, 4)
    # body
    p.setBrush(_ACCENT)
    p.drawRoundedRect(5, 8, 12, 12, 2.0, 2.0)
    p.setBrush(_FG)
    p.drawEllipse(QPointF(8.5, 14.5), 1.6, 1.6)
    p.drawEllipse(QPointF(13.5, 14.5), 1.6, 1.6)
    p.end()
    return QIcon(pm)


_CACHE: Dict[str, QIcon] = {}

_MAKERS = {
    "new": icon_new,
    "open": icon_open,
    "add_video": icon_add_video,
    "play": icon_play,
    "pause": icon_pause,
    "stop": icon_stop,
    "prev": icon_prev,
    "next": icon_next,
    "rect": icon_rect,
    "poly": icon_poly,
    "confirm": icon_confirm,
    "reset": icon_reset,
    "run": icon_run,
    "stop_proc": icon_stop_proc,
    "export": icon_export,
    "satellite": icon_satellite,
    "landmarks": icon_landmarks,
    "path": icon_path,
    "pipeline": icon_pipeline,
    "dataset": icon_dataset,
    "train": icon_train,
    "onnx": icon_onnx,
    "package": icon_package,
    "validate": icon_validate,
    "report": icon_report,
    "test": icon_test,
    "settings": icon_settings,
    "save": icon_save,
    "usb": icon_usb,
}


def get_icon(name: str, size: int = 22) -> QIcon:
    key = f"{name}:{size}"
    if key in _CACHE:
        return _CACHE[key]
    fn = _MAKERS.get(name)
    icon = fn(size) if fn else QIcon()
    _CACHE[key] = icon
    return icon
