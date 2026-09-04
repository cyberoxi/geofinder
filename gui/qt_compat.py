"""Qt binding compatibility: prefer PySide6, fall back to PyQt5."""

from __future__ import annotations

QT_API = None

try:
    from PySide6.QtCore import (  # noqa: F401
        Qt,
        QThread,
        Signal,
        Slot,
        QTimer,
        QMutex,
        QMutexLocker,
        QSize,
        QPoint,
        QPointF,
    )
    from PySide6.QtGui import (  # noqa: F401
        QAction,
        QImage,
        QPixmap,
        QPainter,
        QPen,
        QColor,
        QMouseEvent,
        QKeySequence,
    )
    from PySide6.QtWidgets import (  # noqa: F401
        QApplication,
        QMainWindow,
        QWidget,
        QLabel,
        QVBoxLayout,
        QHBoxLayout,
        QSplitter,
        QToolBar,
        QStatusBar,
        QFileDialog,
        QMessageBox,
        QSlider,
        QSpinBox,
        QDoubleSpinBox,
        QComboBox,
        QCheckBox,
        QGroupBox,
        QFormLayout,
        QDialog,
        QDialogButtonBox,
        QTabWidget,
        QLineEdit,
        QPushButton,
        QSizePolicy,
    )

    QT_API = "PySide6"

    def qt_exec(obj):
        return obj.exec()

    def mouse_xy(event) -> tuple:
        p = event.position()
        return float(p.x()), float(p.y())

except ImportError:  # pragma: no cover
    from PyQt5.QtCore import (  # type: ignore  # noqa: F401
        Qt,
        QThread,
        QTimer,
        QMutex,
        QMutexLocker,
        QSize,
        QPoint,
        QPointF,
    )
    from PyQt5.QtCore import pyqtSignal as Signal  # type: ignore
    from PyQt5.QtCore import pyqtSlot as Slot  # type: ignore
    from PyQt5.QtGui import (  # type: ignore  # noqa: F401
        QImage,
        QPixmap,
        QPainter,
        QPen,
        QColor,
        QMouseEvent,
        QKeySequence,
    )
    from PyQt5.QtWidgets import (  # type: ignore  # noqa: F401
        QApplication,
        QMainWindow,
        QWidget,
        QLabel,
        QVBoxLayout,
        QHBoxLayout,
        QSplitter,
        QToolBar,
        QStatusBar,
        QFileDialog,
        QMessageBox,
        QSlider,
        QSpinBox,
        QDoubleSpinBox,
        QComboBox,
        QCheckBox,
        QGroupBox,
        QFormLayout,
        QDialog,
        QDialogButtonBox,
        QTabWidget,
        QLineEdit,
        QPushButton,
        QSizePolicy,
        QAction,
    )

    QT_API = "PyQt5"

    def qt_exec(obj):
        return obj.exec_()

    def mouse_xy(event) -> tuple:
        p = event.pos()
        return float(p.x()), float(p.y())
