"""Bottom loading strip + short log for interactive map tile fetches."""

from __future__ import annotations

from shared.qt_compat import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    Qt,
    QVBoxLayout,
    QWidget,
)


class MapLoadStatusBar(QWidget):
    """Compact progress + scrolling log under the map (all SatelliteMapPanel hosts)."""

    _MAX_LOG_LINES = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mapLoadStatusBar")
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 4)
        root.setSpacing(2)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._busy = QLabel("")
        self._busy.setFixedWidth(14)
        self._busy.setAlignment(Qt.AlignCenter)
        self._busy.setStyleSheet("color:#6cf; font-weight:bold;")
        self._msg = QLabel("Map idle")
        self._msg.setStyleSheet("color:#ccc;")
        self._msg.setWordWrap(False)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%v / %m")
        self._bar.setMaximumHeight(16)
        self._bar.setMinimumWidth(120)
        self._bar.setMaximumWidth(220)
        row.addWidget(self._busy)
        row.addWidget(self._msg, stretch=1)
        row.addWidget(self._bar)
        root.addLayout(row)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(self._MAX_LOG_LINES)
        self._log.setMaximumHeight(56)
        self._log.setPlaceholderText("Map load log…")
        self._log.setStyleSheet(
            "QPlainTextEdit{background:#1a1a1a;color:#9a9a9a;border:1px solid #333;"
            "font-size:11px;}"
        )
        root.addWidget(self._log)

        self.setStyleSheet(
            "#mapLoadStatusBar{background:#151515;border-top:1px solid #333;}"
        )
        self._set_idle()

    def _set_idle(self) -> None:
        self._busy.setText("")
        self._msg.setText("Map idle")
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._bar.setFormat("—")

    def on_loading(self, active: bool, done: int, total: int, message: str) -> None:
        """Update strip from SatelliteMapWidget.loading_changed."""
        msg = (message or "").strip()
        if not active:
            self._busy.setText("")
            self._msg.setText(msg or "Map ready")
            if total > 0:
                self._bar.setRange(0, total)
                self._bar.setValue(total)
                self._bar.setFormat("%v / %m")
            else:
                self._bar.setRange(0, 1)
                self._bar.setValue(1)
                self._bar.setFormat("done")
            if msg:
                self._append_log(msg)
            return
        self._busy.setText("●")
        self._msg.setText(msg or "Loading map…")
        if total <= 0:
            self._bar.setRange(0, 0)  # indeterminate
            self._bar.setFormat("…")
        else:
            self._bar.setRange(0, total)
            self._bar.setValue(max(0, min(done, total)))
            self._bar.setFormat("%v / %m")
        # Log start + milestones (every tile when few; ~quarter steps when many)
        if not msg:
            return
        step = max(2, total // 4) if total >= 8 else 1
        if done <= 0 or done >= total or done % step == 0:
            self._append_log(msg)

    def _append_log(self, line: str) -> None:
        self._log.appendPlainText(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
