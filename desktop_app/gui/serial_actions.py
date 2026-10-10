"""Serial menu for DesktopMainWindow: port settings + status badge."""

from __future__ import annotations

from shared.icons import get_icon
from shared.qt_compat import QAction, QLabel


class SerialActionsMixin:
    """Expects DesktopMainWindow: menuBar, statusBar, log."""

    _serial_badge: QLabel = None  # type: ignore[assignment]

    def _build_serial_menu(self) -> None:
        menu = self.menuBar().addMenu("&Serial")
        act = QAction(get_icon("settings"), "Serial settings… / تنظیمات سریال", self)
        act.setToolTip("UART output of image-centre → target offset (dx, dy) during Live Test")
        act.triggered.connect(self.open_serial_settings)
        menu.addAction(act)

        self._serial_badge = QLabel()
        self.statusBar().addPermanentWidget(self._serial_badge)
        self._refresh_serial_status()

    def open_serial_settings(self) -> None:
        from desktop_app.gui.serial_dialog import edit_serial_settings

        cfg = edit_serial_settings(self)
        on = bool(cfg.get("enabled"))
        self.log(
            f"Serial {'ON' if on else 'OFF'}"
            + (f" — {cfg.get('port')} @ {cfg.get('baudrate')}" if on else ""),
            "head",
        )
        self._refresh_serial_status()

    def _refresh_serial_status(self) -> None:
        from desktop_app.gui.serial_dialog import load_serial_settings

        if self._serial_badge is None:
            return
        cfg = load_serial_settings()
        if cfg.get("enabled"):
            self._serial_badge.setText(
                f"● SERIAL {cfg.get('port')}@{cfg.get('baudrate')}"
            )
            self._serial_badge.setStyleSheet(
                "color:#38bdf8;font-weight:bold;padding:0 8px;"
            )
        else:
            self._serial_badge.setText("")
            self._serial_badge.setStyleSheet("")
