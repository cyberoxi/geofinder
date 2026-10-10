"""Offline Mode menu for DesktopMainWindow: maps manager, import, on/off toggle + status indicator."""

from __future__ import annotations

from pathlib import Path

from shared.icons import get_icon
from shared.qt_compat import QAction, QFileDialog, QLabel, QMessageBox, qt_exec
from desktop_app.satellite import offline


class OfflineActionsMixin:
    """Expects DesktopMainWindow: menuBar, statusBar, log."""

    _offline_toggle: QAction = None  # type: ignore[assignment]
    _offline_badge: QLabel = None  # type: ignore[assignment]

    def _build_offline_menu(self) -> None:
        menu = self.menuBar().addMenu("&Offline Mode")
        self._offline_toggle = QAction("Offline mode (use offline maps only)", self)
        self._offline_toggle.setCheckable(True)
        self._offline_toggle.setChecked(offline.offline_enabled())
        self._offline_toggle.toggled.connect(self._set_offline_mode)
        menu.addAction(self._offline_toggle)
        menu.addSeparator()
        mgr = QAction(get_icon("satellite"), "Offline Maps (download regions)…", self)
        mgr.triggered.connect(self.open_offline_maps)
        menu.addAction(mgr)
        imp = QAction(get_icon("open"), "Import Offline Map…", self)
        imp.triggered.connect(self.import_offline_map)
        menu.addAction(imp)

        self._offline_badge = QLabel()
        self.statusBar().addPermanentWidget(self._offline_badge)
        self._sync_offline_ui()

    def _sync_offline_ui(self) -> None:
        on = offline.offline_enabled()
        if self._offline_toggle is not None and self._offline_toggle.isChecked() != on:
            self._offline_toggle.blockSignals(True)
            self._offline_toggle.setChecked(on)
            self._offline_toggle.blockSignals(False)
        if self._offline_badge is not None:
            n = len(offline.packages())
            self._offline_badge.setText(f"● OFFLINE ({n} map{'s' if n != 1 else ''})" if on else "")
            self._offline_badge.setStyleSheet("color:#4ade80;font-weight:bold;padding:0 8px;")

    def _set_offline_mode(self, on: bool) -> None:
        if on and not offline.packages():
            QMessageBox.information(
                self, "Offline Mode",
                "No offline maps imported yet. Use Offline Mode → Offline Maps to download a region, "
                "or Import Offline Map.",
            )
        offline.set_offline_enabled(on)
        self.log(f"Offline mode {'ON — imagery from offline maps only' if on else 'OFF'}", "head")
        self._sync_offline_ui()

    def open_offline_maps(self) -> None:
        from desktop_app.gui.offline_dialog import OfflineMapsDialog

        qt_exec(OfflineMapsDialog(self))
        self._sync_offline_ui()

    def import_offline_map(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import offline map", str(Path.cwd()), f"Offline map (*{offline.PACKAGE_EXT});;All files (*)"
        )
        if not paths:
            return
        copy = QMessageBox.question(
            self, "Import", "Copy the file(s) into data/offline_maps?\n\nNo = use the file where it is."
        ) == QMessageBox.Yes
        for p in paths:
            try:
                pkg = offline.import_package(p, copy_into_library=copy)
                self.log(f"Offline map imported: {pkg.summary()}", "ok")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Import", f"{p}\n\n{exc}")
        if not offline.offline_enabled() and QMessageBox.question(
            self, "Offline Mode", "Turn offline mode on now?"
        ) == QMessageBox.Yes:
            self._set_offline_mode(True)
        self._sync_offline_ui()
