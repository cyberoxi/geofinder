"""Dialog: Offline Mode — download map regions at several zooms, import / save packages, toggle offline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from shared.qt_compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    Qt,
    QTimer,
    QVBoxLayout,
    QWidget,
    qt_exec,
)
from desktop_app.gui.atlas_build_dialog import restore_map_view, save_map_view
from desktop_app.gui.workers import StageWorker
from desktop_app.satellite import offline
from desktop_app.satellite.map_widget import SatelliteMapPanel
from desktop_app.satellite.tiles import LAYERS, count_tiles, layer_max_zoom, mosaic_layers

FILE_FILTER = f"Offline map (*{offline.PACKAGE_EXT});;All files (*)"
MAX_TILES = 400_000
CONFIRM_TILES = 40_000


def _poly_fill_ratio(poly) -> float:
    """Polygon area / bbox area (lon/lat plane) — cheap tile-count estimate."""
    lons = [p[0] for p in poly]
    lats = [p[1] for p in poly]
    box = (max(lons) - min(lons)) * (max(lats) - min(lats))
    area = abs(sum(poly[i][0] * poly[i - 1][1] - poly[i - 1][0] * poly[i][1] for i in range(len(poly)))) / 2.0
    return min(1.0, area / box) if box > 0 else 1.0


class OfflineMapsDialog(QDialog):
    """Offline map manager. Closing keeps the library; offline mode applies app-wide."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Offline Mode / حالت آفلاین — Offline Maps")
        self.setMinimumSize(980, 640)
        self.resize(1260, 780)
        self._worker: Optional[StageWorker] = None
        self._stop = False

        root = QHBoxLayout(self)
        self.map_panel = SatelliteMapPanel()
        root.addWidget(self.map_panel, stretch=3)

        side = QWidget()
        side.setMinimumWidth(380)
        sv = QVBoxLayout(side)

        self.chk_offline = QCheckBox("Offline mode enabled / حالت آفلاین فعال")
        self.chk_offline.setStyleSheet("font-weight:bold;font-size:14px;")
        self.chk_offline.setChecked(offline.offline_enabled())
        self.chk_offline.toggled.connect(self._toggle_offline)
        sv.addWidget(self.chk_offline)
        self.mode_label = QLabel()
        self.mode_label.setWordWrap(True)
        sv.addWidget(self.mode_label)
        self._offline_jump_i = 0
        self.btn_goto_offline = QPushButton("Go to offline area / برو به منطقه آفلاین")
        self.btn_goto_offline.setMinimumHeight(32)
        self.btn_goto_offline.setToolTip(
            "Jump the map to a downloaded region. Click again to cycle through areas."
        )
        self.btn_goto_offline.clicked.connect(self._goto_offline_area)
        sv.addWidget(self.btn_goto_offline)

        # ---- download
        g_dl = QGroupBox("Download region / دانلود منطقه")
        dl_v = QVBoxLayout(g_dl)
        form = QFormLayout()
        self.pkg_name = QLineEdit("offline_region")
        form.addRow("Package name", self.pkg_name)
        self.layer = QComboBox()
        for key, title in mosaic_layers().items():
            self.layer.addItem(title, key)
        self.layer.addItem(LAYERS["map"].get("title", "Street map"), "map")
        form.addRow("Imagery", self.layer)
        zrow = QHBoxLayout()
        self.zmin = QSpinBox()
        self.zmin.setRange(1, 20)
        self.zmin.setValue(12)
        self.zmax = QSpinBox()
        self.zmax.setRange(1, 20)
        self.zmax.setValue(17)
        zrow.addWidget(QLabel("from"))
        zrow.addWidget(self.zmin)
        zrow.addWidget(QLabel("to"))
        zrow.addWidget(self.zmax)
        zrow.addStretch(1)
        form.addRow("Zoom levels", zrow)
        self.estimate_label = QLabel("Select a region on the map (Select Region → Polygon / Box).")
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setStyleSheet("color:#ffcc66;")
        form.addRow(self.estimate_label)
        dl_v.addLayout(form)

        self.b_new = QPushButton("Download as new package…")
        self.b_add = QPushButton("Add to selected package")
        self.b_add.setToolTip("Append this region to the package selected in the list below")
        for b in (self.b_new, self.b_add):
            b.setMinimumHeight(32)
            dl_v.addWidget(b)

        prow = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.b_stop = QPushButton("Stop")
        self.b_stop.setEnabled(False)
        prow.addWidget(self.progress, stretch=1)
        prow.addWidget(self.b_stop)
        dl_v.addLayout(prow)
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        dl_v.addWidget(self.progress_label)
        sv.addWidget(g_dl)

        # ---- library
        g_lib = QGroupBox("Offline map packages / بسته‌های نقشه آفلاین")
        lv = QVBoxLayout(g_lib)
        self.pkg_list = QListWidget()
        self.pkg_list.itemDoubleClicked.connect(lambda _i: self._show_selected())
        lv.addWidget(self.pkg_list, stretch=1)
        lrow = QHBoxLayout()
        self.b_import = QPushButton("Import…")
        self.b_export = QPushButton("Save as…")
        self.b_show = QPushButton("Show on map")
        self.b_remove = QPushButton("Remove")
        for b in (self.b_import, self.b_export, self.b_show, self.b_remove):
            lrow.addWidget(b)
        lv.addLayout(lrow)
        sv.addWidget(g_lib, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        sv.addWidget(buttons)
        root.addWidget(side, stretch=1)

        self.b_new.clicked.connect(lambda: self._start_download(new=True))
        self.b_add.clicked.connect(lambda: self._start_download(new=False))
        self.b_stop.clicked.connect(self._request_stop)
        self.b_import.clicked.connect(self._import)
        self.b_export.clicked.connect(self._export)
        self.b_show.clicked.connect(self._show_selected)
        self.b_remove.clicked.connect(self._remove)
        self.pkg_list.currentRowChanged.connect(lambda _r: self._sync_buttons())

        self._est_timer = QTimer(self)
        self._est_timer.setSingleShot(True)
        self._est_timer.timeout.connect(self._update_estimate)
        self.map_panel.map.selection_changed.connect(lambda: self._est_timer.start(250))
        for sb in (self.zmin, self.zmax):
            sb.valueChanged.connect(lambda _v: self._est_timer.start(250))
        self.layer.currentIndexChanged.connect(lambda _i: self._est_timer.start(250))

        restore_map_view(self.map_panel)
        self._refresh_list()
        self._sync_mode_label()

    # ------------------------------------------------------------ offline flag
    def _toggle_offline(self, on: bool) -> None:
        if on and not offline.packages():
            QMessageBox.information(
                self, "Offline Mode",
                "No offline maps yet — download or import a package, otherwise imagery will be empty.",
            )
        offline.set_offline_enabled(on)
        self._sync_mode_label()
        m = self.map_panel.map
        m._failed.clear()
        m._schedule_render()

    def _sync_mode_label(self) -> None:
        if offline.offline_enabled():
            self.mode_label.setText(
                "<span style='color:#4ade80'>● OFFLINE — training-video imagery comes only from the packages "
                "below.</span><br><span style='color:#aaa'>Map shows offline tiles only; turn off to browse new "
                "areas. Downloads below still use the internet.</span>"
            )
        else:
            self.mode_label.setText(
                "<span style='color:#aaa'>Online — imagery is downloaded from the internet. "
                "Enable offline mode to use only the packages below.</span>"
            )

    # ------------------------------------------------------------ library list
    def _refresh_list(self, select: Optional[Path] = None) -> None:
        self.pkg_list.clear()
        for pkg in offline.packages():
            item = QListWidgetItem(pkg.summary())
            item.setData(Qt.UserRole, str(pkg.path))
            item.setToolTip(str(pkg.path))
            self.pkg_list.addItem(item)
            if select is not None and pkg.path == Path(select).resolve():
                self.pkg_list.setCurrentItem(item)
        if self.pkg_list.currentRow() < 0 and self.pkg_list.count():
            self.pkg_list.setCurrentRow(0)
        self._sync_buttons()

    def _selected_pkg(self) -> Optional[offline.OfflinePackage]:
        item = self.pkg_list.currentItem()
        return offline.find_package(item.data(Qt.UserRole)) if item else None

    def _sync_buttons(self) -> None:
        busy = self._worker is not None and self._worker.isRunning()
        has = self._selected_pkg() is not None
        self.b_new.setEnabled(not busy)
        self.b_add.setEnabled(not busy and has)
        self.b_stop.setEnabled(busy)
        for b in (self.b_export, self.b_show, self.b_remove):
            b.setEnabled(has and not busy)
        self.b_import.setEnabled(not busy)

    def _import(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Import offline map", str(Path.cwd()), FILE_FILTER)
        if not paths:
            return
        copy = QMessageBox.question(
            self, "Import",
            "Copy the file(s) into the app's offline maps folder (data/offline_maps)?\n\n"
            "Yes = safe if the source is a USB drive · No = use the file where it is.",
        ) == QMessageBox.Yes
        last = None
        for p in paths:
            try:
                last = offline.import_package(p, copy_into_library=copy).path
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Import", f"{p}\n\n{exc}")
        self._refresh_list(select=last)
        self.map_panel.refresh_overview_offline()

    def _export(self) -> None:
        pkg = self._selected_pkg()
        if pkg is None:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save offline map as", str(Path.cwd() / (pkg.path.stem + offline.PACKAGE_EXT)), FILE_FILTER
        )
        if not dest:
            return
        try:
            if Path(dest).resolve() != pkg.path:
                offline.export_package(pkg, dest)
            QMessageBox.information(self, "Save", f"Saved:\n{dest}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save", str(exc))

    def _goto_offline_area(self) -> None:
        """Fit the map to the next downloaded offline region (cycles if several)."""
        boxes = offline.high_zoom_region_boxes() or offline.region_boxes(imagery_only=False, min_zoom=0)
        if not boxes:
            QMessageBox.information(
                self, "Offline",
                "No offline map regions found yet. Download or import a package first.",
            )
            return
        i = self._offline_jump_i % len(boxes)
        self._offline_jump_i = i + 1
        south, west, north, east = boxes[i]
        self.map_panel.map.fit_bbox(south, west, north, east)
        self.map_panel.map.set_reference_polygon(
            [(west, south), (east, south), (east, north), (west, north)]
        )
        self.map_panel.refresh_overview_offline()
        n = len(boxes)
        self.map_panel.status.setText(
            f"Offline area {i + 1}/{n}: {south:.5f}…{north:.5f} N, {west:.5f}…{east:.5f} E"
        )

    def _show_selected(self) -> None:
        pkg = self._selected_pkg()
        if pkg is None:
            return
        regions = pkg.regions
        if not regions:
            return
        lons = [p[0] for r in regions for p in r]
        lats = [p[1] for r in regions for p in r]
        s, w, n, e = min(lats), min(lons), max(lats), max(lons)
        self.map_panel.map.fit_bbox(s, w, n, e)
        ref = regions[0] if len(regions) == 1 else [(w, s), (e, s), (e, n), (w, n)]
        self.map_panel.map.set_reference_polygon(ref)

    def _remove(self) -> None:
        pkg = self._selected_pkg()
        if pkg is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Remove offline map")
        box.setText(f"Remove '{pkg.name}' from the offline library?")
        b_keep = box.addButton("Remove (keep file)", QMessageBox.AcceptRole)
        b_del = box.addButton("Remove and delete file", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        qt_exec(box)
        clicked = box.clickedButton()
        if clicked not in (b_keep, b_del):
            return
        offline.remove_package(pkg, delete_file=clicked is b_del)
        self.map_panel.map.set_reference_polygon(None)
        self._refresh_list()
        self.map_panel.refresh_overview_offline()

    # ------------------------------------------------------------ download
    def _zooms(self):
        layer = str(self.layer.currentData())
        zmax = min(self.zmax.value(), layer_max_zoom(layer))
        return min(self.zmin.value(), zmax), zmax

    def _estimate(self, poly) -> int:
        lons = [p[0] for p in poly]
        lats = [p[1] for p in poly]
        fill = _poly_fill_ratio(poly)
        zmin, zmax = self._zooms()
        return int(sum(count_tiles(min(lats), min(lons), max(lats), max(lons), z) * fill for z in range(zmin, zmax + 1)))

    def _update_estimate(self) -> None:
        poly = self.map_panel.map.selection_polygon()
        if not poly or len(poly) < 3:
            self.estimate_label.setText("Select a region on the map (Select Region → Polygon / Box).")
            return
        tiles = self._estimate(poly)
        zmin, zmax = self._zooms()
        mb = tiles * offline.AVG_TILE_KB / 1024.0
        warn = "  — too large, lower max zoom or area" if tiles > MAX_TILES else ""
        self.estimate_label.setText(f"z{zmin}–{zmax}: ≈ {tiles:,} tiles · ≈ {mb:,.0f} MB{warn}")

    def _start_download(self, new: bool) -> None:
        poly = self.map_panel.map.selection_polygon()
        if not poly or len(poly) < 3:
            QMessageBox.warning(self, "Offline Mode", "Select a region on the map first (Select Region).")
            return
        layer = str(self.layer.currentData())
        zmin, zmax = self._zooms()
        tiles = self._estimate(poly)
        if tiles > MAX_TILES:
            QMessageBox.warning(self, "Offline Mode", f"≈{tiles:,} tiles is too many. Lower max zoom or the area.")
            return
        if tiles > CONFIRM_TILES and QMessageBox.question(
            self, "Offline Mode", f"Download ≈{tiles:,} tiles (≈{tiles * offline.AVG_TILE_KB / 1024:,.0f} MB)?"
        ) != QMessageBox.Yes:
            return
        if new:
            name = re.sub(r"[^\w.-]+", "_", self.pkg_name.text().strip(), flags=re.UNICODE) or "offline_region"
            offline.OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
            path, _ = QFileDialog.getSaveFileName(
                self, "Save offline map package", str((offline.OFFLINE_DIR / name).resolve()) + offline.PACKAGE_EXT,
                FILE_FILTER,
            )
            if not path:
                return
            existing = offline.find_package(path)
            if Path(path).exists() and existing is None:
                QMessageBox.warning(self, "Offline Mode", "File exists — choose another name or import it first.")
                return
        else:
            pkg = self._selected_pkg()
            if pkg is None:
                return
            if pkg.layer != layer:
                QMessageBox.warning(
                    self, "Offline Mode",
                    f"'{pkg.name}' holds '{pkg.layer}' imagery; pick the same imagery or download a new package.",
                )
                return
            path = str(pkg.path)
        save_map_view(self.map_panel)
        self._stop = False
        region = list(poly)

        def job(progress, _frame):
            return offline.download_region(
                path, region, zmin, zmax, layer=layer, progress_cb=progress, should_stop=lambda: self._stop
            )

        self._worker = StageWorker(job)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._sync_buttons)
        self.progress.setValue(0)
        self.progress_label.setText("Starting…")
        self._worker.start()
        self._sync_buttons()

    def _request_stop(self) -> None:
        self._stop = True
        self.progress_label.setText("Stopping… (downloaded tiles are kept; download again to resume)")

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        self.progress.setValue(int(100 * done / total) if total else 0)
        self.progress_label.setText(msg)

    def _on_done(self, pkg: object) -> None:
        self.progress.setValue(100)
        stopped = " (stopped — run again to resume)" if self._stop else ""
        self.progress_label.setText(f"Done{stopped}: {pkg.summary()}")  # type: ignore[attr-defined]
        self._refresh_list(select=pkg.path)  # type: ignore[attr-defined]
        self._show_selected()
        self.map_panel.refresh_overview_offline()

    def _on_failed(self, err: str) -> None:
        self.progress_label.setText(f"Failed: {err}")
        self._sync_buttons()
        QMessageBox.critical(self, "Offline Mode", err)

    def reject(self) -> None:  # noqa: D401 — also used by the Close button / Esc
        if self._worker is not None and self._worker.isRunning():
            if QMessageBox.question(self, "Offline Mode", "A download is running. Stop it and close?") != QMessageBox.Yes:
                return
            self._stop = True
            self._worker.wait(15000)
        save_map_view(self.map_panel)
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None and self._worker.isRunning():
            event.ignore()
            self.reject()
            return
        super().closeEvent(event)
