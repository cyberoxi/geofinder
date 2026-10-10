"""Visual satellite compare gallery: same region, many sensors — pick scenes for training."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from shared.qt_compat import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPixmap,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
    Qt,
    qt_exec,
)
from desktop_app.gui.archive_cards import PreviewCard, numpy_to_pixmap
from desktop_app.gui.earthdata_auth import (
    EarthdataLoginDialog,
    default_date_range,
    load_credentials,
    save_credentials,
)
from desktop_app.gui.workers import StageWorker
from desktop_app.satellite.archives import SENSOR_TITLES, ArchiveScene, search_archives
from desktop_app.satellite.preview_fetch import COMPARE_BASEMAPS, load_gallery_previews

__all__ = [
    "ArchivePickerDialog",
    "EarthdataLoginDialog",
    "default_date_range",
    "load_credentials",
    "save_credentials",
]


class ArchivePickerDialog(QDialog):
    """One page: basemap + archive quicklooks for the region; check scenes for training."""

    def __init__(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        *,
        sensors: Sequence[str],
        start_date: str,
        end_date: str,
        max_per_sensor: int = 5,
        preselected: Optional[Sequence[ArchiveScene]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Compare satellites — same region")
        self.resize(1100, 720)
        self._bbox = (south, west, north, east)
        self._sensors = list(sensors)
        self._start = start_date
        self._end = end_date
        self._max = max_per_sensor
        self._scenes: List[ArchiveScene] = []
        self.selected: List[ArchiveScene] = list(preselected or [])
        self._cards: Dict[str, PreviewCard] = {}
        self._previews: Dict[str, object] = {}
        self._worker: Optional[StageWorker] = None

        root = QVBoxLayout(self)
        header = QLabel(
            f"Region  S={south:.4f}  W={west:.4f}  N={north:.4f}  E={east:.4f}\n"
            f"Sensors: {', '.join(SENSOR_TITLES.get(s, s) for s in self._sensors)}  ·  "
            f"{start_date} → {end_date}\n"
            "Basemaps = map tiles for this area. Archive cards = Earthdata/ASF quicklooks — "
            "check scenes to include in the training dataset."
        )
        header.setWordWrap(True)
        root.addWidget(header)

        btn_row = QHBoxLayout()
        self.btn_search = QPushButton("Search & load previews")
        self.btn_search.clicked.connect(self._run_search)
        self.btn_login = QPushButton("Earthdata Login…")
        self.btn_login.clicked.connect(self._edit_login)
        self.btn_all = QPushButton("Check all archives")
        self.btn_all.clicked.connect(lambda: self._set_all_checks(True))
        self.btn_none = QPushButton("Uncheck all")
        self.btn_none.clicked.connect(lambda: self._set_all_checks(False))
        for b in (self.btn_search, self.btn_login, self.btn_all, self.btn_none):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        split = QSplitter(Qt.Horizontal)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.gallery_host = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_host)
        self.gallery_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.gallery_host)
        split.addWidget(self.scroll)

        right = QWidget()
        rlay = QVBoxLayout(right)
        self.big_title = QLabel("Click a card to enlarge")
        self.big_title.setWordWrap(True)
        self.big_title.setStyleSheet("font-weight:600;")
        rlay.addWidget(self.big_title)
        self.big_preview = QLabel("Preview")
        self.big_preview.setMinimumSize(360, 360)
        self.big_preview.setAlignment(Qt.AlignCenter)
        self.big_preview.setStyleSheet("background:#141414;color:#888;border:1px solid #333;")
        rlay.addWidget(self.big_preview, stretch=1)
        self.big_meta = QLabel("")
        self.big_meta.setWordWrap(True)
        self.big_meta.setStyleSheet("color:#bbb;")
        rlay.addWidget(self.big_meta)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, stretch=1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Use checked archives")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        if self.selected:
            self._scenes = list(self.selected)
            self._rebuild_gallery()
            self._load_previews_async()
        else:
            self._run_search()

    def _edit_login(self) -> None:
        dlg = EarthdataLoginDialog(self)
        if qt_exec(dlg) == QDialog.Accepted:
            creds = dlg.credentials()
            if not creds.has_auth():
                QMessageBox.warning(self, "Earthdata Login", "Enter username+password or a token.")
                return
            save_credentials(creds)
            self.status.setText("Earthdata credentials saved.")

    def _run_search(self) -> None:
        self.status.setText("Searching CMR / ASF…")
        self.btn_search.setEnabled(False)
        self.progress.setValue(5)
        try:
            south, west, north, east = self._bbox
            self._scenes = search_archives(
                south, west, north, east,
                sensors=self._sensors,
                start_date=self._start,
                end_date=self._end,
                max_per_sensor=self._max,
            )
            self._rebuild_gallery()
            self.status.setText(f"Found {len(self._scenes)} archive scene(s). Loading previews…")
            self._load_previews_async()
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Search failed: {exc}")
            QMessageBox.warning(self, "Archive search", str(exc))
            self.btn_search.setEnabled(True)
            self.progress.setValue(0)

    def _clear_gallery(self) -> None:
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cards.clear()

    def _add_section(self, title: str, cards: List[PreviewCard]) -> None:
        if not cards:
            return
        lab = QLabel(title)
        lab.setStyleSheet("font-size:14px;font-weight:700;color:#ddd;margin-top:8px;")
        self.gallery_layout.addWidget(lab)
        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setSpacing(8)
        for i, card in enumerate(cards):
            grid.addWidget(card, i // 3, i % 3)
            self._cards[card.key] = card
        self.gallery_layout.addWidget(grid_host)

    def _rebuild_gallery(self) -> None:
        self._clear_gallery()
        selected_ids = {s.id for s in self.selected}
        base_cards: List[PreviewCard] = []
        for layer, title in COMPARE_BASEMAPS:
            card = PreviewCard(
                f"basemap:{layer}", title, "Map tiles · same region (compare only)", checkable=False
            )
            card._click_cb = self._on_card_clicked
            base_cards.append(card)
        self._add_section("Basemaps — همین منطقه", base_cards)

        by_sensor: Dict[str, List[ArchiveScene]] = {}
        for sc in self._scenes:
            by_sensor.setdefault(sc.sensor, []).append(sc)
        for sensor, scenes in by_sensor.items():
            cards: List[PreviewCard] = []
            for sc in scenes:
                cloud = f"cloud {sc.cloud_cover:.0f}%" if sc.cloud_cover is not None else "SAR"
                sub = f"{sc.date[:10]} · {cloud}"
                if sc.size_mb:
                    sub += f" · ~{sc.size_mb:.0f} MB"
                card = PreviewCard(
                    f"scene:{sc.id}",
                    sc.title,
                    sub,
                    checkable=True,
                    checked=(sc.id in selected_ids) or (not selected_ids),
                )
                card.scene = sc
                card._click_cb = self._on_card_clicked
                cards.append(card)
            self._add_section(f"Archive — {SENSOR_TITLES.get(sensor, sensor)}", cards)

        if not self._scenes:
            empty = QLabel("No archive scenes for this region / dates. Try a wider range.")
            empty.setStyleSheet("color:#c90;")
            self.gallery_layout.addWidget(empty)
        self.gallery_layout.addStretch(1)

    def _load_previews_async(self) -> None:
        south, west, north, east = self._bbox
        scenes = list(self._scenes)

        def job(progress, _frame):
            return load_gallery_previews(
                south, west, north, east, scenes, include_basemaps=True, progress_cb=progress
            )

        if self._worker and self._worker.isRunning():
            try:
                self._worker.finished_ok.disconnect(self._on_previews_ready)
                self._worker.failed.disconnect(self._on_preview_fail)
                self._worker.progress.disconnect(self._on_preview_progress)
            except (TypeError, RuntimeError):
                pass
        self._worker = StageWorker(job)
        self._worker.progress.connect(self._on_preview_progress)
        self._worker.finished_ok.connect(self._on_previews_ready)
        self._worker.failed.connect(self._on_preview_fail)
        self._worker.start()

    def _on_preview_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.progress.setValue(max(0, min(100, int(100 * done / total))))
        self.status.setText(message)

    def _on_previews_ready(self, mapping: object) -> None:
        self.btn_search.setEnabled(True)
        self.progress.setValue(100)
        if not isinstance(mapping, dict):
            return
        self._previews = mapping
        for key, card in self._cards.items():
            card.set_image(mapping.get(key))
        n_ok = sum(1 for k in self._cards if k in mapping)
        self.status.setText(
            f"Previews ready ({n_ok}/{len(self._cards)}). "
            "Check archive cards, then Use checked archives."
        )
        for key in self._cards:
            if key.startswith("scene:") and key in mapping:
                self._show_large(key)
                return
        for key in self._cards:
            if key in mapping:
                self._show_large(key)
                return

    def _on_preview_fail(self, err: str) -> None:
        self.btn_search.setEnabled(True)
        self.status.setText(f"Preview load failed: {err}")

    def _on_card_clicked(self, card: PreviewCard) -> None:
        self._show_large(card.key)

    def _show_large(self, key: str) -> None:
        for k, c in self._cards.items():
            c.set_selected_style(k == key)
        card = self._cards.get(key)
        img = self._previews.get(key)
        if card:
            self.big_title.setText(card.caption.text())
            meta = card.sub.text()
            if card.scene is not None:
                meta += f"\n{card.scene.id}"
                if card.scene.browse_url:
                    meta += "\n(quicklook — full download on Generate)"
            self.big_meta.setText(meta)
        if img is not None:
            self.big_preview.setPixmap(numpy_to_pixmap(img, 520, 520))
        else:
            self.big_preview.setText("No preview yet")
            self.big_preview.setPixmap(QPixmap())

    def _set_all_checks(self, on: bool) -> None:
        for card in self._cards.values():
            if card.check.isVisible() and card.check.isEnabled():
                card.check.setChecked(on)

    def _accept(self) -> None:
        picked: List[ArchiveScene] = []
        for card in self._cards.values():
            if card.scene is not None and card.check.isChecked():
                picked.append(card.scene)
        if not picked:
            QMessageBox.warning(
                self, "Archives", "Check at least one archive card for training, or Cancel."
            )
            return
        creds = load_credentials()
        if not creds.has_auth():
            dlg = EarthdataLoginDialog(self)
            if qt_exec(dlg) != QDialog.Accepted or not dlg.credentials().has_auth():
                QMessageBox.warning(
                    self,
                    "Earthdata Login",
                    "Login is required to download archive scenes for the dataset.",
                )
                return
            save_credentials(dlg.credentials())
        self.selected = picked
        self.accept()
