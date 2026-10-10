"""Show mosaics of the selected region from several imagery services side by side."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from shared.qt_compat import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPixmap,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
    Qt,
)
from desktop_app.gui.archive_cards import PreviewCard, numpy_to_pixmap
from desktop_app.gui.workers import StageWorker
from desktop_app.satellite.archives import ArchiveScene, SENSOR_TITLES
from desktop_app.satellite.preview_fetch import load_gallery_previews


class ServicePreviewDialog(QDialog):
    """Gallery: one card per selected XYZ service (+ optional archive quicklooks)."""

    def __init__(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        *,
        basemap_layers: Sequence[Tuple[str, str]],
        archive_scenes: Optional[Sequence[ArchiveScene]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Selected satellites — region preview")
        self.resize(1000, 680)
        self._bbox = (south, west, north, east)
        self._layers = list(basemap_layers)
        self._scenes = list(archive_scenes or [])
        self._cards: Dict[str, PreviewCard] = {}
        self._previews: Dict[str, object] = {}
        self._worker: Optional[StageWorker] = None

        root = QVBoxLayout(self)
        names = [t for _, t in self._layers] + [
            f"{SENSOR_TITLES.get(s.sensor, s.sensor)} {s.date[:10]}" for s in self._scenes
        ]
        root.addWidget(
            QLabel(
                f"Region S={south:.4f} W={west:.4f} N={north:.4f} E={east:.4f}\n"
                f"Showing: {', '.join(names) if names else '(none)'}"
            )
        )

        split = QSplitter(Qt.Horizontal)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.host = QWidget()
        self.grid_wrap = QVBoxLayout(self.host)
        self.grid_wrap.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.host)
        split.addWidget(self.scroll)

        right = QWidget()
        rlay = QVBoxLayout(right)
        self.big_title = QLabel("Click a card to enlarge")
        self.big_title.setStyleSheet("font-weight:600;")
        rlay.addWidget(self.big_title)
        self.big = QLabel("Preview")
        self.big.setMinimumSize(340, 340)
        self.big.setAlignment(Qt.AlignCenter)
        self.big.setStyleSheet("background:#141414;color:#888;border:1px solid #333;")
        rlay.addWidget(self.big, stretch=1)
        self.big_meta = QLabel("")
        self.big_meta.setWordWrap(True)
        rlay.addWidget(self.big_meta)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, stretch=1)

        self.progress = QProgressBar()
        root.addWidget(self.progress)
        self.status = QLabel("Loading…")
        root.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._build_cards()
        self._load_async()

    def _build_cards(self) -> None:
        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setSpacing(8)
        cards: List[PreviewCard] = []
        for layer, title in self._layers:
            card = PreviewCard(f"basemap:{layer}", title, "Map tiles · same region", checkable=False)
            card._click_cb = self._on_click
            cards.append(card)
            self._cards[card.key] = card
        for sc in self._scenes:
            cloud = f"cloud {sc.cloud_cover:.0f}%" if sc.cloud_cover is not None else sc.kind
            card = PreviewCard(
                f"scene:{sc.id}",
                sc.title,
                f"{sc.date[:10]} · {cloud}",
                checkable=False,
            )
            card.scene = sc
            card._click_cb = self._on_click
            cards.append(card)
            self._cards[card.key] = card
        for i, card in enumerate(cards):
            grid.addWidget(card, i // 3, i % 3)
        if not cards:
            self.grid_wrap.addWidget(QLabel("No services selected."))
        else:
            self.grid_wrap.addWidget(grid_host)
        self.grid_wrap.addStretch(1)

    def _load_async(self) -> None:
        south, west, north, east = self._bbox
        layers = list(self._layers)
        scenes = list(self._scenes)

        def job(progress, _frame):
            return load_gallery_previews(
                south, west, north, east, scenes,
                basemap_layers=layers,
                include_basemaps=True,
                progress_cb=progress,
            )

        self._worker = StageWorker(job)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_ready)
        self._worker.failed.connect(lambda e: self.status.setText(f"Failed: {e}"))
        self._worker.start()

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress.setValue(int(100 * done / total))
        self.status.setText(msg)

    def _on_ready(self, mapping: object) -> None:
        self.progress.setValue(100)
        if not isinstance(mapping, dict):
            return
        self._previews = mapping
        for key, card in self._cards.items():
            card.set_image(mapping.get(key))
        self.status.setText(f"Loaded {len(mapping)} preview(s)")
        for key in self._cards:
            if key in mapping:
                self._show(key)
                break

    def _on_click(self, card: PreviewCard) -> None:
        self._show(card.key)

    def _show(self, key: str) -> None:
        for k, c in self._cards.items():
            c.set_selected_style(k == key)
        card = self._cards.get(key)
        img = self._previews.get(key)
        if card:
            self.big_title.setText(card.caption.text())
            self.big_meta.setText(card.sub.text())
        if img is not None:
            self.big.setPixmap(numpy_to_pixmap(img, 480, 480))
        else:
            self.big.setText("No preview")
            self.big.setPixmap(QPixmap())
