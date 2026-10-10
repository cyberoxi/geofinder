"""Multi-select imagery services for the satellite training dataset."""

from __future__ import annotations

from typing import List, Tuple

from shared.qt_compat import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSettings,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    Signal,
)
from desktop_app.gui.earthdata_auth import default_date_range
from desktop_app.satellite.archives import (
    SENSOR_HLS_L30,
    SENSOR_HLS_S30,
    SENSOR_S1_GRD,
    SENSOR_TITLES,
)
from desktop_app.satellite.tiles import (
    list_wayback_releases,
    register_azure_maps_layer,
    register_custom_layer,
    register_mapbox_layer,
    register_wayback_layer,
)

from pathlib import Path


class ImageryServicesPanel(QWidget):
    """
    Check which services to use for the dataset (not a single combo).
    Primary mosaic = first resolved XYZ layer; others become extra training views.
    """

    services_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wayback: List[dict] = []
        s = QSettings("GeoFinder", "Desktop")
        self._mapbox_token = str(s.value("imagery/mapbox_token", "") or "")
        self._azure_key = str(s.value("imagery/azure_key", "") or "")
        ignored = getattr(QSizePolicy, "Ignored", None) or QSizePolicy.Policy.Ignored
        preferred = getattr(QSizePolicy, "Preferred", None) or QSizePolicy.Policy.Preferred
        self.setMinimumWidth(0)
        self.setSizePolicy(ignored, preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("Services for dataset")
        box.setToolTip("سرویس‌های دیتاست")
        form = QVBoxLayout(box)
        form.setSpacing(4)

        tip = QLabel(
            "First map tile = main map · others = mini previews (click to switch main)"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#888;")
        form.addWidget(tip)

        form.addWidget(QLabel("Map tiles"))
        self.chk_esri = QCheckBox("Esri World Imagery")
        self.chk_esri.setChecked(False)
        self.chk_clarity = QCheckBox("Esri Clarity")
        self.chk_wayback = QCheckBox("Esri Wayback (pick date)")
        self.chk_mapbox = QCheckBox("Mapbox Satellite (HD)")
        self.chk_azure = QCheckBox("Azure Maps Imagery (HD)")
        self.chk_custom = QCheckBox("Custom XYZ URL")
        for c in (
            self.chk_esri,
            self.chk_clarity,
            self.chk_wayback,
            self.chk_mapbox,
            self.chk_azure,
            self.chk_custom,
        ):
            c.toggled.connect(self._sync_extra_fields)
            form.addWidget(c)

        self.wayback_date = QComboBox()
        self.wayback_date.setVisible(False)
        self.wayback_date.currentIndexChanged.connect(lambda _i: self.services_changed.emit())
        form.addWidget(self.wayback_date)
        self.extra_dates_lab = QLabel("Extra Wayback dates")
        self.extra_dates_lab.setVisible(False)
        form.addWidget(self.extra_dates_lab)
        self.extra_dates = QSpinBox()
        self.extra_dates.setRange(0, 6)
        self.extra_dates.setValue(0)
        self.extra_dates.setToolTip("Extra Wayback seasons when Wayback is checked")
        self.extra_dates.setVisible(False)
        form.addWidget(self.extra_dates)

        self.custom_url = QLineEdit()
        self.custom_url.setPlaceholderText("https://server/tiles/{z}/{x}/{y}.jpg")
        self.custom_url.setVisible(False)
        form.addWidget(self.custom_url)

        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("Mapbox token or Azure key (when those are checked)")
        self.api_key.setVisible(False)
        form.addWidget(self.api_key)

        form.addWidget(QLabel("Archives"))
        self.chk_hls_s30 = QCheckBox(SENSOR_TITLES[SENSOR_HLS_S30])
        self.chk_hls_l30 = QCheckBox(SENSOR_TITLES[SENSOR_HLS_L30])
        self.chk_s1 = QCheckBox(SENSOR_TITLES[SENSOR_S1_GRD])
        for c in (self.chk_hls_s30, self.chk_hls_l30, self.chk_s1):
            c.toggled.connect(self._sync_extra_fields)
            form.addWidget(c)

        _start, _end = default_date_range(180)
        date_row = QHBoxLayout()
        self.arch_start = QLineEdit(_start)
        self.arch_end = QLineEdit(_end)
        date_row.addWidget(self.arch_start)
        date_row.addWidget(QLabel("→"))
        date_row.addWidget(self.arch_end)
        self.arch_dates = QWidget()
        self.arch_dates.setLayout(date_row)
        self.arch_dates.setVisible(False)
        form.addWidget(self.arch_dates)
        self.arch_max_lab = QLabel("Max scenes / sensor")
        self.arch_max_lab.setVisible(False)
        form.addWidget(self.arch_max_lab)
        self.arch_max = QSpinBox()
        self.arch_max.setRange(1, 20)
        self.arch_max.setValue(3)
        self.arch_max.setVisible(False)
        form.addWidget(self.arch_max)

        self.btn_preview = QPushButton("Show selected satellites…")
        self.btn_preview.setToolTip(
            "Download a preview of the selected region from every checked service. / نمایش تصاویر"
        )
        self.btn_preview.setMinimumHeight(34)
        form.addWidget(self.btn_preview)
        self.btn_pick_archive = QPushButton("Pick archive scenes…")
        self.btn_login = QPushButton("Earthdata Login…")
        self.btn_own = QPushButton("Own image file…")
        # Stack vertically so the side panel stays narrow
        arch_btns = QVBoxLayout()
        arch_btns.setContentsMargins(0, 0, 0, 0)
        arch_btns.setSpacing(4)
        arch_btns.addWidget(self.btn_pick_archive)
        arch_btns.addWidget(self.btn_login)
        self.arch_btns = QWidget()
        self.arch_btns.setLayout(arch_btns)
        self.arch_btns.setVisible(False)
        form.addWidget(self.arch_btns)
        form.addWidget(self.btn_own)

        self.status = QLabel("No service selected — map stays blank until you tick one")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#888;")
        form.addWidget(self.status)

        root.addWidget(box)
        self._sync_extra_fields()

    def _sync_extra_fields(self, _checked: bool = False) -> None:
        wb = self.chk_wayback.isChecked()
        self.wayback_date.setVisible(wb)
        self.extra_dates_lab.setVisible(wb)
        self.extra_dates.setVisible(wb)
        if wb and self.wayback_date.count() == 0:
            self._load_wayback()
        self.custom_url.setVisible(self.chk_custom.isChecked())
        need_key = self.chk_mapbox.isChecked() or self.chk_azure.isChecked()
        self.api_key.setVisible(need_key)
        if self.chk_mapbox.isChecked() and not self.chk_azure.isChecked():
            self.api_key.setText(self._mapbox_token)
            self.api_key.setPlaceholderText("Mapbox access token (pk.…)")
        elif self.chk_azure.isChecked() and not self.chk_mapbox.isChecked():
            self.api_key.setText(self._azure_key)
            self.api_key.setPlaceholderText("Azure Maps subscription key")
        elif need_key:
            self.api_key.setPlaceholderText("Token/key for Mapbox and/or Azure")
        has_arch = self.has_archive_sensors()
        self.arch_dates.setVisible(has_arch)
        self.arch_max_lab.setVisible(has_arch)
        self.arch_max.setVisible(has_arch)
        self.arch_btns.setVisible(has_arch)
        self.status.setText(self.summary_text())
        self.services_changed.emit()

    def _load_wayback(self) -> None:
        self._wayback = list_wayback_releases(Path("data") / "satellite_cache")
        self.wayback_date.clear()
        for r in self._wayback:
            self.wayback_date.addItem(r["date"], r["id"])

    def has_archive_sensors(self) -> bool:
        return bool(self.selected_archive_sensors())

    def selected_archive_sensors(self) -> List[str]:
        out: List[str] = []
        if self.chk_hls_s30.isChecked():
            out.append(SENSOR_HLS_S30)
        if self.chk_hls_l30.isChecked():
            out.append(SENSOR_HLS_L30)
        if self.chk_s1.isChecked():
            out.append(SENSOR_S1_GRD)
        return out

    def archive_date_range(self) -> Tuple[str, str]:
        return (
            self.arch_start.text().strip() or default_date_range()[0],
            self.arch_end.text().strip() or default_date_range()[1],
        )

    def summary_text(self) -> str:
        names = [t for _, t in self.preview_layer_titles()]
        names += [SENSOR_TITLES[s] for s in self.selected_archive_sensors()]
        if not names:
            return "No service selected — map stays blank until you tick one"
        return "Selected: " + ", ".join(names)

    def preview_layer_titles(self) -> List[Tuple[str, str]]:
        """Resolved XYZ layer key + title for preview (registers keyed layers)."""
        pairs: List[Tuple[str, str]] = []
        for key, title in self.resolve_xyz_layers(preview_only=True):
            pairs.append((key, title))
        return pairs

    def resolve_xyz_layers(self, *, preview_only: bool = False) -> List[Tuple[str, str]]:
        """
        Return [(layer_key, title), ...] for checked tile services.
        Raises ValueError if a required key/URL is missing (unless preview_only).
        """
        out: List[Tuple[str, str]] = []
        if self.chk_esri.isChecked():
            out.append(("satellite", "Esri World Imagery"))
        if self.chk_clarity.isChecked():
            out.append(("esri_clarity", "Esri Clarity"))
        if self.chk_wayback.isChecked():
            if not self._wayback:
                self._load_wayback()
            rid = self.wayback_date.currentData()
            rel = next((r for r in self._wayback if r["id"] == rid), None)
            if rel:
                key = register_wayback_layer(rel)
                out.append((key, f"Wayback {rel.get('date', '')}"))
            elif not preview_only:
                raise ValueError("Wayback date list unavailable (offline?)")
        if self.chk_mapbox.isChecked():
            # Prefer saved token when Azure is also checked (shared API key field)
            if self.chk_azure.isChecked():
                token = self._mapbox_token or self.api_key.text().strip()
            else:
                token = self.api_key.text().strip() or self._mapbox_token
            if not token:
                if not preview_only:
                    raise ValueError("Mapbox access token required")
            else:
                key = register_mapbox_layer(token)
                self._mapbox_token = token
                QSettings("GeoFinder", "Desktop").setValue("imagery/mapbox_token", token)
                out.append((key, "Mapbox Satellite"))
        if self.chk_azure.isChecked():
            if self.chk_mapbox.isChecked():
                key_txt = self._azure_key or self.api_key.text().strip()
            else:
                key_txt = self.api_key.text().strip() or self._azure_key
            if not key_txt:
                if not preview_only:
                    raise ValueError("Azure Maps subscription key required")
            else:
                key = register_azure_maps_layer(key_txt)
                self._azure_key = key_txt
                QSettings("GeoFinder", "Desktop").setValue("imagery/azure_key", key_txt)
                out.append((key, "Azure Maps Imagery"))
        if self.chk_custom.isChecked():
            url = self.custom_url.text().strip()
            if not url:
                if not preview_only:
                    raise ValueError("Custom XYZ URL required ({z}/{x}/{y})")
            else:
                key = register_custom_layer(url)
                out.append((key, "Custom XYZ"))
        return out

    def primary_and_extras(self) -> Tuple[str, List[str]]:
        """Primary layer + extra XYZ layer keys for prepare_scene."""
        layers = self.resolve_xyz_layers(preview_only=False)
        if not layers and not self.selected_archive_sensors():
            raise ValueError("Select at least one imagery service")
        if not layers:
            # Archives only — landmarks from Esri, archives attached later
            return "satellite", []
        primary = layers[0][0]
        extras = [k for k, _ in layers[1:]]
        if self.chk_wayback.isChecked() and self.extra_dates.value() > 0:
            from desktop_app.satellite.tiles import pick_wayback_dates

            if not self._wayback:
                self._load_wayback()
            extras.extend(
                register_wayback_layer(r)
                for r in pick_wayback_dates(
                    self._wayback, int(self.extra_dates.value()), exclude_key=primary
                )
            )
        # Dedupe preserving order
        seen = {primary}
        uniq: List[str] = []
        for k in extras:
            if k not in seen:
                seen.add(k)
                uniq.append(k)
        return primary, uniq

    def widgets_for_enable(self) -> List[QWidget]:
        return [
            self.chk_esri,
            self.chk_clarity,
            self.chk_wayback,
            self.chk_mapbox,
            self.chk_azure,
            self.chk_custom,
            self.chk_hls_s30,
            self.chk_hls_l30,
            self.chk_s1,
            self.wayback_date,
            self.extra_dates,
            self.custom_url,
            self.api_key,
            self.arch_start,
            self.arch_end,
            self.arch_max,
            self.btn_preview,
            self.btn_pick_archive,
            self.btn_login,
            self.btn_own,
        ]
