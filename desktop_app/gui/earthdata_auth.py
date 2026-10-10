"""Earthdata Login credentials helpers + dialog."""

from __future__ import annotations

from datetime import date, timedelta

from shared.qt_compat import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSettings,
    QVBoxLayout,
)
from desktop_app.satellite.archives import ArchiveCredentials

_SETTINGS_ORG = "GeoFinder"
_SETTINGS_APP = "Desktop"
_KEY_USER = "earthdata/username"
_KEY_TOKEN = "earthdata/token"


def load_credentials() -> ArchiveCredentials:
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    return ArchiveCredentials(
        username=str(s.value(_KEY_USER, "") or ""),
        password=str(s.value("earthdata/password", "") or ""),
        token=str(s.value(_KEY_TOKEN, "") or ""),
    )


def save_credentials(creds: ArchiveCredentials, *, store_password: bool = True) -> None:
    s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
    s.setValue(_KEY_USER, creds.username)
    s.setValue(_KEY_TOKEN, creds.token)
    if store_password:
        s.setValue("earthdata/password", creds.password)
    else:
        s.remove("earthdata/password")


def default_date_range(days: int = 180) -> tuple:
    end = date.today()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


class EarthdataLoginDialog(QDialog):
    """Earthdata Login username/password or bearer token."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Earthdata Login")
        self.resize(420, 220)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Required for NASA HLS and ASF Sentinel-1 downloads.\n"
                "Account: https://urs.earthdata.nasa.gov/\n"
                "Authorize ASF apps at https://search.asf.alaska.edu/"
            )
        )
        form = QFormLayout()
        creds = load_credentials()
        self.username = QLineEdit(creds.username)
        self.password = QLineEdit(creds.password)
        self.password.setEchoMode(QLineEdit.Password)
        self.token = QLineEdit(creds.token)
        self.token.setPlaceholderText("Optional: paste Earthdata bearer token")
        self.token.setEchoMode(QLineEdit.Password)
        form.addRow("Username", self.username)
        form.addRow("Password", self.password)
        form.addRow("Token", self.token)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def credentials(self) -> ArchiveCredentials:
        return ArchiveCredentials(
            username=self.username.text().strip(),
            password=self.password.text(),
            token=self.token.text().strip(),
        )
