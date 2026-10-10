"""Professional dark theme for GeoFinder desktop / tracker GUIs."""

from __future__ import annotations

from typing import Any

# Prefer Persian display / UI faces when installed (common on Windows IR).
_BRAND_FONT_CANDIDATES = (
    "B Titr",
    "NPINazaninTitr",
    "IRANSansX ExtraBold",
    "IRANSansX Bold",
    "IRANSansX",
    "B Nazanin",
    "Tahoma",
    "Segoe UI",
)
_UI_FONT_CANDIDATES = (
    "IRANSansX",
    "IRANSansXFaNum",
    "B Nazanin",
    "Tahoma",
    "Segoe UI",
)

APP_STYLESHEET = """
QWidget {
    background-color: #0f1419;
    color: #e2e8f0;
    font-family: "IRANSansX", "B Nazanin", "Tahoma", "Segoe UI", sans-serif;
    font-size: 13px;
}
QMainWindow, QDialog {
    background-color: #0f1419;
}
QMenuBar {
    background-color: #111827;
    color: #e2e8f0;
    border-bottom: 1px solid #1f2937;
    padding: 2px 0;
}
QMenuBar::item {
    padding: 6px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #1e293b;
}
QMenu {
    background-color: #111827;
    border: 1px solid #334155;
    padding: 4px;
}
QMenu::item {
    padding: 6px 28px 6px 16px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #0d9488;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background: #334155;
    margin: 4px 8px;
}
QToolBar {
    background-color: #111827;
    border-bottom: 1px solid #1f2937;
    spacing: 2px;
    padding: 2px 6px;
}
QToolBar::separator {
    background: #334155;
    width: 1px;
    margin: 4px 4px;
}
QToolButton {
    background-color: transparent;
    color: #e2e8f0;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 4px 6px;
    margin: 0;
}
QToolButton:hover {
    background-color: #1e293b;
    border-color: #334155;
}
QToolButton:pressed, QToolButton:checked {
    background-color: #134e4a;
    border-color: #14b8a6;
    color: #ffffff;
}
QStatusBar {
    background-color: #111827;
    color: #94a3b8;
    border-top: 1px solid #1f2937;
}
QGroupBox {
    background-color: #151b24;
    border: 1px solid #243041;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #14b8a6;
}
QPushButton {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 7px 12px;
    min-height: 18px;
}
QPushButton:hover {
    background-color: #243447;
    border-color: #14b8a6;
}
QPushButton:pressed {
    background-color: #134e4a;
}
QPushButton:disabled {
    color: #64748b;
    background-color: #151b24;
    border-color: #1f2937;
}
QPushButton#primaryButton {
    background-color: #0d9488;
    border-color: #14b8a6;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background-color: #14b8a6;
}
QPushButton#dangerButton {
    background-color: #7f1d1d;
    border-color: #ef4444;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QListWidget {
    background-color: #0b1220;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: #0d9488;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QListWidget:focus {
    border-color: #14b8a6;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background-color: #111827;
    border: 1px solid #334155;
    selection-background-color: #0d9488;
}
QListWidget::item {
    padding: 6px 8px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #134e4a;
    color: #ffffff;
}
QListWidget::item:hover {
    background-color: #1e293b;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #14b8a6;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #0d9488;
    border-radius: 3px;
}
QProgressBar {
    background-color: #0b1220;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #e2e8f0;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: #14b8a6;
    border-radius: 5px;
}
QScrollBar:vertical {
    background: #0f1419;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #0f1419;
    height: 10px;
}
QScrollBar::handle:horizontal {
    background: #334155;
    border-radius: 5px;
    min-width: 24px;
}
QSplitter::handle {
    background: #1f2937;
    width: 2px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #475569;
    background: #0b1220;
}
QCheckBox::indicator:checked {
    background: #14b8a6;
    border-color: #14b8a6;
}
QLabel#panelTitle {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}
QLabel#brandLabel {
    background-color: transparent;
    border: none;
    color: #14b8a6;
    font-family: "B Titr", "IRANSansX ExtraBold", "IRANSansX", "B Nazanin", "Tahoma";
    font-size: 15px;
    font-weight: 700;
    padding: 0 6px;
}
QToolTip {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    padding: 4px 8px;
}
QMessageBox {
    background-color: #111827;
}
QMessageBox QLabel {
    color: #e2e8f0;
    background-color: transparent;
}
QMessageBox QPushButton {
    min-width: 72px;
}
"""


def _font_families() -> set[str]:
    try:
        from shared.qt_compat import QFontDatabase

        return set(QFontDatabase.families())
    except Exception:  # noqa: BLE001
        return set()


def pick_font(candidates: tuple[str, ...], point_size: int, bold: bool = False) -> Any:
    """Return a QFont using the first installed family from *candidates*."""
    from shared.qt_compat import QFont

    available = _font_families()
    family = next((name for name in candidates if name in available), candidates[-1])
    font = QFont(family, point_size)
    if bold:
        font.setBold(True)
    return font


def style_brand_label(label: Any, text: str = "تنگسیری") -> None:
    """Set brand text with a Persian title face (B Titr when available)."""
    from shared.qt_compat import Qt

    label.setObjectName("brandLabel")
    label.setText(f"  {text}  ")
    label.setAutoFillBackground(False)
    try:
        label.setAttribute(Qt.WA_TranslucentBackground)
    except Exception:  # noqa: BLE001
        pass
    try:
        label.setFont(pick_font(_BRAND_FONT_CANDIDATES, 13, bold=True))
    except Exception:  # noqa: BLE001
        pass


def _apply_dark_palette(app: Any) -> None:
    """Keep native Fusion roles readable under the dark stylesheet (esp. QMessageBox)."""
    from shared.qt_compat import QColor, QPalette

    p = QPalette()
    window = QColor("#0f1419")
    base = QColor("#0b1220")
    alt = QColor("#151b24")
    text = QColor("#e2e8f0")
    muted = QColor("#94a3b8")
    highlight = QColor("#0d9488")
    button = QColor("#1e293b")

    p.setColor(QPalette.Window, window)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, alt)
    p.setColor(QPalette.Text, text)
    try:
        p.setColor(QPalette.PlaceholderText, muted)
    except AttributeError:
        pass
    p.setColor(QPalette.Button, button)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.BrightText, QColor("#ffffff"))
    p.setColor(QPalette.ToolTipBase, QColor("#1e293b"))
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Highlight, highlight)
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.Link, QColor("#14b8a6"))
    p.setColor(QPalette.Disabled, QPalette.WindowText, muted)
    p.setColor(QPalette.Disabled, QPalette.Text, muted)
    p.setColor(QPalette.Disabled, QPalette.ButtonText, muted)
    app.setPalette(p)


def apply_theme(app: Any) -> None:
    """Apply Fusion style + dark stylesheet to a QApplication."""
    try:
        app.setStyle("Fusion")
    except Exception:  # noqa: BLE001
        pass
    try:
        _apply_dark_palette(app)
    except Exception:  # noqa: BLE001
        pass
    app.setStyleSheet(APP_STYLESHEET)
    try:
        app.setFont(pick_font(_UI_FONT_CANDIDATES, 10))
    except Exception:  # noqa: BLE001
        pass
