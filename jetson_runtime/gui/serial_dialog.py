"""Serial port settings dialog for Jetson offset output."""

from __future__ import annotations

from typing import Any, Dict

from shared.qt_compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSettings,
    QSpinBox,
    QVBoxLayout,
    qt_exec,
)
from jetson_runtime.serial_out import DEFAULT_SERIAL, list_serial_ports, merge_serial_config

ORG = "GeoFinder"
APP = "JetsonRuntime"


def load_serial_settings() -> Dict[str, Any]:
    s = QSettings(ORG, APP)
    cfg = dict(DEFAULT_SERIAL)
    cfg["enabled"] = s.value("serial/enabled", cfg["enabled"], type=bool)
    cfg["port"] = s.value("serial/port", cfg["port"], type=str)
    cfg["baudrate"] = int(s.value("serial/baudrate", cfg["baudrate"]))
    cfg["bytesize"] = int(s.value("serial/bytesize", cfg["bytesize"]))
    cfg["parity"] = s.value("serial/parity", cfg["parity"], type=str)
    cfg["stopbits"] = float(s.value("serial/stopbits", cfg["stopbits"]))
    cfg["timeout"] = float(s.value("serial/timeout", cfg["timeout"]))
    cfg["write_every_n"] = int(s.value("serial/write_every_n", cfg["write_every_n"]))
    cfg["send_when_lost"] = s.value("serial/send_when_lost", cfg["send_when_lost"], type=bool)
    cfg["line_format"] = s.value("serial/line_format", cfg["line_format"], type=str)
    return merge_serial_config(cfg)


def save_serial_settings(cfg: Dict[str, Any]) -> None:
    s = QSettings(ORG, APP)
    for key in DEFAULT_SERIAL:
        if key in cfg:
            s.setValue(f"serial/{key}", cfg[key])


class SerialSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Serial output (dx, dy)")
        self.setMinimumWidth(420)
        cfg = load_serial_settings()

        form = QFormLayout()
        self.enabled = QCheckBox("Enable serial send")
        self.enabled.setChecked(bool(cfg["enabled"]))
        form.addRow(self.enabled)

        port_row = QHBoxLayout()
        self.port = QComboBox()
        self.port.setEditable(True)
        ports = list_serial_ports()
        if cfg["port"] and cfg["port"] not in ports:
            ports = [str(cfg["port"])] + ports
        if not ports:
            ports = [str(cfg["port"] or "/dev/ttyTHS1"), "/dev/ttyUSB0", "/dev/ttyACM0", "COM3"]
        self.port.addItems(ports)
        self.port.setCurrentText(str(cfg["port"]))
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_ports)
        port_row.addWidget(self.port, stretch=1)
        port_row.addWidget(refresh)
        form.addRow("Port", port_row)

        self.baud = QComboBox()
        for b in (9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600):
            self.baud.addItem(str(b), b)
        idx = self.baud.findData(int(cfg["baudrate"]))
        self.baud.setCurrentIndex(idx if idx >= 0 else self.baud.findData(115200))
        form.addRow("Baud rate", self.baud)

        self.bytesize = QComboBox()
        for n in (5, 6, 7, 8):
            self.bytesize.addItem(str(n), n)
        bi = self.bytesize.findData(int(cfg["bytesize"]))
        self.bytesize.setCurrentIndex(bi if bi >= 0 else self.bytesize.findData(8))
        form.addRow("Data bits", self.bytesize)

        self.parity = QComboBox()
        for p, label in (("N", "None"), ("E", "Even"), ("O", "Odd")):
            self.parity.addItem(label, p)
        pi = self.parity.findData(str(cfg["parity"]).upper())
        self.parity.setCurrentIndex(pi if pi >= 0 else 0)
        form.addRow("Parity", self.parity)

        self.stopbits = QComboBox()
        for n in (1, 1.5, 2):
            self.stopbits.addItem(str(n), n)
        si = self.stopbits.findData(float(cfg["stopbits"]))
        self.stopbits.setCurrentIndex(si if si >= 0 else 0)
        form.addRow("Stop bits", self.stopbits)

        self.every_n = QSpinBox()
        self.every_n.setRange(1, 120)
        self.every_n.setValue(int(cfg["write_every_n"]))
        form.addRow("Send every N frames", self.every_n)

        self.send_lost = QCheckBox("Send 0,0 when lost")
        self.send_lost.setChecked(bool(cfg["send_when_lost"]))
        form.addRow(self.send_lost)

        self.fmt = QLineEdit(str(cfg["line_format"]))
        self.fmt.setToolTip("Placeholders: {dx} {dy} {dist} {status} {dx_f} {dy_f} {dist_f}")
        form.addRow("Line format", self.fmt)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _refresh_ports(self) -> None:
        current = self.port.currentText()
        self.port.clear()
        ports = list_serial_ports() or ["/dev/ttyTHS1", "/dev/ttyUSB0", "/dev/ttyACM0"]
        self.port.addItems(ports)
        self.port.setCurrentText(current)

    def values(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled.isChecked(),
            "port": self.port.currentText().strip(),
            "baudrate": int(self.baud.currentData()),
            "bytesize": int(self.bytesize.currentData()),
            "parity": str(self.parity.currentData()),
            "stopbits": float(self.stopbits.currentData()),
            "timeout": 0.1,
            "write_every_n": int(self.every_n.value()),
            "send_when_lost": self.send_lost.isChecked(),
            "line_format": self.fmt.text() or DEFAULT_SERIAL["line_format"],
        }


def edit_serial_settings(parent=None) -> Dict[str, Any]:
    dlg = SerialSettingsDialog(parent)
    if qt_exec(dlg) == QDialog.Accepted:
        cfg = dlg.values()
        save_serial_settings(cfg)
        return cfg
    return load_serial_settings()
