"""Serial offset output: image-centre → target-centre (dx, dy)."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

from shared.logging import get_logger

logger = get_logger("serial")

_DEFAULT_PORT = "COM3" if sys.platform == "win32" else "/dev/ttyTHS1"

DEFAULT_SERIAL: Dict[str, Any] = {
    "enabled": False,
    "port": _DEFAULT_PORT,
    "baudrate": 115200,
    "bytesize": 8,
    "parity": "N",
    "stopbits": 1,
    "timeout": 0.1,
    "write_every_n": 1,
    "send_when_lost": False,
    # ASCII line: signed pixel offsets (target − image centre); +x right, +y down
    "line_format": "{dx},{dy}\n",
}


def merge_serial_config(
    runtime_serial: Optional[Dict[str, Any]] = None,
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(DEFAULT_SERIAL)
    if runtime_serial:
        cfg.update({k: v for k, v in runtime_serial.items() if v is not None})
    if override:
        cfg.update({k: v for k, v in override.items() if v is not None})
    return cfg


def list_serial_ports() -> List[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [p.device for p in list_ports.comports()]


def image_to_target_offset(
    frame_w: int,
    frame_h: int,
    center_x: float,
    center_y: float,
) -> Tuple[float, float, float]:
    """Return (dx, dy, distance) in pixels from image centre to target centre."""
    icx = frame_w * 0.5
    icy = frame_h * 0.5
    dx = float(center_x) - icx
    dy = float(center_y) - icy
    dist = float((dx * dx + dy * dy) ** 0.5)
    return dx, dy, dist


class SerialOffsetSender:
    """Open a UART and emit dx/dy each tracking update."""

    def __init__(self, config: Dict[str, Any]):
        self.config = merge_serial_config(config)
        self._ser = None
        self._frame_count = 0
        self._opened = False

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled"))

    def open(self) -> bool:
        if not self.enabled:
            return False
        try:
            import serial
        except ImportError:
            logger.error("pyserial not installed — pip install pyserial")
            return False
        port = str(self.config.get("port") or "")
        if not port:
            logger.error("Serial enabled but port is empty")
            return False
        parity_map = {
            "N": serial.PARITY_NONE,
            "E": serial.PARITY_EVEN,
            "O": serial.PARITY_ODD,
            "M": serial.PARITY_MARK,
            "S": serial.PARITY_SPACE,
        }
        parity = parity_map.get(str(self.config.get("parity", "N")).upper(), serial.PARITY_NONE)
        stop_map = {1: serial.STOPBITS_ONE, 1.5: serial.STOPBITS_ONE_POINT_FIVE, 2: serial.STOPBITS_TWO}
        stopbits = stop_map.get(float(self.config.get("stopbits", 1)), serial.STOPBITS_ONE)
        bytesize_map = {5: serial.FIVEBITS, 6: serial.SIXBITS, 7: serial.SEVENBITS, 8: serial.EIGHTBITS}
        bytesize = bytesize_map.get(int(self.config.get("bytesize", 8)), serial.EIGHTBITS)
        try:
            self._ser = serial.Serial(
                port=port,
                baudrate=int(self.config.get("baudrate", 115200)),
                bytesize=bytesize,
                parity=parity,
                stopbits=stopbits,
                timeout=float(self.config.get("timeout", 0.1)),
                write_timeout=float(self.config.get("timeout", 0.1)),
            )
            self._opened = True
            logger.info(
                "Serial open %s @ %s baud (dx,dy = target − image centre)",
                port,
                self.config.get("baudrate"),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Serial open failed (%s): %s", port, exc)
            self._ser = None
            self._opened = False
            return False

    def send_offset(
        self,
        dx: float,
        dy: float,
        *,
        distance: float = 0.0,
        status: str = "",
        tracking: bool = True,
    ) -> bool:
        if not self._opened or self._ser is None:
            return False
        self._frame_count += 1
        every = max(1, int(self.config.get("write_every_n", 1)))
        if self._frame_count % every != 0:
            return False
        if not tracking and not self.config.get("send_when_lost", False):
            return False
        if not tracking:
            dx, dy, distance = 0.0, 0.0, 0.0
        fmt = str(self.config.get("line_format") or DEFAULT_SERIAL["line_format"])
        try:
            line = fmt.format(
                dx=int(round(dx)),
                dy=int(round(dy)),
                dist=int(round(distance)),
                status=status or "",
                dx_f=dx,
                dy_f=dy,
                dist_f=distance,
            )
            self._ser.write(line.encode("ascii", errors="replace"))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Serial write failed: %s", exc)
            return False

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001
                pass
        self._ser = None
        self._opened = False
