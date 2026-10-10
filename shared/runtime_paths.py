"""Resolve repo / frozen-app paths (PyInstaller onedir/onefile)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def bundle_dir() -> Path:
    """Read-only payload dir (PyInstaller _MEIPASS) or repo root."""
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def app_dir() -> Path:
    """Writable directory next to the .exe (frozen) or repo root (dev)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def find_bundled_file(*names: str) -> Optional[Path]:
    """Prefer file next to the exe, then inside the bundle."""
    roots = (app_dir(), bundle_dir())
    for name in names:
        for root in roots:
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None
