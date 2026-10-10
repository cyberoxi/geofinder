"""PyInstaller runtime hook: fix cwd and packaging shadowing."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap() -> None:
    # Prefer the folder that contains the .exe for relative model / config paths.
    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
        try:
            os.chdir(app_dir)
        except OSError:
            pass
        # Keep PyPI `packaging` importable if a local packaging/ folder is on path.
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bad = str(Path(meipass) / "desktop_app" / "packaging")
            sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(bad).resolve()]


_bootstrap()
