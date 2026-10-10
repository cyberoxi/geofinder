#!/usr/bin/env python3
"""Desktop Dataset and Deployment Manager entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")

if FROZEN:
    ROOT = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    APP_DIR = Path(sys.executable).resolve().parent
    try:
        os.chdir(APP_DIR)
    except OSError:
        pass
else:
    ROOT = HERE.parent
    # `python desktop_app/app.py` puts desktop_app/ first on sys.path, where our
    # desktop_app/packaging/ would shadow the PyPI `packaging` library that
    # ultralytics / torch import ("No module named 'packaging.version'").
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

# PySide6/shiboken replaces builtins.__import__. Importing torch *after* that
# makes torch.cuda init crawl through inspect/getsource (Windows + Py 3.13+)
# and looks hung until Ctrl+C. Load torch first; later CUDA probes are cheap.
try:
    import torch  # noqa: F401
except Exception:
    pass

from desktop_app.gui.main_window import main

if __name__ == "__main__":
    raise SystemExit(main())
