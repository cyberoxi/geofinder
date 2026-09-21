#!/usr/bin/env python3
"""Desktop Dataset and Deployment Manager entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
# `python desktop_app/app.py` puts desktop_app/ first on sys.path, where our
# desktop_app/packaging/ would shadow the PyPI `packaging` library that
# ultralytics / torch import ("No module named 'packaging.version'").
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.gui.main_window import main

if __name__ == "__main__":
    raise SystemExit(main())
