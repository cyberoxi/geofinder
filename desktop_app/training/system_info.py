"""Lightweight host hardware probe for training advice."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, Optional


def _gpu_name() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _cuda_torch_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def probe_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cpu_cores": os.cpu_count() or 0,
        "ram_gb": None,
        "gpu_name": _gpu_name(),
        "gpu_vram_mb": None,
        "cuda_torch": _cuda_torch_ok(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:  # noqa: BLE001
        pass
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run(
                [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                info["gpu_vram_mb"] = int(float(out.stdout.strip().splitlines()[0].strip()))
        except Exception:  # noqa: BLE001
            pass
    return info
