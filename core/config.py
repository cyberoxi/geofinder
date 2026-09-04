"""Configuration loader and helpers."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "default_config.yaml"


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path: str | Path, base: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    base = base or ROOT_DIR
    return (base / p).resolve()


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = resolve_path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Resolve relative paths against project root
    paths = config.setdefault("paths", {})
    for key, value in list(paths.items()):
        if isinstance(value, str):
            paths[key] = str(resolve_path(value))

    # Ensure directories exist
    for key in ("output_dir", "failed_frames_dir", "projects_dir", "logs_dir", "input_dir", "models_dir"):
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    return config


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)


def get_nested(config: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_nested(config: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def env_override(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optional environment overrides for headless / CI use."""
    if os.environ.get("GRT_HEADLESS", "").lower() in ("1", "true", "yes"):
        config.setdefault("app", {})["headless"] = True
    device = os.environ.get("GRT_DEVICE")
    if device:
        config.setdefault("yolo", {})["device"] = device
    return config
