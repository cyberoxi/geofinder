"""Load and verify deployment packages on Jetson."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from shared.package_utils import validate_package
from shared.config import default_runtime_config, load_yaml
from shared.logging import get_logger
from shared.schemas import Manifest

logger = get_logger("jetson.package")


class PackageLoader:
    def __init__(self, package_path: str | Path, extract_dir: Optional[str | Path] = None):
        self.package_path = Path(package_path)
        self.extract_dir = Path(extract_dir) if extract_dir else self.package_path.parent / f"loaded_{self.package_path.stem}"
        self.root: Optional[Path] = None
        self.manifest: Optional[Manifest] = None
        self.runtime_config: Dict[str, Any] = default_runtime_config()
        self.validation: Dict[str, Any] = {}

    def load(self) -> Path:
        self.validation = validate_package(self.package_path)
        if not self.validation.get("ok"):
            raise RuntimeError(f"Package validation failed: {self.validation.get('mismatches')}")
        self.root = Path(self.validation["root"])
        # If validated into temp folder under packages, copy/move to extract_dir for stable path
        if self.root.resolve() != self.extract_dir.resolve():
            if self.extract_dir.exists():
                shutil.rmtree(self.extract_dir)
            shutil.copytree(self.root, self.extract_dir)
            self.root = self.extract_dir
        self.manifest = Manifest.from_dict(json.loads((self.root / "manifest.json").read_text(encoding="utf-8")))
        cfg_path = self.root / "config" / "runtime.yaml"
        if cfg_path.exists():
            self.runtime_config = load_yaml(cfg_path)
        logger.info("Loaded package %s format=%s", self.manifest.project_name, self.manifest.model_format)
        return self.root

    def model_candidates(self) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        assert self.root is not None
        engine = self.root / "model" / "model.engine"
        onnx = self.root / "model" / "model.onnx"
        pt = self.root / "model" / "best.pt"
        return (
            engine if engine.exists() else None,
            onnx if onnx.exists() else None,
            pt if pt.exists() else None,
        )

    def reference_dir(self) -> Path:
        assert self.root is not None
        return self.root / "reference"

    def descriptors_path(self) -> Path:
        return self.reference_dir() / "descriptors.npz"
