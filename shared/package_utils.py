"""Package checksum and validation utilities (shared by Desktop and Jetson)."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_package(package_path: str | Path) -> Dict[str, Any]:
    path = Path(package_path)
    tmp_root = path.parent / f".validate_{path.stem}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"Unsafe path in zip: {name}")
            zf.extractall(tmp_root)
        root = tmp_root
        children = [p for p in root.iterdir()]
        if len(children) == 1 and children[0].is_dir() and (children[0] / "manifest.json").exists():
            root = children[0]
        elif not (root / "manifest.json").exists():
            found = list(root.rglob("manifest.json"))
            if not found:
                raise FileNotFoundError("manifest.json missing")
            root = found[0].parent
    else:
        root = path

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.json missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksum_path = root / "checksums" / "sha256.json"
    if not checksum_path.exists():
        raise FileNotFoundError("checksums/sha256.json missing")
    expected = json.loads(checksum_path.read_text(encoding="utf-8"))
    mismatches = []
    for rel, digest in expected.items():
        if rel.endswith("sha256.json"):
            continue
        fp = root / rel
        if not fp.exists():
            mismatches.append({"file": rel, "error": "missing"})
            continue
        if sha256_file(fp) != digest:
            mismatches.append({"file": rel, "error": "checksum_mismatch"})

    size = path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return {
        "ok": len(mismatches) == 0,
        "root": str(root),
        "manifest": manifest,
        "mismatches": mismatches,
        "size_bytes": size,
    }
