"""Validate a YOLO dataset directory structure."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Validate YOLO dataset layout")
    p.add_argument("--data", required=True, help="Dataset root or data.yaml")
    args = p.parse_args()

    root = Path(args.data)
    if root.is_file() and root.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        with open(root, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        print("data.yaml:", data)
        for split in ("train", "val", "test"):
            if split in data:
                print(f"  {split}: {data[split]}")
        return

    issues = []
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            issues.append(f"Missing {img_dir}")
            continue
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        print(f"{split}: {len(images)} images, {len(labels)} labels")
        # Check orphan labels
        img_stems = {i.stem for i in images}
        lbl_stems = {l.stem for l in labels}
        missing = img_stems - lbl_stems
        if missing:
            issues.append(f"{split}: {len(missing)} images without labels")

    if issues:
        print("Issues:")
        for i in issues:
            print(" -", i)
        raise SystemExit(1)
    print("Dataset OK")


if __name__ == "__main__":
    main()
