"""Load training summary, metrics CSV, and dataset counts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

METRIC_LABELS = {
    "metrics/precision(B)": "Precision (box)",
    "metrics/recall(B)": "Recall (box)",
    "metrics/mAP50(B)": "mAP50 (box)",
    "metrics/mAP50-95(B)": "mAP50-95 (box)",
    "metrics/precision(M)": "Precision (mask)",
    "metrics/recall(M)": "Recall (mask)",
    "metrics/mAP50(M)": "mAP50 (mask)",
    "metrics/mAP50-95(M)": "mAP50-95 (mask)",
}


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def settings_from(train: Dict[str, Any], results_dir: Optional[Path]) -> Dict[str, Any]:
    settings = dict(train.get("settings") or {})
    if results_dir:
        cfg_path = results_dir.parent / "training_config.yaml"
        if cfg_path.exists():
            try:
                from shared.config import load_yaml

                cfg = load_yaml(cfg_path) or {}
                for k in ("epochs", "batch", "imgsz", "device", "model", "data", "model_type"):
                    if k in cfg and k not in settings:
                        settings[k] = cfg[k]
            except Exception:  # noqa: BLE001
                pass
    if "device" not in settings and train.get("device"):
        settings["device"] = train["device"]
    return settings


def dataset_info(results_dir: Optional[Path], settings: Dict[str, Any]) -> Dict[str, Any]:
    data_yaml = settings.get("data")
    candidates: List[Path] = []
    if data_yaml:
        root = Path(data_yaml).parent
        candidates.extend([root / "split_by_video.json", root / "dataset_meta.json", root / "meta.json"])
    if results_dir and not data_yaml:
        cfg = results_dir.parent / "training_config.yaml"
        if cfg.exists():
            try:
                from shared.config import load_yaml

                data_yaml = (load_yaml(cfg) or {}).get("data")
                if data_yaml:
                    root = Path(data_yaml).parent
                    candidates.extend([root / "split_by_video.json", root / "dataset_meta.json"])
            except Exception:  # noqa: BLE001
                pass
    for p in candidates:
        if p.exists():
            meta = load_json(p)
            if meta:
                return {
                    "counts": meta.get("counts") or {},
                    "classes": meta.get("classes") or [],
                    "path": str(p.parent),
                }
    if data_yaml and Path(data_yaml).exists():
        root = Path(data_yaml).parent
        counts = {}
        for split in ("train", "val", "test"):
            img_dir = root / "images" / split
            if img_dir.is_dir():
                counts[split] = len(list(img_dir.glob("*.*")))
        if counts:
            return {"counts": counts, "classes": [], "path": str(root)}
    return {}


def read_results_csv(results_dir: Path) -> Tuple[List[Dict[str, float]], Optional[bool]]:
    path = results_dir / "results.csv"
    if not path.exists():
        return [], None
    rows: List[Dict[str, float]] = []
    try:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row = {}
                for k, v in raw.items():
                    if k is None:
                        continue
                    key = k.strip()
                    try:
                        row[key] = float(str(v).strip())
                    except (TypeError, ValueError):
                        continue
                if row:
                    rows.append(row)
    except Exception:  # noqa: BLE001
        return [], None
    if len(rows) < 4:
        return rows, None
    key = next((k for k in rows[-1] if "mAP50(" in k and "95" not in k), None)
    if not key:
        return rows, None
    recent = [r.get(key, 0.0) for r in rows[-5:]]
    earlier = sum(recent[:2]) / 2.0
    later = sum(recent[-2:]) / 2.0
    return rows, later > earlier + 0.01


def extract_metrics(train: Dict[str, Any], csv_rows: List[Dict[str, float]]) -> Dict[str, float]:
    raw = ((train or {}).get("last_metrics") or {}).get("metrics") or {}
    out = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    if not out and csv_rows:
        last = csv_rows[-1]
        for k in METRIC_LABELS:
            if k in last:
                out[k] = last[k]
    return out


def primary_map(metrics: Dict[str, float]) -> Optional[float]:
    for k in ("metrics/mAP50(M)", "metrics/mAP50(B)"):
        if k in metrics:
            return metrics[k]
    return None


def list_result_files(train: Dict[str, Any], results_dir: Optional[Path]) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    best = train.get("best_weights")
    if best:
        files.append(("best.pt", str(best)))
        last = Path(best).with_name("last.pt")
        if last.exists():
            files.append(("last.pt", str(last)))
    if results_dir:
        for name in (
            "results.csv",
            "results.png",
            "confusion_matrix.png",
            "BoxPR_curve.png",
            "MaskPR_curve.png",
            "args.yaml",
        ):
            p = results_dir / name
            if p.exists():
                files.append((name, str(p)))
    return files
