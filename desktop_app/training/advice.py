"""Suggest training improvements from metrics + this PC's hardware."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from desktop_app.training.results_parse import (
    METRIC_LABELS,
    dataset_info,
    extract_metrics,
    list_result_files,
    load_json,
    primary_map,
    read_results_csv,
    settings_from,
)
from desktop_app.training.system_info import probe_hardware

GRADE_FA = {
    "excellent": "عالی",
    "good": "خوب",
    "fair": "متوسط",
    "weak": "ضعیف",
    "unknown": "نامشخص",
}


@dataclass
class Tip:
    priority: str  # high | medium | low
    title: str
    detail: str
    apply: Dict[str, int] = field(default_factory=dict)


@dataclass
class TrainAdviceReport:
    grade: str
    grade_fa: str
    headline: str
    metrics: List[Tuple[str, float]]
    hardware: Dict[str, Any]
    settings: Dict[str, Any]
    dataset: Dict[str, Any]
    tips: List[Tip]
    files: List[Tuple[str, str]]
    suggested: Dict[str, int]
    still_improving: Optional[bool] = None


def _grade(map50: Optional[float]) -> str:
    if map50 is None:
        return "unknown"
    if map50 >= 0.75:
        return "excellent"
    if map50 >= 0.50:
        return "good"
    if map50 >= 0.30:
        return "fair"
    return "weak"


def _vram_targets(vram_mb: Optional[int]) -> Tuple[int, int, int]:
    """Suggested (batch, imgsz, epochs_floor) for this GPU memory."""
    if vram_mb is None:
        return 4, 640, 60
    if vram_mb < 4500:
        return 2, 512, 50
    if vram_mb < 8500:
        return 8, 640, 60
    if vram_mb < 13000:
        return 16, 640, 80
    return 24, 960, 100


def _collect_tips(
    *,
    grade: str,
    hw: Dict[str, Any],
    settings: Dict[str, Any],
    metrics_raw: Dict[str, float],
    dataset: Dict[str, Any],
    train: Dict[str, Any],
    still_improving: Optional[bool],
) -> Tuple[List[Tip], Dict[str, int]]:
    tips: List[Tip] = []
    suggested: Dict[str, int] = {}
    batch = int(settings.get("batch") or 4)
    epochs = int(settings.get("epochs") or 40)
    imgsz = int(settings.get("imgsz") or 640)
    device = str(settings.get("device") or train.get("device") or "?")
    sug_batch, sug_imgsz, sug_epochs = _vram_targets(hw.get("gpu_vram_mb"))

    if device == "cpu" and hw.get("gpu_name") and hw.get("cuda_torch"):
        tips.append(
            Tip(
                "high",
                "از GPU استفاده کنید",
                f"آموزش الان روی CPU اجرا می‌شود، در حالی که {hw['gpu_name']} با CUDA در دسترس است. "
                "در تنظیمات، گزینهٔ Train on را روی GPU بگذارید تا آموزش خیلی سریع‌تر شود.",
            )
        )
    elif hw.get("gpu_name") and not hw.get("cuda_torch"):
        tips.append(
            Tip(
                "high",
                "پشتیبانی CUDA در PyTorch نصب نیست",
                f"کارت {hw['gpu_name']} دیده می‌شود، ولی نسخهٔ فعلی torch از GPU استفاده نمی‌کند. "
                "نسخهٔ CUDA مربوط به PyTorch را نصب کنید و سپس Train on را روی GPU بگذارید.",
            )
        )
    elif not hw.get("gpu_name"):
        tips.append(
            Tip(
                "medium",
                "بدون GPU — تنظیمات سبک‌تر",
                "GPU پیدا نشد. Batch را کم نگه دارید (۲ تا ۴)، Imgsz را حدود ۵۱۲ بگذارید و Epochs را معقول انتخاب کنید. "
                "آموزش کندتر است، اما وزن نهایی برای Jetson همان خواهد بود.",
                apply={"batch": 2, "imgsz": 512},
            )
        )
        suggested.update({"batch": 2, "imgsz": 512})

    if hw.get("cuda_torch") and hw.get("gpu_vram_mb"):
        if batch < sug_batch // 2:
            tips.append(
                Tip(
                    "medium",
                    f"افزایش Batch (حدود {sug_batch})",
                    f"حدود {hw['gpu_vram_mb']} مگابایت VRAM دارید و batch فعلی ({batch}) کم است. "
                    f"batch نزدیک {sug_batch} معمولاً آموزش پایدارتر و سریع‌تری می‌دهد.",
                    apply={"batch": sug_batch},
                )
            )
            suggested["batch"] = sug_batch
        elif batch > sug_batch * 2:
            tips.append(
                Tip(
                    "high",
                    f"کاهش Batch (حدود {sug_batch})",
                    f"با حدود {hw['gpu_vram_mb']} مگابایت VRAM، batch برابر {batch} احتمال کمبود حافظه (OOM) دارد. "
                    f"آن را به حدود {sug_batch} برسانید.",
                    apply={"batch": sug_batch},
                )
            )
            suggested["batch"] = sug_batch
        if imgsz < sug_imgsz - 64 and hw["gpu_vram_mb"] >= 8000:
            tips.append(
                Tip(
                    "low",
                    f"افزایش Imgsz تا {sug_imgsz}",
                    "اگر VRAM کافی باشد، رزولوشن بالاتر جزئیات ماهواره‌ای را بهتر می‌گیرد.",
                    apply={"imgsz": sug_imgsz},
                )
            )
            suggested["imgsz"] = sug_imgsz

    if grade == "weak":
        tips.append(
            Tip(
                "high",
                "دقت پایین است — داده و Epoch بیشتر",
                "مقدار mAP50 ضعیف است. ویدیو و برچسب بیشتری بسازید، Augment را روی ۲ یا ۳ بگذارید "
                f"و Epochs را حداقل تا {max(epochs + 40, sug_epochs)} افزایش دهید.",
                apply={"epochs": max(epochs + 40, sug_epochs)},
            )
        )
        suggested["epochs"] = max(epochs + 40, sug_epochs)
    elif grade == "fair":
        more = max(epochs + 20, sug_epochs)
        tips.append(
            Tip(
                "medium",
                "نتیجه متوسط — آموزش را ادامه دهید",
                f"mAP50 قابل قبول است، ولی جا برای بهبود دارد. Epochs را به حدود {more} برسانید "
                "و کیفیت برچسب نقاط عطف و هدف را بررسی کنید.",
                apply={"epochs": more},
            )
        )
        suggested["epochs"] = more
    elif grade in ("good", "excellent"):
        tips.append(
            Tip(
                "low",
                "نتیجه خوب است",
                "متریک‌ها مناسب‌اند. با Live Test روی ویدیوی واقعی بررسی کنید؛ "
                "اگر جابه‌جایی زیاد دیدید، داده‌های سخت‌تر (زاویه یا زوم متفاوت) اضافه کنید.",
            )
        )

    prec_b = metrics_raw.get("metrics/precision(B)") or metrics_raw.get("metrics/precision(M)")
    rec_b = metrics_raw.get("metrics/recall(B)") or metrics_raw.get("metrics/recall(M)")
    if prec_b is not None and rec_b is not None:
        if prec_b > 0.7 and rec_b < 0.4:
            tips.append(
                Tip(
                    "medium",
                    "پایین بودن Recall",
                    "مدل محتاط است و هدف را کمتر پیدا می‌کند. نمونه‌های مثبت بیشتر، "
                    "Augment بالاتر، یا Epochs بیشتر معمولاً کمک می‌کند.",
                )
            )
        elif rec_b > 0.7 and prec_b < 0.4:
            tips.append(
                Tip(
                    "medium",
                    "پایین بودن Precision",
                    "مدل بیش از حد تشخیص اشتباه می‌دهد. ویدیوهای منفی (بدون هدف) بیشتری بسازید "
                    "و برچسب‌های نادرست را اصلاح کنید.",
                )
            )

    if still_improving and grade not in ("excellent",):
        more = epochs + 30
        tips.append(
            Tip(
                "medium",
                "هنوز در حال پیشرفت است",
                "منحنی mAP در آخرین epochها هنوز صعودی است. با Epochs بیشتر دوباره آموزش دهید.",
                apply={"epochs": more},
            )
        )
        suggested["epochs"] = max(suggested.get("epochs", 0), more)

    counts = dataset.get("counts") or {}
    n_train = int(counts.get("train") or 0)
    if n_train and n_train < 80:
        tips.append(
            Tip(
                "high",
                "دیتاست کوچک است",
                f"تنها حدود {n_train} نمونهٔ train دارید. از Build Test Videos و Augment بیشتر استفاده کنید "
                "تا دست‌کم چند صد نمونه داشته باشید.",
            )
        )

    if hw.get("ram_gb") is not None and hw["ram_gb"] < 8 and batch > 4:
        tips.append(
            Tip(
                "medium",
                "رم سیستم کم است",
                f"حدود {hw['ram_gb']} گیگابایت RAM دارید؛ batch را کم کنید تا سیستم پایدار بماند.",
                apply={"batch": min(batch, 2)},
            )
        )
        suggested["batch"] = min(suggested.get("batch", batch), 2)

    if not tips:
        tips.append(
            Tip(
                "low",
                "تنظیمات فعلی مناسب به نظر می‌رسد",
                "تغییر اجباری نیست؛ با Live Test کیفیت واقعی را بسنجید.",
            )
        )
    return tips, suggested


def build_train_advice(
    train: Optional[Dict[str, Any]] = None,
    *,
    runs_dir: Optional[str | Path] = None,
    hardware: Optional[Dict[str, Any]] = None,
) -> TrainAdviceReport:
    train = dict(train or {})
    if not train and runs_dir:
        summary_path = Path(runs_dir) / "train_summary.json"
        if summary_path.exists():
            train = load_json(summary_path)

    results_dir = Path(train["results_dir"]) if train.get("results_dir") else None
    if results_dir is None and runs_dir:
        cand = Path(runs_dir) / "train"
        if cand.is_dir():
            results_dir = cand

    hw = hardware or probe_hardware()
    settings = settings_from(train, results_dir)
    csv_rows, still_improving = read_results_csv(results_dir) if results_dir else ([], None)
    metrics_raw = extract_metrics(train, csv_rows)
    metrics = [(METRIC_LABELS[k], metrics_raw[k]) for k in METRIC_LABELS if k in metrics_raw]
    map50 = primary_map(metrics_raw)
    grade = _grade(map50)
    dataset = dataset_info(results_dir, settings)
    tips, suggested = _collect_tips(
        grade=grade,
        hw=hw,
        settings=settings,
        metrics_raw=metrics_raw,
        dataset=dataset,
        train=train,
        still_improving=still_improving,
    )
    batch = int(settings.get("batch") or 4)
    imgsz = int(settings.get("imgsz") or 640)
    device = str(settings.get("device") or train.get("device") or "?")
    map_txt = f"{map50:.3f}" if map50 is not None else "—"
    headline = (
        f"رتبه: {GRADE_FA[grade]} · mAP50={map_txt} · "
        f"دستگاه={device} · epochs={settings.get('epochs', '?')} · "
        f"batch={batch} · imgsz={imgsz}"
    )
    return TrainAdviceReport(
        grade=grade,
        grade_fa=GRADE_FA[grade],
        headline=headline,
        metrics=metrics,
        hardware=hw,
        settings=settings,
        dataset=dataset,
        tips=tips,
        files=list_result_files(train, results_dir),
        suggested=suggested,
        still_improving=still_improving,
    )
