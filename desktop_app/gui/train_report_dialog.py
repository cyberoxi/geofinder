"""Dialog: full training results + hardware-aware improvement tips."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from desktop_app.training.advice import GRADE_COLOR, TrainAdviceReport, build_train_advice
from shared.qt_compat import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    Qt,
    qt_exec,
)
_PRIO_COLOR = {"high": "#ef4444", "medium": "#eab308", "low": "#64748b"}
_PRIO_FA = {"high": "مهم", "medium": "متوسط", "low": "کم"}

# Unicode BiDi: force RTL paragraph even when the first strong char is Latin
# (e.g. "Batch را افزایش دهید" / "PyTorch با …").
_RLI, _PDI, _RLM = "\u2067", "\u2069", "\u200f"


def _force_rtl_text(text: str) -> str:
    if not text:
        return text
    # Already wrapped / marked
    if text.startswith(_RLI) or text.startswith(_RLM):
        return text
    return f"{_RLI}{text}{_PDI}"


def _rtl_label(text: str, *, style: str = "", wrap: bool = True) -> QLabel:
    lbl = QLabel(_force_rtl_text(text))
    lbl.setLayoutDirection(Qt.RightToLeft)
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter if not wrap else Qt.AlignRight | Qt.AlignTop)
    if wrap:
        lbl.setWordWrap(True)
    if style:
        lbl.setStyleSheet(style)
    return lbl


def _open_path(path: str) -> None:
    p = Path(path)
    target = str(p if p.is_dir() else p.parent)
    try:
        if sys.platform.startswith("win"):
            os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except Exception:  # noqa: BLE001
        pass


class TrainReportDialog(QDialog):
    """Show metrics, PC specs, and actionable tips after training."""

    def __init__(self, report: TrainAdviceReport, parent=None, title: str = "گزارش آموزش / Training Report"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setLayoutDirection(Qt.RightToLeft)
        self.setMinimumSize(720, 560)
        self.resize(820, 640)
        self.report = report
        self.applied_settings: Optional[Dict[str, int]] = None

        root = QVBoxLayout(self)
        root.setSpacing(10)

        grade = _rtl_label(
            f"{report.grade_fa}  ·  {report.grade.upper()}",
            style=(
                f"font-size:20px;font-weight:700;"
                f"color:{GRADE_COLOR.get(report.grade, '#94a3b8')};"
            ),
            wrap=False,
        )
        root.addWidget(grade)
        root.addWidget(_rtl_label(report.headline, style="color:#94a3b8;"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        no_frame = getattr(QFrame, "NoFrame", None) or QFrame.Shape.NoFrame
        scroll.setFrameShape(no_frame)
        body = QWidget()
        body.setLayoutDirection(Qt.RightToLeft)
        bv = QVBoxLayout(body)
        bv.setSpacing(12)

        bv.addWidget(self._section("نتایج / Results", self._metrics_block(report)))
        bv.addWidget(self._section("مشخصات این کامپیوتر / This PC", self._hw_block(report)))
        if report.dataset.get("counts"):
            bv.addWidget(self._section("دیتاست / Dataset", self._dataset_block(report)))
        bv.addWidget(self._section("راهنمای بهبود / What to do next", self._tips_block(report)))
        if report.files:
            bv.addWidget(self._section("فایل‌های ذخیره‌شده / Saved files", self._files_block(report)))

        bv.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        btns = QHBoxLayout()
        if report.suggested:
            apply_btn = QPushButton("اعمال تنظیمات پیشنهادی / Apply suggested settings")
            apply_btn.setObjectName("primaryButton")
            apply_btn.setLayoutDirection(Qt.RightToLeft)
            apply_btn.setToolTip(
                "مقادیر پیشنهادی Epochs، Batch و Imgsz را روی پنل Training Settings می‌گذارد"
            )
            apply_btn.clicked.connect(self._apply_suggested)
            btns.addWidget(apply_btn)
        btns.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.Ok)
        box.accepted.connect(self.accept)
        btns.addWidget(box)
        root.addLayout(btns)

    @staticmethod
    def _section(title: str, widget: QWidget) -> QWidget:
        wrap = QFrame()
        wrap.setLayoutDirection(Qt.RightToLeft)
        wrap.setStyleSheet(
            "QFrame{background:#0f172a;border:1px solid #243041;border-radius:10px;}"
        )
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(12, 10, 12, 10)
        t = _rtl_label(title, style="font-weight:600;color:#e2e8f0;border:none;", wrap=False)
        lay.addWidget(t)
        lay.addWidget(widget)
        return wrap

    @staticmethod
    def _metrics_block(report: TrainAdviceReport) -> QWidget:
        text = QPlainTextEdit()
        text.setLayoutDirection(Qt.LeftToRight)
        text.setReadOnly(True)
        text.setMaximumHeight(140)
        lines = []
        if report.metrics:
            for name, val in report.metrics:
                lines.append(f"{name:22s}  {val:.3f}")
        else:
            lines.append("متریکی پیدا نشد / No metrics found")
        if report.still_improving is True:
            lines.append("")
            lines.append("روند: mAP هنوز در حال پیشرفت است / still rising")
        elif report.still_improving is False:
            lines.append("")
            lines.append("روند: منحنی mAP تقریباً ثابت شده / plateaued")
        text.setPlainText("\n".join(lines))
        return text

    @staticmethod
    def _hw_block(report: TrainAdviceReport) -> QLabel:
        hw = report.hardware
        vram = hw.get("gpu_vram_mb")
        vram_s = f"{vram} MB" if vram else "—"
        cuda = "بله" if hw.get("cuda_torch") else "خیر"
        lines = [
            f"هسته‌های CPU: {hw.get('cpu_cores') or '—'}",
            f"رم: {hw.get('ram_gb') or '—'} GB",
            f"GPU: {hw.get('gpu_name') or 'ندارد'}",
            f"VRAM: {vram_s}",
            f"PyTorch با CUDA: {cuda}",
            f"دستگاه آموزش: {report.settings.get('device') or '—'}",
        ]
        lbl = _rtl_label(
            "\n".join(lines),
            style="color:#cbd5e1;border:none;font-family:Consolas,monospace;",
        )
        return lbl

    @staticmethod
    def _dataset_block(report: TrainAdviceReport) -> QLabel:
        counts = report.dataset.get("counts") or {}
        parts = [f"{k}={v}" for k, v in counts.items()]
        n_cls = len(report.dataset.get("classes") or [])
        return _rtl_label(
            f"نمونه‌ها: {', '.join(parts) or '—'}   کلاس‌ها: {n_cls or '—'}",
            style="color:#cbd5e1;border:none;",
        )

    def _tips_block(self, report: TrainAdviceReport) -> QWidget:
        wrap = QWidget()
        wrap.setLayoutDirection(Qt.RightToLeft)
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        for tip in report.tips:
            color = _PRIO_COLOR.get(tip.priority, "#94a3b8")
            prio = _PRIO_FA.get(tip.priority, tip.priority)
            title = _rtl_label(
                f"[{prio}]  {tip.title}",
                style=f"color:{color};font-weight:600;border:none;",
                wrap=True,
            )
            detail = _rtl_label(
                tip.detail,
                style="color:#cbd5e1;border:none;",
                wrap=True,
            )
            lay.addWidget(title)
            lay.addWidget(detail)
        return wrap

    def _files_block(self, report: TrainAdviceReport) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        for label, path in report.files:
            row = QHBoxLayout()
            name = QLabel(f"{label}")
            name.setStyleSheet("color:#e2e8f0;border:none;")
            row.addWidget(name)
            row.addStretch(1)
            btn = QPushButton("باز کردن پوشه")
            btn.setFixedWidth(120)
            btn.setLayoutDirection(Qt.RightToLeft)
            btn.clicked.connect(lambda _=False, p=path: _open_path(p))
            row.addWidget(btn)
            lay.addLayout(row)
            pth = QLabel(path)
            pth.setStyleSheet("color:#64748b;border:none;font-size:11px;")
            pth.setWordWrap(True)
            lay.addWidget(pth)
        return wrap

    def _apply_suggested(self) -> None:
        self.applied_settings = dict(self.report.suggested)
        self.accept()


def show_train_report(
    parent,
    train: Optional[Dict[str, Any]] = None,
    *,
    runs_dir: Optional[str | Path] = None,
    title: str = "گزارش آموزش / Training Report",
) -> Optional[Dict[str, int]]:
    """Build advice and show dialog. Returns suggested settings if user applied them."""
    report = build_train_advice(train, runs_dir=runs_dir)
    dlg = TrainReportDialog(report, parent=parent, title=title)
    qt_exec(dlg)
    return dlg.applied_settings
