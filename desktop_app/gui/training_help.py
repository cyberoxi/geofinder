"""Persian RTL help for Training Settings panel items."""

from __future__ import annotations

import html
from typing import Dict

from shared.qt_compat import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
    Qt,
    qt_exec,
)

HELP: Dict[str, Dict[str, str]] = {
    "model_type": {
        "title": "نوع مدل (Model type)",
        "body": (
            "تعیین می‌کند مدل YOLO چه چیزی یاد بگیرد.\n\n"
            "• segmentation (قطعه‌بندی): علاوه بر محل هدف، شکل/ماسک ناحیه را هم یاد می‌گیرد. "
            "برای مناطق جغرافیایی با مرز مشخص معمولاً دقیق‌تر است و در ردیابی روی Jetson "
            "ماسک غنی‌تری می‌دهد.\n\n"
            "• detection (تشخیص): فقط جعبهٔ محدودکننده (bounding box) را پیش‌بینی می‌کند. "
            "سبک‌تر و گاهی سریع‌تر آموزش می‌بیند، ولی شکل دقیق هدف را برنمی‌گرداند.\n\n"
            "این انتخاب باید با دیتاست و بستهٔ نهایی Jetson یکی باشد؛ بعد از تغییر نوع مدل، "
            "دیتاست و آموزش را دوباره بسازید."
        ),
    },
    "every_n": {
        "title": "هر N فریم (Every N)",
        "body": (
            "هنگام ساخت دیتاست، از حاشیه‌نویسی‌های میانی (درون‌یابی‌شده بین کلیدفریم‌ها) "
            "فقط هر N فریم یک نمونه برداشته می‌شود.\n\n"
            "• مقدار ۱: تقریباً همهٔ فریم‌های برچسب‌دار وارد دیتاست می‌شوند "
            "(داده بیشتر، زمان ساخت و آموزش بیشتر، احتمال تکرار زیاد).\n\n"
            "• مقدار بالاتر (مثلاً ۲ تا ۵): نمونه‌ها خلوت‌تر می‌شوند؛ برای ویدیوهای بلند "
            "یا وقتی فریم‌های پشت‌سرهم خیلی شبیه‌اند مفید است.\n\n"
            "این تنظیم روی «ساخت دیتاست» اثر دارد، نه مستقیم روی حلقهٔ آموزش YOLO."
        ),
    },
    "aug_n": {
        "title": "تقویت به‌ازای هر تصویر (Augment / image)",
        "body": (
            "برای هر تصویر آموزشی، چند نسخهٔ تقویت‌شده (augmentation) اضافه شود؛ "
            "مثلاً چرخش جزئی، تغییر روشنایی/کنتراست، برش یا نویز ملایم.\n\n"
            "• ۰: بدون تقویت مصنوعی — فقط همان تصاویر اصلی.\n\n"
            "• ۱ تا ۳: معمولاً برای دیتاست کوچک ماهواره‌ای مفید است و به تعمیم بهتر "
            "(زاویه، نور، زوم متفاوت) کمک می‌کند.\n\n"
            "مقدار خیلی بالا دیتاست را باد می‌کند و ممکن است نویز اضافه کند. "
            "این مقدار هنگام ساخت دیتاست اعمال می‌شود."
        ),
    },
    "epochs": {
        "title": "تعداد دور آموزش (Epochs)",
        "body": (
            "یک Epoch یعنی یک بار عبور کامل از کل مجموعهٔ آموزشی.\n\n"
            "تعداد بیشتر معمولاً دقت را تا جایی بالا می‌برد، بعد از آن منحنی mAP ثابت "
            "می‌شود یا بیش‌برازش (overfit) رخ می‌دهد.\n\n"
            "پیشنهاد عملی: با مقدار متوسط (مثلاً حدود ۴۰) شروع کنید؛ بعد از Training Report "
            "اگر mAP هنوز در حال رشد بود Epoch را بیشتر کنید، و اگر ثابت شده بود همان را نگه دارید "
            "یا داده/تقویت را بهتر کنید. آموزش طولانی‌تر زمان بیشتری روی GPU/CPU می‌گیرد."
        ),
    },
    "batch": {
        "title": "اندازهٔ دسته (Batch)",
        "body": (
            "تعداد تصاویری که در هر گام به مدل داده می‌شود.\n\n"
            "• Batch بالاتر: معمولاً گرادیان پایدارتر و استفادهٔ بهتر از GPU، "
            "ولی حافظهٔ ویدئویی (VRAM) بیشتری لازم دارد.\n\n"
            "• Batch خیلی بالا: خطای کمبود حافظه (OOM) و توقف آموزش.\n\n"
            "• Batch خیلی پایین (۱–۲): آموزش کندتر و ناپایدارتر، مخصوصاً روی دیتاست کوچک.\n\n"
            "اگر GPU ندارید یا VRAM کم است، Batch را ۲ تا ۴ نگه دارید. "
            "گزارش آموزش اغلب مقدار پیشنهادی متناسب با کارت شما را پیشنهاد می‌دهد."
        ),
    },
    "imgsz": {
        "title": "اندازهٔ ورودی تصویر (Imgsz)",
        "body": (
            "طول ضلع مربعی که تصاویر قبل از ورود به YOLO به آن تغییر اندازه داده می‌شوند "
            "(مثلاً ۶۴۰ یعنی ۶۴۰×۶۴۰).\n\n"
            "• مقدار بالاتر: جزئیات ماهواره‌ای و اهداف کوچک بهتر دیده می‌شوند، "
            "ولی VRAM و زمان آموزش/استنتاج بیشتر می‌شود.\n\n"
            "• مقدار پایین‌تر: سریع‌تر و سبک‌تر، با ریسک از دست رفتن جزئیات ریز.\n\n"
            "مقادیر معمولاً مضرب ۳۲ هستند (۳۲۰ تا ۱۲۸۰). مهم: اندازهٔ export به ONNX و "
            "ورودی TensorRT روی Jetson باید با همین Imgsz آموزش یکی باشد؛ بعد از تغییر، "
            "دوباره آموزش و بسته‌سازی کنید."
        ),
    },
    "device": {
        "title": "دستگاه آموزش (Train on)",
        "body": (
            "مشخص می‌کند آموزش روی کدام سخت‌افزار اجرا شود.\n\n"
            "• Auto: اگر GPU با CUDA در دسترس باشد از آن استفاده می‌کند، وگرنه CPU.\n\n"
            "• GPU: خیلی سریع‌تر از CPU است؛ برای Epoch و Imgsz بالاتر عملاً ضروری است. "
            "اگر کارت دیده شود ولی کنارش «needs CUDA PyTorch» نوشته شده، باید نسخهٔ "
            "CUDA مربوط به PyTorch نصب شود.\n\n"
            "• CPU: همیشه کار می‌کند ولی بسیار کند است؛ وزن نهایی مدل برای Jetson "
            "از نظر محتوا با آموزش GPU یکسان است — فقط زمان ساخت فرق دارد.\n\n"
            "این گزینه فقط محل اجرای آموزش دسکتاپ را عوض می‌کند، نه سخت‌افزار Jetson در میدان."
        ),
    },
}


def _rtl_html(text: str) -> str:
    """HTML block that keeps Persian help fully right-aligned (RTL)."""
    escaped = html.escape(text or "").replace("\n", "<br>")
    return (
        '<div dir="rtl" style="text-align:right; direction:rtl;'
        f'unicode-bidi:embed;">{escaped}</div>'
    )


def make_setting_label(parent: QWidget, text: str, help_key: str) -> QWidget:
    """Label + compact «?» button for a QFormLayout row."""
    wrap = QWidget(parent)
    lay = QHBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lab = QLabel(text)
    btn = QToolButton(wrap)
    btn.setText("?")
    btn.setCursor(Qt.PointingHandCursor)
    btn.setToolTip("توضیح این تنظیم")
    btn.setFixedSize(22, 22)
    btn.setStyleSheet(
        "QToolButton{color:#94a3b8;background:#1e293b;border:1px solid #334155;"
        "border-radius:11px;font-weight:700;font-size:12px;}"
        "QToolButton:hover{color:#e2e8f0;background:#334155;border-color:#64748b;}"
    )
    btn.clicked.connect(lambda: show_training_help(parent.window() or parent, help_key))
    lay.addWidget(lab)
    lay.addWidget(btn)
    lay.addStretch(1)
    return wrap


def show_training_help(parent: QWidget, key: str) -> None:
    info = HELP.get(key)
    if not info:
        return
    dlg = QDialog(parent)
    dlg.setLayoutDirection(Qt.RightToLeft)
    dlg.setWindowTitle(info["title"])
    dlg.setMinimumWidth(420)
    dlg.setMaximumWidth(520)
    root = QVBoxLayout(dlg)
    root.setContentsMargins(16, 14, 16, 12)
    root.setSpacing(12)

    title = QLabel()
    title.setLayoutDirection(Qt.RightToLeft)
    title.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    title.setWordWrap(True)
    title.setTextFormat(Qt.RichText)
    title.setText(
        '<div dir="rtl" style="text-align:right;font-weight:700;font-size:14px;'
        f'color:#e2e8f0;">{html.escape(info["title"])}</div>'
    )
    root.addWidget(title)

    body = QLabel()
    body.setLayoutDirection(Qt.RightToLeft)
    body.setAlignment(Qt.AlignRight | Qt.AlignTop)
    body.setWordWrap(True)
    body.setTextFormat(Qt.RichText)
    body.setTextInteractionFlags(Qt.TextSelectableByMouse)
    body.setText(_rtl_html(info["body"]))
    body.setStyleSheet(
        "QLabel{color:#cbd5e1;font-size:13px;"
        "background:#0f172a;border:1px solid #243041;border-radius:8px;padding:12px;}"
    )
    root.addWidget(body)

    box = QDialogButtonBox(QDialogButtonBox.Ok)
    box.setLayoutDirection(Qt.RightToLeft)
    box.accepted.connect(dlg.accept)
    root.addWidget(box)
    qt_exec(dlg)
