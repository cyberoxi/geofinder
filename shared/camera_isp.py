"""Jetson Argus (nvarguscamerasrc) image settings for CSI cameras such as the IMX477.

The settings are baked into the GStreamer pipeline, so changing them re-opens the camera.
``enabled = False`` keeps the plain pipeline (Argus defaults).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

WB_MODES = {
    0: "off", 1: "auto", 2: "incandescent", 3: "fluorescent", 4: "warm-fluorescent",
    5: "daylight", 6: "cloudy-daylight", 7: "twilight", 8: "shade",
}
NR_MODES = {0: "off", 1: "fast", 2: "high quality"}

DEFAULT_ISP: Dict[str, Any] = {
    "enabled": False,
    "wbmode": 1,
    "awblock": False,
    "saturation": 1.0,  # 0 … 2
    "exposure_auto": True,
    "exposure_min_us": 50,  # used when exposure_auto is False
    "exposure_max_us": 4000,  # short max exposure → less motion blur from a moving platform
    "gain_min": 1.0,
    "gain_max": 16.0,
    "exposure_comp": 0.0,  # −2 … +2 EV
    "aelock": False,
    "tnr_mode": 1,  # temporal noise reduction
    "tnr_strength": -1.0,  # −1 = sensor default, else 0 … 1
    "ee_mode": 1,  # edge enhancement (sharpening)
    "ee_strength": -1.0,
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "Default / پیش‌فرض": {},
    "Fast motion — short exposure / حرکت سریع": {
        "exposure_auto": False, "exposure_max_us": 2000, "gain_max": 22.0, "tnr_mode": 1, "ee_mode": 1,
    },
    "Bright sun / آفتاب شدید": {
        "wbmode": 5, "exposure_auto": False, "exposure_max_us": 1000, "gain_max": 4.0, "exposure_comp": -0.5,
    },
    "Cloudy / ابری": {"wbmode": 6, "exposure_comp": 0.3},
    "Low light — less noise / نور کم": {
        "exposure_auto": True, "gain_max": 16.0, "tnr_mode": 2, "tnr_strength": 0.6, "ee_mode": 0,
    },
    "Sharper detail / جزئیات بیشتر": {"ee_mode": 2, "ee_strength": 0.4, "tnr_mode": 1, "tnr_strength": 0.2},
}


def merge_isp(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(DEFAULT_ISP)
    for k, v in (cfg or {}).items():
        if k in out:
            out[k] = type(out[k])(v)
    return out


def argus_properties(cfg: Optional[Dict[str, Any]]) -> str:
    """Extra ``nvarguscamerasrc`` properties (leading space) or "" when disabled."""
    c = merge_isp(cfg)
    if not c["enabled"]:
        return ""
    props = [
        f"wbmode={int(c['wbmode'])}",
        f"awblock={'true' if c['awblock'] else 'false'}",
        f"saturation={max(0.0, min(2.0, c['saturation'])):.2f}",
        f"exposurecompensation={max(-2.0, min(2.0, c['exposure_comp'])):.2f}",
        f"aelock={'true' if c['aelock'] else 'false'}",
        f"gainrange=\"{max(1.0, c['gain_min']):.1f} {max(c['gain_min'], c['gain_max']):.1f}\"",
        f"tnr-mode={int(c['tnr_mode'])}",
        f"ee-mode={int(c['ee_mode'])}",
    ]
    if not c["exposure_auto"]:
        lo = max(13, int(c["exposure_min_us"])) * 1000  # ns
        hi = max(lo // 1000 + 1, int(c["exposure_max_us"])) * 1000
        props.append(f"exposuretimerange=\"{lo} {hi}\"")
    if c["tnr_strength"] >= 0:
        props.append(f"tnr-strength={min(1.0, c['tnr_strength']):.2f}")
    if c["ee_strength"] >= 0:
        props.append(f"ee-strength={min(1.0, c['ee_strength']):.2f}")
    return " " + " ".join(props)


def describe_isp(cfg: Optional[Dict[str, Any]]) -> str:
    c = merge_isp(cfg)
    if not c["enabled"]:
        return "Camera ISP: Argus defaults"
    exp = "auto" if c["exposure_auto"] else f"≤{c['exposure_max_us']} µs"
    return (
        f"Camera ISP: WB {WB_MODES.get(c['wbmode'], c['wbmode'])}, exposure {exp}, gain ≤{c['gain_max']:g}, "
        f"NR {NR_MODES.get(c['tnr_mode'], c['tnr_mode'])}, sharpen {NR_MODES.get(c['ee_mode'], c['ee_mode'])}"
    )
