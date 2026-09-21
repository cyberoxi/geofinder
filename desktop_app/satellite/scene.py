"""Scene construction: exact target polygon + automatic landmark discovery on the mosaic."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from desktop_app.satellite.tiles import TILE_SIZE, lonlat_to_tile
from shared.landmarks import SceneLayout
from shared.logging import get_logger

logger = get_logger("desktop.satellite.scene")

Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1 (mosaic px)


def geo_to_mosaic_px(
    lat: float,
    lon: float,
    fetch_bbox: Tuple[float, float, float, float],
    zoom: int,
) -> Tuple[float, float]:
    """Map lat/lon into pixels of a mosaic returned by ``fetch_mosaic`` for ``fetch_bbox``."""
    s, w, n, e = fetch_bbox
    ox, oy = lonlat_to_tile(w, n, zoom)
    # fetch_mosaic crops at int() of the fractional pixel origin
    tx0, ty0 = math.floor(ox), math.floor(oy)
    px0 = int((ox - tx0) * TILE_SIZE)
    py0 = int((oy - ty0) * TILE_SIZE)
    x, y = lonlat_to_tile(lon, lat, zoom)
    return (x - tx0) * TILE_SIZE - px0, (y - ty0) * TILE_SIZE - py0


def selection_rect_px(
    selection: Tuple[float, float, float, float],
    fetch_bbox: Tuple[float, float, float, float],
    zoom: int,
    mosaic_shape: Tuple[int, ...],
) -> Rect:
    s, w, n, e = selection
    x0, y0 = geo_to_mosaic_px(n, w, fetch_bbox, zoom)
    x1, y1 = geo_to_mosaic_px(s, e, fetch_bbox, zoom)
    H, W = mosaic_shape[:2]
    x0, x1 = max(0.0, min(x0, x1)), min(float(W), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(H), max(y0, y1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("Selected region is too small at this download zoom")
    return x0, y0, x1, y1


def meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2.0**zoom)


def _rect_poly(r: Rect) -> List[List[float]]:
    x0, y0, x1, y1 = r
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _rects_overlap(a: Rect, b: Rect, pad: float = 0.0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _window_features(bgr: np.ndarray, hsv: np.ndarray, lab: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Appearance descriptor of a (downsampled) window — colour, texture, material cues."""
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    edges = cv2.Canny(gray, 60, 160)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    green = ((h > 30) & (h < 90) & (sat > 40)).mean()
    red = (((h < 12) | (h > 165)) & (sat > 70)).mean()
    blue = ((h > 90) & (h < 130) & (sat > 50)).mean()
    bright = (val > 215).mean()
    dark = (val < 45).mean()
    hist = cv2.calcHist([hsv], [0, 1], None, [6, 3], [0, 180, 0, 256]).ravel()
    hist = hist / max(1.0, hist.sum())
    return np.concatenate(
        [
            [L.mean(), A.mean(), B.mean(), L.std(), A.std(), B.std()],
            [(edges > 0).mean() * 100, mag.mean() / 4.0],
            np.array([green, red, blue, bright, dark]) * 100,
            hist * 30,
        ]
    ).astype(np.float32)


def discover_landmarks(
    mosaic: np.ndarray,
    target: Rect,
    k: int = 4,
    size_factors: Tuple[float, ...] = (0.8, 1.5, 2.5),
    max_radius_factor: float = 6.0,
    work_px: int = 40,
) -> List[Rect]:
    """
    Find ``k`` distinctive, non-overlapping patches around the target.

    Score = rarity (how unlike this patch is from every other window of the same
    size in the whole mosaic — dense residential blocks are common, a stadium or
    a green-roofed complex is rare) + local uniqueness (no look-alike nearby by
    NCC) + structure (enough texture to be detectable).  Picks are spread
    around the target so several are visible at every zoom level.
    """
    if k <= 0:
        return []
    H, W = mosaic.shape[:2]
    tx0, ty0, tx1, ty1 = target
    tcx, tcy = (tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0
    T = max(24.0, math.sqrt((tx1 - tx0) * (ty1 - ty0)))
    gray_full = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=200, fastThreshold=12)

    scored = []  # (score, rect)
    for sf in size_factors:
        S = max(32.0, T * sf)
        if S * 2 > min(W, H):
            continue
        # Downsample once so each window becomes ~work_px wide
        ds = work_px / S
        small = cv2.resize(mosaic, (max(1, int(W * ds)), max(1, int(H * ds))), interpolation=cv2.INTER_AREA)
        hsv_s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        lab_s = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        step = S * 0.5
        feats, rects, is_cand, stds = [], [], [], []
        for cy in np.arange(S / 2, H - S / 2 + 1e-3, step):
            for cx in np.arange(S / 2, W - S / 2 + 1e-3, step):
                rect = (cx - S / 2, cy - S / 2, cx + S / 2, cy + S / 2)
                a0, b0 = int(rect[0] * ds), int(rect[1] * ds)
                sl = (slice(b0, b0 + work_px), slice(a0, a0 + work_px))
                g = gray_s[sl]
                if g.shape[0] < work_px // 2 or g.shape[1] < work_px // 2:
                    continue
                feats.append(_window_features(small[sl], hsv_s[sl], lab_s[sl], g))
                rects.append(rect)
                stds.append(float(g.std()))
                d = math.hypot(cx - tcx, cy - tcy)
                is_cand.append(
                    T * 0.5 + S * 0.6 <= d <= T * max_radius_factor
                    and not _rects_overlap(rect, target, pad=T * 0.1)
                )
        if not any(is_cand) or len(feats) < 4:
            continue
        F = np.asarray(feats, np.float32)
        F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-3)
        centers = np.array([((r[0] + r[2]) / 2, (r[1] + r[3]) / 2) for r in rects], np.float32)
        cand_idx = [i for i, c in enumerate(is_cand) if c and stds[i] >= 10.0]
        rarity = {}
        for i in cand_idx:
            dist = np.linalg.norm(F - F[i], axis=1)
            near = np.linalg.norm(centers - centers[i], axis=1) < S * 0.9  # overlapping windows
            dist[near] = np.inf
            nn = np.sort(dist)[: min(6, int(np.isfinite(dist).sum()))]
            rarity[i] = float(nn.mean()) if len(nn) else 0.0
        if not rarity:
            continue
        r_vals = np.array(list(rarity.values()))
        r_norm = max(1e-6, float(np.percentile(r_vals, 97)))
        # Pre-select by rarity, then check local look-alikes with NCC
        pre = sorted(rarity.items(), key=lambda kv: -kv[1])[: max(30, k * 8)]
        for i, rar in pre:
            rect = rects[i]
            x0, y0, x1, y1 = (int(round(v)) for v in rect)
            sx0, sy0 = max(0, int(x0 - 2.5 * S)), max(0, int(y0 - 2.5 * S))
            sx1, sy1 = min(W, int(x1 + 2.5 * S)), min(H, int(y1 + 2.5 * S))
            region = cv2.resize(
                gray_full[sy0:sy1, sx0:sx1],
                (max(work_px + 1, int((sx1 - sx0) * ds)), max(work_px + 1, int((sy1 - sy0) * ds))),
                interpolation=cv2.INTER_AREA,
            )
            tmpl = cv2.resize(gray_full[y0:y1, x0:x1], (work_px, work_px), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
            px, py = int((x0 - sx0) * ds), int((y0 - sy0) * ds)
            r = int(work_px * 0.6)
            res[max(0, py - r) : py + r + 1, max(0, px - r) : px + r + 1] = -1.0
            uniqueness = max(0.0, 1.0 - max(0.0, float(res.max()) if res.size else 0.0))
            kp = orb.detect(cv2.resize(tmpl, (work_px * 2, work_px * 2)), None)
            structure = 0.5 * min(1.0, stds[i] / 45.0) + 0.5 * min(1.0, len(kp) / 40.0)
            score = 0.55 * min(1.0, rar / r_norm) + 0.25 * uniqueness + 0.20 * structure
            scored.append((score, rect))

    if not scored:
        logger.warning("No landmark candidates found around target")
        return []

    scored.sort(key=lambda x: -x[0])
    chosen: List[Rect] = []
    chosen_angles: List[float] = []
    for strict in (True, False):
        for score, rect in scored:
            if len(chosen) >= k:
                break
            if rect in chosen or any(_rects_overlap(rect, c, pad=T * 0.2) for c in chosen):
                continue
            cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
            ang = math.atan2(cy - tcy, cx - tcx)
            # Encourage angular spread around the target
            if strict and any(
                abs((ang - a + math.pi) % (2 * math.pi) - math.pi) < math.radians(40) for a in chosen_angles
            ):
                continue
            chosen.append(rect)
            chosen_angles.append(ang)
            logger.info("Landmark %d: rect=%s score=%.3f", len(chosen), tuple(round(v) for v in rect), score)
    return chosen


def build_scene_layout(
    mosaic: np.ndarray,
    target: Rect,
    *,
    num_landmarks: int = 4,
    target_class: str = "target_region",
    meters_per_px: float = 0.0,
    meta: Optional[dict] = None,
) -> SceneLayout:
    landmarks = discover_landmarks(mosaic, target, k=num_landmarks)
    classes = [target_class] + [f"landmark_{i + 1}" for i in range(len(landmarks))]
    polygons = [_rect_poly(target)] + [_rect_poly(r) for r in landmarks]
    H, W = mosaic.shape[:2]
    return SceneLayout(
        classes=classes,
        polygons=polygons,
        mosaic_size=(W, H),
        meters_per_px=meters_per_px,
        meta=dict(meta or {}),
    )


def draw_scene_preview(mosaic: np.ndarray, layout: SceneLayout, max_side: int = 1400) -> np.ndarray:
    """Mosaic thumbnail with target (green) and landmarks (orange) drawn."""
    H, W = mosaic.shape[:2]
    s = min(1.0, max_side / float(max(W, H)))
    img = cv2.resize(mosaic, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else mosaic.copy()
    for cid, name in enumerate(layout.classes):
        pts = np.round(layout.polygon(cid) * s).astype(np.int32)
        color = (0, 220, 0) if cid == 0 else (0, 160, 255)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        cv2.putText(img, name, (int(x), max(12, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img
