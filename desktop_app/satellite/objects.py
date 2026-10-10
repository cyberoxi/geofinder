"""Real-object landmark proposals on a satellite mosaic.

Instead of arbitrary square patches, landmarks are actual objects with their
own outline: sports fields / tracks, parks, water, large distinctive roofs,
roundabouts and major road intersections.  Each candidate is scored by how
rare it looks in the whole mosaic (so a unique stadium beats one of a hundred
similar roofs), by a per-kind prior and by shape quality, then selected in
near / mid / far distance bands so the target can be located at every zoom.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from shared.logging import get_logger

logger = get_logger("desktop.satellite.objects")

Rect = Tuple[float, float, float, float]

KIND_PRIOR: Dict[str, float] = {
    "sports_field": 1.0,
    "roundabout": 0.95,
    "water": 0.9,
    "building": 0.75,
    "green_area": 0.7,
    "open_area": 0.6,
    "intersection": 0.55,
    "patch": 0.3,
}


@dataclass(eq=False)  # identity equality (holds numpy arrays)
class ObjectCandidate:
    polygon: np.ndarray  # (N, 2) mosaic px
    kind: str
    score: float = 0.0
    distance: float = 0.0  # px from target centre
    shape: float = 1.0  # 0..1 compactness / regularity
    rarity: float = 0.0
    meta: Dict = field(default_factory=dict)

    @property
    def size(self) -> float:
        return math.sqrt(max(1.0, abs(cv2.contourArea(self.polygon.astype(np.float32)))))

    @property
    def center(self) -> np.ndarray:
        return self.polygon.mean(axis=0)

    def bbox(self) -> Rect:
        x0, y0 = self.polygon.min(axis=0)
        x1, y1 = self.polygon.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    def to_dict(self) -> Dict:
        return {
            "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in self.polygon],
            "kind": self.kind,
            "score": round(float(self.score), 4),
            "distance": round(float(self.distance), 2),
            "shape": round(float(self.shape), 3),
            "rarity": round(float(self.rarity), 3),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ObjectCandidate":
        return cls(
            polygon=np.asarray(d["polygon"], np.float32),
            kind=d.get("kind", "patch"),
            score=float(d.get("score", 0.0)),
            distance=float(d.get("distance", 0.0)),
            shape=float(d.get("shape", 1.0)),
            rarity=float(d.get("rarity", 0.0)),
        )


# ------------------------------------------------------------------ helpers


def _bbox_iou(a: Rect, b: Rect) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _overlaps(a: Rect, b: Rect, pad: float = 0.0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _clean(mask: np.ndarray, k: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)


def _shape_stats(contour: np.ndarray) -> Tuple[float, float, float]:
    area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) or 1.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    rect_area = max(1.0, rw * rh)
    return area, area / hull_area, area / rect_area


def _simplify(contour: np.ndarray, frac: float = 0.012, max_pts: int = 24) -> np.ndarray:
    peri = cv2.arcLength(contour, True)
    eps = frac * peri
    approx = cv2.approxPolyDP(contour, eps, True)
    while len(approx) > max_pts:
        eps *= 1.5
        approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 4:
        approx = cv2.boxPoints(cv2.minAreaRect(contour)).reshape(-1, 1, 2)
    return approx.reshape(-1, 2).astype(np.float32)


def _zhang_suen(img: np.ndarray, max_iter: int = 60) -> np.ndarray:
    """Binary thinning (1-px skeleton), vectorised."""
    sk = (img > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(sk, 1)
            P2, P3, P4 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, :-2]
            P8, P9 = p[1:-1, :-2], p[:-2, :-2]
            nb = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(n.astype(np.int16) for n in nb)
            seq = nb + [P2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                c = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rm = (sk == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                sk[rm] = 0
                changed = True
        if not changed:
            break
    return sk


def _window_features(bgr: np.ndarray) -> np.ndarray:
    from desktop_app.satellite.scene import _window_features as wf

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return wf(bgr, hsv, lab, gray)


# ------------------------------------------------------------------ proposals


def _odd(n: float, lo: int = 3) -> int:
    n = max(lo, int(round(n)))
    return n if n % 2 else n + 1


class _SceneStats:
    """Per-mosaic statistics so thresholds adapt to the imagery (tint, season, sensor)."""

    def __init__(self, small: np.ndarray, mpp: float):
        self.smooth = cv2.bilateralFilter(small, 7, 40, 7)
        self.lab = cv2.cvtColor(self.smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.hsv = cv2.cvtColor(self.smooth, cv2.COLOR_BGR2HSV)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = _odd(12.0 / mpp, 5)  # ~12 m window
        mu = cv2.blur(g, (k, k))
        self.texture = np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mu * mu, 0))
        self.median_lab = np.median(self.lab.reshape(-1, 3), axis=0)
        self.median_tex = float(np.median(self.texture))
        self.L = self.lab[..., 0]
        self.dark = self.L < (self.median_lab[0] - 45)  # asphalt: clearly darker than typical ground
        self.mpp = mpp


def _flat_proposals(st: _SceneStats, px_area: Tuple[float, float]) -> List[ObjectCandidate]:
    """Large flat, compact, uniformly coloured regions: fields, plazas, parking, big flat roofs, water."""
    thr = float(np.clip(0.4 * st.median_tex, 6.0, 20.0))
    flat = (st.texture < thr).astype(np.uint8)
    # Line markings on pitches / roof fixtures: close small gaps inside flat surfaces
    flat = cv2.morphologyEx(flat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Colour clusters (k-means on a/b + L) → components of (flat ∧ same colour)
    lab_small = st.lab
    H, W = flat.shape
    sample = lab_small.reshape(-1, 3)[:: max(1, (H * W) // 60000)]
    k = 10
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(sample.astype(np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    d = ((lab_small.reshape(-1, 1, 3) - centers.reshape(1, k, 3)) ** 2).sum(-1)
    labels = d.argmin(1).reshape(H, W)

    k_open = _odd(8.0 / st.mpp, 3)
    se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(6.0 / st.mpp, 3), _odd(6.0 / st.mpp, 3)))
    a_min, a_max = px_area
    out = []
    for ci in range(k):
        m = ((labels == ci) & (flat > 0)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se_close)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se_open)  # cut thin links (roads, alleys)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area, solidity, rect = _shape_stats(c)
            if area < a_min or area > a_max or solidity < 0.78 or rect < 0.6:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            if max(rw, rh) / max(1.0, min(rw, rh)) > 4.0:
                continue  # road segment / canal
            region = np.zeros(m.shape, np.uint8)
            cv2.drawContours(region, [c], -1, 1, -1)
            mean = lab_small[region > 0].mean(axis=0)
            dL, da, db = mean - st.median_lab
            chroma = math.hypot(da, db)
            size_m = math.sqrt(area) * st.mpp
            if da < -6:
                kind = "sports_field" if rect > 0.7 else "green_area"
            elif dL > 35:
                kind = "building"  # bright flat roof
            elif dL < -30 and db < -10:
                kind = "water"
            elif dL < -35:
                continue  # dark & neutral: shadow / asphalt, not distinctive
            elif chroma > 22:
                kind = "sports_field" if (rect > 0.7 and size_m > 40) else "building"
            elif rect > 0.75 and size_m > 60:
                kind = "sports_field"  # large regular flat ground (pitch, track, dirt field)
            else:
                kind = "open_area"
            shape = 0.5 * solidity + 0.5 * rect
            out.append(
                ObjectCandidate(polygon=_simplify(c), kind=kind, shape=float(shape), meta={"dL": float(dL), "da": float(da)})
            )
    return out


def _classify_region(st: _SceneStats, region: np.ndarray, area: float, solidity: float, rect: float, circ: float) -> Optional[str]:
    """Object kind from colour / texture / shape relative to the scene statistics."""
    lab = st.lab[region > 0]
    if lab.size == 0:
        return None
    dL, da, db = lab.mean(axis=0) - st.median_lab
    chroma = math.hypot(da, db)
    tex = float(st.texture[region > 0].mean())
    flat = tex < max(8.0, 0.55 * st.median_tex)
    size_m = math.sqrt(area) * st.mpp
    if not flat and size_m > 250:
        return None  # a textured region this big is a city block / mask leak, not one object
    if circ > 0.85 and 20 <= size_m <= 200 and not flat:
        return "roundabout"
    if da < -6:
        if flat and rect > 0.7:
            return "sports_field"
        return "green_area" if size_m >= 45 else None
    if dL < -30 and db < -10 and flat:
        return "water" if size_m >= 40 else None  # small dark blobs are shadows
    if dL < -35:
        return None  # shadow / asphalt
    if flat and rect > 0.72 and size_m > 55:
        return "sports_field"
    if dL > 30 or chroma > 22:
        return "building"
    if rect > 0.7 and size_m > 35:
        return "building" if not flat else "open_area"
    return "open_area" if flat else None


_SAM_CACHE: Dict[str, object] = {}


def sam_available(weights: str = "FastSAM-s.pt") -> bool:
    from pathlib import Path as _P

    from shared.runtime_paths import find_bundled_file

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    if find_bundled_file(weights) is not None:
        return True
    return _P(weights).exists() or (_P(__file__).resolve().parents[2] / weights).exists()


def _sam_proposals(
    st: _SceneStats,
    small: np.ndarray,
    px_area: Tuple[float, float],
    weights: str = "FastSAM-s.pt",
    tile: int = 1024,
    overlap: int = 256,
    device: str = "cpu",
) -> List[ObjectCandidate]:
    """Segment-everything (FastSAM) on overlapping tiles → real object outlines."""
    from pathlib import Path as _P

    from shared.runtime_paths import find_bundled_file
    from ultralytics import FastSAM

    w = find_bundled_file(weights) or _P(weights)
    if not w.exists():
        w = _P(__file__).resolve().parents[2] / weights
    key = str(w)
    if key not in _SAM_CACHE:
        _SAM_CACHE[key] = FastSAM(str(w))
    model = _SAM_CACHE[key]

    H, W = small.shape[:2]
    a_min, a_max = px_area
    step = max(256, tile - overlap)
    xs = list(range(0, max(1, W - tile), step)) + [max(0, W - tile)]
    ys = list(range(0, max(1, H - tile), step)) + [max(0, H - tile)]
    out: List[ObjectCandidate] = []
    for y0 in sorted(set(ys)):
        for x0 in sorted(set(xs)):
            crop = small[y0 : y0 + tile, x0 : x0 + tile]
            ch, cw = crop.shape[:2]
            res = model.predict(crop, imgsz=tile, conf=0.35, iou=0.8, retina_masks=True, device=device, verbose=False)
            if not res or res[0].masks is None:
                continue
            masks = res[0].masks.data.cpu().numpy()
            for mk in masks:
                mk = (mk > 0.5).astype(np.uint8)
                if mk.shape != (ch, cw):
                    mk = cv2.resize(mk, (cw, ch), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                c = max(contours, key=cv2.contourArea)
                area, solidity, rect = _shape_stats(c)
                if area < a_min or area > a_max or solidity < 0.8:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                # Drop objects cut by an interior tile edge (the neighbouring tile sees them whole)
                if (bx <= 1 and x0 > 0) or (by <= 1 and y0 > 0) or (bx + bw >= cw - 1 and x0 + cw < W) or (by + bh >= ch - 1 and y0 + ch < H):
                    continue
                (_, _), (rw, rh), _ = cv2.minAreaRect(c)
                if max(rw, rh) / max(1.0, min(rw, rh)) > 4.0:
                    continue
                peri = cv2.arcLength(c, True) or 1.0
                circ = 4 * math.pi * area / (peri * peri)
                c_full = c + np.array([[[x0, y0]]])
                region = np.zeros((H, W), np.uint8)
                cv2.drawContours(region, [c_full], -1, 1, -1)
                kind = _classify_region(st, region, area, solidity, rect, circ)
                if kind is None:
                    continue
                out.append(
                    ObjectCandidate(
                        polygon=_simplify(c_full.astype(np.int32)),
                        kind=kind,
                        shape=float(0.5 * solidity + 0.5 * rect),
                        meta={"source": "fastsam"},
                    )
                )
    return out


def _vegetation_proposals(st: _SceneStats, px_area: Tuple[float, float]) -> List[ObjectCandidate]:
    """Parks / tree groves: clearly greener than the scene (textured is fine)."""
    green = (st.lab[..., 1] < st.median_lab[1] - 7) & (st.L > 25)
    k = _odd(10.0 / st.mpp, 5)
    m = cv2.morphologyEx(green.astype(np.uint8) * 255, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    a_min = max(px_area[0], (50.0 / st.mpp) ** 2)  # parks must be sizeable to be distinctive
    out = []
    for c in contours:
        area, solidity, rect = _shape_stats(c)
        if area < a_min or area > px_area[1] or solidity < 0.65:
            continue
        out.append(ObjectCandidate(polygon=_simplify(c), kind="green_area", shape=float(0.5 * solidity + 0.5 * rect)))
    return out


def _roundabout_proposals(st: _SceneStats, r_m: Tuple[float, float] = (10.0, 80.0)) -> List[ObjectCandidate]:
    """Circular islands enclosed by a dark ring road (hole analysis + Hough, verified)."""
    r_min, r_max = r_m[0] / st.mpp, r_m[1] / st.mpp
    road = st.dark & (st.texture < max(12.0, 0.8 * st.median_tex))
    road = cv2.morphologyEx(road.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    found: List[Tuple[float, float, float]] = []
    # 1) holes inside the road network that are round
    contours, hier = cv2.findContours(road, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is not None:
        for c, h in zip(contours, hier[0]):
            if h[3] < 0 or len(c) < 12:  # only holes
                continue
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True) or 1.0
            circ = 4 * math.pi * area / (peri * peri)
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if r_min <= r <= r_max and circ > 0.8 and area / (math.pi * r * r) > 0.7:
                found.append((cx, cy, r))
    # 2) Hough circles (islands with weak contrast)
    gray = cv2.medianBlur(cv2.cvtColor(st.smooth, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(8, int(r_min * 3)), param1=100, param2=30,
        minRadius=int(max(4, r_min)), maxRadius=int(max(6, r_max)),
    )
    if circles is not None:
        found += [tuple(map(float, c)) for c in circles[0][:300]]

    H, W = road.shape
    out = []
    for cx, cy, r in found:
        x0, x1 = int(max(0, cx - 2 * r)), int(min(W, cx + 2 * r + 1))
        y0, y1 = int(max(0, cy - 2 * r)), int(min(H, cy + 2 * r + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - cx, yy - cy)
        ring = (d >= 1.05 * r) & (d <= 1.6 * r)
        inner = d <= 0.85 * r
        if ring.sum() < 20 or inner.sum() < 10:
            continue
        ring_dark = float((road[y0:y1, x0:x1][ring] > 0).mean())
        inner_dark = float((road[y0:y1, x0:x1][inner] > 0).mean())
        if ring_dark < 0.5 or inner_dark > 0.3:
            continue
        theta = np.arctan2(yy - cy, xx - cx)
        sectors = [
            float((road[y0:y1, x0:x1][ring & (theta >= a) & (theta < a + np.pi / 4)] > 0).mean())
            for a in np.arange(-np.pi, np.pi, np.pi / 4)
        ]
        if min(sectors) < 0.3:
            continue  # one-sided darkness = shadow, not a ring road
        if any(math.hypot(cx - o[0], cy - o[1]) < max(r, o[2]) for o in [(c.meta["cx"], c.meta["cy"], c.meta["r"]) for c in out]):
            continue
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        R = 1.6 * r
        poly = np.stack([cx + R * np.cos(ang), cy + R * np.sin(ang)], 1).astype(np.float32)
        out.append(ObjectCandidate(polygon=poly, kind="roundabout", shape=min(1.0, 0.4 + ring_dark * 0.6), meta={"cx": cx, "cy": cy, "r": r}))
    return out


def _intersection_proposals(st: _SceneStats, min_road_m: float = 18.0, limit: int = 120) -> List[ObjectCandidate]:
    """Junctions of wide (major) roads."""
    road = (st.dark & (st.texture < max(14.0, st.median_tex))).astype(np.uint8)
    road = _clean(road > 0, 3)
    dist = cv2.distanceTransform(road, cv2.DIST_L2, 5)
    min_road_px = min_road_m / st.mpp
    half = max(1.5, min_road_px / 2.0)
    major = (dist >= half).astype(np.uint8)
    if major.sum() < 50:
        return []
    k = _odd(2 * half + 1)
    major = cv2.dilate(major, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) & (road > 0).astype(np.uint8)
    sk = _zhang_suen(major)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk
    junction = ((sk == 1) & (nb >= 3)).astype(np.uint8)
    junction = cv2.dilate(junction, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(min_road_px), _odd(min_road_px))))
    n, _, _, cents = cv2.connectedComponentsWithStats(junction)
    out = []
    for i in range(1, n):
        cx, cy = cents[i]
        w = float(dist[int(cy), int(cx)]) * 2.0
        if w < min_road_px:
            continue
        side = max(3.5 * w, 3.0 * min_road_px)
        poly = np.array(
            [[cx - side / 2, cy - side / 2], [cx + side / 2, cy - side / 2], [cx + side / 2, cy + side / 2], [cx - side / 2, cy + side / 2]],
            np.float32,
        )
        out.append(ObjectCandidate(polygon=poly, kind="intersection", shape=0.7, meta={"road_w": w}))
    out.sort(key=lambda c: -c.meta.get("road_w", 0))
    return out[:limit]


def propose_objects(
    mosaic: np.ndarray,
    target: Rect,
    meters_per_px: float,
    *,
    work_max: int = 1600,
    min_obj_m: float = 25.0,
    max_obj_m: float = 600.0,
    seed: int = 0,
    include_intersections: bool = False,
    use_sam: bool = True,
    sam_weights: str = "FastSAM-s.pt",
    distance_visibility: bool = True,
) -> List[ObjectCandidate]:
    """All plausible landmark objects around the target, scored (best first).

    ``distance_visibility=False`` skips the size-vs-distance-to-target rule (used when
    scanning a whole area without a real target, e.g. region atlas chunks).
    """
    H, W = mosaic.shape[:2]
    ds = min(1.0, work_max / float(max(W, H)))
    small = cv2.resize(mosaic, (max(1, int(W * ds)), max(1, int(H * ds))), interpolation=cv2.INTER_AREA) if ds < 1 else mosaic
    mpp = max(1e-3, (meters_per_px or 0.6) / ds)  # metres per small px
    a_min = (min_obj_m / mpp) ** 2
    a_max = (max_obj_m / mpp) ** 2

    st = _SceneStats(small, mpp)
    cands: List[ObjectCandidate] = []
    if use_sam and sam_available(sam_weights):
        try:
            cands += _sam_proposals(st, small, (a_min, a_max), weights=sam_weights)
            logger.info("FastSAM proposals: %d", len(cands))
        except Exception as exc:  # noqa: BLE001
            logger.warning("FastSAM unavailable (%s) — classical proposals only", exc)
    cands += _flat_proposals(st, (a_min, a_max))
    cands += _vegetation_proposals(st, (a_min, a_max))
    cands += _roundabout_proposals(st)
    if include_intersections:
        cands += _intersection_proposals(st)

    # back to mosaic pixels + geometric filters
    tx0, ty0, tx1, ty1 = target
    tc = np.array([(tx0 + tx1) / 2, (ty0 + ty1) / 2], np.float32)
    T = math.sqrt(max(1.0, (tx1 - tx0) * (ty1 - ty0)))
    kept: List[ObjectCandidate] = []
    for c in cands:
        c.polygon = (c.polygon / ds).astype(np.float32)
        c.polygon[:, 0] = np.clip(c.polygon[:, 0], 0, W - 1)
        c.polygon[:, 1] = np.clip(c.polygon[:, 1], 0, H - 1)
        bb = c.bbox()
        if _overlaps(bb, target, pad=T * 0.05):
            continue
        c.distance = float(np.linalg.norm(c.center - tc))
        # Visibility rule: when the view is wide enough to include both the
        # target and this object, the object must still be >= ~14 px in a 720p frame.
        if distance_visibility and c.size < 0.035 * c.distance:
            continue
        kept.append(c)
    if not kept:
        return []

    # Rarity against random reference windows of comparable size
    rng = random.Random(seed)
    Hs, Ws = small.shape[:2]
    for c in kept:
        side = max(12, int(c.size * ds * 1.4))
        side = min(side, Hs - 1, Ws - 1)
        cx, cy = (c.center * ds).astype(int)
        x0, y0 = max(0, min(Ws - side, cx - side // 2)), max(0, min(Hs - side, cy - side // 2))
        f = _window_features(cv2.resize(small[y0 : y0 + side, x0 : x0 + side], (32, 32), interpolation=cv2.INTER_AREA))
        refs = []
        for _ in range(120):
            rx, ry = rng.randint(0, Ws - side), rng.randint(0, Hs - side)
            if abs(rx - x0) < side and abs(ry - y0) < side:
                continue
            refs.append(_window_features(cv2.resize(small[ry : ry + side, rx : rx + side], (32, 32), interpolation=cv2.INTER_AREA)))
        R = np.asarray(refs, np.float32)
        mu, sd = R.mean(0), R.std(0) + 1e-3
        d = np.linalg.norm((R - f) / sd, axis=1)
        c.rarity = float(np.sort(d)[:6].mean())
    r95 = max(1e-6, float(np.percentile([c.rarity for c in kept], 95)))
    for c in kept:
        size_m = c.size * (meters_per_px or 0.6)
        c.score = (
            0.40 * min(1.0, c.rarity / r95)
            + 0.25 * KIND_PRIOR.get(c.kind, 0.5)
            + 0.20 * c.shape
            + 0.15 * min(1.0, size_m / 120.0)  # big objects stay visible from high altitude
        )

    # Non-maximum suppression on overlapping proposals
    kept.sort(key=lambda c: -c.score)
    final: List[ObjectCandidate] = []
    for c in kept:
        if any(_bbox_iou(c.bbox(), f.bbox()) > 0.25 for f in final):
            continue
        final.append(c)
    counts: Dict[str, int] = {}
    for c in final:
        counts[c.kind] = counts.get(c.kind, 0) + 1
    logger.info("Landmark candidates: %d %s", len(final), counts)
    return final


def select_landmarks(
    candidates: Sequence[ObjectCandidate],
    target: Rect,
    mosaic_size: Tuple[int, int],
    k: int = 8,
    band_fractions: Tuple[float, float, float] = (0.4, 0.35, 0.25),
) -> List[ObjectCandidate]:
    """Pick ``k`` landmarks spread over near / mid / far distance bands and around the target."""
    if k <= 0 or not candidates:
        return []
    W, H = mosaic_size
    tx0, ty0, tx1, ty1 = target
    tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
    T = math.sqrt(max(1.0, (tx1 - tx0) * (ty1 - ty0)))
    R = max(T * 2, min(tcx, W - tcx, tcy, H - tcy, max(W, H) / 2))
    edges = [0.0, max(3 * T, 0.3 * R), max(6 * T, 0.6 * R), float("inf")]
    quotas = [int(round(k * f)) for f in band_fractions]
    quotas[0] += k - sum(quotas)

    def band(c: ObjectCandidate) -> int:
        return next(i for i in range(3) if edges[i] <= c.distance < edges[i + 1])

    chosen: List[ObjectCandidate] = []
    angles: List[float] = []

    def ok(c: ObjectCandidate, strict: bool) -> bool:
        if any(_overlaps(c.bbox(), s.bbox(), pad=T * 0.1) for s in chosen):
            return False
        if strict:
            a = math.atan2(c.center[1] - tcy, c.center[0] - tcx)
            if any(abs((a - b + math.pi) % (2 * math.pi) - math.pi) < math.radians(30) for b in angles):
                return False
        return True

    ranked = sorted(candidates, key=lambda c: -c.score)
    for b in range(3):
        taken = 0
        for strict in (True, False):
            for c in ranked:
                if taken >= quotas[b]:
                    break
                if c in chosen or band(c) != b or not ok(c, strict):
                    continue
                chosen.append(c)
                angles.append(math.atan2(c.center[1] - tcy, c.center[0] - tcx))
                taken += 1
    # Unused quota (sparse band) → best remaining anywhere
    for c in ranked:
        if len(chosen) >= k:
            break
        if c not in chosen and ok(c, False):
            chosen.append(c)
    return chosen[:k]
