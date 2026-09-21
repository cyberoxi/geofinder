"""Align an image from another satellite / sensor (incl. real night imagery) to the scene mosaic.

Once warped into the mosaic pixel frame, every target / landmark label of the
scene applies to it unchanged, so it can be rendered into extra training videos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from shared.logging import get_logger

logger = get_logger("desktop.satellite.align")


@dataclass
class Alignment:
    H: np.ndarray  # 3x3, image px → mosaic px
    inliers: int
    method: str  # "auto" | "points"
    reproj_error: float = 0.0


def _prep(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def auto_align(image: np.ndarray, mosaic: np.ndarray, max_side: int = 1800, min_inliers: int = 25) -> Optional[Alignment]:
    """SIFT + RANSAC homography.  Works for other *daytime* sources; night↔day usually needs points."""
    def small(img):
        s = min(1.0, max_side / float(max(img.shape[:2])))
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return _prep(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g), s

    a, sa = small(image)
    b, sb = small(mosaic)
    sift = cv2.SIFT_create(nfeatures=6000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good]) / sa
    dst = np.float32([kb[m.trainIdx].pt for m in good]) / sb
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0 / sb)
    if H is None or mask is None:
        return None
    n_in = int(mask.sum())
    if n_in < min_inliers or n_in < 0.15 * len(good):
        return None
    err = _reproj(H, src[mask.ravel() > 0], dst[mask.ravel() > 0])
    logger.info("Auto-align: %d/%d inliers, reprojection %.1f px", n_in, len(good), err)
    return Alignment(H=H, inliers=n_in, method="auto", reproj_error=err)


def _reproj(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    if len(src) == 0:
        return 0.0
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - dst, axis=1)))


def align_from_points(image_pts: Sequence[Sequence[float]], mosaic_pts: Sequence[Sequence[float]]) -> Alignment:
    """Homography from ≥4 user-clicked point pairs (affine fallback for exactly 3)."""
    src = np.float32(image_pts).reshape(-1, 2)
    dst = np.float32(mosaic_pts).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("Need at least 3 matching point pairs (4+ recommended)")
    if len(src) == 3:
        A = cv2.getAffineTransform(src, dst)
        H = np.vstack([A, [0, 0, 1]])
    else:
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(src) > 5 else 0, 8.0)
        if H is None:
            raise ValueError("Points are degenerate (collinear?) — pick points spread over the image")
    return Alignment(H=H, inliers=len(src), method="points", reproj_error=_reproj(H, src, dst))


def warp_to_mosaic(image: np.ndarray, alignment: Alignment, mosaic_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (image in mosaic frame, coverage mask uint8 {0,255})."""
    h, w = mosaic_shape[:2]
    warped = cv2.warpPerspective(image, alignment.H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    ones = np.full(image.shape[:2], 255, np.uint8)
    mask = cv2.warpPerspective(ones, alignment.H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    if warped.ndim == 2:
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    return warped, mask


def covered_classes(layout, mask: np.ndarray, min_cover: float = 0.7) -> set:
    """Class ids whose polygon is at least ``min_cover`` inside the coverage mask."""
    ok = set()
    for cid in range(len(layout.classes)):
        poly = np.round(layout.polygon(cid)).astype(np.int32)
        m = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(m, [poly], 1)
        area = int(m.sum())
        if area and float((mask[m > 0] > 0).mean()) >= min_cover:
            ok.add(cid)
    return ok
