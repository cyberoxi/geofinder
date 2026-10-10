"""Landmark tours: one camera flight per video that visits EVERY landmark of a chunk.

Focus zooms only show the area around one point, so landmarks far from it are never seen up
close (and never learned). A tour flies a nearest-neighbour route over all of them, pausing over
each one at a varying altitude / heading, so the base model sees every atlas class.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from desktop_app.atlas.model import RegionAtlas, bbox_of, lonlat_to_mosaic
from desktop_app.satellite.synth import CameraKey

# Per tour: view height as a multiple of the landmark size (close pass, high pass, mid pass, …)
ALTITUDES = (3.0, 5.5, 2.2, 4.0)


def tour_lights(n_tours: int, night_fraction: float, ir_fraction: float) -> List[str]:
    """Light per tour: first is daylight, then night / IR (if enabled) alternate with day."""
    extra = [lt for lt, f in (("night", night_fraction), ("ir", ir_fraction)) if f > 0]
    cycle = ["day"] + [x for lt in extra for x in (lt, "day")] if extra else ["day"]
    return [cycle[i % len(cycle)] for i in range(max(0, n_tours))]


def _landmark_spots(
    atlas: RegionAtlas, chunk: Dict[str, Any], keep: Sequence[int]
) -> List[Tuple[float, float, float]]:
    """(cx, cy, size) in chunk-mosaic px for every kept landmark class id."""
    out = []
    for cid in sorted(keep):
        poly = lonlat_to_mosaic(atlas.landmarks[cid - 1]["polygon_lonlat"], chunk["swne"], atlas.zoom)
        x0, y0, x1, y1 = bbox_of(poly)
        out.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, max(x1 - x0, y1 - y0, 8.0)))
    return out


def tour_keys(
    atlas: RegionAtlas,
    chunk: Dict[str, Any],
    keep: Sequence[int],
    mosaic_wh: Tuple[int, int],
    tour_idx: int,
    rng: random.Random,
) -> List[CameraKey]:
    """Wide start over the chunk → every landmark (nearest-neighbour order) → wide end."""
    W, H = mosaic_wh
    spots = _landmark_spots(atlas, chunk, keep)
    if not spots:
        return []
    side = float(min(W, H))
    alt = ALTITUDES[tour_idx % len(ALTITUDES)]
    heading = rng.uniform(0, 360) if tour_idx else 0.0
    wide = side * 0.9
    # Alternate the start corner so tours cross the chunk in different directions
    start = [(W * 0.5, H * 0.5), (W * 0.2, H * 0.2), (W * 0.8, H * 0.8), (W * 0.8, H * 0.2)][tour_idx % 4]
    keys = [CameraKey(start[0], start[1], wide, heading, 0.0)]
    todo = list(spots)
    pos = np.array(start)
    while todo:
        j = min(range(len(todo)), key=lambda i: float(np.hypot(todo[i][0] - pos[0], todo[i][1] - pos[1])))
        cx, cy, size = todo.pop(j)
        view = float(np.clip(size * alt * rng.uniform(0.8, 1.25), side * 0.08, side * 0.6))
        dist = float(np.hypot(cx - pos[0], cy - pos[1]))
        heading += rng.uniform(-35, 35)
        # ~1–2.5 s per hop: time to travel plus a short hover (ease in/out) over the landmark
        keys.append(CameraKey(cx, cy, view, heading, float(np.clip(0.8 + dist / max(side, 1.0) * 2.0, 1.0, 2.5))))
        pos = np.array((cx, cy))
    keys.append(CameraKey(W * 0.5, H * 0.5, wide, heading + rng.uniform(-30, 30), 2.0))
    return keys

