"""Annotation interpolation between manual keyframes."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from shared.geometry import interpolate_polygons, polygon_area, polygon_to_bbox
from shared.schemas import Annotation, AnnotationSource


def interpolate_between_keyframes(
    keyframes: List[Annotation],
    class_name: str = "target_region",
) -> List[Annotation]:
    """Generate interpolated annotations for every frame between sorted manual keyframes."""
    manuals = sorted(
        [a for a in keyframes if a.annotation_source in (AnnotationSource.MANUAL.value, AnnotationSource.CONFIRMED.value)],
        key=lambda a: a.frame_id,
    )
    if len(manuals) < 2:
        return []
    generated: List[Annotation] = []
    for i in range(len(manuals) - 1):
        a, b = manuals[i], manuals[i + 1]
        if a.video_id != b.video_id:
            continue
        gap = b.frame_id - a.frame_id
        if gap <= 1:
            continue
        pa = np.asarray(a.polygon_points, dtype=np.float32)
        pb = np.asarray(b.polygon_points, dtype=np.float32)
        area_a = max(polygon_area(pa), 1.0)
        for f in range(a.frame_id + 1, b.frame_id):
            t = (f - a.frame_id) / gap
            pts = interpolate_polygons(pa, pb, t)
            bbox = list(polygon_to_bbox(pts))
            zoom = float(np.sqrt(polygon_area(pts) / area_a) * a.zoom_level)
            ts = a.timestamp + t * (b.timestamp - a.timestamp)
            generated.append(
                Annotation(
                    video_id=a.video_id,
                    frame_id=f,
                    timestamp=ts,
                    image_width=a.image_width,
                    image_height=a.image_height,
                    roi_type=a.roi_type,
                    polygon_points=[[float(x), float(y)] for x, y in pts],
                    bounding_box=bbox,
                    annotation_source=AnnotationSource.INTERPOLATED.value,
                    zoom_level=zoom,
                    quality_score=0.5,
                    class_name=class_name,
                )
            )
    return generated


def confirm_interpolated(ann: Annotation) -> Annotation:
    out = Annotation.from_dict(ann.to_dict())
    out.annotation_source = AnnotationSource.CONFIRMED.value
    out.quality_score = max(out.quality_score, 0.8)
    return out
