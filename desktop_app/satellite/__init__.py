"""Satellite imagery → synthetic video helpers.

Public API is re-exported here for stable imports. Implementation lives in:
- generate.py — simple zoom/pan generation
- prepare.py — mosaic + landmark scene preparation
- render.py — labeled video rendering
- test_videos.py — evaluation clips with / without target in view
- archives.py — Earthdata / ASF archive search + attach extras
"""

from __future__ import annotations

from desktop_app.satellite.archives import (
    ArchiveCredentials,
    ArchiveScene,
    attach_archive_extras,
    search_archives,
)
from desktop_app.satellite.generate import DEFAULT_ZOOM_OUT_FACTOR, generate_satellite_videos
from desktop_app.satellite.prepare import (
    PreparedScene,
    load_prepared_scene,
    prepare_scene,
    prepare_scene_from_image,
    read_world_file_mpp,
)
from desktop_app.satellite.render import render_path_videos, render_scene_videos
from desktop_app.satellite.test_videos import render_test_videos

__all__ = [
    "ArchiveCredentials",
    "ArchiveScene",
    "DEFAULT_ZOOM_OUT_FACTOR",
    "PreparedScene",
    "attach_archive_extras",
    "generate_satellite_videos",
    "load_prepared_scene",
    "prepare_scene",
    "prepare_scene_from_image",
    "read_world_file_mpp",
    "render_path_videos",
    "render_scene_videos",
    "render_test_videos",
    "search_archives",
]
