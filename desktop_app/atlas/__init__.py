"""Region atlas: train a base model on a large area once, then fine-tune sub-regions quickly.

- model.py — RegionAtlas record (atlas.json) + lon/lat ↔ mosaic pixel helpers
- build.py — chunk planning, global landmarks, chunk videos, base project
- finetune.py — sub-region scene/project linked to the atlas base model
"""

from __future__ import annotations

from desktop_app.atlas.build import (
    AtlasBuildOptions,
    estimate,
    finalize_base_model,
    new_atlas,
    plan_chunks,
    prepare_atlas_data,
)
from desktop_app.atlas.finetune import (
    finetune_source,
    prepare_subregion_project,
    prepare_subregion_scene,
    validate_target,
)
from desktop_app.atlas.model import ATLAS_FILE, RegionAtlas

__all__ = [
    "ATLAS_FILE",
    "AtlasBuildOptions",
    "RegionAtlas",
    "estimate",
    "finalize_base_model",
    "finetune_source",
    "new_atlas",
    "plan_chunks",
    "prepare_atlas_data",
    "prepare_subregion_project",
    "prepare_subregion_scene",
    "validate_target",
]
