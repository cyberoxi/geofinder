# GeoFinder — agent map (keep context small)

Dual app: **desktop_app/** (dataset + package) and **jetson_runtime/** (offline track). Shared code in **shared/**. Legacy **core/** + **gui/** mirror older single-app paths.

## Where to edit

| Task | Open these (not whole tree) |
|------|-----------------------------|
| Desktop main UI | `desktop_app/gui/main_window.py` |
| Train / pipeline / package UI actions | `desktop_app/gui/pipeline_actions.py` |
| Training results + advice dialog | `desktop_app/gui/train_report_dialog.py`, `desktop_app/training/advice.py` (+ `system_info.py`, `results_parse.py`) |
| BG workers | `desktop_app/gui/workers.py` |
| Satellite dialog | `desktop_app/gui/satellite_dialog.py` |
| Test video dialog | `desktop_app/gui/test_video_dialog.py` |
| Live model test (play + highlight) | `desktop_app/gui/live_test_dialog.py` |
| Frame ROI canvas | `desktop_app/gui/frame_canvas.py` |
| Satellite API | `desktop_app/satellite/__init__.py` (re-exports only) |
| Fetch + landmarks | `desktop_app/satellite/prepare.py` |
| Labeled video render | `desktop_app/satellite/render.py` |
| Test videos (pos/neg) | `desktop_app/satellite/test_videos.py` |
| Simple zoom/pan gen | `desktop_app/satellite/generate.py` |
| Tiles / Wayback | `desktop_app/satellite/tiles.py` |
| Offline maps (MBTiles packages, offline mode flag, region download) | `desktop_app/satellite/offline.py` |
| Offline Mode menu / dialog | `desktop_app/gui/offline_actions.py`, `desktop_app/gui/offline_dialog.py` |
| Desktop serial (dx/dy during live test) | `desktop_app/gui/serial_actions.py`, `serial_dialog.py`; core `shared/serial_out.py` |
| Auto pipeline logic | `desktop_app/pipeline/auto.py` |
| Region atlas record / geo helpers | `desktop_app/atlas/model.py` |
| Atlas build (chunks, global landmarks, videos) | `desktop_app/atlas/build.py` |
| Atlas sub-region fine-tune | `desktop_app/atlas/finetune.py` |
| Atlas UI actions / dialogs | `desktop_app/gui/atlas_actions.py`, `atlas_build_dialog.py`, `atlas_finetune_dialog.py` |
| Jetson runtime | `jetson_runtime/runtime/pipeline.py` |
| Jetson serial dx/dy | `shared/serial_out.py`, `jetson_runtime/gui/serial_dialog.py` (re-export: `jetson_runtime/serial_out.py`) |

## Token rules

- Prefer **one focused module** over reading a whole package.
- Do **not** pull `data/`, weights, videos, or caches into context.
- Keep new files **≤ ~400 lines**; split by responsibility when larger.
- Re-export stable names from package `__init__.py`; put logic in sibling modules.
