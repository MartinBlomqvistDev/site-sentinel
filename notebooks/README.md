# notebooks/

These are the development-phase explorations written during the project. They're
kept here because they show the real iterative process — the parser went through
four versions, the feature engineering through three, and the video renderer
through 23 iterations before the timing and visual clarity felt right.

**They are not pipeline code.** They won't run without modification against the
current project structure. The production pipeline lives in `pipeline/` and the
shared package lives in `site_sentinel/`.

## Files

| File | What it was for |
|---|---|
| `1_data_parser.py` | First attempt at a DFS Viewer CSV parser — good enough to prove the format |
| `2_event_analyzer.py` | Early event ranking using a simpler risk score |
| `2c_vhelper.py` | Small helper for extracting video clips around events |
| `2d_extract_coords.py` | Script to extract raw coordinates for visualisation |
| `3b_feature_eng_for_demo.py` | Feature engineering pass targeting a single demo session |
| `3c_feature_eng_for_demo_restore.py` | Recovery version of the above after a botched run |
| `explore_concords.py` | Initial exploration of the CONCOR-D dataset structure |
| `extract_top_clips.py` | Extract video clips for the top-ranked events |
| `6x_render_video_old.py` | Abandoned renderer iteration (pre-hysteresis state machine) |
| `6b_visualize_demo.py` | Debug visualisation for checking homography alignment |
