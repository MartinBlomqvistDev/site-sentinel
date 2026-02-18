# Architecture

Site Sentinel is a full-stack ML system for predicting vehicle-worker near-misses
at road construction zones. This document describes the data flow from raw footage
to live web app.

## Pipeline overview

```
CONCOR-D Dataset
(129 annotated CSVs)
        │
        ▼
pipeline/01_find_events.py
  ├── Scans all sessions for high-risk interactions
  └── Ranks by composite risk score → top_events.csv
        │
        ▼
pipeline/02_build_dataset.py
  ├── Parses every CSV with the shared parser
  ├── Computes motion features (velocity, acceleration)
  ├── Computes pairwise interaction features for each (vehicle, worker) pair
  └── Concatenates → master_training_dataset_full.csv
        │
        ▼
pipeline/03a_train_xgboost.py   ─┐
pipeline/03b_train_lstm.py       ├── Baseline comparison (same master CSV, same CV setup)
pipeline/03c_train_tcn.py       ─┘
        │
        ▼
pipeline/04_train_random_forest.py
  ├── 5-fold stratified CV with SMOTE resampling
  ├── RandomizedSearchCV over n_estimators, max_depth, etc.
  ├── Trains dual-target models: Y_preventive + Y_standard
  └── Saves dict pkl → models/rf_master_predictor_dual_lead_tuned.pkl
        │
        ▼
pipeline/05_calibrate_camera.py
  ├── Reads pixel-annotated CSV (9 cols per timestep, includes image coords)
  ├── Estimates rotation angle via linear regression on centred UTM cloud
  ├── RANSAC findHomography (UTM → pixel, ~2500+ inlier points)
  └── Saves homography_matrix.npy + transform_params.json
        │
        ▼
pipeline/06_render_video.py
  ├── Loads model, homography, and pixel-annotated trajectories
  ├── Projects UTM positions to pixels frame by frame
  ├── Runs model inference on every (vehicle, worker) pair
  ├── Temporal smoothing: 15-frame risk history window
  ├── Hysteresis state machine: SAFE → APPROACHING → IMMINENT
  └── Writes annotated MP4 → data/analysis_results/final_demo_v16.mp4
        │
        ▼
Streamlit App (src/)
  ├── Dashboard: embeds demo video from GCS, shows model metrics
  └── About: full technical write-up of dataset, model, and results
```

## Key technical decisions

### Dual-target Random Forest

The model is trained to predict two things simultaneously:

- **Y_standard** — Is TTC ≤ 2s right now? (immediate danger)
- **Y_preventive** — Will TTC ≤ 2s within the next 4 seconds? (early warning)

The preventive label is created by a forward-looking rolling max: each frame is
labelled 1 if danger arrives within the lookahead window. This means the model
learns to fire before the situation becomes immediately dangerous, giving workers
enough time to react.

### Why not LSTM/TCN?

All four architectures were evaluated on the same master dataset with identical
5-fold stratified CV setup. The Random Forest came out ahead on F1. The sequential
models (LSTM, TCN) didn't gain anything beyond RF despite having explicit
sequence awareness. The rolling features in the tabular dataset already encode
the temporal context the recurrent models were hoping to learn.

### Homography calibration

The CONCOR-D dataset provides both UTM coordinates (from GPS/GNSS) and pixel
coordinates (from the DFS Viewer's own calibration). These paired coordinates
are used directly to fit a perspective transform with RANSAC, giving a very
stable homography with thousands of inlier point correspondences.

### Temporal smoothing

Raw per-frame risk scores are noisy. The render pipeline maintains a sliding
window of the last 15 frames (≈0.5s at 30fps) and uses the smoothed average
as the actual risk signal. Alert state transitions use separate enter/exit
thresholds (hysteresis) to prevent chattering:

```
SAFE ──[risk > 0.50]──▶ APPROACHING ──[risk > 0.85 for 10 frames]──▶ IMMINENT
     ◀──[risk < 0.40]──             ◀──[risk < 0.65]──────────────────
```

## Repository structure

```
Site_Sentinel/
├── site_sentinel/        Shared Python package (parser, features, transforms)
│   ├── config.py         YAML config loader
│   ├── logging_utils.py  Consistent log format across all scripts
│   ├── data/parser.py    Single authoritative DFS CSV parser
│   ├── features/
│   │   ├── engineering.py  Motion + interaction feature computation
│   │   └── targets.py      Dual-target label creation
│   └── vision/transform.py UTM → pixel coordinate transform
│
├── pipeline/             ML pipeline scripts (run in numbered order)
│   ├── 01_find_events.py
│   ├── 02_build_dataset.py
│   ├── 03a/b/c_train_*.py
│   ├── 04_train_random_forest.py
│   ├── 05_calibrate_camera.py
│   └── 06_render_video.py
│
├── src/                  Streamlit app
│   ├── main_app.py       Entry point, CSS, navigation
│   ├── dashboard.py      Dashboard page
│   └── about.py          About / technical deep-dive page
│
├── configs/              All tunable values (no magic numbers in source)
│   ├── pipeline.yaml
│   ├── model_training.yaml
│   └── app.yaml
│
├── tests/                Unit tests (parser, features, model loading)
├── notebooks/            Archived development explorations (not pipeline code)
└── docs/                 This file and future documentation
```

## 11 input features

| Feature | Description |
|---|---|
| `rel_distance` | Euclidean distance between worker and vehicle (m) |
| `rel_speed` | Magnitude of relative velocity (m/s) |
| `speed_ms_vuln` | Worker scalar speed (m/s) |
| `speed_ms_car` | Vehicle scalar speed (m/s) |
| `accel_ms2_vuln` | Worker acceleration (m/s²) |
| `accel_ms2_car` | Vehicle acceleration (m/s²) |
| `approach_speed` | Rate of gap closure — dot product of separation and velocity vectors |
| `ttc` | Time-to-collision assuming constant velocity (s); 100 if diverging |
| `rel_dist_avg_2s` | 2s rolling mean of `rel_distance` |
| `rel_speed_avg_2s` | 2s rolling mean of `rel_speed` |
| `future_rel_dist_avg_2s` | 2s rolling mean of projected distance after 4s |
