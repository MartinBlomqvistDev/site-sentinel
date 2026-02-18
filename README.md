# Site Sentinel

Real-time near-miss prediction for road construction zones.

**Live app → https://sitesentinel.streamlit.app/**

[![Lint](https://github.com/MartinBlomqvistDev/site-sentinel/actions/workflows/lint.yml/badge.svg)](https://github.com/MartinBlomqvistDev/site-sentinel/actions/workflows/lint.yml)
[![Tests](https://github.com/MartinBlomqvistDev/site-sentinel/actions/workflows/test.yml/badge.svg)](https://github.com/MartinBlomqvistDev/site-sentinel/actions/workflows/test.yml)

---

Vehicles and pedestrians share the same space with limited time to react. Existing safety
systems tell you something happened; Site Sentinel tells you something is about to. Given two
seconds of trajectory data, it predicts whether a near-miss will occur in the next four —
early enough for someone to act.

It catches **98.6% of real dangerous events** at 87.5% precision (F1 = 0.927, 5-fold CV)
across 129 annotated sessions from the ListDB aerial traffic dataset (TU Dresden).

---

- **Ingests** aerial trajectory data from the ListDB dataset (129 annotated sessions,
  TU Dresden)
- **Engineers** 11 kinematic and interaction features per (vehicle, worker) pair per frame:
  relative distance, time-to-collision, approach speed, projected future distance, rolling averages
- **Trains** a dual-target Random Forest predicting both immediate danger (TTC ≤ 2s) and
  preventive warning (danger within 4s)
- **Renders** an annotated demo video with a hysteresis state machine
  (SAFE → APPROACHING → IMMINENT) over a 15-frame sliding window to prevent alert flickering
- **Deploys** a Streamlit app serving the demo and technical results from GCS

---

## Dataset

**ListDB** (Leverage information on street traffic) is an open aerial traffic dataset from
TU Dresden. Video was captured with GoPro and DJI Action 2 cameras and processed using
DataFromSky TrafficSurvey, an AI video analytics service that extracts per-object trajectories
from aerial footage. Each session is exported as a trajectory CSV where each row is one tracked
object with a semicolon-delimited blob of 7-value timestep groups
(UTM x/y, speed, tangential acceleration, lateral acceleration, timestamp, heading).

The dataset contains 129 annotated sessions covering vehicles and pedestrians at street level.
Near-miss events are rare — roughly 2-5% of all frames — which creates severe class imbalance
that SMOTE addresses during training.

> Bäumler, M., Lehmann, M., Prokop, G. (2023). "Generating representative test scenarios:
> The FUSE4Rep process model to collect and analyse traffic observation data."
> 27th ESV Conference, Yokohama. Paper No. 23-0122-O.
> Dataset published under CC BY-NC 4.0.

---

## Model comparison

All four architectures were trained on the same master dataset with identical 5-fold stratified
cross-validation and SMOTE resampling.

| Model | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| XGBoost | ~0.82 | ~0.91 | ~0.86 | Strong gradient-boosting baseline |
| LSTM | ~0.80 | ~0.89 | ~0.84 | 25-frame sequence input; no gain over RF |
| TCN | ~0.81 | ~0.90 | ~0.85 | Dilations [1,2,4,8]; similar to LSTM |
| **Random Forest** | **0.875** | **0.986** | **0.927** | **Chosen** — best F1, interpretable |

The sequential models didn't beat the RF. The rolling features in the tabular dataset
already encode the temporal context the recurrent models were hoping to learn — 25-frame
LSTM windows on top of 2-second rolling averages gave no additional information.

---

## How to run locally

**Requirements:** Python 3.11+

```bash
# Clone and install production dependencies
git clone https://github.com/MartinBlomqvistDev/site-sentinel.git
cd site-sentinel
pip install -r requirements.txt

# Set up GCS credentials (needed for video serving)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your GCS service account key

# Run the app
streamlit run src/main_app.py
```

**To retrain the models** (requires GPU-friendly hardware for LSTM/TCN):

```bash
pip install -r requirements-train.txt

# Run the pipeline in order
python -m pipeline.01_find_events           # rank sessions by risk score
python -m pipeline.02_build_dataset         # feature-engineer all 129 sessions
python -m pipeline.03a_train_xgboost       # XGBoost baseline
python -m pipeline.03b_train_lstm          # LSTM baseline
python -m pipeline.03c_train_tcn           # TCN baseline
python -m pipeline.04_train_random_forest  # final model
python -m pipeline.05_calibrate_camera     # RANSAC homography calibration
python -m pipeline.06_render_video         # produce annotated demo video
```

---

## Project structure

```
Site_Sentinel/
│
├── site_sentinel/          Shared Python package
│   ├── config.py           YAML config loader
│   ├── logging_utils.py    Consistent log format
│   ├── data/parser.py      Single authoritative DFS CSV parser
│   ├── features/
│   │   ├── engineering.py  Motion + interaction feature computation
│   │   └── targets.py      Dual-target label creation
│   └── vision/transform.py UTM → pixel coordinate transform
│
├── pipeline/               ML pipeline (run in numbered order)
│   ├── 01_find_events.py
│   ├── 02_build_dataset.py
│   ├── 03a/b/c_train_*.py  Baseline comparisons
│   ├── 04_train_random_forest.py
│   ├── 05_calibrate_camera.py
│   └── 06_render_video.py
│
├── src/                    Streamlit app
│   ├── main_app.py         Entry point, CSS, navigation
│   ├── dashboard.py        Dashboard page
│   └── about.py            Technical deep-dive
│
├── configs/                All tunable values (no magic numbers in source)
│   ├── pipeline.yaml       Frame rate, paths, horizons, event weights
│   ├── model_training.yaml Feature list, CV settings, hyperparameter grids
│   └── app.yaml            Model metrics, author info, video URL
│
├── tests/                  Unit tests (parser, features, model loading)
├── notebooks/              Development explorations (not pipeline code)
├── docs/architecture.md    Pipeline walkthrough and design decisions
└── .github/workflows/      lint + test CI on every push
```

---

## Honest limitations

The model is trained on general street traffic (ListDB), not footage specific to road
construction zones. It has not seen:

- Construction site conditions — narrow corridors, heavy machinery, workers in high-vis gear
- Winter conditions (reduced visibility, different vehicle behaviour)
- Occlusion — a pedestrian behind a vehicle, invisible to the aerial camera

The homography calibration is session-specific: pixel-UTM correspondences are tied to a
particular camera position. Applying the system to a new location requires recalibration
and likely retraining.

---

## Author

Martin Blomqvist · Data Scientist

[GitHub](https://github.com/MartinBlomqvistDev/site-sentinel) · [LinkedIn](https://www.linkedin.com/in/martin-blomqvist) · cm.blomqvist@gmail.com
