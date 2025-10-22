# SITE SENTINEL: Predictive Safety for Construction Zones

**Slogan:** The Intelligent Watchdog for Construction Zones.

---
### Keywords & Hashtags

`#MachineLearning` `#ComputerVision` `#PredictiveAnalytics` `#RandomForest` `#XGBoost` `#LSTM` `#TCN` `#ModelSelection` `#SiteSafety` `#TrafficMonitoring` `#SoloDeveloper` `#Streamlit` `#Time-Series`

---

### Project Summary & Impact

**Site Sentinel** is a full-stack, AI-driven monitoring system designed to solve a critical, high-value safety problem: **preventing vehicle-personnel Near-Miss events in high-speed roadside construction areas.**

The core achievement was architecting a robust model evaluation pipeline to identify the optimal algorithm for this task. After competitively training multiple models, a final **Dual-Lead Random Forest** classifier was selected. It forecasts immediate risk based on TTC and preventive risk **4.0 seconds in advance** based on future proximity. This provides a significant lead time for intervention, drastically reducing liability and enhancing worker safety.

As a solo developer, **I owned the entire ML lifecycle**, from building a master training dataset across all available data to feature engineering, comparative model training, and final visualization.

---
### Project Structure

```
site-sentinel/
│
├── data/
│   ├── raw/                          # Raw CONCOR-D trajectory CSVs
│   └── analysis_results/
│       ├── top_20_dynamic_events.csv # List of highest-risk events found
│       └── master_training_dataset_full.csv # Unified dataset for training
│
├── models/
│   └── rf_master_predictor_dual_lead_tuned.pkl # Final serialized model
│
├── scripts/
│   ├── 2b_find_best_dynamic_event.py # Scans all data to find high-risk events
│   ├── 3_feature_eng.py              # Engineers features for a single event
│   ├── 4a_train_xgboost.py           # Trains XGBoost model (comparative)
│   ├── 4b_train_lstm.py              # Trains LSTM model (comparative)
│   ├── 4c_train_tcn.py               # Trains TCN model (comparative)
│   ├── 4d_train_randomforest.py      # ✅ Trains the final selected RF model
│   ├── 5_build_master_dataset.py     # Builds the master training dataset
│   └── 6_render_video.py             # Renders demo video with predictions
│
└── README.md
```

---

### End-to-End Pipeline

| Stage | Script(s) | Purpose |
| :--- | :--- | :--- |
| **1. Event Identification** | `2b_find_best_dynamic_event.py` | Scans all raw CSVs to auto-detect and rank the most dynamically dangerous interaction events based on a physics-driven risk score. |
| **2. Master Dataset Creation**| `5_build_master_dataset.py` | Processes **all 100+ raw trajectory files**, auto-detects actors, calculates motion and interaction features, and builds a single, unified master training dataset. |
| **3. Comparative Training**| `4a` to `4c` | Trains and evaluates multiple baseline models (XGBoost, LSTM, TCN) on the master dataset to establish performance benchmarks. |
| **4. Final Model Training**| `4d_train_randomforest.py` | Trains, tunes, and cross-validates the final **Dual-Lead Random Forest** model on the full master dataset using SMOTE for class imbalance. |
| **5. Visualization** | `6_render_video.py` | Loads the trained production model and overlays its risk predictions onto the source video for a final polished demonstration. |

---

### Core Innovation: From Heuristics to a Selected Model

This project showcases a professional, data-driven approach to model selection. The final predictive engine is the result of a multi-stage refinement process.

| Stage | Technical Achievement | Value & Rationale |
| :--- | :--- | :--- |
| **1. Heuristic Risk Scoring** | Developed a physics-based risk score combining relative distance, approach speed, and vehicle speed to automatically find the best training data. | This initial step created a non-ML baseline and a robust method for automatically identifying the most dynamically interesting events for later training. |
| **2. Scalable Feature Engineering** | Built an automated pipeline (`5_build_master_dataset.py`) to process **all raw data files** into a single, unified master training dataset. | This demonstrates the ability to scale from a single-event prototype to a robust model trained on comprehensive, diverse interaction scenarios. |
| **3. Competitive Model Evaluation** | Trained and cross-validated four distinct model architectures: XGBoost, Random Forest, LSTM, and TCN. | This proves a deep understanding of the ML landscape, from classic tree-based models to advanced deep learning architectures for time-series data. |
| **4. The Dual-Risk System**| Engineered the selected Random Forest model with a unique **dual-target system**: one for *preventive* risk (future proximity) and one for *standard* risk (immediate TTC). | This nuanced approach provides a more complete safety picture, capturing both slowly developing threats and sudden dangers. |

---

### Model Selection & Performance

The **Dual-Lead Random Forest** was selected as the production model. It offered the best balance of **predictive performance (high F1-Score)** and **interpretability**. Its dual-target structure aligns perfectly with the project's goal of providing both early warnings and immediate alerts, making it the most functionally valuable for a real-world safety application.

#### Final Model Performance

The model was validated using 5-fold stratified cross-validation. **Precision** is prioritized as it is critical that every alert is credible to avoid "alarm fatigue" in a safety-critical system.

| Metric (Preventive Risk) | Result | Goal & Interpretation |
| :--- | :--- | :--- |
| **Precision** | **0.875** | **(CRITICAL METRIC)** Measures how often a "future risk" alarm is correct. A high score proves the system is trustworthy. |
| **Recall** | **0.986** | Measures how many real risk events the model successfully identified in advance. |
| **F1-Score** | **0.927** | Harmonic mean of Precision and Recall, showing a balanced model. |

---
### Features & Target Variables

The models are trained on a rich set of kinematic and interaction features.

| Feature | Description |
| :--- | :--- |
| `rel_distance`, `rel_speed` | Instantaneous relative distance and speed between actors. |
| `approach_speed` | The closing velocity between actors (rate at which the gap is shrinking). |
| `ttc` | Time-to-Collision, a classic safety metric. |
| `speed_ms_vuln`, `speed_ms_car` | Absolute speeds of each actor. |
| `accel_ms2_vuln`, `accel_ms2_car`| Absolute acceleration of each actor. |
| `*_avg_2s` | Features smoothed over a 2-second rolling window for stability. |

#### Dual-Lead Target Variables

The final model predicts two distinct targets:
* **`Y_standard`**: An immediate risk flag, active if `ttc <= 2.0` seconds.
* **`Y_preventive`**: A future risk flag, active if the actors are predicted to be within a `2.0m` distance threshold **4.0 seconds in the future**.

---

### Usage

1.  **Generate Master Dataset:**
    ```bash
    python scripts/5_build_master_dataset.py
    ```
2.  **Train Final Model:**
    ```bash
    python scripts/4d_train_randomforest.py
    ```
3.  **Render Demo Video:**
    ```bash
    python scripts/6_render_video.py
    ```

---

### Demo & Contact

*Note: Insert a live video or image here once available.*

**I am ready for a role in Machine Learning Engineering/Data Science, especially in areas involving predictive analysis on time-series or sensor data.**

* Project Repository: [**https://github.com/MartinBlomqvistDev/site-sentinel**]
* LinkedIn: [**https://www.linkedin.com/in/martin-blomqvist**]
* Email: [**cm.blomqvist@gmail.com**]