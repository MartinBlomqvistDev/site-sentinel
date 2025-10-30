# SITE SENTINEL: Predictive Safety for Construction Zones

**Predictive Safety for Construction Zones**

---

## Overview

**Site Sentinel** is a full-stack, AI-driven monitoring system to prevent vehicle-personnel near-miss events in high-speed roadside construction.  
The Streamlit app demonstrates the end-to-end solution, from data ingestion to live predictive visualization.

- **Live App:** [https://sitesentinel.streamlit.app/](https://sitesentinel.streamlit.app/)
- **Video Demo:** Embedded in the dashboard, streams from Google Cloud Storage

---

## Project Summary

Site Sentinel predicts vehicle-worker collisions up to **3 seconds in advance**.  
It uses computer vision, physics-based feature engineering, and a machine learning pipeline to identify dangerous situations before they escalate.

**Key Features:**
- Real-time risk prediction with a 3s lead time
- Model selection pipeline: XGBoost, Random Forest, LSTM, TCN
- Dual-target system: Immediate vs. Preventive risk (3s ahead)
- Public demo via Streamlit + GCS video streaming

---

## Model Performance

| Metric (Preventive Risk) | Result |
|--------------------------|--------|
| Precision                | 0.875  |
| Recall                   | 0.986  |
| F1-Score                 | 0.927  |

- Validated with 5-fold stratified cross-validation
- Prioritizes precision to avoid alarm fatigue

---

## Repository Structure

```
site-sentinel/
│
├── data/
│   ├── raw/                          # Raw CONCOR-D trajectory CSVs
│   └── analysis_results/
│       ├── top_20_dynamic_events.csv
│       └── master_training_dataset_full.csv
│
├── models/
│   └── rf_master_predictor_dual_lead_tuned.pkl
│
├── scripts/
│   ├── 2b_find_best_dynamic_event.py
│   ├── 3_feature_eng.py
│   ├── 4a_train_xgboost.py
│   ├── 4b_train_lstm.py
│   ├── 4c_train_tcn.py
│   ├── 4d_train_randomforest.py
│   ├── 5_build_master_dataset.py
│   ├── 6_render_video.py
│   ├── 6a_get_homography.py
│   └── 6b_visualize_demo.py
│
├── src/
│   ├── __init__.py
│   ├── about_page.py
│   ├── gcs_utils.py
│   ├── main_app.py
│   └── main_dashboard.py
│
├── .streamlit/
│
├── requirements.txt
├── site_sentinel.yaml
└── README.md
```

---

## Pipeline Overview

| Stage                     | Script(s)                   | Purpose                                                                 |
|---------------------------|-----------------------------|-------------------------------------------------------------------------|
| Event Identification      | 2b_find_best_dynamic_event.py | Detect and rank high-risk events using physics-based scoring            |
| Master Dataset Creation   | 5_build_master_dataset.py   | Build master dataframe from all raw data and feature engineering        |
| Comparative Training      | 4a–4c                       | Train and evaluate XGBoost, LSTM, TCN baselines                        |
| Final Model Training      | 4d_train_randomforest.py    | Train and cross-validate selected Random Forest model                   |
| Visualization             | 6_render_video.py, 6b_visualize_demo.py | Overlay predictions on video and create demo outputs     |

---

## Features & Target Variables

**Engineered Features:**
- Relative distance, speed, approach speed
- Time-to-collision (TTC)
- Absolute speeds and accelerations (worker and vehicle)
- 2-second rolling averages for smoothness

**Targets:**
- `Y_standard`: Immediate risk (TTC ≤ 2.0s)
- `Y_preventive`: Preventive risk (predicted proximity < 2.0m in 3.0s)

---

## Usage

1. **Generate Master Dataset**
    ```bash
    python scripts/5_build_master_dataset.py
    ```
2. **Train Final Model**
    ```bash
    python scripts/4d_train_randomforest.py
    ```
3. **Render Demo Video**
    ```bash
    python scripts/6_render_video.py
    ```
4. **Run Streamlit App Locally**
    ```bash
    streamlit run src/main_app.py
    ```

---

## Tech Stack

- Machine Learning: scikit-learn, XGBoost, TensorFlow/Keras, imbalanced-learn (SMOTE)
- Computer Vision: OpenCV, NumPy, Polars
- Deployment: Streamlit, Google Cloud Storage, GitHub Actions
- Language: Python 3.13

---

## Contact

Martin Blomqvist | Data Scientist & Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/martin-blomqvist) | [cm.blomqvist@gmail.com](mailto:cm.blomqvist@gmail.com) | [GitHub](https://github.com/MartinBlomqvistDev)

Open to roles in Machine Learning Engineering, Data Science, Computer Vision, and Predictive Analytics.

---

## Keywords

`#MachineLearning` `#ComputerVision` `#RandomForest` `#XGBoost` `#LSTM` `#TCN` `#ModelSelection` `#PredictiveAnalytics` `#ConstructionSafety` `#TimeSeries` `#Streamlit` `#SoloDeveloper` `#FullStack` `#Python` `#OpenCV` `#SafetyCriticalSystems`

---

## Reflection & Critical Review

### Model Choice & Motivation

- Multiple architectures were evaluated: XGBoost, Random Forest, LSTM, and TCN.
- The final choice was Random Forest, motivated by its strong interpretability, robust performance, and suitability for safety-critical environments, where clear explanations for alerts are required.
- The selection was based on both quantitative metrics (precision, recall, F1-score) and qualitative needs (ease of explaining predictions to end-users).

### Application of Previous Knowledge

- The project applies concepts from earlier courses, including data ingestion, feature engineering, machine learning, model selection, and deployment.
- The pipeline structure and coding practices reflect a foundation built through previous coursework and projects.

### Agile Process

- The project was developed iteratively, with regular feedback and continuous improvement.
- Tasks were managed in phases: data preparation, feature engineering, model experimentation, and deployment.
- Adjustments and refinements were made based on model outcomes and evaluation results.

### Critical Assessment

- **Strengths:** The model achieved high recall and balanced precision, making it suitable for real-world deployment with minimal false alarms.
- **Limitations:** Further improvements could be made by increasing the dataset size and exploring additional feature engineering or model architectures.
- **Agility:** Iterative development allowed for rapid identification and resolution of bottlenecks, especially in data parsing and feature engineering.

---