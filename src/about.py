"""
About page — technical deep-dive into the Site Sentinel project.

Covers the dataset, data pipeline, model architecture, evaluation results,
and the tech stack.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from site_sentinel.config import load_config  # noqa: E402

_cfg = load_config("app")
_perf = _cfg["model_performance"]
_author = _cfg["author"]


def show_about_page(render_footer: Callable[[], None]) -> None:
    # Page title
    st.markdown(
        """
        <div class="hero" style="padding-bottom: 1rem;">
            <div class="page-title">About Site Sentinel</div>
            <div class="page-subtitle">A full-stack ML project on real-world safety data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overview
    st.markdown('<h2 class="section-header">Overview</h2>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="text-block">
        Site Sentinel started with {_perf['training_sessions']} annotated trajectory sessions
        from the ListDB dataset (TU Dresden) — real aerial traffic footage, real vehicles,
        real near-misses. The data was captured using GoPro and DJI Action 2 cameras and
        processed with DataFromSky TrafficSurvey, which extracts per-object trajectories
        (position, speed, heading) for every tracked object in the scene.
        </div>
        <div class="text-block">
        The question was straightforward: given what's happened in the last two seconds,
        can you predict whether a vehicle and a pedestrian are going to get dangerously close
        in the next four? The answer is yes — {_perf['recall']*100:.1f}% recall and
        {_perf['precision']*100:.1f}% precision with a dual-target Random Forest.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="callout">
            <div class="callout-title">In one sentence</div>
            <div class="callout-text">
            A {_perf['model_type']} model, trained on {_perf['feature_count']} engineered
            features across {_perf['training_sessions']} sessions, predicts near-miss events
            {_perf['prediction_lead_time_s']:.0f} seconds ahead with an F1 of {_perf['f1_score']:.3f}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # What was built
    st.markdown('<h2 class="section-header">What Was Built</h2>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="achievement">
            <div class="achievement-title">Data Pipeline</div>
            <div class="achievement-text">
            The CONCOR-D trajectory files use a non-standard format — each row is one tracked
            object, with a semicolon-delimited blob of 7-value timestep groups after the
            metadata columns. A custom parser handles this reliably across all 129 session files.
            Camera calibration is done automatically using RANSAC homography, reading the pixel
            coordinates already embedded in the annotation data — no manual calibration step.
            </div>
        </div>

        <div class="achievement">
            <div class="achievement-title">Model Development</div>
            <div class="achievement-text">
            Four architectures were trained and evaluated on the same master dataset with
            5-fold stratified CV and SMOTE resampling: XGBoost, LSTM, TCN, and Random Forest.
            RF came out ahead on F1 and was chosen for interpretability — in a safety-critical
            system, feature importances matter as much as the score.
            </div>
        </div>

        <div class="achievement">
            <div class="achievement-title">Video Renderer</div>
            <div class="achievement-text">
            The demo video is produced by a rendering pipeline that runs the trained model
            frame by frame, projects positions through the homography matrix, and uses a
            hysteresis state machine (SAFE → APPROACHING → IMMINENT) with a 15-frame smoothing
            window to keep alert states stable. The renderer went through 23 iterations before
            the timing and visual clarity felt right.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Technical tabs
    st.markdown('<h2 class="section-header">Technical Details</h2>', unsafe_allow_html=True)

    tab_data, tab_model, tab_results = st.tabs(["Dataset & Features", "Model", "Evaluation"])

    with tab_data:
        st.markdown(
            f"""
            #### ListDB Dataset

            - **Source**: Open aerial traffic dataset — TU Dresden (CC BY-NC 4.0)
            - **Sessions**: {_perf['training_sessions']} annotated trajectory files
            - **Cameras**: GoPro and DJI Action 2, processed with DataFromSky TrafficSurvey
            - **Format**: Trajectory CSV — each row is one tracked object,
              with a 7-field-per-timestep blob (x_utm, y_utm, speed, tangential_acc,
              lateral_acc, timestamp, heading)
            - **Objects**: Vehicles (cars, trucks) and pedestrians

            > Bäumler et al. (2023). "Generating representative test scenarios."
            > 27th ESV Conference, Yokohama. Paper No. 23-0122-O.

            #### Feature Engineering

            {_perf['feature_count']} features per (vehicle, worker) pair per timestep:

            | Feature | Description |
            |---|---|
            | `rel_distance` | Euclidean distance between worker and vehicle (m) |
            | `rel_speed` | Magnitude of relative velocity (m/s) |
            | `approach_speed` | Rate of gap closure — dot product of separation and velocity vectors |
            | `ttc` | Time-to-collision assuming constant velocity (s); 100 if diverging |
            | `future_rel_dist` | Projected distance after {_perf['prediction_lead_time_s']:.0f}s (linear extrapolation) |
            | `speed_ms_vuln` / `speed_ms_car` | Per-object scalar speed |
            | `accel_ms2_vuln` / `accel_ms2_car` | Per-object acceleration |
            | `rel_dist_avg_2s` | {_perf['rolling_window_s']:.0f}s rolling mean of `rel_distance` |
            | `rel_speed_avg_2s` | {_perf['rolling_window_s']:.0f}s rolling mean of `rel_speed` |
            | `future_rel_dist_avg_2s` | {_perf['rolling_window_s']:.0f}s rolling mean of `future_rel_dist` |
            """,
            unsafe_allow_html=True,
        )

    with tab_model:
        st.markdown(
            f"""
            #### Architecture comparison

            All four models were trained on the same master dataset with identical CV setup:

            | Model | Notes |
            |---|---|
            | XGBoost | Strong baseline; gradient boosting on tabular features |
            | LSTM | Sequence model with 25-frame windows; no significant gain over RF |
            | TCN | Temporal Convolutional Network; dilations [1,2,4,8]; similar to LSTM |
            | **Random Forest** | **Selected** — best F1, interpretable feature importances |

            The sequential models didn't beat the RF despite having access to the same
            2-second rolling windows as additional inputs. The rolling features already
            encode the temporal context the recurrent models were hoping to learn.

            #### Dual-target training

            - **Y_standard** — 1 when TTC ≤ {_perf['ttc_threshold_s']:.0f}s at the current frame
              (immediate danger)
            - **Y_preventive** — 1 when Y_standard will be 1 within the next
              {_perf['prediction_lead_time_s']:.0f}s (the label used for early warnings)

            Y_preventive is computed with a forward-looking rolling max — each frame is
            labelled positive if danger arrives within the lookahead window, so the model
            learns to fire before the situation becomes immediately dangerous.

            #### Training setup

            - SMOTE resampling (near-misses are rare in normal traffic — severe class imbalance)
            - {_perf['cv_folds']}-fold stratified cross-validation
            - RandomizedSearchCV over n_estimators, max_depth, min_samples_split,
              min_samples_leaf, max_features
            - Optimised for F1-score
            """,
            unsafe_allow_html=True,
        )

    with tab_results:
        st.markdown(
            f"""
            #### Preventive risk model — {_perf['cv_folds']}-fold cross-validation

            | Metric | Score | What it means |
            |---|---|---|
            | Precision | **{_perf['precision']:.3f}** | {_perf['precision']*100:.1f}% of alerts are genuine near-misses |
            | Recall | **{_perf['recall']:.3f}** | Catches {_perf['recall']*100:.1f}% of real dangerous events |
            | F1-Score | **{_perf['f1_score']:.3f}** | Harmonic mean of the above |

            Recall is the primary metric. A false alarm is annoying; a missed
            near-miss is a worker who didn't get a warning.

            #### Honest limitations

            The dataset covers three seasons at one German road construction site.
            The model has not seen:
            - Winter conditions (reduced visibility, ice, different vehicle behaviour)
            - Multilane highways or urban construction zones
            - Occlusion (a worker behind a truck, invisible to the camera)

            Generalisation to other sites would need retraining — the homography
            calibration is also site-specific.
            """,
            unsafe_allow_html=True,
        )

    # Tech stack
    st.markdown('<h2 class="section-header">Tech Stack</h2>', unsafe_allow_html=True)

    col_ml, col_data, col_deploy = st.columns(3)
    with col_ml:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-name">Machine Learning</div>
                <div class="tech-desc">scikit-learn · XGBoost · Keras (LSTM, TCN) · imbalanced-learn</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_data:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-name">Data & Vision</div>
                <div class="tech-desc">pandas · NumPy · OpenCV · custom feature engineering</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_deploy:
        st.markdown(
            """
            <div class="tech-card">
                <div class="tech-name">Deployment</div>
                <div class="tech-desc">Streamlit · Google Cloud Storage · GitHub Actions · joblib</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()
