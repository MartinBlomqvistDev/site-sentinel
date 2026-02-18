"""
Dashboard page — the main landing page of the Site Sentinel app.
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
_video_url = _cfg["video"]["public_url"]


def show_dashboard_page(render_footer: Callable[[], None]) -> None:
    # Hero
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Site <span>Sentinel</span></div>
            <div class="hero-subtitle">Predictive safety for road construction zones</div>
            <div class="hero-tag">4-second warning window &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; 129 annotated sessions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Video
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    st.video(_video_url)
    st.markdown("</div>", unsafe_allow_html=True)

    # Context — no "The Problem" header, just prose
    st.markdown(
        """
        <div class="text-block" style="margin-top: 1.5rem;">
        Construction workers and vehicles share tight spaces, often with poor sightlines on
        both sides. Four seconds is roughly the gap between "that vehicle is heading the wrong
        way" and "too late to move." This system predicts danger before it becomes immediate —
        not as a technical exercise, but because that window is what actually gives someone
        time to step aside.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model performance
    st.markdown('<h2 class="section-header">Performance</h2>', unsafe_allow_html=True)

    col_p, col_r, col_f = st.columns(3)
    with col_p:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{_perf['precision']:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{_perf['recall']:.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_f:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{_perf['f1_score']:.3f}</div>
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
            Recall is the number that matters most — a missed event means a worker who didn't
            get a warning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs
    st.markdown('<h2 class="section-header">How it works</h2>', unsafe_allow_html=True)

    tab_approach, tab_model, tab_results = st.tabs(["The pipeline", "The model", "Results"])

    with tab_approach:
        st.markdown(
            f"""
            #### End-to-end, no manual steps

            Raw trajectory CSV files go in, risk-annotated video comes out.

            1. **Parse** — custom parser for the DFS Viewer format used by CONCOR-D.
               Each row is one tracked object with a flattened time-series blob,
               not standard CSV.
            2. **Calibrate** — camera-to-world alignment from RANSAC homography,
               using the coordinate pairs already in the annotations. No manual
               calibration step.
            3. **Engineer features** — {_perf['feature_count']} features per frame:
               velocity, acceleration, relative distance, approach speed,
               time-to-collision, and {_perf['rolling_window_s']:.0f}-second rolling averages.
            4. **Train** — four architectures evaluated head-to-head; the best one ships.
            5. **Render** — frame-by-frame risk overlay with a hysteresis state machine
               to stop alert flickering.

            #### A few design choices worth noting

            - **Dual targets**: one model for immediate danger (TTC ≤ {_perf['ttc_threshold_s']:.0f}s),
              one for the {_perf['prediction_lead_time_s']:.0f}-second warning window. The preventive
              model is what actually matters for giving workers time to react.
            - **SMOTE resampling** — near-misses are rare in {_perf['training_sessions']} sessions of
              ordinary traffic. Without resampling, the model learns to call everything safe.
            - **{_perf['cv_folds']}-fold stratified CV** — the numbers on this page are honest,
              not cherry-picked from the best fold.
            """,
            unsafe_allow_html=True,
        )

    with tab_model:
        st.markdown(
            f"""
            #### Why Random Forest?

            Four architectures competed on the same dataset: XGBoost, LSTM, TCN, and
            Random Forest. RF came out ahead on F1 and gives you feature importances —
            in a safety-critical system, "the model said so" isn't good enough when
            someone asks why a warning fired. Relative distance and approach speed
            dominate, which is exactly what physics predicts.

            #### Two predictions per frame

            - **Immediate risk** — is TTC already ≤ {_perf['ttc_threshold_s']:.0f}s?
              (the situation is already dangerous)
            - **Preventive risk** — will TTC drop below {_perf['ttc_threshold_s']:.0f}s
              within the next {_perf['prediction_lead_time_s']:.0f}s?
              (time to warn before it gets dangerous)

            #### {_perf['feature_count']} engineered features

            Relative distance and speed, per-object velocity and acceleration,
            time-to-collision, approach speed (the dot product of the separation vector
            and relative velocity), projected future distance, and
            {_perf['rolling_window_s']:.0f}-second rolling averages for the three key metrics.
            """,
            unsafe_allow_html=True,
        )

    with tab_results:
        st.markdown(
            f"""
            #### Preventive risk model — {_perf['cv_folds']}-fold CV

            - **Precision: {_perf['precision']:.3f}** — {_perf['precision']*100:.1f}% of alerts
              are genuine near-misses
            - **Recall: {_perf['recall']:.3f}** — catches {_perf['recall']*100:.1f}% of real
              dangerous events
            - **F1: {_perf['f1_score']:.3f}**

            Recall is the primary metric. A false alarm is annoying;
            a missed near-miss is a worker who didn't get a warning.

            #### What the demo shows

            The video runs the {_perf['prediction_lead_time_s']:.0f}-second preventive model on
            a real near-miss event from the CONCOR-D dataset. Bounding boxes cycle
            green (safe) → orange (approaching) → red (imminent) with a 15-frame
            smoothing window. The "WARNING" overlay appears {_perf['prediction_lead_time_s']:.0f}
            seconds before the actual closest-approach moment.
            """,
            unsafe_allow_html=True,
        )

    # Skills section
    st.markdown('<h2 class="section-header">What went into this</h2>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(
            """
            <div class="skill-card">
                <div class="skill-label">Raw data to result</div>
                <div class="skill-text">
                Started with a safety question and built the full pipeline from scratch —
                parser, calibration, feature engineering, training, rendering. No templates,
                no hand-holding.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            """
            <div class="skill-card">
                <div class="skill-label">Head-to-head comparison</div>
                <div class="skill-text">
                Ran XGBoost, LSTM, TCN, and Random Forest against the same dataset with
                identical CV setup, then made a reasoned call on which one to ship — and
                wrote down why.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s3:
        st.markdown(
            """
            <div class="skill-card">
                <div class="skill-label">Production thinking</div>
                <div class="skill-text">
                Config-driven pipeline, temporal smoothing to prevent alert flicker,
                hysteresis states to avoid threshold ping-pong, and CI that catches
                regressions before they reach main.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()
