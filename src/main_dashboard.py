# src/main_dashboard.py (The Live Demo Page)

import streamlit as st
import pandas as pd
import numpy as np
# Import necessary ML artifacts (Placeholder for V43 integration)
# import joblib
# from ultralytics import YOLO

PROJECT_SLOGAN = "The Intelligent Watchdog for Construction Zones."

def show_dashboard_page():
    """Renders the main real-time surveillance dashboard."""
    
    # Display the header unique to this page
    st.title("Site Sentinel")
    st.subheader(PROJECT_SLOGAN)
    st.markdown("---")
    
    st.header("Active Surveillance Dashboard")

    # --- Top Row: Risk Indicator and Key Metrics ---
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])

    with col1:
        st.subheader("Near-Miss Risk Status")
        # **RISK INDICATOR Placeholder (V43)** - Static for now, dynamic later
        st.markdown(
            f"""
            <div style="background-color: #008000; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h2>STATUS: SAFE</h2>
                <h3>Risk Score: 0.15 (LOW)</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        st.metric(label="Total Objects Tracked", value="15", delta="^ 3 since last frame")
        st.metric(label="Frames Processed (FPS)", value="28.5", delta_color="normal")
        
    with col3:
        st.metric(label="Personnel Count", value="4", delta="0")
        st.metric(label="Vehicle Hazard Count", value="11", delta="^ 3")
        
    with col4:
        st.metric(label="Average Vehicle Speed", value="65 km/h", delta="-5 km/h")
        st.metric(label="Mean TTC (Hazard Zone)", value="9.2 sec", delta="v 1.1 sec")


    st.markdown("---")

    # --- Middle Row: Video and Data Visualization ---
    col_video, col_chart = st.columns([3, 2])

    with col_video:
        st.subheader("Live Video Feed & Tracking")
        # **VIDEO FEED Placeholder (V41)** - Will be replaced by live video loop
        st.image("https://via.placeholder.com/800x450.png?text=SITE+SENTINEL+VIDEO+FEED+LOADS+HERE", 
                 caption="Live feed showing Bounding Boxes and Trajectories")

    with col_chart:
        st.subheader("Flow and Density Metrics")
        # **CHART Placeholder (V42)**
        chart_data = pd.DataFrame(
           np.random.randn(20, 3),
           columns=['TTC_mean', 'Density', 'Alerts'])

        st.line_chart(chart_data)