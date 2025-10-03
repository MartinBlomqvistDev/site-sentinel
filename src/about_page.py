# src/about_page.py (The Portfolio Page)

import streamlit as st

def show_about_page():
    """Renders the portfolio-focused 'About' page content."""
    
    st.header("About Site Sentinel: Project Context & Impact")
    st.markdown("---")

    # --- Section 1: The Problem and Solution ---
    st.subheader("The Challenge: Proactive Safety in Construction")
    st.markdown("""
    Roadway construction sites are among the most dangerous working environments due to high-speed vehicle traffic near static personnel. The problem is predicting **high-risk interactions** in time for human intervention.
    
    **Site Sentinel** solves this by shifting from reactive monitoring to **predictive decision support**.
    """)

    # --- Section 2: Technical Value & Achievement ---
    st.subheader("Technical Achievement: The Predictive Engine")
    st.markdown("""
    The core innovation is the **Predictive Analytics Engine (XGBoost)**, which I built from scratch:
    
    * **Feature Engineering (P31):** I engineered custom features critical for risk assessment, including **Time-to-Collision (TTC)** and **Relative Velocity**.
    * **Target Definition (P30):** The model is trained to predict a Near-Miss event **4.0 seconds** into the future with a **5.0 meter** risk perimeter.
    * **Architecture:** I developed a full, professional ML pipeline, from cloud data access (GCS) to a live web dashboard (Streamlit).
    """)
    st.markdown("---")
    
    # --- Section 3: My Role & Skills Showcase ---
    st.subheader("My Role: Full-Stack ML Engineer")
    
    st.info("""
    **As a solo developer, I owned the full lifecycle of this project:**
    """)
    
    col_vision, col_ml, col_dev = st.columns(3)
    
    with col_vision:
        st.markdown("**Computer Vision & Tracking**")
        st.caption("YOLOv8 Fine-Tuning, Multi-Object Tracking (MOT), and frame processing.")
    
    with col_ml:
        st.markdown("**Core ML & Analytics**")
        st.caption("Custom Feature Engineering, XGBoost Training, and Model Validation (Precision Focus).")
        
    with col_dev:
        st.markdown("**Deployment & Infrastructure**")
        st.caption("Cloud Storage (GCS) configuration, data logging, and Streamlit dashboard.")
        
    st.markdown("---")
    
    st.subheader("Contact & Future Work")
    st.markdown("Project Repository: [**https://github.com/MartinBlomqvistDev/site-sentinel**]")
    st.markdown("LinkedIn: [**https://www.linkedin.com/in/martin-blomqvist**]")
    st.markdown("Email: [**cm.blomqvist@gmail.com**]")