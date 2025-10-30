import streamlit as st
import os

PROJECT_SLOGAN = "Predictive Safety for Construction Zones"

# Use public GCS URL for video
VIDEO_URL = "https://storage.googleapis.com/site-sentinel-roadwork-data/output/final_demo_smooth_v15.mp4"

st.set_page_config(
    page_title="Site Sentinel",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def show_dashboard_page():
    
    st.markdown("""
    <div class="title-block">
        <div class="main-title">Site Sentinel</div>
        <div class="main-subtitle">Predictive Safety for Construction Zones</div>
        <div class="tag">Solo Developer | Full-Stack ML</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="video-section">', unsafe_allow_html=True)
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    st.video(VIDEO_URL)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Model Performance</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="column-item">
            <div class="column-label">Precision</div>
            <div class="column-value">0.875</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="column-item">
            <div class="column-label">Recall</div>
            <div class="column-value">0.986</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="column-item">
            <div class="column-label">F1-Score</div>
            <div class="column-value">0.927</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">The Problem</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="text-block">
    Vehicle-worker near-miss incidents are a leading cause of injury in roadside construction. Workers have limited visibility of approaching vehicles, and drivers often can't see workers in blind spots or through equipment.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight">
        <div class="highlight-title">The Opportunity</div>
        <div class="highlight-text">
        Can we predict when a collision is likely to occur before it happens? With a 4-second lead time, there's enough time to warn workers and prevent injury.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">The Solution</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="text-block">
    I built a system that processes video in real-time to predict vehicle-worker collisions. It combines computer vision, physics-based feature engineering, and a machine learning model to identify dangerous situations before they escalate.
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["My Approach", "The Model", "Results"])
    
    with tab1:
        st.markdown("""
        #### What I Built
        
        **End-to-End Pipeline** — Working as a solo developer, I handled every stage:
        
        1. **Data Parsing** — Custom parser for complex trajectory files (DFS format)
        2. **Automatic Calibration** — Aligned real-world coordinates to video pixels using homography and dynamic rotation
        3. **Feature Engineering** — Extracted velocity, acceleration, distance, approach speed, time-to-collision
        4. **Model Training** — Trained and compared XGBoost, Random Forest, LSTM, and TCN
        5. **Visualization** — Real-time rendering with worker-focused risk display
        6. **Deployment** — Streamlit app with Google Cloud Storage integration
        
        #### Key Decisions
        
        - **Random Forest** instead of LSTM/TCN for interpretability in a safety-critical context
        - **Dual-target system**: immediate risk (TTC ≤ 2s) plus future risk (4 seconds ahead)
        - **SMOTE resampling** to handle class imbalance in the training data
        - **5-fold stratified cross-validation** to validate the model properly
        """)
    
    with tab2:
        st.markdown("""
        #### Random Forest Model
        
        **Why Random Forest?**
        
        I tested four different model architectures. Random Forest came out ahead because of its F1-Score and interpretability — critical in safety systems where people need to understand why an alert was triggered.
        
        **Two Risk Predictions**
        
        - **Immediate Risk**: Is a collision happening right now? (TTC ≤ 2.0 seconds)
        - **Preventive Risk**: Will a worker enter a danger zone in the next 4 seconds? (early warning)
        
        **Features** — 11 engineered per frame:
        
        - Relative distance and speed between vehicle and worker
        - Time-to-Collision (TTC)
        - Vehicle and worker velocity and acceleration
        - 2-second rolling averages for stability
        - Predicted future proximity
        
        **Training Details**
        
        - SMOTE resampling for class balance
        - Stratified 5-fold cross-validation
        - Hyperparameter tuning via RandomizedSearchCV
        - Optimized for F1-Score as the primary metric
        """)
    
    with tab3:
        st.markdown("""
        #### Performance & Impact
        
        **Preventive Risk Model**
        
        - **Precision: 0.875** — When the system alerts, it's correct 87.5% of the time
        - **Recall: 0.986** — Catches 98.6% of real near-miss events
        - **F1-Score: 0.927** — Good balance between precision and recall
        
        **What This Means**
        
        The system identifies almost all real hazards while keeping false alarms low. With a 4-second lead time, workers have time to get out of danger.
        
        **Real-World Numbers**
        
        - Runs at 60 FPS on standard hardware
        - Trained on 100+ trajectory files with diverse scenarios
        - Handles multi-worker, multi-vehicle interactions
        - Fully extensible for live feeds and drone systems
        """)
    
    st.markdown('<h2 class="section-header">Why This Matters</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="text-block">
    This project shows I can take a real safety problem, turn it into a data problem, build a complete solution, and ship it. Here's what it demonstrates:
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col_skill1, col_skill2, col_skill3 = st.columns(3)
    
    with col_skill1:
        st.markdown("""
        <div class="skill-box">
            <div class="skill-label">Problem Framing</div>
            <div class="skill-text">
            Took a safety challenge and formulated it as a prediction problem with clear metrics and business value.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_skill2:
        st.markdown("""
        <div class="skill-box">
            <div class="skill-label">Full-Stack Execution</div>
            <div class="skill-text">
            Owned the entire pipeline: parsing, feature engineering, model training, validation, and deployment.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_skill3:
        st.markdown("""
        <div class="skill-box">
            <div class="skill-label">Model Selection</div>
            <div class="skill-text">
            Tested multiple approaches and selected the best model for the specific job and context.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <p>Martin Blomqvist | Data Scientist | Solo Developer</p>
        <p>
        <a href="https://github.com/MartinBlomqvistDev/site-sentinel" target="_blank">GitHub</a> | 
        <a href="https://www.linkedin.com/in/martin-blomqvist" target="_blank">LinkedIn</a> | 
        <a href="mailto:cm.blomqvist@gmail.com">Email</a>
        </p>
    </div>
    """, unsafe_allow_html=True)