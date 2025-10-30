import streamlit as st

def show_about_page():
    
    st.markdown("""
    <div class="title-block">
        <div class="page-title">About Site Sentinel</div>
        <div class="page-subtitle">Full-Stack ML Engineering Project</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Overview</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="text-block">
    Site Sentinel is a system I built to prevent vehicle-worker collisions in roadside construction. It uses computer vision and machine learning to predict accidents before they happen, giving workers time to get out of danger.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
        <div class="highlight-title">What I Built</div>
        <div class="highlight-text">
        A Random Forest model that predicts immediate risk and future risk with 87.5% precision and 98.6% recall. As a solo developer, I handled everything from data parsing to model training to deployment.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">My Work</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="achievement-item">
        <div class="achievement-title">Data Pipeline</div>
        <div class="achievement-text">
        Built a custom parser for complex trajectory data, automatic calibration to map real-world coordinates to video pixels, and engineered 11 features from raw sensor data.
        </div>
    </div>
    
    <div class="achievement-item">
        <div class="achievement-title">Model Development</div>
        <div class="achievement-text">
        Trained and compared XGBoost, Random Forest, LSTM, and TCN models on a dataset of 100+ trajectory files. Selected Random Forest for better performance and interpretability.
        </div>
    </div>
    
    <div class="achievement-item">
        <div class="achievement-title">Production System</div>
        <div class="achievement-text">
        Built a real-time video processing pipeline that runs at 60 FPS. Integrated with Google Cloud Storage for storage and created a Streamlit interface for the model.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Technical Details</h2>', unsafe_allow_html=True)
    
    tab_data, tab_model, tab_eval = st.tabs(["Data", "Model", "Results"])
    
    with tab_data:
        st.markdown("""
        #### Data Pipeline
        
        **Input**: Complex nested trajectory CSV files from DFS Viewer
        
        **Process**:
        1. Custom parser for multi-level data hierarchy
        2. Extract motion features (velocity, acceleration)
        3. Detect actors (vehicle vs. pedestrian/bicycle)
        4. Match vehicle-worker pairs
        5. Calculate relative features (distance, approach speed, TTC)
        6. Smooth with 2-second rolling windows
        7. Predict future positions for 4-second horizon
        
        **Output**: Master dataset with 11 engineered features per frame
        
        **Key Decision**: Automatic calibration using homography transformation. No manual camera setup needed.
        """)
    
    with tab_model:
        st.markdown("""
        #### Model Details
        
        **Why Random Forest?**
        
        I tested four architectures. Random Forest had the best F1-Score and is interpretable — important for safety-critical work where people need to understand the decisions.
        
        **Two Risk Systems**:
        
        - **Immediate**: Is a collision happening now? (TTC ≤ 2 seconds)
        - **Preventive**: Will a worker enter a danger zone in the next 4 seconds?
        
        **Training**:
        - SMOTE resampling for class imbalance
        - Stratified 5-fold cross-validation
        - Hyperparameter tuning with RandomizedSearchCV
        """)
    
    with tab_eval:
        st.markdown("""
        #### Performance
        
        - **Precision: 0.875** — 87.5% of alerts are correct
        - **Recall: 0.986** — Catches 98.6% of real hazards
        - **F1-Score: 0.927** — Good balance
        
        **What It Means**:
        The model finds almost all real near-misses while keeping false alarms low. Workers get warnings with time to react.
        
        **Validation**: Tested on multiple scenarios with different numbers of workers and vehicles.
        """)
    
    st.markdown('<h2 class="section-header">Tech Stack</h2>', unsafe_allow_html=True)
    
    col_ml, col_eng, col_deploy = st.columns(3)
    
    with col_ml:
        st.markdown("""
        <div class="skill-card">
            <div class="skill-name">Machine Learning</div>
            <div class="skill-desc">Scikit-learn, XGBoost, TensorFlow, Imbalanced-learn</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_eng:
        st.markdown("""
        <div class="skill-card">
            <div class="skill-name">Data</div>
            <div class="skill-desc">Pandas, NumPy, OpenCV, Custom feature engineering</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_deploy:
        st.markdown("""
        <div class="skill-card">
            <div class="skill-name">Deployment</div>
            <div class="skill-desc">Streamlit, Google Cloud Storage, Joblib</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Learning</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="achievement-item">
        <div class="achievement-title">Problem to Model</div>
        <div class="achievement-text">
        Took a real safety challenge and turned it into a clear prediction problem with measurable success metrics.
        </div>
    </div>
    
    <div class="achievement-item">
        <div class="achievement-title">Testing Ideas</div>
        <div class="achievement-text">
        Built and evaluated multiple model types to find the best solution for this specific job.
        </div>
    </div>
    
    <div class="achievement-item">
        <div class="achievement-title">Complete Ownership</div>
        <div class="achievement-text">
        Handled all stages: problem definition, data processing, feature engineering, model training, validation, and getting it into production.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">Links</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="contact-links">
        <a href="https://github.com/MartinBlomqvistDev/site-sentinel" target="_blank">GitHub Repository</a>
        <a href="https://www.linkedin.com/in/martin-blomqvist" target="_blank">LinkedIn</a>
        <a href="mailto:cm.blomqvist@gmail.com">Email</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <p>Martin Blomqvist | Data Scientist | Solo Developer</p>
        <p>Site Sentinel © 2025 | Predictive Safety for Construction Zones</p>
    </div>
    """, unsafe_allow_html=True)