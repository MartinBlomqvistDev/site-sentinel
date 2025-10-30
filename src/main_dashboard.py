import streamlit as st
import os
import gcs_utils

PROJECT_SLOGAN = "Predictive Safety for Construction Zones"

LOCAL_VIDEO_PATH = "data/analysis_results/final_demo_smooth_v15.mp4"
GCS_VIDEO_BLOB_NAME = "output/final_demo_smooth_v15.mp4"
USE_GCS_FOR_VIDEO = True

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
    
    video_bytes = None
    video_load_error = None

    if USE_GCS_FOR_VIDEO:
        if gcs_utils.check_gcs_connection():
            with st.spinner("Loading video..."):
                video_bytes = gcs_utils.download_bytes_from_gcs(GCS_VIDEO_BLOB_NAME)
                if video_bytes is None:
                    video_load_error = "Failed to load video from GCS"
                    # Show detailed error
                    if 'gcs_download_error' in st.session_state:
                        with st.expander("🔍 Debug Info"):
                            st.code(st.session_state['gcs_download_error'])
        else:
            video_load_error = "GCS connection unavailable"
            init_error = gcs_utils.get_init_error()
            if init_error:
                with st.expander("🔍 Debug Info"):
                    st.code(init_error)
    else:
        if os.path.exists(LOCAL_VIDEO_PATH):
            try:
                with open(LOCAL_VIDEO_PATH, 'rb') as video_file:
                    video_bytes = video_file.read()
            except Exception as e:
                video_load_error = str(e)
        else:
            video_load_error = "Video not found"

    if video_bytes:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(video_bytes)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error(f"Video unavailable: {video_load_error}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Rest of the file stays the same...