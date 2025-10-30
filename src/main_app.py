import streamlit as st
from main_dashboard import show_dashboard_page 
from about_page import show_about_page 
import gcs_utils

PROJECT_TITLE = "Site Sentinel"

def set_page_config():
    st.set_page_config(
        page_title=PROJECT_TITLE,
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def main():
    set_page_config()
    
    # GLOBAL CSS - loaded once and stays for the entire session
    st.markdown("""
    <style>
        * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
        
        body {
            background-color: #0a0a0a;
            color: #ffffff;
        }
        
        .block-container {
            max-width: 900px !important;
            margin: 0 auto !important;
            padding-top: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            background-color: #0a0a0a;
        }
        
        [data-testid="stAppViewContainer"] {
            background-color: #0a0a0a;
        }
        
        .title-block {
            margin-bottom: 4rem;
            text-align: center;
        }
        
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            padding: 0;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #ff8c00 0%, #ffaa33 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .main-subtitle {
            font-size: 1rem;
            color: #cccccc;
            margin-top: 0.5rem;
            font-weight: 400;
            letter-spacing: 0.5px;
        }
        
        .page-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(135deg, #ff8c00 0%, #ffaa33 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .page-subtitle {
            font-size: 0.95rem;
            color: #999999;
            margin-top: 0.5rem;
            letter-spacing: 0.5px;
        }
        
        .tag {
            display: inline-block;
            background: rgba(255, 140, 0, 0.1);
            border: 1px solid #ff8c00;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            color: #ff8c00;
            margin-top: 1rem;
        }
        
        .video-section {
            margin: 3rem 0;
        }
        
        .video-container {
            background: #1a1a1a;
            border: 1px solid #ff8c00;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            margin-top: 4rem;
            margin-bottom: 3rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #ff8c00;
            color: #ffffff;
        }
        
        .text-block {
            font-size: 1rem;
            line-height: 1.7;
            color: #dddddd;
            margin-bottom: 1.5rem;
        }
        
        .highlight {
            background: rgba(255, 140, 0, 0.05);
            border-left: 4px solid #ff8c00;
            padding: 1.5rem;
            margin: 2rem 0;
            border-radius: 4px;
        }
        
        .highlight-title {
            font-weight: 600;
            color: #ff8c00;
            margin-bottom: 0.75rem;
        }
        
        .highlight-text {
            color: #dddddd;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        .highlight-box {
            background: rgba(255, 140, 0, 0.05);
            border-left: 4px solid #ff8c00;
            padding: 1.5rem;
            margin: 2rem 0;
            border-radius: 4px;
        }
        
        .column-item {
            padding: 1.5rem;
            background: #1a1a1a;
            border: 1px solid #ff8c00;
            border-radius: 6px;
            text-align: center;
        }
        
        .column-label {
            font-size: 0.9rem;
            color: #ff8c00;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }
        
        .column-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
        }
        
        /* Skills boxes - good height */
        .skill-box {
            padding: 1.5rem;
            background: #1a1a1a;
            border: 1px solid #ff8c00;
            border-radius: 6px;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .skill-label {
            font-size: 0.9rem;
            color: #ff8c00;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 1rem;
        }
        
        .skill-text {
            color: #dddddd;
            font-size: 0.9rem;
            line-height: 1.6;
            flex-grow: 1;
        }
        
        .skill-card {
            background: #1a1a1a;
            border: 1px solid #ff8c00;
            border-radius: 6px;
            padding: 1.5rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .skill-name {
            font-weight: 600;
            color: #ff8c00;
            margin-bottom: 0.5rem;
        }
        
        .skill-desc {
            color: #999999;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        .achievement-item {
            background: #1a1a1a;
            border: 1px solid #ff8c00;
            border-radius: 6px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .achievement-title {
            font-weight: 600;
            color: #ff8c00;
            margin-bottom: 0.5rem;
        }
        
        .achievement-text {
            color: #dddddd;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        .footer {
            text-align: center;
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid #333333;
            color: #666666;
            font-size: 0.85rem;
        }
        
        .contact-links {
            text-align: center;
            margin: 2rem 0;
        }
        
        .contact-links a {
            display: inline-block;
            margin: 0 1rem;
            color: #ff8c00;
            text-decoration: none;
            font-weight: 500;
        }
        
        .contact-links a:hover {
            text-decoration: underline;
            color: #ffaa33;
        }
        
        /* Streamlit tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
            border-bottom: 1px solid #333333;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #999999;
            font-weight: 500;
            padding: 1rem 1rem;
            font-size: 0.95rem;
        }
        
        .stTabs [aria-selected="true"] {
            color: #ff8c00;
            border-bottom: 2px solid #ff8c00;
        }
        
        /* Streamlit buttons */
        .stButton > button {
            background-color: #1a1a1a !important;
            border: 1px solid #ff8c00 !important;
            color: #ff8c00 !important;
            border-radius: 6px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            background-color: rgba(255, 140, 0, 0.1) !important;
            border-color: #ffaa33 !important;
            color: #ffaa33 !important;
        }
        
        /* Typography */
        h4 {
            color: #ffffff;
            font-size: 1.2rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        p {
            color: #dddddd;
            line-height: 1.7;
        }
        
        ul {
            color: #dddddd;
        }
        
        li {
            margin-bottom: 0.5rem;
        }
        
        a {
            color: #ff8c00;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
            color: #ffaa33;
        }
        
        /* Hide sidebar */
        [data-testid="stSidebarNav"] { display: none; }
        [data-testid="stSidebar"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
    
    # Simple navigation buttons
    btn1, btn2 = st.columns([1, 1], gap="small")
    
    with btn1:
        if st.button("Dashboard", use_container_width=True, key="btn_dashboard"):
            st.session_state.page = 'dashboard'
    
    with btn2:
        if st.button("About", use_container_width=True, key="btn_about"):
            st.session_state.page = 'about'
    
    # Add spacing
    st.markdown("")
    
    # Render the selected page
    if st.session_state.page == 'dashboard':
        show_dashboard_page()
    elif st.session_state.page == 'about':
        show_about_page()
    else:
        show_dashboard_page()

if __name__ == "__main__":
    main()