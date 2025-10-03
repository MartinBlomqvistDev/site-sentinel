# src/main_app.py (The Controller)

import streamlit as st
# Import logic from the two separate page files
from main_dashboard import show_dashboard_page 
from about_page import show_about_page 

# --- CONFIGURATION ---
PROJECT_TITLE = "Site Sentinel"
# The slogan will be displayed by the dashboard/about pages
# ---------------------

def set_page_config():
    """Sets up the initial appearance, relying completely on config.toml for theme."""
    st.set_page_config(
        page_title=PROJECT_TITLE,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # We rely on config.toml for color theme (Black, White Text, Red Primary)
    # No CSS injection here.


def display_navigation():
    """Creates the sidebar navigation and returns the selected page."""
    
    st.sidebar.title("Site Sentinel")
    
    # --- NAVIGATION CONTROL ---
    page = st.sidebar.radio(
        "Select View",
        ["Dashboard (Live Feed)", "About (Portfolio)"],
        # Start on the Dashboard
        index=0 
    )
    st.sidebar.markdown("---")
    
    # Display fixed parameters in the sidebar only if the Dashboard is active
    if page == "Dashboard (Live Feed)":
        st.sidebar.subheader("Prediction Parameters")
        st.sidebar.metric(label="Safety Distance Threshold", value="5.0 meters", delta_color="off")
        st.sidebar.metric(label="Prediction Lead Time", value="4.0 seconds", delta_color="off")
    
    return page

def main():
    """Main execution function that runs the entire Streamlit application."""
    set_page_config()
    
    # 1. Get the current page selection
    page = display_navigation()

    # 2. Render the selected page content
    if page == "Dashboard (Live Feed)":
        show_dashboard_page() 
    elif page == "About (Portfolio)":
        show_about_page()

if __name__ == "__main__":
    main()