"""
Site Sentinel — Streamlit application entry point.

Run locally with:
    streamlit run src/main_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the project root importable so site_sentinel.* works regardless of
# what directory streamlit is launched from.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from about import show_about_page  # noqa: E402
from dashboard import show_dashboard_page  # noqa: E402
from site_sentinel.config import load_config  # noqa: E402

_cfg = load_config("app")
_author = _cfg["author"]
_project = _cfg["project"]


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg:           #f8fafc;
            --surface:      #ffffff;
            --surface-up:   #f1f5f9;
            --border:       #e2e8f0;
            --accent:       #0284c7;
            --accent-hover: #0369a1;
            --accent-light: #e0f2fe;
            --text:         #0f172a;
            --text-muted:   #475569;
            --text-faint:   #94a3b8;
        }

        /* ---- Layout ---- */
        .block-container {
            max-width: 860px !important;
            margin: 0 auto !important;
            padding-top: 1.75rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        [data-testid="stAppViewContainer"] { background-color: var(--bg); }
        [data-testid="stSidebarNav"]       { display: none; }
        [data-testid="stSidebar"]          { display: none; }

        /* ---- Typography ---- */
        * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

        h4 {
            color: var(--text);
            font-size: 1.1rem;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
            font-weight: 600;
        }

        p  { color: var(--text-muted); line-height: 1.7; }
        ul { color: var(--text-muted); }
        li { margin-bottom: 0.4rem; }
        a  { color: var(--accent); text-decoration: none; }
        a:hover { text-decoration: underline; color: var(--accent-hover); }

        /* ---- Hero block ---- */
        .hero {
            text-align: center;
            padding: 2.5rem 0 2rem;
            margin-bottom: 2rem;
        }

        .hero-title {
            font-size: 2.75rem;
            font-weight: 800;
            color: var(--text);
            letter-spacing: -1.5px;
            margin: 0;
            line-height: 1.1;
        }

        .hero-title span { color: var(--accent); }

        .hero-subtitle {
            font-size: 1.05rem;
            color: var(--text-muted);
            margin-top: 0.6rem;
            font-weight: 400;
            letter-spacing: 0.2px;
        }

        .hero-tag {
            display: inline-block;
            background: var(--accent-light);
            color: var(--accent);
            border: 1px solid #bae6fd;
            padding: 0.3rem 0.9rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 1rem;
            letter-spacing: 0.3px;
        }

        /* ---- Page title (About page) ---- */
        .page-title {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--text);
            letter-spacing: -1px;
            margin: 0;
        }

        .page-subtitle {
            font-size: 0.95rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
            letter-spacing: 0.2px;
        }

        /* ---- Section header ---- */
        .section-header {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text);
            margin-top: 3rem;
            margin-bottom: 1.25rem;
            padding-bottom: 0.6rem;
            border-bottom: 2px solid var(--accent);
        }

        /* ---- Cards ---- */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
        }

        /* Metric card — left accent stripe */
        .metric-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }

        .metric-label {
            font-size: 0.78rem;
            color: var(--accent);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--text);
            letter-spacing: -0.5px;
        }

        /* Skill / tech cards */
        .skill-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.4rem;
            min-height: 10rem;
            display: flex;
            flex-direction: column;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: box-shadow 0.2s ease;
        }

        .skill-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }

        .skill-label {
            font-size: 0.78rem;
            color: var(--accent);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 0.75rem;
        }

        .skill-text {
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.6;
            flex-grow: 1;
        }

        /* Tech stack cards (About page) */
        .tech-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .tech-name {
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.4rem;
            font-size: 0.95rem;
        }

        .tech-desc {
            color: var(--text-faint);
            font-size: 0.85rem;
            line-height: 1.5;
        }

        /* Achievement / timeline items */
        .achievement {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        .achievement-title {
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.4rem;
            font-size: 0.95rem;
        }

        .achievement-text {
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.6;
        }

        /* Callout block */
        .callout {
            background: var(--accent-light);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin: 1.5rem 0;
        }

        .callout-title {
            font-weight: 700;
            color: var(--accent-hover);
            margin-bottom: 0.4rem;
            font-size: 0.9rem;
        }

        .callout-text {
            color: var(--text);
            font-size: 0.92rem;
            line-height: 1.6;
        }

        /* General text block */
        .text-block {
            font-size: 1rem;
            line-height: 1.75;
            color: var(--text-muted);
            margin-bottom: 1.25rem;
        }

        /* Video container */
        .video-container {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
            margin: 1.5rem 0 2rem;
        }

        /* ---- Navigation ---- */
        .nav-active > button {
            background-color: var(--accent) !important;
            color: #ffffff !important;
            border-color: var(--accent) !important;
        }

        .stButton > button {
            background-color: var(--surface) !important;
            border: 1.5px solid var(--border) !important;
            color: var(--text-muted) !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.25rem !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
            transition: all 0.15s ease !important;
        }

        .stButton > button:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent;
            border-bottom: 1px solid var(--border);
            gap: 0.25rem;
        }

        .stTabs [data-baseweb="tab"] {
            color: var(--text-faint);
            font-weight: 500;
            padding: 0.75rem 1rem;
            font-size: 0.92rem;
            background: transparent;
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom: 2px solid var(--accent) !important;
            background: transparent !important;
        }

        /* ---- Footer ---- */
        .footer {
            text-align: center;
            margin-top: 4rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            color: var(--text-faint);
            font-size: 0.82rem;
            line-height: 1.8;
        }

        .footer a { color: var(--text-faint); }
        .footer a:hover { color: var(--accent); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer">
            <p>{_author['name']} &nbsp;·&nbsp; {_author['role']}</p>
            <p>
                <a href="{_author['github']}" target="_blank">GitHub</a>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <a href="{_author['linkedin']}" target="_blank">LinkedIn</a>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <a href="mailto:{_author['email']}">{_author['email']}</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title=_project["title"],
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _inject_css()

    if "page" not in st.session_state:
        st.session_state.page = "dashboard"

    # Navigation — compact buttons on the left, rest of the row is empty
    nav_dash, nav_about, _ = st.columns([1, 1, 4])

    with nav_dash:
        if st.session_state.page == "dashboard":
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        if st.button("Dashboard", use_container_width=True, key="btn_dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
        if st.session_state.page == "dashboard":
            st.markdown("</div>", unsafe_allow_html=True)

    with nav_about:
        if st.session_state.page == "about":
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        if st.button("About", use_container_width=True, key="btn_about"):
            st.session_state.page = "about"
            st.rerun()
        if st.session_state.page == "about":
            st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.page == "dashboard":
        show_dashboard_page(_render_footer)
    else:
        show_about_page(_render_footer)


if __name__ == "__main__":
    main()
