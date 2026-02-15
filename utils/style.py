import streamlit as st


def init_theme():
    """Initialize theme in session state (call once at app start)."""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"


def toggle_theme():
    """Callback for the dark-mode toggle widget."""
    st.session_state.theme = "dark" if st.session_state.toggle_dark_mode else "light"


def get_theme_config():
    """Return a dict of colour tokens for the active theme."""
    if st.session_state.get("theme", "dark") == "dark":
        return {
            "BG_COLOR": "#0f172a",
            "CARD_BG": "rgba(30, 41, 59, 0.7)",
            "TEXT_COLOR": "#f8fafc",
            "TEXT_SUB": "#94a3b8",
            "BORDER_COLOR": "rgba(255, 255, 255, 0.1)",
            "COLOR_BLUE_DARK": "#60a5fa",
            "COLOR_BLUE": "#3b82f6",
            "COLOR_ORANGE": "#f97316",
            "PLOT_TEMPLATE": "plotly_dark",
            "GRID_COLOR": "#334155",
            "SIDEBAR_BG": "#1e293b",
        }
    return {
        "BG_COLOR": "#F8FAFC",
        "CARD_BG": "#ffffff",
        "TEXT_COLOR": "#1e293b",
        "TEXT_SUB": "#64748b",
        "BORDER_COLOR": "#e2e8f0",
        "COLOR_BLUE_DARK": "#003366",
        "COLOR_BLUE": "#137fec",
        "COLOR_ORANGE": "#F37021",
        "PLOT_TEMPLATE": "plotly_white",
        "GRID_COLOR": "#e2e8f0",
        "SIDEBAR_BG": "#003366",
    }


def load_css():
    """Inject the global stylesheet (call once per render)."""
    t = get_theme_config()
    is_dark = st.session_state.get("theme", "dark") == "dark"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        .stApp {{
            font-family: 'Inter', sans-serif;
            background-color: {t['BG_COLOR']};
            color: {t['TEXT_COLOR']};
        }}
        html, body {{ background-color: {t['BG_COLOR']}; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {t['SIDEBAR_BG']};
            border-right: 1px solid {t['BORDER_COLOR']};
        }}
        [data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}

        .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}

        h1, h2, h3 {{
            color: {t['TEXT_COLOR'] if is_dark else '#003366'} !important;
            font-weight: 700;
        }}

        [data-testid="stMetricValue"] {{ color: {t['TEXT_COLOR']} !important; }}

        .stDateInput label, .stSelectbox label, .stRadio label {{
            color: {t['TEXT_COLOR']} !important;
        }}
        .stDateInput input {{
            color: #334155 !important;
            font-weight: 600;
        }}

        {'div[data-testid="stExpander"] { border: 1px solid ' + t['BORDER_COLOR'] + '; border-radius: 8px; }' if is_dark else ''}

        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}

        /* KPI cards */
        .kpi-card {{
            background-color: {t['CARD_BG']};
            border: 1px solid {t['BORDER_COLOR']};
            border-radius: 0.75rem;
            padding: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            backdrop-filter: blur(10px);
        }}
        .kpi-title {{
            color: {t['TEXT_SUB']};
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 800;
            color: {t['TEXT_COLOR']};
        }}
        .kpi-delta {{
            font-size: 0.75rem;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        .kpi-sub {{
            font-size: 0.7rem;
            color: {t['TEXT_SUB']};
            font-style: italic;
            margin-top: 0.5rem;
        }}

        /* Sidebar Buttons */
        [data-testid="stSidebar"] .stButton > button {
            color: #ffffff !important;
            background: linear-gradient(90deg, {t['COLOR_BLUE']}, {t['COLOR_BLUE_DARK']});
            border: none;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            background: linear-gradient(90deg, {t['COLOR_BLUE_DARK']}, {t['COLOR_BLUE']});
        }
        [data-testid="stSidebar"] .stButton > button:active {
            transform: translateY(0);
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }

        /* Input fields in Sidebar */
        [data-testid="stSidebar"] input {
            color: #334155 !important;
            background-color: #ffffff !important;
        }
        
        /* Mobile */
        @media (max-width: 768px) {{
            .block-container {{
                padding: 1rem;
            }}
            .kpi-card {{
                padding: 0.75rem;
                margin-bottom: 0.5rem;
            }}
            .kpi-value {{ font-size: 1.4rem; }}
        }}
        @media only screen and (orientation: landscape) and (max-width: 900px) {{
            .block-container {{
                padding: 0rem 0.5rem !important;
            }}
            .kpi-value {{ font-size: 1.2rem; }}
        }}
    </style>
    """, unsafe_allow_html=True)
