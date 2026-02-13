import streamlit as st

def init_theme():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.toggle_dark_mode else 'light'

def get_theme_config():
    """Returns a dictionary with theme colors based on session state."""
    if st.session_state.get('theme', 'dark') == 'dark':
        return {
            "BG_COLOR": "#0f172a", # Slate 900
            "CARD_BG": "rgba(30, 41, 59, 0.7)", # Slate 800 + Opacity
            "TEXT_COLOR": "#f8fafc", # Slate 50
            "TEXT_SUB": "#94a3b8", # Slate 400
            "BORDER_COLOR": "rgba(255, 255, 255, 0.1)",
            "COLOR_BLUE_DARK": "#60a5fa", # Blue 400 (Lighter for dark mode)
            "COLOR_BLUE": "#3b82f6",
            "COLOR_ORANGE": "#f97316", # Orange 500
            "PLOT_TEMPLATE": "plotly_dark",
            "GRID_COLOR": "#334155",
            "SIDEBAR_BG": "#1e293b"
        }
    else:
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
            "SIDEBAR_BG": "#003366"
        }

def load_css():
    """Injects custom CSS based on the current theme."""
    theme = get_theme_config()
    
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global Styles */
        .stApp {{
            font-family: 'Inter', sans-serif;
            background-color: {theme['BG_COLOR']};
            color: {theme['TEXT_COLOR']};
        }}
        
        html, body {{
            background-color: {theme['BG_COLOR']};
        }}
        
        /* Sidebar Styles */
        [data-testid="stSidebar"] {{
            background-color: {theme['SIDEBAR_BG']};
            border-right: 1px solid {theme['BORDER_COLOR']};
        }}
        [data-testid="stSidebar"] * {{
            color: #e2e8f0 !important;
        }}
        
        /* Remove top padding */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        /* Headers */
        h1, h2, h3 {{
            color: {theme['TEXT_COLOR'] if st.session_state.theme == 'dark' else '#003366'} !important;
            font-weight: 700;
        }}
        
        /* Metrics / Cards - Streamlit Native Override */
        [data-testid="stMetricValue"] {{
            color: {theme['TEXT_COLOR']} !important;
        }}
        
        /* Streamlit widgets */
        .stDateInput label, .stSelectbox label, .stRadio label {{
            color: {theme['TEXT_COLOR']} !important;
        }}
        
        /* FIX: Input text color should be dark if background is white-ish (default streamlit input) */
        .stDateInput input {{
            color: #334155 !important; /* Dark Slate to ensure visibility on light input bg */
            font-weight: 600;
        }}
        
        /* Additional Dark Mode Tweaks */
        {'div[data-testid="stExpander"] { border: 1px solid ' + theme['BORDER_COLOR'] + '; border-radius: 8px; }' if st.session_state.theme == 'dark' else ''}
        
        /* Hide Footer Only */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Radio Button Styling - Pills/Tabs Look */
        div[data-testid="stRadio"] > div {
            display: flex;
            flex-direction: row;
            gap: 8px;
        }
        
        div[data-testid="stRadio"] label {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 4px 12px;
            border-radius: 4px;
            cursor: pointer;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.2s;
            text-align: center;
            margin-right: 0 !important;
            color: {theme['TEXT_COLOR']} !important;
        }

        div[data-testid="stRadio"] label:hover {
            background-color: rgba(255, 255, 255, 0.1);
            border-color: {theme['COLOR_BLUE']};
        }
        
        /* Highlight the active radio button (This relies on Streamlit's internal structure which may vary, 
           but usually the active one has a specific child or state. 
           Since we can't easily target the checked state of the parent label via CSS only in all browsers without :has,
           we will rely on basic styling.
           However, let's try to target the 'p' inside the label if possible or just rely on the dot being hidden 
           and the label being visible.
        */
        
        /* Hide the default radio circle */
        div[data-testid="stRadio"] label div[role="radio"] {
            display: none;
        }
        
    </style>
    """, unsafe_allow_html=True)
