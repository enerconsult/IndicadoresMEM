import streamlit as st
from utils.style import get_theme_config

def kpi_card_html(title, value, delta, sub_text="", color_bar=None, progress=1.0):
    """
    Generates HTML for a custom KPI card matching the design.
    Adapts to theme via global variables.
    progress: float 0.0 to 1.0 (determines how many bars are filled)
    """
    theme = get_theme_config()
    
    if color_bar is None:
        color_bar = theme['COLOR_BLUE']

    delta_color = "#10b981" if delta >= 0 else "#f43f5e" # Emerald vs Rose
    delta_sign = "+" if delta >= 0 else ""
    delta_str = f"{delta_sign}{delta:.1f}%"
    
    # Dynamic Styles
    card_style = f"background-color: {theme['CARD_BG']}; border: 1px solid {theme['BORDER_COLOR']}; border-radius: 0.75rem; padding: 1.25rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); height: 100%; display: flex; flex-direction: column; justify-content: space-between; backdrop-filter: blur(10px);"
    title_style = f"color: {theme['TEXT_SUB']}; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;"
    value_style = f"font-size: 1.8rem; font-weight: 800; color: {theme['TEXT_COLOR']};"
    sub_style = f"font-size: 0.7rem; color: {theme['TEXT_SUB']}; font-style: italic; margin-top: 0.5rem;"
    
    # Progress Logic (3 Bars)
    # < 0.33: 1 bar (Low intensity)
    # < 0.66: 2 bars (Med intensity)
    # >= 0.66: 3 bars (High intensity)
    
    # Base colors for inactive bars
    bg_inactive = f"{theme['TEXT_SUB']}30"
    
    # Bar 1 (Always active if progress > 0, else inactive)
    c1 = color_bar if progress > 0 else bg_inactive
    
    # Bar 2
    c2 = color_bar if progress > 0.33 else bg_inactive
    
    # Bar 3
    c3 = color_bar if progress > 0.66 else bg_inactive
    
    html = f"""
    <div style="{card_style}">
        <div>
            <p style="{title_style}">{title}</p>
            <div style="display: flex; align-items: baseline; gap: 0.75rem;">
                <span style="{value_style}">{value}</span>
                <span style="font-size: 0.75rem; font-weight: 700; color: {delta_color}; background-color: {delta_color}20; padding: 2px 6px; border-radius: 4px;">{delta_str}</span>
            </div>
            {f'<p style="{sub_style}">{sub_text}</p>' if sub_text else ''}
        </div>
        <div style="display: flex; align-items: flex-end; gap: 4px; height: 6px; margin-top: 1.5rem;">
            <div style="flex: 1; background-color: {c1}; height: 40%; border-radius: 2px;"></div>
            <div style="flex: 1; background-color: {c2}; height: 70%; border-radius: 2px;"></div>
            <div style="flex: 1; background-color: {c3}; height: 100%; border-radius: 2px; box-shadow: 0 0 8px {c3}60;"></div>
        </div>
    </div>
    """
    return html

def render_chart_controls(key_prefix, options=["1D", "1M", "1Y"]):
    return st.radio(
        "Periodicidad", 
        options, 
        horizontal=True, 
        key=key_prefix, 
        label_visibility="collapsed"
    )

def render_sidebar():
    """Renders common sidebar elements."""
    # Logo
    try:
        st.sidebar.image("logo_empresa.png", width=150)
    except:
        st.sidebar.title("Indicadores MEM") # Fallback
        
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

def render_sidebar_footer():
    st.markdown(f"""
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 32px; height: 32px; background-color: rgba(255,255,255,0.1); border-radius: 50%; display: flex; align-items: center; justify-content: center;">📊</div>
        <div>
            <div style="font-weight: 700; font-size: 0.8rem; color: white;">Fuente de Datos</div>
            <div style="font-size: 0.65rem; color: #94a3b8;">XM S.A. E.S.P.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; color: #94a3b8; font-size: 0.75rem;">
        <div>
            <b>POWERED BY STREAMLIT</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
