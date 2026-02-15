import streamlit as st
from utils.style import get_theme_config


def kpi_card_html(title, value, delta, sub_text="", color_bar=None, progress=1.0):
    """Generate HTML for a KPI card. Uses CSS classes defined in style.load_css()."""
    t = get_theme_config()
    if color_bar is None:
        color_bar = t["COLOR_BLUE"]

    delta_color = "#10b981" if delta >= 0 else "#f43f5e"
    delta_sign = "+" if delta >= 0 else ""
    delta_str = f"{delta_sign}{delta:.1f}%"

    inactive = f"{t['TEXT_SUB']}30"
    c1 = color_bar if progress > 0 else inactive
    c2 = color_bar if progress > 0.33 else inactive
    c3 = color_bar if progress > 0.66 else inactive

    return f"""
    <div class="kpi-card">
        <div>
            <p class="kpi-title">{title}</p>
            <div style="display:flex;align-items:baseline;gap:0.75rem;">
                <span class="kpi-value">{value}</span>
                <span class="kpi-delta" style="color:{delta_color};background-color:{delta_color}20;">{delta_str}</span>
            </div>
            {'<p class="kpi-sub">' + sub_text + '</p>' if sub_text else ''}
        </div>
        <div style="display:flex;align-items:flex-end;gap:4px;height:6px;margin-top:1.5rem;">
            <div style="flex:1;background-color:{c1};height:40%;border-radius:2px;"></div>
            <div style="flex:1;background-color:{c2};height:70%;border-radius:2px;"></div>
            <div style="flex:1;background-color:{c3};height:100%;border-radius:2px;box-shadow:0 0 8px {c3}60;"></div>
        </div>
    </div>
    """


def render_chart_controls(key_prefix, options=None):
    """Inline radio for chart periodicity."""
    if options is None:
        options = ["1D", "1M", "1Y"]
    return st.radio(
        "Periodicidad",
        options,
        horizontal=True,
        key=key_prefix,
        label_visibility="collapsed",
    )
