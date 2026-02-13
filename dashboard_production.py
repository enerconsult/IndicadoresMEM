import streamlit as st
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils.style import init_theme, toggle_theme, get_theme_config, load_css
from utils.data import (
    fetch_metric_data, get_catalog, calculate_periodicity, get_chart_val_col,
    get_spot_price_metric, get_scarcity_metric, get_robust_demand_metric, get_offer_metric
)
from utils.ui import kpi_card_html, render_chart_controls, render_sidebar, render_footer, render_sidebar_footer
from utils.charts import style_fig, plot_scarcity_line

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Indicadores MEM - Enerconsult",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME MANAGEMENT ---
init_theme()
load_css()
theme = get_theme_config()

# --- SIDEBAR ---
with st.sidebar:
    render_sidebar()
    st.toggle("Modo Dark Premium 🌙", value=(st.session_state.theme == 'dark'), key="toggle_dark_mode", on_change=toggle_theme)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Global Date Filter
    today = dt.datetime.now()
    start_default = today - dt.timedelta(days=365)
    
    st.markdown("### 📅 PERIODO", unsafe_allow_html=True)
    start_date = st.date_input("Inicio", start_default)
    end_date = st.date_input("Fin", today)

    st.markdown("---")
    
    # Navigation
    st.markdown("### PRINCIPALES", unsafe_allow_html=True)
    menu_options = ["Resumen", "Explorador"]
    selection = st.radio("Navegación", menu_options, label_visibility="collapsed")
    
    render_sidebar_footer()

# --- MAIN APP LOGIC ---

if selection == "Resumen":
    # Header
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem;">
            <h1 style="margin: 0; font-size: 1.5rem;">Indicadores MEM - Colombia</h1>
            <span style="background-color: {theme['COLOR_ORANGE']}20; color: {theme['COLOR_ORANGE']}; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;">En Vivo</span>
        </div>
        <p style="color: {theme['TEXT_SUB']}; font-size: 0.8rem; margin-top: 0.25rem;">Vista general del mercado para el periodo seleccionado.</p>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div style="text-align: right; color: {theme['TEXT_SUB']}; font-size: 0.8rem;">
            <span class="material-symbols-outlined" style="vertical-align: bottom; font-size: 1rem;">calendar_today</span>
            {dt.datetime.now().strftime('%d %B, %Y')}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. KPI CARDS
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
        <div style="width: 4px; height: 1.5rem; background-color: {theme['COLOR_ORANGE']}; border-radius: 2px;"></div>
        <h2 style="margin: 0; font-size: 1.25rem;">Indicadores Clave del Día</h2>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Data Fetching
    with st.spinner("Actualizando indicadores..."):
        # 1. Spot Price
        df_bolsa = fetch_metric_data("PrecBolsNaci", "Sistema", start_date, end_date)
        val_bolsa, delta_bolsa, date_bolsa, prog_bolsa = get_spot_price_metric(df_bolsa)
        
        # 2. Scarcity Price
        df_escasez = fetch_metric_data("PrecEsca", "Sistema", start_date, end_date)
        df_escasez_sup = fetch_metric_data("PrecEscaSup", "Sistema", start_date, end_date)
        df_escasez_inf = fetch_metric_data("PrecEscaInf", "Sistema", start_date, end_date)
        
        val_escasez, delta_escasez, prog_escasez = get_scarcity_metric(df_escasez, df_escasez_sup)
        
        # 3. System Demand
        df_demanda = fetch_metric_data("DemaCome", "Sistema", start_date, end_date)
        val_demanda, delta_demanda, date_demanda, prog_demanda = get_robust_demand_metric(df_demanda)
        
        # 4. Max Offer
        df_oferta = fetch_metric_data("MaxPrecOferNal", "Sistema", start_date, end_date)
        max_oferta, min_oferta, avg_oferta, date_oferta, prog_oferta = get_offer_metric(df_oferta)

    with c1: st.markdown(kpi_card_html("Precio Bolsa (Promedio Dia)", f"${val_bolsa:,.1f}", delta_bolsa, f"Fecha: {date_bolsa}", progress=prog_bolsa), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card_html("Precio Escasez (Mes)", f"${val_escasez:,.1f}", delta_escasez, "Activación", progress=prog_escasez), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card_html("Demanda Comercial (GWh)", f"{val_demanda/1e6:,.1f}", delta_demanda, f"Fecha: {date_demanda}", progress=prog_demanda), unsafe_allow_html=True) 
    with c4: st.markdown(kpi_card_html("Máximo precio Ofertado", f"${avg_oferta:,.1f}", 0, f"Max: ${max_oferta:,.0f} | Min: ${min_oferta:,.0f} | Fecha: {date_oferta}", progress=prog_oferta), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. CHARTS SECTION
    
    # --- CHART 1: Spot vs Scarcity ---
    st.markdown("### Precio de Bolsa vs Escasez")
    col_p, col_c = st.columns([1, 4])
    with col_p:
        period_price = render_chart_controls("period_price")
    
    df_bolsa_chart = calculate_periodicity(df_bolsa, period_price, 'mean')
    df_escasez_chart = calculate_periodicity(df_escasez, period_price, 'mean')
    df_escasez_sup_chart = calculate_periodicity(df_escasez_sup, period_price, 'mean')
    df_escasez_inf_chart = calculate_periodicity(df_escasez_inf, period_price, 'mean')
    
    if df_bolsa_chart is not None and not df_bolsa_chart.empty:
        fig_price = go.Figure()
        
        val_col_bolsa = get_chart_val_col(df_bolsa_chart)
        
        # Spot Price
        fig_price.add_trace(go.Scatter(
            x=df_bolsa_chart['Date'], y=df_bolsa_chart[val_col_bolsa], 
            name='Precio Bolsa', mode='lines+markers', line=dict(color=theme['COLOR_ORANGE'])
        ))
        
        # Scarcity Lines
        broadcast_x = df_bolsa_chart['Date'] if period_price == '1D' else None
        
        # 1. Base
        trace_base = plot_scarcity_line(df_escasez_chart, 'Precio Escasez', theme['COLOR_BLUE_DARK'], 'solid', broadcast_x)
        if trace_base: fig_price.add_trace(trace_base)
        
        # 2. Superior
        trace_sup = plot_scarcity_line(df_escasez_sup_chart, 'Precio Escasez Superior', '#ef4444', 'dot', broadcast_x) # Red
        if trace_sup: fig_price.add_trace(trace_sup)
        
        # 3. Inferior
        trace_inf = plot_scarcity_line(df_escasez_inf_chart, 'Precio Escasez Inferior', '#22c55e', 'dot', broadcast_x) # Green
        if trace_inf: fig_price.add_trace(trace_inf)
        
        fig_price = style_fig(fig_price, "COP / kWh")
        st.plotly_chart(fig_price, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- CHART 2: Demand vs Generation ---
    st.markdown("### Demanda Comercial vs Generación (GWh)")
    col_d, col_dc = st.columns([1, 4])
    with col_d:
        period_dem = render_chart_controls("period_dem")
        
    df_gen = fetch_metric_data("Gene", "Sistema", start_date, end_date)
    
    # Aggregation: Sum for GWh
    df_dem_chart = calculate_periodicity(df_demanda, period_dem, 'sum')
    df_gen_chart = calculate_periodicity(df_gen, period_dem, 'sum')

    if df_dem_chart is not None and not df_dem_chart.empty:
        fig_dg = go.Figure()
        val_col_dem = get_chart_val_col(df_dem_chart)
        
        # Simple Drop Last Logic
        if len(df_dem_chart) > 2:
             last_val = df_dem_chart.iloc[-1][val_col_dem]
             prev_val = df_dem_chart.iloc[-2][val_col_dem]
             if last_val < 0.6 * prev_val:
                 df_dem_chart = df_dem_chart.iloc[:-1]
        
        # Scale to GWh
        y_dem = df_dem_chart[val_col_dem] / 1e6
        
        fig_dg.add_trace(go.Scatter(
            x=df_dem_chart['Date'], y=y_dem,
            name='Demanda Comercial', mode='lines+markers', line=dict(color=theme['COLOR_BLUE'])
        ))
        
        if df_gen_chart is not None and not df_gen_chart.empty:
            val_col_gen = get_chart_val_col(df_gen_chart)
            y_gen = df_gen_chart[val_col_gen] / 1e6
            fig_dg.add_trace(go.Scatter(
                x=df_gen_chart['Date'], y=y_gen,
                name='Generación Real', mode='lines+markers', line=dict(color=theme['COLOR_ORANGE'])
            ))
            
        fig_dg = style_fig(fig_dg, "GWh")
        st.plotly_chart(fig_dg, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CHART 3: Capacity vs Volume ---
    st.markdown("### Capacidad Útil vs Volumen Útil (GWh)")
    col_c, col_cc = st.columns([1, 4])
    with col_c:
        period_cap = render_chart_controls("period_cap", options=["1M", "1Y"])
    
    df_cap = fetch_metric_data("CapaUtilDiarEner", "Sistema", start_date, end_date)
    df_vol = fetch_metric_data("VoluUtilDiarEner", "Sistema", start_date, end_date)

    df_cap_chart = calculate_periodicity(df_cap, period_cap, 'mean')
    df_vol_chart = calculate_periodicity(df_vol, period_cap, 'mean')
    
    if df_vol_chart is not None and not df_vol_chart.empty:
        fig_cv = go.Figure()
        
        val_col_vol = get_chart_val_col(df_vol_chart)
        y_vol = df_vol_chart[val_col_vol] / 1e6
        fig_cv.add_trace(go.Scatter(
            x=df_vol_chart['Date'], y=y_vol,
            name='Volumen Útil', mode='none', fill='tozeroy', fillcolor='rgba(19, 127, 236, 0.4)'
        ))

        if df_cap_chart is not None and not df_cap_chart.empty:
            val_col_cap = get_chart_val_col(df_cap_chart)
            y_cap = df_cap_chart[val_col_cap] / 1e6
            fig_cv.add_trace(go.Scatter(
                x=df_cap_chart['Date'], y=y_cap,
                name='Capacidad Útil', mode='none', fill='tonexty', fillcolor='rgba(148, 163, 184, 0.2)'
            ))
            
        fig_cv = style_fig(fig_cv, "GWh")
        st.plotly_chart(fig_cv, use_container_width=True)
    else:
        st.info("Cargando datos de Embalses...")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CHART 4: Contributions (Aportes) ---
    st.markdown("### Aportes hídricos vs Media histórica (GWh)")
    col_a, col_ac = st.columns([1, 4])
    with col_a:
        period_apor = render_chart_controls("period_apor", options=["1M", "1Y"])
        
    df_apor = fetch_metric_data("AporEner", "Sistema", start_date, end_date)
    df_media = fetch_metric_data("AporEnerMediHist", "Sistema", start_date, end_date)
    
    df_apor_chart = calculate_periodicity(df_apor, period_apor, 'sum')
    df_media_chart = calculate_periodicity(df_media, period_apor, 'sum')

    if df_apor_chart is not None and not df_apor_chart.empty:
        fig_ap = go.Figure()
        
        val_col_ap = get_chart_val_col(df_apor_chart)
        y_ap = df_apor_chart[val_col_ap] / 1e6
        
        fig_ap.add_trace(go.Bar(
            x=df_apor_chart['Date'], y=y_ap, 
            name='Aportes Hídricos', marker_color=theme['COLOR_BLUE']
        ))
        
        if df_media_chart is not None and not df_media_chart.empty:
             val_col_media = get_chart_val_col(df_media_chart)
             y_media = df_media_chart[val_col_media] / 1e6
             fig_ap.add_trace(go.Scatter(
                 x=df_media_chart['Date'], y=y_media,
                 name='Media Histórica', mode='lines', line=dict(color=theme['COLOR_ORANGE'], dash='solid', width=3)
             ))
             
        fig_ap = style_fig(fig_ap, "GWh")
        st.plotly_chart(fig_ap, use_container_width=True)
    else:
         st.info("Cargando datos de Aportes...")


elif selection == "Explorador":
    st.title("🔍 Explorador Avanzado")
    with st.spinner("Cargando catálogo..."):
        df_vars = get_catalog()
    
    if df_vars is not None and not df_vars.empty:
        df_vars['DisplayName'] = df_vars['MetricName'] + " (" + df_vars['Entity'] + ")"
        var_map = df_vars.set_index('DisplayName')[['MetricId', 'Entity']].to_dict('index')

        metric_option = st.selectbox("Seleccione Variable", options=sorted(list(var_map.keys())), index=0)

        if st.button("Consultar"):
            meta = var_map[metric_option]
            df = fetch_metric_data(meta['MetricId'], meta['Entity'], start_date, end_date)
            
            if df is not None:
                cols = [c for c in df.columns if c not in ['Date', 'Id', 'Entity', 'MetricId', 'Values_code']]
                if cols:
                    fig = px.line(df, x='Date', y=cols[0], title=metric_option, color_discrete_sequence=[theme['COLOR_BLUE']])
                    st.plotly_chart(style_fig(fig, ""), use_container_width=True)
                st.dataframe(df)
    else:
        st.error("No se pudo cargar el catálogo de variables.")

render_footer()
