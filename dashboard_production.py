import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime as dt

from utils.style import init_theme, toggle_theme, get_theme_config, load_css
from utils.data import (
    fetch_metrics_parallel,
    fetch_single_metric,
    get_catalog,
    get_value_col,
    calculate_periodicity,
    extract_spot_price,
    extract_scarcity,
    extract_demand,
    extract_offer,
    trim_partial_tail,
)
from utils.charts import style_fig, plot_scarcity_line
from utils.ui import kpi_card_html, render_chart_controls

# ======================================================================
# PAGE CONFIG (must be first Streamlit call)
# ======================================================================
st.set_page_config(
    page_title="Indicadores MEM - Enerconsult",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# THEME + CSS
# ======================================================================
init_theme()
load_css()

# ======================================================================
# METRICS CATALOG (tuple so it's hashable for the cache key)
# ======================================================================
SUMMARY_METRICS = (
    ("PrecBolsNaci", "Sistema"),
    ("PrecEsca", "Sistema"),
    ("PrecEscaSup", "Sistema"),
    ("PrecEscaInf", "Sistema"),
    ("DemaCome", "Sistema"),
    ("MaxPrecOferNal", "Sistema"),
    ("Gene", "Sistema"),
    ("CapaUtilDiarEner", "Sistema"),
    ("VoluUtilDiarEner", "Sistema"),
    ("AporEner", "Sistema"),
    ("AporEnerMediHist", "Sistema"),
)


def _classify_risk(pressure_pct, hydro_dev_pct, util_pct):
    if pressure_pct > 100 and hydro_dev_pct < -15:
        return "CRITICO"
    if (90 <= pressure_pct <= 100) or util_pct > 80:
        return "ALERTA"
    return "NORMAL"


def _pct_change(current, previous):
    if previous in (None, 0):
        return 0.0
    return ((current - previous) / previous) * 100.0


def _metric_to_daily_value(df, mode="mean"):
    if df is None or df.empty or "Date" not in df.columns:
        return None
    tmp = df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    hcols = [c for c in tmp.columns if "Hour" in c]

    if hcols:
        if mode == "sum":
            tmp["_value"] = tmp[hcols].sum(axis=1)
        else:
            tmp["_value"] = tmp[hcols].mean(axis=1)
    else:
        col = get_value_col(tmp)
        if col is None:
            return None
        tmp["_value"] = pd.to_numeric(tmp[col], errors="coerce")

    agg = "sum" if mode == "sum" else "mean"
    out = tmp[["Date", "_value"]].dropna().groupby("Date", as_index=False)["_value"].agg(agg)
    return out


def _aggregate_period_sum(df):
    daily = _metric_to_daily_value(df, mode="sum")
    if daily is None or daily.empty:
        return 0.0
    return float(daily["_value"].sum())


def _add_chart_stats(fig, df, value_col, label, color):
    if df is None or df.empty or value_col not in df.columns:
        return
    data = df[["Date", value_col]].dropna()
    if data.empty:
        return

    vmax_idx = data[value_col].idxmax()
    vmin_idx = data[value_col].idxmin()
    vmax = data.loc[vmax_idx]
    vmin = data.loc[vmin_idx]
    vavg = float(data[value_col].mean())

    same_point = (vmax["Date"] == vmin["Date"]) and (float(vmax[value_col]) == float(vmin[value_col]))
    fig.add_trace(go.Scatter(
        x=[vmax["Date"]], y=[vmax[value_col]],
        name=f"Máx {label}", mode="markers",
        marker=dict(color=color, size=9, symbol="diamond"),
        hovertemplate=f"Máx {label}: "+"%{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>",
    ))
    if not same_point:
        fig.add_trace(go.Scatter(
            x=[vmin["Date"]], y=[vmin[value_col]],
            name=f"Mín {label}", mode="markers",
            marker=dict(color=color, size=9, symbol="diamond-open"),
            hovertemplate=f"Mín {label}: "+"%{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=data["Date"], y=[vavg] * len(data),
        name=f"Prom {label}", mode="lines",
        line=dict(color=color, width=1.5, dash="dot"),
        opacity=0.45,
        hovertemplate=f"Prom {label}: {vavg:.2f}<extra></extra>",
    ))

# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.image("logo_empresa.png", width=150)
    st.markdown("<br>", unsafe_allow_html=True)

    st.toggle(
        "Modo Dark Premium 🌙",
        value=(st.session_state.theme == "dark"),
        key="toggle_dark_mode",
        on_change=toggle_theme,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    today = dt.datetime.now()
    start_default = today - dt.timedelta(days=30)

    st.markdown("### 📅 PERIODO")
    start_date = st.date_input("Inicio", start_default)
    end_date = st.date_input("Fin", today)

    st.markdown("---")
    st.markdown("### PRINCIPALES")
    selection = st.radio(
        "Navegación",
        ["Resumen", "Informe MEM", "Explorador"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("""
    <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.1);
                display:flex;align-items:center;gap:0.75rem;">
        <div style="width:32px;height:32px;background-color:rgba(255,255,255,0.1);
                    border-radius:50%;display:flex;align-items:center;justify-content:center;">📊</div>
        <div>
            <div style="font-weight:700;font-size:0.8rem;color:white;">Fuente de Datos</div>
            <div style="font-size:0.65rem;color:#94a3b8;">XM S.A. E.S.P.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================================
# CHART FRAGMENTS  (only the chart re-runs when its radio changes)
# ======================================================================

@st.fragment
def _chart_price_vs_scarcity(df_bolsa, df_escasez, df_esc_sup, df_esc_inf):
    """Precio de Bolsa vs Bandas de Escasez."""
    t = get_theme_config()
    st.markdown("### Precio de Bolsa vs Escasez")
    col_ctrl, _ = st.columns([1, 4])
    with col_ctrl:
        period = render_chart_controls("period_price")

    df_b = calculate_periodicity(df_bolsa, period, "mean")
    df_e = calculate_periodicity(df_escasez, period, "mean")
    df_es = calculate_periodicity(df_esc_sup, period, "mean")
    df_ei = calculate_periodicity(df_esc_inf, period, "mean")

    if df_b is not None and not df_b.empty:
        fig = go.Figure()
        vcol = get_value_col(df_b)
        fig.add_trace(go.Scatter(
            x=df_b["Date"], y=df_b[vcol],
            name="Precio Bolsa", mode="lines+markers",
            line=dict(color=t["COLOR_ORANGE"]),
        ))
        _add_chart_stats(fig, df_b, vcol, "Bolsa", t["COLOR_ORANGE"])

        bx = df_b["Date"] if period == "1D" else None
        for src, name, color, dash in [
            (df_e, "Precio Escasez", t["COLOR_BLUE_DARK"], "solid"),
            (df_es, "Escasez Superior", "#ef4444", "dot"),
            (df_ei, "Escasez Inferior", "#22c55e", "dot"),
        ]:
            trace = plot_scarcity_line(src, name, color, dash, bx)
            if trace:
                fig.add_trace(trace)

        st.plotly_chart(style_fig(fig, "COP / kWh"), use_container_width=True)


@st.fragment
def _chart_demand_vs_gen(df_demanda, df_gen):
    """Demanda Comercial vs Generación Real."""
    t = get_theme_config()
    st.markdown("### Demanda Comercial vs Generación (GWh)")
    col_ctrl, _ = st.columns([1, 4])
    with col_ctrl:
        period = render_chart_controls("period_dem")

    df_d = calculate_periodicity(df_demanda, period, "sum")
    df_g = calculate_periodicity(df_gen, period, "sum")

    if df_d is not None and not df_d.empty:
        vcol = get_value_col(df_d)
        df_d = trim_partial_tail(df_d, vcol)
        y_dem = df_d[vcol] / 1e6

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_d["Date"], y=y_dem,
            name="Demanda Comercial", mode="lines+markers",
            line=dict(color=t["COLOR_BLUE"]),
        ))
        df_dem_stats = df_d[["Date"]].copy()
        df_dem_stats["_stat"] = y_dem.values
        _add_chart_stats(fig, df_dem_stats, "_stat", "Demanda", t["COLOR_BLUE"])

        if df_g is not None and not df_g.empty:
            gcol = get_value_col(df_g)
            fig.add_trace(go.Scatter(
                x=df_g["Date"], y=df_g[gcol] / 1e6,
                name="Generación Real", mode="lines+markers",
                line=dict(color=t["COLOR_ORANGE"]),
            ))

        st.plotly_chart(style_fig(fig, "GWh"), use_container_width=True)


@st.fragment
def _chart_capacity_volume(df_cap, df_vol):
    """Capacidad Útil vs Volumen Útil."""
    st.markdown("### Capacidad Útil vs Volumen Útil (GWh)")
    col_ctrl, _ = st.columns([1, 4])
    with col_ctrl:
        period = render_chart_controls("period_cap", options=["1M", "1Y"])

    df_c = calculate_periodicity(df_cap, period, "mean")
    df_v = calculate_periodicity(df_vol, period, "mean")

    if df_v is not None and not df_v.empty:
        fig = go.Figure()
        vcol = get_value_col(df_v)
        fig.add_trace(go.Scatter(
            x=df_v["Date"], y=df_v[vcol] / 1e6,
            name="Volumen Útil", mode="none",
            fill="tozeroy", fillcolor="rgba(19,127,236,0.4)",
        ))
        df_vol_stats = df_v[["Date"]].copy()
        df_vol_stats["_stat"] = (df_v[vcol] / 1e6).values
        _add_chart_stats(fig, df_vol_stats, "_stat", "Volumen", "#137fec")
        if df_c is not None and not df_c.empty:
            ccol = get_value_col(df_c)
            fig.add_trace(go.Scatter(
                x=df_c["Date"], y=df_c[ccol] / 1e6,
                name="Capacidad Útil", mode="none",
                fill="tonexty", fillcolor="rgba(148,163,184,0.2)",
            ))
        st.plotly_chart(style_fig(fig, "GWh"), use_container_width=True)
    else:
        st.info("Cargando datos de Embalses...")


@st.fragment
def _chart_hydro_contributions(df_apor, df_media):
    """Aportes hídricos vs Media Histórica."""
    t = get_theme_config()
    st.markdown("### Aportes Hídricos vs Media Histórica (GWh)")
    col_ctrl, _ = st.columns([1, 4])
    with col_ctrl:
        period = render_chart_controls("period_apor", options=["1M", "1Y"])

    df_a = calculate_periodicity(df_apor, period, "sum")
    df_m = calculate_periodicity(df_media, period, "sum")

    if df_a is not None and not df_a.empty:
        fig = go.Figure()
        acol = get_value_col(df_a)
        fig.add_trace(go.Bar(
            x=df_a["Date"], y=df_a[acol] / 1e6,
            name="Aportes Hídricos", marker_color=t["COLOR_BLUE"],
        ))
        df_apor_stats = df_a[["Date"]].copy()
        df_apor_stats["_stat"] = (df_a[acol] / 1e6).values
        _add_chart_stats(fig, df_apor_stats, "_stat", "Aportes", t["COLOR_BLUE"])
        if df_m is not None and not df_m.empty:
            mcol = get_value_col(df_m)
            fig.add_trace(go.Scatter(
                x=df_m["Date"], y=df_m[mcol] / 1e6,
                name="Media Histórica", mode="lines",
                line=dict(color=t["COLOR_ORANGE"], width=3),
            ))
        st.plotly_chart(style_fig(fig, "GWh"), use_container_width=True)
    else:
        st.info("Cargando datos de Aportes...")


@st.fragment
def _chart_scarcity_risk(df_bolsa, df_escasez, df_esc_sup):
    """Índice de presión del mercado vs precio de escasez superior."""
    t = get_theme_config()
    st.markdown("### Índice de Presión del Mercado")
    col_ctrl, _ = st.columns([1, 4])
    with col_ctrl:
        period = render_chart_controls("period_risk", options=["1M", "1Y"])

    df_b = calculate_periodicity(df_bolsa, period, "mean")
    df_e = calculate_periodicity(df_escasez, period, "mean")
    df_s = calculate_periodicity(df_esc_sup, period, "mean")

    if any(x is None or x.empty for x in [df_b, df_e, df_s]):
        st.info("Sin datos suficientes para construir el índice.")
        return

    bcol = get_value_col(df_b)
    ecol = get_value_col(df_e)
    scol = get_value_col(df_s)

    risk = (
        df_b[["Date", bcol]].rename(columns={bcol: "Bolsa"})
        .merge(df_e[["Date", ecol]].rename(columns={ecol: "Escasez"}), on="Date", how="inner")
        .merge(df_s[["Date", scol]].rename(columns={scol: "EscSup"}), on="Date", how="inner")
    )
    risk = risk[(risk["EscSup"] > 0) & (risk["Bolsa"] > 0)]
    if risk.empty:
        st.info("Sin datos válidos para el índice.")
        return

    risk["Indice"] = (risk["Bolsa"] / risk["EscSup"]) * 100.0
    risk["Brecha"] = risk["Bolsa"] - risk["Escasez"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=risk["Date"], y=risk["Indice"],
        name="Presión (%)", mode="lines+markers",
        line=dict(color=t["COLOR_ORANGE"], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=risk["Date"], y=[100.0] * len(risk),
        name="Umbral 100%", mode="lines",
        line=dict(color="#ef4444", dash="dash"),
    ))
    st.plotly_chart(style_fig(fig, "%"), use_container_width=True)

    last = risk.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Presión Actual", f"{last['Indice']:.1f}%")
    c2.metric("Brecha Bolsa-Escasez", f"${last['Brecha']:.1f}")
    c3.metric("Precio Escasez Superior", f"${last['EscSup']:.1f}")


@st.fragment
def _chart_hydro_efficiency(df_cap, df_vol, df_apor, df_media):
    """Uso de embalse y desviación de aportes frente a media histórica."""
    t = get_theme_config()
    st.markdown("### Eficiencia Hidrológica del Sistema")
    col_ctrl, _ = st.columns([1, 4])
    with col_ctrl:
        period = render_chart_controls("period_hydro_eff", options=["1M", "1Y"])

    df_c = calculate_periodicity(df_cap, period, "mean")
    df_v = calculate_periodicity(df_vol, period, "mean")
    df_a = calculate_periodicity(df_apor, period, "sum")
    df_m = calculate_periodicity(df_media, period, "sum")

    if any(x is None or x.empty for x in [df_c, df_v, df_a, df_m]):
        st.info("Sin datos suficientes para el tablero hidrológico.")
        return

    ccol = get_value_col(df_c)
    vcol = get_value_col(df_v)
    acol = get_value_col(df_a)
    mcol = get_value_col(df_m)

    util = (
        df_v[["Date", vcol]].rename(columns={vcol: "Vol"})
        .merge(df_c[["Date", ccol]].rename(columns={ccol: "Cap"}), on="Date", how="inner")
    )
    util = util[util["Cap"] > 0]
    util["Utilizacion"] = (util["Vol"] / util["Cap"]) * 100.0

    aportes = (
        df_a[["Date", acol]].rename(columns={acol: "Aportes"})
        .merge(df_m[["Date", mcol]].rename(columns={mcol: "Media"}), on="Date", how="inner")
    )
    aportes = aportes[aportes["Media"] > 0]
    aportes["Desvio"] = ((aportes["Aportes"] / aportes["Media"]) - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=aportes["Date"], y=aportes["Desvio"],
        name="Desvío Aportes vs Media (%)", marker_color=t["COLOR_BLUE"],
        opacity=0.45,
    ))
    fig.add_trace(go.Scatter(
        x=util["Date"], y=util["Utilizacion"],
        name="Utilización de Embalse (%)", mode="lines+markers",
        line=dict(color=t["COLOR_ORANGE"], width=3),
    ))
    _add_chart_stats(fig, util, "Utilizacion", "Embalse", t["COLOR_ORANGE"])
    st.plotly_chart(style_fig(fig, "%"), use_container_width=True)

    if not util.empty and not aportes.empty:
        u = util.iloc[-1]["Utilizacion"]
        d = aportes.iloc[-1]["Desvio"]
        c1, c2 = st.columns(2)
        c1.metric("Utilización Embalse", f"{u:.1f}%")
        c2.metric("Desvío de Aportes", f"{d:.1f}%")


def _build_market_risk_frame(df_bolsa, df_escasez, df_esc_sup, period):
    if period in ("1M", "1Y"):
        df_b = calculate_periodicity(df_bolsa, period, "mean")
        df_e = calculate_periodicity(df_escasez, period, "mean")
        df_s = calculate_periodicity(df_esc_sup, period, "mean")
        b = _metric_to_daily_value(df_b, mode="mean")
        e = _metric_to_daily_value(df_e, mode="mean")
        s = _metric_to_daily_value(df_s, mode="mean")
    else:
        b = _metric_to_daily_value(df_bolsa, mode="mean")
        e = _metric_to_daily_value(df_escasez, mode="mean")
        s = _metric_to_daily_value(df_esc_sup, mode="mean")

    if any(x is None or x.empty for x in [b, e, s]):
        return None

    risk = (
        b.rename(columns={"_value": "Bolsa"})
        .merge(e.rename(columns={"_value": "Escasez"}), on="Date", how="inner")
        .merge(s.rename(columns={"_value": "EscSup"}), on="Date", how="inner")
    )
    risk = risk[(risk["EscSup"] > 0) & (risk["Bolsa"] > 0)].sort_values("Date")
    if risk.empty:
        return None
    risk["PresionPct"] = (risk["Bolsa"] / risk["EscSup"]) * 100.0
    risk["Brecha"] = risk["Bolsa"] - risk["Escasez"]
    return risk


def _build_hydro_frame(df_cap, df_vol, df_apor, df_media, period):
    if period in ("1M", "1Y"):
        df_c = calculate_periodicity(df_cap, period, "mean")
        df_v = calculate_periodicity(df_vol, period, "mean")
        df_a = calculate_periodicity(df_apor, period, "sum")
        df_m = calculate_periodicity(df_media, period, "sum")
        c = _metric_to_daily_value(df_c, mode="mean")
        v = _metric_to_daily_value(df_v, mode="mean")
        a = _metric_to_daily_value(df_a, mode="sum")
        m = _metric_to_daily_value(df_m, mode="sum")
    else:
        c = _metric_to_daily_value(df_cap, mode="mean")
        v = _metric_to_daily_value(df_vol, mode="mean")
        a = _metric_to_daily_value(df_apor, mode="sum")
        m = _metric_to_daily_value(df_media, mode="sum")

    if any(x is None or x.empty for x in [c, v, a, m]):
        return None

    hydro = (
        v.rename(columns={"_value": "Vol"})
        .merge(c.rename(columns={"_value": "Cap"}), on="Date", how="inner")
        .merge(a.rename(columns={"_value": "Aportes"}), on="Date", how="inner")
        .merge(m.rename(columns={"_value": "Media"}), on="Date", how="inner")
    )
    hydro = hydro[(hydro["Cap"] > 0) & (hydro["Media"] > 0)].sort_values("Date")
    if hydro.empty:
        return None
    hydro["UtilPct"] = (hydro["Vol"] / hydro["Cap"]) * 100.0
    hydro["HydroDevPct"] = ((hydro["Aportes"] / hydro["Media"]) - 1.0) * 100.0
    return hydro


# ======================================================================
# VIEWS
# ======================================================================

if selection == "Resumen":
    t = get_theme_config()

    # --- Header ---
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;">
            <h1 style="margin:0;font-size:1.5rem;">Indicadores MEM - Colombia</h1>
            <span style="background-color:{t['COLOR_ORANGE']}20;color:{t['COLOR_ORANGE']};
                         padding:2px 8px;border-radius:4px;font-size:0.65rem;
                         font-weight:700;text-transform:uppercase;">En Vivo</span>
        </div>
        <p style="color:{t['TEXT_SUB']};font-size:0.8rem;margin-top:0.25rem;">
            Vista general del mercado para el periodo seleccionado.
        </p>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div style="text-align:right;color:{t['TEXT_SUB']};font-size:0.8rem;">
            {dt.datetime.now().strftime('%d %B, %Y')}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Parallel data fetch (single call for all 11 metrics) ---
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    with st.spinner("Actualizando indicadores..."):
        data = fetch_metrics_parallel(SUMMARY_METRICS, start_str, end_str)

    df_bolsa       = data.get("PrecBolsNaci")
    df_escasez     = data.get("PrecEsca")
    df_escasez_sup = data.get("PrecEscaSup")
    df_escasez_inf = data.get("PrecEscaInf")
    df_demanda     = data.get("DemaCome")
    df_oferta      = data.get("MaxPrecOferNal")
    df_gen         = data.get("Gene")
    df_cap         = data.get("CapaUtilDiarEner")
    df_vol         = data.get("VoluUtilDiarEner")
    df_apor        = data.get("AporEner")
    df_media       = data.get("AporEnerMediHist")

    # --- KPI Cards ---
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
        <div style="width:4px;height:1.5rem;background-color:{t['COLOR_ORANGE']};border-radius:2px;"></div>
        <h2 style="margin:0;font-size:1.25rem;">Indicadores Clave del Día</h2>
    </div>
    """, unsafe_allow_html=True)

    val_b, delta_b, date_b, prog_b       = extract_spot_price(df_bolsa)
    val_e, delta_e, prog_e               = extract_scarcity(df_escasez, df_escasez_sup)
    val_d, delta_d, date_d, prog_d       = extract_demand(df_demanda)
    mx_o, mn_o, avg_o, date_o, prog_o    = extract_offer(df_oferta)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card_html(
            "Precio Bolsa (Promedio Día)", f"${val_b:,.1f}", delta_b,
            f"Fecha: {date_b}", progress=prog_b,
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card_html(
            "Precio Escasez (Mes)", f"${val_e:,.1f}", delta_e,
            "Activación", progress=prog_e,
        ), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card_html(
            "Demanda Comercial (GWh)", f"{val_d / 1e6:,.1f}", delta_d,
            f"Fecha: {date_d}", progress=prog_d,
        ), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card_html(
            "Máximo Precio Ofertado", f"${avg_o:,.1f}", 0,
            f"Max: ${mx_o:,.0f} | Min: ${mn_o:,.0f} | Fecha: {date_o}",
            progress=prog_o,
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts (each is an independent fragment) ---
    _chart_price_vs_scarcity(df_bolsa, df_escasez, df_escasez_sup, df_escasez_inf)
    st.markdown("<br>", unsafe_allow_html=True)

    _chart_demand_vs_gen(df_demanda, df_gen)
    st.markdown("<br>", unsafe_allow_html=True)

    _chart_capacity_volume(df_cap, df_vol)
    st.markdown("<br>", unsafe_allow_html=True)

    _chart_hydro_contributions(df_apor, df_media)


elif selection == "Informe MEM":
    st.title("📝 Informe MEM")
    st.caption("Análisis automático del periodo seleccionado con semáforo y hallazgos operativos.")
    st.info(
        "El informe compara el periodo seleccionado contra el periodo anterior de igual duración "
        "y resume presión de precios, balance de energía y condición hidrológica."
    )

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    days = max((end_date - start_date).days + 1, 1)
    prev_end = start_date - dt.timedelta(days=1)
    prev_start = prev_end - dt.timedelta(days=days - 1)

    with st.spinner("Construyendo informe..."):
        cur = fetch_metrics_parallel(SUMMARY_METRICS, start_str, end_str)
        prev = fetch_metrics_parallel(
            SUMMARY_METRICS,
            prev_start.strftime("%Y-%m-%d"),
            prev_end.strftime("%Y-%m-%d"),
        )

    market_cur = _build_market_risk_frame(cur.get("PrecBolsNaci"), cur.get("PrecEsca"), cur.get("PrecEscaSup"), "RANGE")
    hydro_cur = _build_hydro_frame(cur.get("CapaUtilDiarEner"), cur.get("VoluUtilDiarEner"), cur.get("AporEner"), cur.get("AporEnerMediHist"), "RANGE")
    market_prev = _build_market_risk_frame(prev.get("PrecBolsNaci"), prev.get("PrecEsca"), prev.get("PrecEscaSup"), "RANGE")
    hydro_prev = _build_hydro_frame(prev.get("CapaUtilDiarEner"), prev.get("VoluUtilDiarEner"), prev.get("AporEner"), prev.get("AporEnerMediHist"), "RANGE")

    pressure_now = util_now = hydro_dev_now = 0.0
    state = "SIN DATOS"
    if market_cur is not None and hydro_cur is not None and not market_cur.empty and not hydro_cur.empty:
        pressure_now = float(market_cur.iloc[-1]["PresionPct"])
        util_now = float(hydro_cur.iloc[-1]["UtilPct"])
        hydro_dev_now = float(hydro_cur.iloc[-1]["HydroDevPct"])
        state = _classify_risk(pressure_now, hydro_dev_now, util_now)

    def _safe_metric(df, col, agg="mean"):
        if df is None or df.empty or col not in df.columns:
            return 0.0
        return float(df[col].mean() if agg == "mean" else df[col].sum())

    price_cur = _safe_metric(market_cur, "Bolsa", "mean")
    price_prev = _safe_metric(market_prev, "Bolsa", "mean")
    pressure_prev = _safe_metric(market_prev, "PresionPct", "mean")
    demand_cur = _aggregate_period_sum(cur.get("DemaCome"))
    demand_prev = _aggregate_period_sum(prev.get("DemaCome"))
    gen_cur = _aggregate_period_sum(cur.get("Gene"))
    gen_prev = _aggregate_period_sum(prev.get("Gene"))
    util_prev = _safe_metric(hydro_prev, "UtilPct", "mean")

    st.markdown("### Resumen Ejecutivo")
    st.markdown(f"- Estado general del periodo: **{state}**.")
    st.markdown(
        f"- Precio bolsa promedio: **${price_cur:,.1f}** "
        f"({ _pct_change(price_cur, price_prev):+.1f}% vs periodo previo)."
    )
    st.markdown(
        f"- Presión de mercado actual: **{pressure_now:.1f}%** "
        f"({ _pct_change(pressure_now, pressure_prev):+.1f}% vs periodo previo)."
    )
    balance_cur = (gen_cur - demand_cur) / 1e6
    balance_prev = (gen_prev - demand_prev) / 1e6
    st.markdown(
        f"- Balance generación-demanda: **{balance_cur:,.1f} GWh** "
        f"({ _pct_change(balance_cur, balance_prev if balance_prev != 0 else 1):+.1f}% vs previo)."
    )
    st.markdown(
        f"- Utilización de embalse: **{util_now:.1f}%** y desvío de aportes: **{hydro_dev_now:.1f}%**."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semáforo", state)
    c2.metric("Precio Bolsa Prom.", f"${price_cur:,.1f}", f"{_pct_change(price_cur, price_prev):+.1f}%")
    c3.metric("Presión Mercado", f"{pressure_now:.1f}%", f"{_pct_change(pressure_now, pressure_prev):+.1f}%")
    c4.metric("Utilización Embalse", f"{util_now:.1f}%", f"{_pct_change(util_now, util_prev):+.1f}%")

    st.markdown("### ¿Cómo se calcula cada valor?")
    last_market_date = market_cur.iloc[-1]["Date"].strftime("%Y-%m-%d") if market_cur is not None and not market_cur.empty else end_str
    st.markdown(
        f"- **${price_cur:,.1f} (Precio Bolsa Prom.)**: promedio aritmético de los precios horarios de bolsa "
        f"entre **{start_str}** y **{end_str}**. Se suma cada valor horario del periodo y se divide por el total de horas."
    )
    st.markdown(
        f"- **{pressure_now:.1f}% (Presión Mercado)**: relación del último día disponible "
        f"(**{last_market_date}**) entre Precio Bolsa y Precio de Escasez Superior: "
        f"`(Bolsa / Escasez Superior) x 100`."
    )
    st.markdown(
        f"- **{balance_cur:,.1f} GWh (Balance Generación-Demanda)**: diferencia acumulada del periodo "
        f"entre generación total y demanda total: `(Σ Generación - Σ Demanda) / 1e6`."
    )
    st.markdown(
        f"- **{util_now:.1f}% (Utilización Embalse)**: porcentaje de uso del embalse en el último día disponible: "
        f"`(Volumen Útil / Capacidad Útil) x 100`."
    )
    st.markdown(
        f"- **{hydro_dev_now:.1f}% (Desvío de Aportes)**: desviación porcentual de aportes frente a su media histórica: "
        f"`((Aportes / Media Histórica) - 1) x 100`."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _chart_price_vs_scarcity(cur.get("PrecBolsNaci"), cur.get("PrecEsca"), cur.get("PrecEscaSup"), cur.get("PrecEscaInf"))
    st.markdown("<br>", unsafe_allow_html=True)
    _chart_demand_vs_gen(cur.get("DemaCome"), cur.get("Gene"))
    st.markdown("<br>", unsafe_allow_html=True)
    _chart_hydro_efficiency(
        cur.get("CapaUtilDiarEner"),
        cur.get("VoluUtilDiarEner"),
        cur.get("AporEner"),
        cur.get("AporEnerMediHist"),
    )

    st.markdown("### Hallazgos Automáticos")
    findings = []
    if pressure_now > 100:
        findings.append("Presión de mercado por encima de 100%: tensión crítica de precio.")
    elif pressure_now >= 90:
        findings.append("Presión de mercado en banda de alerta (90%-100%).")
    if util_now > 80:
        findings.append("Utilización de embalse superior a 80%: holgura operativa reducida.")
    if hydro_dev_now < -15:
        findings.append("Aportes hídricos con desvío menor a -15%: riesgo hidrológico elevado.")
    if balance_cur < 0:
        findings.append("Balance neto generación-demanda negativo en el periodo.")
    if not findings:
        findings.append("No se detectaron eventos críticos con las reglas actuales para este periodo.")
    for item in findings:
        st.markdown(f"- {item}")

    if market_cur is not None and hydro_cur is not None:
        risk_tbl = market_cur[["Date", "PresionPct", "Brecha"]].merge(
            hydro_cur[["Date", "UtilPct", "HydroDevPct"]],
            on="Date",
            how="inner",
        ).sort_values("Date")
        if not risk_tbl.empty:
            risk_tbl["Estado"] = risk_tbl.apply(
                lambda r: _classify_risk(r["PresionPct"], r["HydroDevPct"], r["UtilPct"]),
                axis=1,
            )
            out = risk_tbl.tail(30).copy()
            out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
            out = out.rename(columns={
                "Date": "Fecha",
                "PresionPct": "Presión (%)",
                "Brecha": "Brecha Bolsa-Escasez",
                "UtilPct": "Utilización Embalse (%)",
                "HydroDevPct": "Desvío Aportes (%)",
            })
            st.markdown("### Evidencias del Periodo")
            st.dataframe(out, use_container_width=True, hide_index=True)


elif selection == "Explorador":
    st.title("🔍 Explorador Avanzado")

    with st.spinner("Cargando catálogo..."):
        df_vars = get_catalog()

    if df_vars is not None and not df_vars.empty:
        df_vars["DisplayName"] = df_vars["MetricName"] + " (" + df_vars["Entity"] + ")"
        var_map = df_vars.set_index("DisplayName")[["MetricId", "Entity"]].to_dict("index")

        metric_option = st.selectbox(
            "Seleccione Variable",
            options=sorted(var_map.keys()),
            index=0,
        )

        if st.button("Consultar"):
            meta = var_map[metric_option]
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            with st.spinner("Consultando..."):
                df = fetch_single_metric(
                    meta["MetricId"], meta["Entity"], start_str, end_str,
                )

            if df is not None and not df.empty:
                t = get_theme_config()
                cols = [
                    c for c in df.columns
                    if c not in ("Date", "Id", "Entity", "MetricId", "Values_code")
                ]
                if cols:
                    fig = px.line(
                        df, x="Date", y=cols[0],
                        title=metric_option,
                        color_discrete_sequence=[t["COLOR_BLUE"]],
                    )
                    st.plotly_chart(style_fig(fig), use_container_width=True)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("Sin datos para la variable seleccionada en este periodo.")
    else:
        st.error("No se pudo cargar el catálogo de variables.")


# ======================================================================
# FOOTER
# ======================================================================
st.markdown("---")
st.markdown("""
<div style="display:flex;justify-content:center;color:#94a3b8;font-size:0.75rem;">
    <b>POWERED BY STREAMLIT</b>
</div>
""", unsafe_allow_html=True)
