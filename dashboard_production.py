import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime as dt
import requests
import html as pyhtml

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


def _call_ceo_consultant(api_key, user_question, report_context, history):
    instruction = (
        "Actúa como consultor experto del Mercado de Energía Mayorista (MEM) de Colombia.\n"
        "Reglas:\n"
        "- Responde SOLO sobre mercado de energía, operación del sistema, precios, escasez, demanda, generación, embalses y aportes.\n"
        "- Usa primero el contexto del informe provisto.\n"
        "- Si la pregunta está fuera de ese dominio, indícalo brevemente.\n"
        "- No inventes cifras; si no hay dato en el contexto, dilo con claridad.\n"
        "- Responde en español ejecutivo, claro y accionable.\n"
    )

    history_text = []
    for msg in history[-8:]:
        role = "Usuario" if msg.get("role") == "user" else "Consultor"
        history_text.append(f"{role}: {msg.get('content', '')}")
    history_block = "\n".join(history_text)

    prompt = (
        f"{instruction}\n"
        f"Contexto del informe:\n{report_context}\n\n"
        f"Historial reciente:\n{history_block}\n\n"
        f"Pregunta actual del usuario:\n{user_question}\n\n"
        "Entrega una respuesta directa con 3 bloques: 1) lectura del dato, 2) implicación de negocio, 3) recomendación."
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }

    res = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    data = res.json()
    if not res.ok:
        msg = data.get("error", {}).get("message", "Error al consultar Gemini.")
        raise RuntimeError(msg)
    return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

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
        ["Resumen", "Informe del CEO", "Explorador"],
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


elif selection == "Informe del CEO":
    st.markdown("""
    <style>
      .ceo-wrap {max-width: 980px; margin: 0 auto;}
      .ceo-hero {
        background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 52%, #0d9488 100%);
        color: #f8fafc; border-radius: 20px; padding: 20px 24px; margin-bottom: 14px;
        border: 1px solid rgba(56,189,248,0.35);
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.45);
      }
      .ceo-hero h2 {margin: 0; font-size: 1.35rem; font-weight: 900; letter-spacing: 0.01em;}
      .ceo-hero p {margin: 8px 0 0; color: #dbeafe; font-size: 0.92rem;}
      .ceo-chip-row {display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; margin: 12px 0 14px;}
      .ceo-chip {
        border-radius: 14px; padding: 10px 12px; background: rgba(2,6,23,0.6);
        border: 1px solid rgba(148,163,184,0.25);
      }
      .ceo-chip-k {font-size: 11px; color:#93c5fd; text-transform: uppercase; letter-spacing: .08em;}
      .ceo-chip-v {font-size: 18px; color:#f8fafc; font-weight: 800; margin-top: 2px;}
      .ceo-box {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 18px;
        padding: 14px;
      }
      .ceo-chat-shell {
        margin-top: 14px;
        border-radius: 22px;
        border: 1px solid rgba(100,116,139,0.28);
        background: rgba(2,6,23,0.55);
        overflow: hidden;
        box-shadow: 0 16px 36px rgba(2,6,23,0.45);
      }
      .ceo-chat-head {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: #f8fafc;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .ceo-chat-title {font-weight: 800; font-size: 1.1rem; margin: 0;}
      .ceo-chat-sub {font-size: .85rem; color: #dbeafe; margin: 2px 0 0;}
      .ceo-chat-body {
        background: #eef2f7;
        min-height: 420px;
        max-height: 520px;
        overflow-y: auto;
        padding: 16px;
      }
      .ceo-msg-row {
        display: flex;
        gap: 10px;
        margin-bottom: 12px;
        align-items: flex-start;
      }
      .ceo-msg-row.user {justify-content: flex-end;}
      .ceo-avatar {
        width: 34px; height: 34px; border-radius: 999px; flex: 0 0 34px;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px;
      }
      .ceo-avatar.bot {background: #dbeafe; color: #1e3a8a;}
      .ceo-avatar.user {background: #fee2e2; color: #b91c1c;}
      .ceo-bubble {
        border-radius: 16px;
        padding: 12px 14px;
        max-width: 84%;
        line-height: 1.45;
        font-size: 1rem;
      }
      .ceo-bubble.bot {
        background: #ffffff;
        border: 1px solid #d1d5db;
        color: #334155;
      }
      .ceo-bubble.user {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        color: #374151;
      }
      .ceo-input-shell {
        background: #f8fafc;
        border-top: 1px solid #cbd5e1;
        padding: 10px 12px;
      }
      .ceo-prompt-note {font-size: .8rem; color: #94a3b8; margin: 8px 2px 0;}
      div[data-testid="stTextInput"] input {
        background: #0b1220 !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
      }
      div[data-testid="stTextInput"] input::placeholder {
        color: #94a3b8 !important;
      }
      div[data-testid="stExpander"] {
        border: 1px solid rgba(148,163,184,0.28) !important;
        border-radius: 14px !important;
        background: rgba(15,23,42,0.48) !important;
      }
      div[data-testid="stChatInput"] textarea {
        background: #0b1220 !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 14px !important;
      }
      div[data-testid="stChatMessage"] {
        border-radius: 14px !important;
      }
      div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1d4ed8, #0d9488) !important;
        color: #f8fafc !important;
        border: 0 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
      }
      div[data-testid="stForm"] div[data-testid="stTextInput"] input {
        background: #0b1220 !important;
        border-radius: 999px !important;
      }
      @media (max-width: 840px) {
        .ceo-chip-row {grid-template-columns: repeat(2, minmax(0,1fr));}
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ceo-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="ceo-hero">
      <h2>Informe del CEO | Chat Consultor MEM</h2>
      <p>Asistente experto en mercado de energía. Consulta solo información del MEM y análisis ejecutivo del periodo seleccionado.</p>
    </div>
    """, unsafe_allow_html=True)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    with st.spinner("Preparando contexto del periodo para el consultor..."):
        cur = fetch_metrics_parallel(SUMMARY_METRICS, start_str, end_str)
    market_cur = _build_market_risk_frame(cur.get("PrecBolsNaci"), cur.get("PrecEsca"), cur.get("PrecEscaSup"), "RANGE")
    hydro_cur = _build_hydro_frame(cur.get("CapaUtilDiarEner"), cur.get("VoluUtilDiarEner"), cur.get("AporEner"), cur.get("AporEnerMediHist"), "RANGE")

    pressure_now = util_now = hydro_dev_now = 0.0
    price_cur = 0.0
    state = "SIN DATOS"
    if market_cur is not None and hydro_cur is not None and not market_cur.empty and not hydro_cur.empty:
        price_cur = float(market_cur["Bolsa"].mean())
        pressure_now = float(market_cur.iloc[-1]["PresionPct"])
        util_now = float(hydro_cur.iloc[-1]["UtilPct"])
        hydro_dev_now = float(hydro_cur.iloc[-1]["HydroDevPct"])
        state = _classify_risk(pressure_now, hydro_dev_now, util_now)
    demand_cur = _aggregate_period_sum(cur.get("DemaCome"))
    gen_cur = _aggregate_period_sum(cur.get("Gene"))
    balance_cur = (gen_cur - demand_cur) / 1e6

    findings = []
    if pressure_now > 100:
        findings.append("Presión de mercado > 100%")
    elif pressure_now >= 90:
        findings.append("Presión en banda de alerta (90%-100%)")
    if util_now > 80:
        findings.append("Utilización de embalse > 80%")
    if hydro_dev_now < -15:
        findings.append("Aportes por debajo de -15% vs media")
    if balance_cur < 0:
        findings.append("Balance neto generación-demanda negativo")
    if not findings:
        findings.append("Sin alertas críticas para el periodo")

    report_context = (
        f"Periodo analizado: {start_str} a {end_str}\n"
        f"Estado general: {state}\n"
        f"Precio bolsa promedio: {price_cur:.2f} COP/kWh\n"
        f"Presión de mercado: {pressure_now:.2f}%\n"
        f"Balance generación-demanda: {balance_cur:.2f} GWh\n"
        f"Utilización embalse: {util_now:.2f}%\n"
        f"Desvío aportes: {hydro_dev_now:.2f}%\n"
        f"Hallazgos: {' | '.join(findings)}"
    )

    if "ceo_api_key" not in st.session_state:
        st.session_state.ceo_api_key = ""
    current_period_key = f"{start_str}|{end_str}"
    if "ceo_period_key" not in st.session_state:
        st.session_state.ceo_period_key = current_period_key
    if "ceo_chat_messages" not in st.session_state:
        st.session_state.ceo_chat_messages = [
            {
                "role": "assistant",
                "content": "Listo. Soy tu consultor experto del MEM. Pregúntame sobre precios, escasez, demanda, generación, embalses o implicaciones ejecutivas del periodo.",
            }
        ]
    elif st.session_state.ceo_period_key != current_period_key:
        st.session_state.ceo_period_key = current_period_key
        st.session_state.ceo_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Periodo actualizado a {start_str} - {end_str}. "
                    "Inicié un nuevo chat para evitar mezclar conversaciones."
                ),
            }
        ]

    st.markdown('<div class="ceo-box">', unsafe_allow_html=True)
    with st.expander("Configuración de API Key (Gemini)", expanded=(not bool(st.session_state.ceo_api_key))):
        key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.ceo_api_key,
            placeholder="AIza...",
            key="ceo_key_input",
        )
        c_key_1, c_key_2 = st.columns(2)
        if c_key_1.button("Guardar API Key", key="save_ceo_key"):
            st.session_state.ceo_api_key = key_input.strip()
            if st.session_state.ceo_api_key:
                st.success("API Key guardada para esta sesión.")
            else:
                st.warning("Ingresa una API Key válida.")
        if c_key_2.button("Limpiar API Key", key="clear_ceo_key"):
            st.session_state.ceo_api_key = ""
            st.warning("API Key eliminada de la sesión.")
    st.markdown("</div>", unsafe_allow_html=True)

    quick_col_1, quick_col_2, quick_col_3 = st.columns(3)
    if quick_col_1.button("¿Cuál es el principal riesgo del periodo?", use_container_width=True):
        st.session_state.ceo_quick_q = "¿Cuál es el principal riesgo del periodo y por qué?"
    if quick_col_2.button("¿Qué decisión tomarías hoy?", use_container_width=True):
        st.session_state.ceo_quick_q = "Con este estado del mercado, ¿qué decisión ejecutiva tomarías hoy?"
    if quick_col_3.button("Proyección 7 días", use_container_width=True):
        st.session_state.ceo_quick_q = "Dame una lectura ejecutiva para los próximos 7 días, con riesgos y recomendaciones."

    action_col_1, action_col_2 = st.columns(2)
    if action_col_1.button("Nuevo chat", use_container_width=True):
        st.session_state.ceo_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Nuevo chat iniciado para el periodo {start_str} - {end_str}. "
                    "¿Qué deseas analizar del MEM?"
                ),
            }
        ]
        st.session_state.ceo_quick_q = ""
        st.rerun()
    if action_col_2.button("Limpiar historial", use_container_width=True):
        st.session_state.ceo_chat_messages = []
        st.session_state.ceo_quick_q = ""
        st.rerun()

    st.markdown("""
    <div class="ceo-chat-shell">
      <div class="ceo-chat-head">
        <div>
          <p class="ceo-chat-title">Consultor MEM AI</p>
          <p class="ceo-chat-sub">Experto en mercado eléctrico colombiano</p>
        </div>
        <div style="font-size:1.15rem;">✦</div>
      </div>
      <div class="ceo-chat-body">
    """, unsafe_allow_html=True)
    for msg in st.session_state.ceo_chat_messages:
        role = msg.get("role", "assistant")
        safe_text = pyhtml.escape(msg.get("content", "")).replace("\n", "<br>")
        if role == "user":
            st.markdown(
                f"""
                <div class="ceo-msg-row user">
                  <div class="ceo-bubble user">{safe_text}</div>
                  <div class="ceo-avatar user">☺</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="ceo-msg-row">
                  <div class="ceo-avatar bot">🤖</div>
                  <div class="ceo-bubble bot">{safe_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="ceo-input-shell">', unsafe_allow_html=True)
        with st.form("ceo_chat_form", clear_on_submit=True):
            in_col, send_col = st.columns([10, 1])
            with in_col:
                question = st.text_input(
                    "Pregunta al consultor MEM",
                    value="",
                    placeholder="Escribe tu consulta ejecutiva del MEM...",
                    label_visibility="collapsed",
                )
            with send_col:
                send_clicked = st.form_submit_button("➤", use_container_width=True)
        st.markdown('<p class="ceo-prompt-note">Tip: pregunta por riesgos, precio, escasez, embalses o decisiones ejecutivas.</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not send_clicked:
        question = question if question else None
    if not question and "ceo_quick_q" in st.session_state and st.session_state.ceo_quick_q:
        question = st.session_state.ceo_quick_q
        st.session_state.ceo_quick_q = ""

    if question:
        last_user = None
        for m in reversed(st.session_state.ceo_chat_messages):
            if m.get("role") == "user":
                last_user = m.get("content", "").strip()
                break
        if last_user and last_user == question.strip():
            question = None

    if question:
        st.session_state.ceo_chat_messages.append({"role": "user", "content": question})

        if not st.session_state.ceo_api_key:
            answer = "Primero configura tu API Key de Gemini para activar el consultor."
        else:
            try:
                with st.spinner("Analizando el MEM..."):
                    answer = _call_ceo_consultant(
                        st.session_state.ceo_api_key,
                        question,
                        report_context,
                        st.session_state.ceo_chat_messages,
                    )
                    if not answer:
                        answer = "No recibí contenido del modelo en esta consulta. Intenta reformular la pregunta."
            except Exception as e:
                answer = f"No pude completar la consulta: {e}"

        st.session_state.ceo_chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)


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
