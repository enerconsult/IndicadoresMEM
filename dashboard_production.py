import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import datetime as dt
import requests
import re

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


def _extract_question_range(question, global_start, global_end):
    """Parse explicit date ranges from user question and clamp to selected global period."""
    if not question:
        return None

    text = question.lower()
    parsed = []
    patterns = [
        (r"\b\d{4}-\d{2}-\d{2}\b", "%Y-%m-%d"),
        (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "%d/%m/%Y"),
        (r"\b\d{1,2}-\d{1,2}-\d{4}\b", "%d-%m-%Y"),
    ]
    for pat, fmt in patterns:
        for token in re.findall(pat, text):
            try:
                parsed.append(dt.datetime.strptime(token, fmt).date())
            except ValueError:
                pass

    if len(parsed) >= 2:
        start_q = min(parsed[0], parsed[1])
        end_q = max(parsed[0], parsed[1])
    elif len(parsed) == 1:
        start_q = end_q = parsed[0]
    else:
        m_days = re.search(r"\bultim[oa]s?\s+(\d+)\s+d[ií]as\b", text)
        if m_days:
            days = max(1, int(m_days.group(1)))
            end_q = global_end
            start_q = end_q - dt.timedelta(days=days - 1)
        elif "última semana" in text or "ultima semana" in text:
            end_q = global_end
            start_q = end_q - dt.timedelta(days=6)
        elif "último mes" in text or "ultimo mes" in text:
            end_q = global_end
            start_q = end_q - dt.timedelta(days=29)
        else:
            return None

    start_q = max(start_q, global_start)
    end_q = min(end_q, global_end)
    if start_q > end_q:
        return None
    return start_q, end_q


def _fmt_or_nd(v, fmt):
    if v is None:
        return "N/D"
    return format(v, fmt)


def _build_subperiod_context(question, global_start, global_end, cur, market_cur, hydro_cur):
    sub_range = _extract_question_range(question, global_start, global_end)
    if not sub_range:
        return ""
    sub_start, sub_end = sub_range

    def _between(df):
        if df is None or df.empty or "Date" not in df.columns:
            return None
        tmp = df.copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"]).dt.date
        return tmp[(tmp["Date"] >= sub_start) & (tmp["Date"] <= sub_end)]

    price_val = pressure_val = util_val = hydro_dev_val = None
    state_val = "SIN DATOS"

    market_sub = _between(market_cur)
    if market_sub is not None and not market_sub.empty:
        price_val = float(market_sub["Bolsa"].mean()) if "Bolsa" in market_sub.columns else None
        pressure_val = float(market_sub.iloc[-1]["PresionPct"]) if "PresionPct" in market_sub.columns else None

    hydro_sub = _between(hydro_cur)
    if hydro_sub is not None and not hydro_sub.empty:
        util_val = float(hydro_sub.iloc[-1]["UtilPct"]) if "UtilPct" in hydro_sub.columns else None
        hydro_dev_val = float(hydro_sub.iloc[-1]["HydroDevPct"]) if "HydroDevPct" in hydro_sub.columns else None

    if pressure_val is not None and hydro_dev_val is not None and util_val is not None:
        state_val = _classify_risk(pressure_val, hydro_dev_val, util_val)

    dem_daily = _metric_to_daily_value(cur.get("DemaCome"), mode="sum")
    gen_daily = _metric_to_daily_value(cur.get("Gene"), mode="sum")
    balance_val = None
    if dem_daily is not None and gen_daily is not None and not dem_daily.empty and not gen_daily.empty:
        dem_sub = _between(dem_daily)
        gen_sub = _between(gen_daily)
        if dem_sub is not None and gen_sub is not None and not dem_sub.empty and not gen_sub.empty:
            demand_sum = float(dem_sub["_value"].sum())
            gen_sum = float(gen_sub["_value"].sum())
            balance_val = (gen_sum - demand_sum) / 1e6

    return (
        "\n"
        "SUBRANGO SOLICITADO (prioritario para responder):\n"
        f"- Desde: {sub_start.strftime('%Y-%m-%d')}\n"
        f"- Hasta: {sub_end.strftime('%Y-%m-%d')}\n"
        f"- Estado general subrango: {state_val}\n"
        f"- Precio bolsa promedio subrango: {_fmt_or_nd(price_val, '.2f')} COP/kWh\n"
        f"- Presión de mercado subrango (último punto): {_fmt_or_nd(pressure_val, '.2f')}%\n"
        f"- Balance generación-demanda subrango: {_fmt_or_nd(balance_val, '.2f')} GWh\n"
        f"- Utilización embalse subrango (último punto): {_fmt_or_nd(util_val, '.2f')}%\n"
        f"- Desvío aportes subrango (último punto): {_fmt_or_nd(hydro_dev_val, '.2f')}%\n"
    )


def _call_ceo_consultant(api_key, user_question, report_context, history):
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("API Key vacía. Configura una API Key válida de Gemini.")

    instruction = (
        "Actúa como consultor experto del Mercado de Energía Mayorista (MEM) de Colombia.\n"
        "Reglas:\n"
        "1. PRIORIDAD - DATOS CARGADOS: Si la pregunta se refiere al periodo o datos específicos del 'Contexto del informe', responde basándote estrictamente en esos números provided.\n"
        "2. ANÁLISIS GENERAL / PREDICCIÓN: Si la pregunta excede el rango de fechas cargado, pide predicciones futuras, o trata sobre conceptos generales del mercado (regulación, fenómenos climáticos, tendencias globales), DEBES usar tu conocimiento general como modelo de IA (similar a Gemini Web). NO te limites a decir 'no tengo datos'.\n"
        "3. INTEGRACIÓN: Si es posible, combina los datos cargados con tu conocimiento general para dar una respuesta más completa.\n"
        "4. FORMATO: Responde en español ejecutivo, claro y accionable.\n"
        "5. NO ALUCINES CIFRAS EXACTAS: Para fechas fuera del contexto cargado, usa estimaciones o tendencias generales, no inventes valores precisos si no están en el contexto.\n"
    )

    history_text = []
    for msg in history[-8:]:
        role = "Usuario" if msg.get("role") == "user" else "Consultor"
        history_text.append(f"{role}: {msg.get('content', '')}")
    history_block = "\n".join(history_text)

    prompt = (
        f"{instruction}\n"
        f"--- CONTEXTO DE DATOS CARGADOS (Verdad absoluta para este periodo) ---\n{report_context}\n"
        "----------------------------------------------------------------------\n\n"
        f"Historial reciente:\n{history_block}\n\n"
        f"Pregunta actual del usuario:\n{user_question}\n\n"
        "Instrucción final: Analiza la pregunta. Si puedes responder con los datos cargados, hazlo con precisión. "
        "Si la pregunta requiere conocimiento externo, teoría, o proyección futura fuera de los datos, responde con tu conocimiento general experto. "
        "Estructura tu respuesta de forma ejecutiva (Lectura de situación -> Implicación -> Recomendación/Análisis)."
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
    candidates = data.get("candidates", [])
    if not candidates:
        pf = data.get("promptFeedback", {})
        block_reason = pf.get("blockReason")
        if block_reason:
            raise RuntimeError(f"Gemini bloqueó la respuesta ({block_reason}). Ajusta la pregunta e intenta de nuevo.")
        raise RuntimeError("Gemini no devolvió candidatos en la respuesta.")

    parts = candidates[0].get("content", {}).get("parts", [])
    answer = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if not answer:
        raise RuntimeError("Gemini respondió sin contenido utilizable.")
    return answer

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
    
    nav_options = ["Resumen"]
    if "shared_data" in st.session_state:
        nav_options.append("Informe del CEO")
    
    # "Explorador" remains fixed and independent
    nav_options.append("Explorador")

    selection = st.radio(
        "Navegación",
        nav_options,
        label_visibility="collapsed",
    )
    
    # If user was on CEO report but page reloaded and data lost, default back to Resumen
    if selection == "Informe del CEO" and "shared_data" not in st.session_state:
        st.experimental_rerun() 
    st.markdown("---")

    # --- Gemini API Key (always visible in sidebar) ---
    if "ceo_api_key" not in st.session_state:
        st.session_state.ceo_api_key = ""
    if "ceo_key_input" not in st.session_state:
        st.session_state.ceo_key_input = st.session_state.ceo_api_key
    if "ceo_reset_key_input" not in st.session_state:
        st.session_state.ceo_reset_key_input = False

    if st.session_state.ceo_reset_key_input:
        st.session_state.ceo_key_input = ""
        st.session_state.ceo_reset_key_input = False

    st.markdown("### 🔑 API KEY")
    st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        key="ceo_key_input",
        label_visibility="collapsed",
    )
    sb_k1, sb_k2 = st.columns(2)
    if sb_k1.button("Guardar", key="save_ceo_key", use_container_width=True):
        saved = (st.session_state.ceo_key_input or "").strip()
        st.session_state.ceo_api_key = saved
        if saved:
            st.success("Activa")
        else:
            st.warning("Vacia")
    if sb_k2.button("Limpiar", key="clear_ceo_key", use_container_width=True):
        st.session_state.ceo_api_key = ""
        st.session_state.ceo_reset_key_input = True
        st.rerun()
    if st.session_state.ceo_api_key:
        st.caption("✅ API Key activa")
    else:
        st.caption("⚠️ Sin API Key")

    st.markdown("---")
    st.markdown("""
    <div style="margin-top:1rem;padding-top:0.5rem;
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
        st.session_state.shared_data = data

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
    # ------------------------------------------------------------------
    # Modern AI-chat — transparent messages, gradient accents, no footer
    # ------------------------------------------------------------------
    t = get_theme_config()
    is_dark = st.session_state.get("theme", "dark") == "dark"

    _accent = "#818cf8" if is_dark else "#4f46e5"
    _accent2 = "#38bdf8" if is_dark else "#0ea5e9"
    _text = t["TEXT_COLOR"]
    _sub = t["TEXT_SUB"]
    _bg = t["BG_COLOR"]

    st.markdown(f"""
    <style>
      /* ---- Hide footer in chat view ---- */
      .appview-container > section > div > div > div > div:last-child hr,
      #mem-footer-section {{
        display: none !important;
      }}

      /* ---- Assistant messages: transparent, no card ---- */
      div[data-testid="stChatMessage"] {{
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.8rem 0 !important;
        margin-bottom: 0 !important;
        border-bottom: 1px solid {'rgba(255,255,255,0.04)' if is_dark else 'rgba(0,0,0,0.04)'} !important;
      }}
      div[data-testid="stChatMessage"] p,
      div[data-testid="stChatMessage"] li,
      div[data-testid="stChatMessage"] span,
      div[data-testid="stChatMessage"] td {{
        color: {_text} !important;
        font-size: 0.9rem !important;
        line-height: 1.7 !important;
      }}
      div[data-testid="stChatMessage"] strong {{
        color: {_accent} !important;
      }}
      div[data-testid="stChatMessage"] h1,
      div[data-testid="stChatMessage"] h2,
      div[data-testid="stChatMessage"] h3 {{
        color: {_accent} !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        margin-top: 0.5rem !important;
      }}
      div[data-testid="stChatMessage"] ul {{
        padding-left: 1.2rem !important;
      }}
      div[data-testid="stChatMessage"] code {{
        background: {'rgba(129,140,248,0.12)' if is_dark else 'rgba(79,70,229,0.08)'} !important;
        color: {_accent} !important;
        border-radius: 4px !important;
        padding: 1px 6px !important;
        font-size: 0.85rem !important;
      }}

      /* ---- Bottom bar: kill white stripe ---- */
      div[data-testid="stBottom"] {{
        background: {_bg} !important;
        border-top: none !important;
      }}
      div[data-testid="stBottom"] > div {{
        background: transparent !important;
      }}

      /* ---- Input bar: floating glass ---- */
      div[data-testid="stChatInput"] {{
        background: transparent !important;
        border-top: none !important;
        padding: 0.4rem 0 !important;
      }}
      div[data-testid="stChatInput"] textarea {{
        background: {'rgba(30,41,59,0.7)' if is_dark else '#ffffff'} !important;
        border: 1.5px solid {'rgba(129,140,248,0.3)' if is_dark else 'rgba(79,70,229,0.2)'} !important;
        border-radius: 28px !important;
        color: {_text} !important;
        font-size: 0.9rem !important;
        padding: 0.7rem 1.2rem !important;
        box-shadow: {'0 0 20px rgba(129,140,248,0.08)' if is_dark else '0 2px 8px rgba(0,0,0,0.06)'} !important;
        backdrop-filter: blur(12px) !important;
      }}
      div[data-testid="stChatInput"] textarea:focus {{
        border-color: {_accent} !important;
        box-shadow: 0 0 0 3px {'rgba(129,140,248,0.15)' if is_dark else 'rgba(79,70,229,0.12)'},
                    {'0 0 24px rgba(129,140,248,0.12)' if is_dark else '0 2px 12px rgba(0,0,0,0.08)'} !important;
      }}
      div[data-testid="stChatInput"] textarea::placeholder {{
        color: {_sub} !important;
      }}
      div[data-testid="stChatInput"] button {{
        background: linear-gradient(135deg, {_accent}, {_accent2}) !important;
        color: #fff !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 10px {'rgba(129,140,248,0.3)' if is_dark else 'rgba(79,70,229,0.25)'} !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
      }}
      div[data-testid="stChatInput"] button:hover {{
        transform: scale(1.08) !important;
        box-shadow: 0 4px 16px {'rgba(129,140,248,0.4)' if is_dark else 'rgba(79,70,229,0.35)'} !important;
      }}

      /* ---- Welcome hero ---- */
      .mem-hero {{
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
      }}
      .mem-hero-icon {{
        width: 56px; height: 56px; margin: 0 auto 0.8rem;
        border-radius: 16px; display: flex; align-items: center; justify-content: center;
        font-size: 1.6rem;
        background: linear-gradient(135deg, {_accent}, {_accent2});
        box-shadow: 0 4px 20px {'rgba(129,140,248,0.25)' if is_dark else 'rgba(79,70,229,0.2)'};
      }}
      .mem-hero h2 {{
        margin: 0 0 0.3rem; font-size: 1.4rem; font-weight: 800;
        color: {_text} !important;
        -webkit-text-fill-color: {_text} !important;
      }}
      .mem-hero p {{
        color: {_sub}; font-size: 0.82rem; margin: 0;
      }}

      /* ---- Suggested question cards ---- */
      .mem-suggestions {{
        display: flex; justify-content: center; flex-wrap: wrap;
        gap: 0.6rem; margin: 1.2rem auto 1.5rem; max-width: 640px;
        padding: 0 0.5rem;
      }}
      /* Style the Streamlit buttons inside the suggestions area */
      .mem-suggestions-row button {{
        background: {'rgba(129,140,248,0.06)' if is_dark else 'rgba(79,70,229,0.04)'} !important;
        border: 1px solid {'rgba(129,140,248,0.18)' if is_dark else 'rgba(79,70,229,0.12)'} !important;
        border-radius: 14px !important;
        color: {_text} !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
        text-align: left !important;
        transition: all 0.2s !important;
        height: auto !important;
        min-height: 56px !important;
        white-space: normal !important;
        line-height: 1.4 !important;
      }}
      .mem-suggestions-row button:hover {{
        background: {'rgba(129,140,248,0.14)' if is_dark else 'rgba(79,70,229,0.08)'} !important;
        border-color: {_accent} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px {'rgba(129,140,248,0.15)' if is_dark else 'rgba(79,70,229,0.1)'} !important;
      }}
      .mem-suggestions-row button p {{
        color: {_text} !important;
        font-size: 0.78rem !important;
      }}

      /* ---- Status bar ---- */
      .mem-status {{
        display: flex; justify-content: center; flex-wrap: wrap;
        gap: 0.4rem; margin: 0 auto 0.5rem; max-width: 600px;
      }}
      .mem-status .pill {{
        font-size: 0.65rem; font-weight: 600; padding: 2px 9px;
        border-radius: 999px; display: inline-flex; align-items: center; gap: 4px;
        color: {_sub};
        background: {'rgba(255,255,255,0.03)' if is_dark else 'rgba(0,0,0,0.02)'};
        border: 1px solid {'rgba(255,255,255,0.06)' if is_dark else 'rgba(0,0,0,0.06)'};
      }}
      .pill .d {{ width: 6px; height: 6px; border-radius: 50%; display: inline-block; }}
      .d-ok   {{ background: #22c55e; }}
      .d-warn {{ background: #eab308; }}
      .d-crit {{ background: #ef4444; }}
    </style>
    """, unsafe_allow_html=True)

    # --- Data context (backend unchanged) ---
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    # Check if data was loaded in Resumen
    if "shared_data" not in st.session_state:
        st.warning("⚠️ Para acceder al consultor, primero debes cargar los datos en la pestaña **Resumen**.")
        st.info("Ve a la sección 'Resumen' en el menú lateral para inicializar el tablero.")
        st.stop()
    
    cur = st.session_state.shared_data
    
    market_cur = _build_market_risk_frame(
        cur.get("PrecBolsNaci"), cur.get("PrecEsca"), cur.get("PrecEscaSup"), "RANGE")
    hydro_cur = _build_hydro_frame(
        cur.get("CapaUtilDiarEner"), cur.get("VoluUtilDiarEner"),
        cur.get("AporEner"), cur.get("AporEnerMediHist"), "RANGE")

    # --- Define specific dataframes for charts ---
    df_bolsa       = cur.get("PrecBolsNaci")
    df_escasez     = cur.get("PrecEsca")
    df_escasez_sup = cur.get("PrecEscaSup")
    df_demanda     = cur.get("DemaCome")
    df_gen         = cur.get("Gene")
    df_vol         = cur.get("VoluUtilDiarEner")
    df_cap         = cur.get("CapaUtilDiarEner")
    df_apor        = cur.get("AporEner")
    df_media       = cur.get("AporEnerMediHist")

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
        findings.append("Presion de mercado > 100%")
    elif pressure_now >= 90:
        findings.append("Presion en banda de alerta (90%-100%)")
    if util_now > 80:
        findings.append("Utilizacion de embalse > 80%")
    if hydro_dev_now < -15:
        findings.append("Aportes por debajo de -15% vs media")
    if balance_cur < 0:
        findings.append("Balance neto generacion-demanda negativo")
    if not findings:
        findings.append("Sin alertas criticas para el periodo")

    report_context = (
        f"Periodo analizado: {start_str} a {end_str}\n"
        f"Estado general: {state}\n"
        f"Precio bolsa promedio: {price_cur:.2f} COP/kWh\n"
        f"Presion de mercado: {pressure_now:.2f}%\n"
        f"Balance generacion-demanda: {balance_cur:.2f} GWh\n"
        f"Utilizacion embalse: {util_now:.2f}%\n"
        f"Desvio aportes: {hydro_dev_now:.2f}%\n"
        f"Hallazgos: {' | '.join(findings)}"
    )

    # --- Session state ---
    current_period_key = f"{start_str}|{end_str}"
    if "ceo_period_key" not in st.session_state:
        st.session_state.ceo_period_key = current_period_key
    if "ceo_chat_messages" not in st.session_state:
        st.session_state.ceo_chat_messages = []
    elif st.session_state.ceo_period_key != current_period_key:
        st.session_state.ceo_period_key = current_period_key
        st.session_state.ceo_chat_messages = []

    has_messages = len(st.session_state.ceo_chat_messages) > 0

    # Predefined suggestions
    _SUGGESTIONS = [
        "Cual es el estado actual del mercado y los principales riesgos?",
        "Como se comporta el precio de bolsa frente al precio de escasez?",
        "Cual es el nivel de los embalses y como estan los aportes hidricos?",
    ]

    # Handle suggestion click from previous run
    if "ceo_suggestion" not in st.session_state:
        st.session_state.ceo_suggestion = None
    pending_suggestion = st.session_state.ceo_suggestion
    st.session_state.ceo_suggestion = None

    # --- Welcome hero (only when chat is empty) ---
    if not has_messages:
        dot_cls = "d-ok" if state == "NORMAL" else ("d-warn" if state == "ALERTA" else "d-crit")
        st.markdown(f"""
        <div class="mem-hero">
          <div class="mem-hero-icon">⚡</div>
          <h2>Consultor MEM AI</h2>
          <p>Analisis inteligente del mercado electrico colombiano</p>
        </div>
        <div class="mem-status">
          <span class="pill"><span class="d {dot_cls}"></span> {state}</span>
          <span class="pill">{start_str} &rarr; {end_str}</span>
          <span class="pill">Presion {pressure_now:.1f}%</span>
          <span class="pill">Embalse {util_now:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Clickable suggestion buttons
        st.markdown('<div class="mem-suggestions-row">', unsafe_allow_html=True)
        sg_cols = st.columns(3)
        for i, txt in enumerate(_SUGGESTIONS):
            if sg_cols[i].button(txt, key=f"sg_{i}", use_container_width=True):
                st.session_state.ceo_suggestion = txt
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        dot_cls = "d-ok" if state == "NORMAL" else ("d-warn" if state == "ALERTA" else "d-crit")
        st.markdown(f"""
        <div class="mem-status">
          <span class="pill"><span class="d {dot_cls}"></span> {state}</span>
          <span class="pill">{start_str} &rarr; {end_str}</span>
          <span class="pill">Presion {pressure_now:.1f}%</span>
          <span class="pill">Embalse {util_now:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # --- Chat history ---
    for msg in st.session_state.ceo_chat_messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role, avatar="\u26a1" if role == "assistant" else "\U0001f464"):
            st.markdown(msg.get("content", ""))

    # --- ACTION BUTTONS (Charts) ---
    # Render immediately after history, before the input box
    # Only show if:
    # 1. Chat is not empty
    # 2. Last message is from assistant
    # 3. API Key is present
    # 4. Last message is NOT an error message
    if st.session_state.ceo_chat_messages:
        last_msg = st.session_state.ceo_chat_messages[-1].get("content", "")
        last_role = st.session_state.ceo_chat_messages[-1].get("role")
        is_error = last_msg.startswith("Error") or last_msg.startswith("No pude") or "API Key" in last_msg

        if (last_role == "assistant" and 
            st.session_state.ceo_api_key and 
            not is_error):
            
            # 1. Determine Context
            context_type = "general"
            if "precio" in last_msg.lower() or "bolsa" in last_msg.lower():
                context_type = "precio"
            elif "demanda" in last_msg.lower() or "consumo" in last_msg.lower():
                context_type = "demanda"
            elif "embalse" in last_msg.lower() or "aportes" in last_msg.lower() or "hidrico" in last_msg.lower():
                context_type = "hidro"
            
            # Suggestion UI
            st.markdown(f"""
            <div style="margin-top:0.5rem;margin-bottom:0.5rem;text-align:center;">
                 <span style="font-size:0.8rem;color:{t['TEXT_SUB']};">
                     ¿Quieres profundizar en estos datos?
                 </span>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"📊 Generar Gráficos de {context_type.capitalize()}", key="btn_gen_charts", use_container_width=True):
                 st.session_state.ceo_chart_request = {"context": context_type, "timestamp": dt.datetime.now().timestamp()}
                 st.rerun()

    # --- Render Auto-Generated Charts (Container) ---
    if "ceo_chart_request" in st.session_state and st.session_state.ceo_chart_request:
        ctx = st.session_state.ceo_chart_request.get("context", "general")
        
        # --- Enforce 30-day window for these charts ---
        end_d = dt.datetime.now()
        start_d = end_d - dt.timedelta(days=30)
        s_30 = start_d.strftime("%Y-%m-%d")
        e_30 = end_d.strftime("%Y-%m-%d")

        with st.spinner("Generando gráficos (últimos 30 días)..."):
            data_30 = fetch_metrics_parallel(SUMMARY_METRICS, s_30, e_30)

        # Override dataframes for plotting scope
        df_bolsa   = data_30.get("PrecBolsNaci")
        df_escasez = data_30.get("PrecEsca")
        df_demanda = data_30.get("DemaCome")
        df_gen     = data_30.get("Gene")
        df_vol     = data_30.get("VoluUtilDiarEner")
        df_cap     = data_30.get("CapaUtilDiarEner")
        df_apor    = data_30.get("AporEner")
        df_media   = data_30.get("AporEnerMediHist")

        st.markdown(f"### 📊 Análisis Gráfico: {ctx.capitalize()}")
        st.markdown(f"<span style='color:{t['TEXT_SUB']}'>Visualización de tendencia reciente (últimos 30 días).</span>", unsafe_allow_html=True)
        
        g1, g2 = st.columns(2)
        
        try:
             # --- LOGIC: PRECIO ---
            if ctx == "precio":
                df_p = calculate_periodicity(df_bolsa, "1D", "mean")
                if df_p is not None:
                    fig1 = px.line(df_p, x="Date", y=get_value_col(df_p), title="Evolución Precio Bolsa")
                    fig1.update_traces(line_color=t["COLOR_ORANGE"], line_width=2)
                    g1.plotly_chart(style_fig(fig1), use_container_width=True)

                df_esc = calculate_periodicity(df_escasez, "1D", "mean")
                if df_p is not None and df_esc is not None:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df_p["Date"], y=df_p[get_value_col(df_p)], name="Bolsa", line=dict(color=t["COLOR_ORANGE"])))
                    fig2.add_trace(go.Scatter(x=df_esc["Date"], y=df_esc[get_value_col(df_esc)], name="Escasez", line=dict(color=t["COLOR_BLUE_DARK"], dash="dot")))
                    fig2.update_layout(title="Precio vs Escasez")
                    g2.plotly_chart(style_fig(fig2), use_container_width=True)

            # --- LOGIC: DEMANDA ---
            elif ctx == "demanda":
                df_d = calculate_periodicity(df_demanda, "1D", "sum")
                df_g = calculate_periodicity(df_gen, "1D", "sum")
                
                if df_d is not None and df_g is not None:
                     fig1 = go.Figure()
                     fig1.add_trace(go.Scatter(x=df_d["Date"], y=df_d[get_value_col(df_d)], name="Demanda", fill='tozeroy', line=dict(color=t["COLOR_BLUE"])))
                     fig1.add_trace(go.Scatter(x=df_g["Date"], y=df_g[get_value_col(df_g)], name="Generación", line=dict(color=t["COLOR_ORANGE"])))
                     fig1.update_layout(title="Demanda vs Generación")
                     g1.plotly_chart(style_fig(fig1), use_container_width=True)

                if df_d is not None:
                     vcol = get_value_col(df_d)
                     df_d["Weekday"] = pd.to_datetime(df_d["Date"]).dt.day_name()
                     df_w = df_d.groupby("Weekday")[vcol].mean().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
                     fig2 = px.bar(df_w, x="Weekday", y=vcol, title="Perfil Semanal Promedio")
                     fig2.update_traces(marker_color=t["COLOR_BLUE"])
                     g2.plotly_chart(style_fig(fig2), use_container_width=True)

            # --- LOGIC: HIDRO (Embalses) ---
            elif ctx == "hidro":
                df_c = calculate_periodicity(df_cap, "1D", "mean")
                df_v = calculate_periodicity(df_vol, "1D", "mean")
                if df_v is not None and df_c is not None:
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df_c["Date"], y=df_c[get_value_col(df_c)], name="Capacidad Total", line=dict(dash='dot', color="#94a3b8")))
                    fig1.add_trace(go.Scatter(x=df_v["Date"], y=df_v[get_value_col(df_v)], name="Volumen Actual", fill='tonexty', line=dict(color="#22c55e")))
                    fig1.update_layout(title="Nivel de Llenado (%)")
                    g1.plotly_chart(style_fig(fig1), use_container_width=True)

                df_a = calculate_periodicity(df_apor, "1D", "sum")
                df_m = calculate_periodicity(df_media, "1D", "sum")
                if df_a is not None and df_m is not None:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(x=df_a["Date"], y=df_a[get_value_col(df_a)], name="Aportes", marker_color=t["COLOR_BLUE"]))
                    fig2.add_trace(go.Scatter(x=df_m["Date"], y=df_m[get_value_col(df_m)], name="Media Hist", line=dict(color=t["COLOR_ORANGE"])))
                    fig2.update_layout(title="Aportes vs Media")
                    g2.plotly_chart(style_fig(fig2), use_container_width=True)

            # --- LOGIC: GENERAL (Fallback) ---
            else:
                df_p = calculate_periodicity(df_bolsa, "1D", "mean")
                if df_p is not None:
                     fig1 = px.line(df_p, x="Date", y=get_value_col(df_p), title="Precio Bolsa")
                     fig1.update_traces(line_color=t["COLOR_ORANGE"])
                     g1.plotly_chart(style_fig(fig1), use_container_width=True)
                
                df_v = calculate_periodicity(df_vol, "1D", "mean")
                if df_v is not None:
                     fig2 = px.line(df_v, x="Date", y=get_value_col(df_v), title="Nivel Embalses")
                     fig2.update_traces(line_color="#22c55e")
                     g2.plotly_chart(style_fig(fig2), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generando gráficos: {e}")

        if st.button("Cerrar Visualización", key="close_auto_charts"):
            st.session_state.ceo_chart_request = None
            st.rerun()

    # --- Chat input (MOVED TO END) ---
    question = st.chat_input(
        "Pregunta sobre precios, demanda, embalses, riesgos..."
        if st.session_state.ceo_api_key
        else "Configura tu API Key en el panel lateral para comenzar"
    )

    if question:
        question = question.strip() or None

    # Accept suggestion if no typed question
    if not question and pending_suggestion:
        question = pending_suggestion

    if question:
        st.session_state.ceo_chat_messages.append({"role": "user", "content": question})
        st.session_state.ceo_chart_request = None # Reset charts on new question
        
        # Immediate display of user message (needed because we are at end of script)
        # But we will rerun anyway, so it matters little.
        
        with st.chat_message("user", avatar="\U0001f464"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="\u26a1"):
            if not st.session_state.ceo_api_key:
                answer = "Para activar el consultor, configura tu **API Key de Gemini** en el panel lateral."
                st.markdown(answer)
            else:
                with st.spinner("Analizando..."):
                    try:
                        specific_context = _build_subperiod_context(
                            question=question,
                            global_start=start_date,
                            global_end=end_date,
                            cur=cur,
                            market_cur=market_cur,
                            hydro_cur=hydro_cur,
                        )
                        final_context = report_context + specific_context
                        answer = _call_ceo_consultant(
                            st.session_state.ceo_api_key,
                            question,
                            final_context,
                            st.session_state.ceo_chat_messages,
                        )
                        if not answer:
                            answer = "No recibi contenido del modelo."
                    except Exception as e:
                        answer = f"Error: {e}"
                st.markdown(answer)

        st.session_state.ceo_chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()

elif selection == "Explorador":
   # Redirigir al inicio o mostrar mensaje de que ahora está integrado en el chat
   t = get_theme_config()
   st.markdown(f"""
   <div style="text-align:center;padding:3rem;">
       <div style="font-size:3rem;margin-bottom:1rem;">🤖</div>
       <h2>¡El explorador ahora es inteligente!</h2>
       <p style="color:{t['TEXT_SUB']};max-width:500px;margin:0 auto;">
          Ya no necesitas buscar variables manualmente. 
          Ve a la pestaña <b>Informe del CEO</b> y pídele al consultor los datos que necesitas.
          Él generará los gráficos automáticamente para ti.
       </p>
   </div>
   """, unsafe_allow_html=True)



# ======================================================================
# FOOTER  (hidden in CEO chat view via CSS)
# ======================================================================
if selection != "Informe del CEO":
    st.markdown("---")
    st.markdown("""
    <div style="display:flex;justify-content:center;color:#94a3b8;font-size:0.75rem;">
        <b>POWERED BY STREAMLIT</b>
    </div>
    """, unsafe_allow_html=True)
