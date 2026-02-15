import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydataxm.pydataxm import ReadDB
import datetime as dt

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Indicadores MEM - Enerconsult",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME MANAGEMENT ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.toggle_dark_mode else 'light'

# Sidebar Toggle
with st.sidebar:
    # Logo moved to top
    st.image("logo_empresa.png", width=150)
    st.markdown("<br>", unsafe_allow_html=True)

# --- DYNAMIC COLORS ---
if st.session_state.theme == 'dark':
    # Dark Premium Palette
    BG_COLOR = "#0f172a" # Slate 900
    CARD_BG = "rgba(30, 41, 59, 0.7)" # Slate 800 + Opacity
    TEXT_COLOR = "#f8fafc" # Slate 50
    TEXT_SUB = "#94a3b8" # Slate 400
    BORDER_COLOR = "rgba(255, 255, 255, 0.1)"
    COLOR_BLUE_DARK = "#60a5fa" # Blue 400 (Lighter for dark mode)
    COLOR_BLUE = "#3b82f6"
    COLOR_ORANGE = "#f97316" # Orange 500
    PLOT_TEMPLATE = "plotly_dark"
    GRID_COLOR = "#334155"
else:
    # Light Classic Palette
    BG_COLOR = "#F8FAFC"
    CARD_BG = "#ffffff"
    TEXT_COLOR = "#1e293b"
    TEXT_SUB = "#64748b"
    BORDER_COLOR = "#e2e8f0"
    COLOR_BLUE_DARK = "#003366"
    COLOR_BLUE = "#137fec"
    COLOR_ORANGE = "#F37021"
    PLOT_TEMPLATE = "plotly_white"
    GRID_COLOR = "#e2e8f0"

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    .stApp {{
        font-family: 'Inter', sans-serif;
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}
    
    html, body {{
        background-color: {BG_COLOR};
    }}
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {{
        background-color: {'#1e293b' if st.session_state.theme == 'dark' else '#003366'};
        border-right: 1px solid {BORDER_COLOR};
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
        color: {TEXT_COLOR if st.session_state.theme == 'dark' else '#003366'} !important;
        font-weight: 700;
    }}
    
    /* Metrics / Cards - Streamlit Native Override */
    [data-testid="stMetricValue"] {{
        color: {TEXT_COLOR} !important;
    }}
    
    /* Streamlit widgets */
    .stDateInput label, .stSelectbox label, .stRadio label {{
        color: {TEXT_COLOR} !important;
    }}
    
    /* FIX: Input text color should be dark if background is white-ish (default streamlit input) */
    /* Or if we want to ensure visibility, let's force a dark color for the text inside the input box */
    .stDateInput input {{
        color: #334155 !important; /* Dark Slate to ensure visibility on light input bg */
        font-weight: 600;
    }}
    
    /* Additional Dark Mode Tweaks */
    {'div[data-testid="stExpander"] { border: 1px solid ' + BORDER_COLOR + '; border-radius: 8px; }' if st.session_state.theme == 'dark' else ''}
    
    /* Hide Footer Only */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
</style>
""", unsafe_allow_html=True)

# --- KPI CARD COMPONENT ---
# --- KPI CARD COMPONENT ---
def kpi_card_html(title, value, delta, sub_text="", color_bar=COLOR_BLUE, progress=1.0):
    """
    Generates HTML for a custom KPI card matching the design.
    Adapts to theme via global variables.
    progress: float 0.0 to 1.0 (determines how many bars are filled)
    """
    delta_color = "#10b981" if delta >= 0 else "#f43f5e" # Emerald vs Rose
    delta_sign = "+" if delta >= 0 else ""
    delta_str = f"{delta_sign}{delta:.1f}%"
    
    # Dynamic Styles
    card_style = f"background-color: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 0.75rem; padding: 1.25rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); height: 100%; display: flex; flex-direction: column; justify-content: space-between; backdrop-filter: blur(10px);"
    title_style = f"color: {TEXT_SUB}; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;"
    value_style = f"font-size: 1.8rem; font-weight: 800; color: {TEXT_COLOR};"
    sub_style = f"font-size: 0.7rem; color: {TEXT_SUB}; font-style: italic; margin-top: 0.5rem;"
    
    # Progress Logic (3 Bars)
    # < 0.33: 1 bar (Low intensity)
    # < 0.66: 2 bars (Med intensity)
    # >= 0.66: 3 bars (High intensity)
    
    # Base colors for inactive bars
    bg_inactive = f"{TEXT_SUB}30"
    
    # Bar 1 (Always active if progress > 0, else inactive)
    c1 = color_bar if progress > 0 else bg_inactive
    
    # Bar 2
    c2 = color_bar if progress > 0.33 else bg_inactive
    
    # Bar 3
    c3 = color_bar if progress > 0.66 else bg_inactive
    
    # Opacity for "filling" effect or just solid? Let's keep the solid look but vary the active bars.
    # Actually, the user liked the look of different opacities. Let's make active bars FULL color, inactive bars formatted.
    
    
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

# --- API & DATA FUNCTIONS ---
@st.cache_resource
def get_api():
    return ReadDB()

api = get_api()

@st.cache_data(ttl=3600) 
def fetch_metric_data(metric_id, entity, start_date, end_date):
    try:
        df = api.request_data(metric_id, entity, start_date, end_date)
        if df is not None and not df.empty:
             # Ensure Date column is datetime
             if 'Date' in df.columns:
                 df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=86400) 
def get_catalog():
    return api.get_collections()

# --- HELPER FUNCTIONS ---
def get_id_col(df):
    """Dynamically find the identifier column."""
    for col in ['Values_code', 'Name', 'Entity', 'Id', 'To']:
        if col in df.columns:
            return col
    return 'Id' # Fallback

def get_latest_value(df, value_col=None):
    if df is None or df.empty:
        return 0, None, 0
    
    cols = [c for c in df.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId', 'Name', 'To']]
    col = value_col if value_col else (cols[0] if cols else None)
    
    if not col:
        return 0, None, 0

    # Date conversion is now handled in fetch, but double check
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    if 'Date' in df.columns:
        df = df.sort_values('Date')
        latest_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
        latest_val = df[col].iloc[-1]
        
        prev_val = df[col].iloc[-2] if len(df) > 1 else latest_val
        delta = ((latest_val - prev_val) / prev_val) * 100 if prev_val != 0 else 0
        
        return latest_val, latest_date, delta
    return 0, dt.datetime.now().strftime('%Y-%m-%d'), 0

def calculate_periodicity(df, period, agg_func='sum'):
    """
    Aggregates dataframe based on periodicity:
    1D: Last available day (hourly if available)
    1M: Last 30 days or Last Month (Daily aggregation)
    1Y: Last 365 days or Last Year (Monthly aggregation)
    """
    if df is None or df.empty: return df
    
    if 'Date' not in df.columns: return df
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    
    # Identify hourly columns (Values_Hour01... or Hour01...)
    hour_cols = [c for c in df.columns if 'Hour' in c]
    
    # Value columns (exclude metadata AND Date AND Hourly columns)
    val_cols = [c for c in df.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId', 'Name', 'To'] and c not in hour_cols]
    
    # 1D: Return last 24h (or last day records)
    if period == '1D':
        last_date = df['Date'].max().date()
        df_filtered = df[df['Date'].dt.date == last_date].copy()
        
        # If we have hourly columns, melt them to get 24 data points
        if hour_cols:
            # Melt: Date, Hour, Value
            df_melted = df_filtered.melt(id_vars=['Date'], value_vars=hour_cols, var_name='Hour', value_name='Value')
            # Convert HourXX to integer/time
            df_melted['HourNum'] = df_melted['Hour'].str.extract(r'(\d+)').astype(int) - 1 # 0-23
            df_melted = df_melted.sort_values('HourNum')
            # Create a full datetime for plotting if needed, or just use Hour as x
            # Let's use a constructed datetime for the X axis to show time of day
            df_melted['DateTime'] = df_melted.apply(lambda x: x['Date'] + pd.Timedelta(hours=x['HourNum']), axis=1)
            # Rename for consistency with other views which might expect 'Date' to be the x-axis
            df_melted = df_melted[['DateTime', 'Value']].rename(columns={'DateTime': 'Date'})
            return df_melted
            
        return df_filtered
    
    # 1M: Daily Aggregation for last month
    elif period == '1M':
        # Filter last 30 days
        last_date = df['Date'].max()
        start_date = last_date - dt.timedelta(days=30)
        df_filtered = df[df['Date'] >= start_date].copy()
        
        # If hourly data exists, we need to sum/mean across hours FIRST, then group by Date
        if hour_cols:
            if agg_func == 'sum':
                df_filtered['DailyValue'] = df_filtered[hour_cols].sum(axis=1)
            else: # mean
                df_filtered['DailyValue'] = df_filtered[hour_cols].mean(axis=1)
            # Now we have one value per day per entity
            val_cols = ['DailyValue']
            
        # Aggregation
        df_filtered['Date'] = df_filtered['Date'].dt.date
        if agg_func == 'sum':
            df_agg = df_filtered.groupby('Date')[val_cols].sum().reset_index()
        else:
            df_agg = df_filtered.groupby('Date')[val_cols].mean().reset_index()
        return df_agg

    # 1Y: Monthly Aggregation for last year
    elif period == '1Y':
        last_date = df['Date'].max()
        start_date = last_date - dt.timedelta(days=365)
        df_filtered = df[df['Date'] >= start_date].copy()
        
        # Handle hourly columns if present
        if hour_cols:
            if agg_func == 'sum':
                df_filtered['DailyValue'] = df_filtered[hour_cols].sum(axis=1)
            else:
                df_filtered['DailyValue'] = df_filtered[hour_cols].mean(axis=1)
            val_cols = ['DailyValue']

        # Aggregation to Month-Year using Resample for safety
        df_filtered = df_filtered.set_index('Date')
        
        # Select only numeric columns for resampling to avoid errors
        cols_to_resample = val_cols
        
        if agg_func == 'sum':
            df_agg = df_filtered[cols_to_resample].resample('ME').sum().reset_index()
        else:
            df_agg = df_filtered[cols_to_resample].resample('ME').mean().reset_index()
            
        # Shift date to start of month for better chart alignment (User reported visual shift)
        if not df_agg.empty:
            df_agg['Date'] = df_agg['Date'].dt.to_period('ME').dt.to_timestamp()
            
        return df_agg
    
    return df

def render_chart_controls(key_prefix, options=["1D", "1M", "1Y"]):
    return st.radio(
        "Periodicidad", 
        options, 
        horizontal=True, 
        key=key_prefix, 
        label_visibility="collapsed"
    )

# --- MAIN APP LOGIC ---

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.75rem;">
        <div style="background-color: white; padding: 4px; border-radius: 4px;">
            <span style="color: {COLOR_BLUE_DARK}; font-weight: bold; font-size: 1.2rem;">⚡</span>
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.1rem; letter-spacing: -0.05em; color: white;">ENERCONSULT</div>
            <div style="font-size: 0.6rem; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase;">Dashboards Pro</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Toggle moved to here
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
    menu_options = [
        "Resumen", # Renamed for cleaner menu
        "Precios",
        "Demanda",
        "Generación",
        "Embalses",
        "Explorador"
    ]
    selection = st.radio("Navegación", menu_options, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Reports section (Visual only)
    st.markdown("### REPORTES", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color: #cbd5e1; padding: 5px 0; font-size: 0.9rem; cursor: pointer;">📄 Informes XM</div>
    <div style="color: #cbd5e1; padding: 5px 0; font-size: 0.9rem; cursor: pointer;">📊 Datos Históricos</div>
    """, unsafe_allow_html=True)
    
    # Footer User
    st.markdown(f"""
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 32px; height: 32px; background-color: rgba(255,255,255,0.1); border-radius: 50%; display: flex; align-items: center; justify-content: center;">📊</div>
        <div>
            <div style="font-weight: 700; font-size: 0.8rem; color: white;">Fuente de Datos</div>
            <div style="font-size: 0.65rem; color: #94a3b8;">XM S.A. E.S.P.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- VIEWS ---

# Common Plot Styling
# Common Plot Styling
def style_fig(fig, y_axis_title=None):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        font_family="Inter",
        # title_font_family="Inter", 
        # title_font_weight=700,
        # title_font_color=TEXT_COLOR,
        title_text="", # FORCE EMPTY TITLE
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified", # Enable unified hover tooltip
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color=TEXT_SUB)
        )
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER_COLOR, tickfont=dict(color=TEXT_SUB))
    
    # Check if a title was provided
    if y_axis_title:
        fig.update_layout(yaxis_title=y_axis_title)
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color=TEXT_SUB), title_text=y_axis_title)
    else:
        fig.update_layout(yaxis_title="")
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, linecolor=BORDER_COLOR, tickfont=dict(color=TEXT_SUB), title_text="")
        
    return fig

def get_chart_val_col(df):
    if 'Value' in df.columns: return 'Value'
    if 'DailyValue' in df.columns: return 'DailyValue'
    # Fallback: Exclude Date/Id/Metadata
    v = [c for c in df.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId', 'Name', 'To']]
    return v[0] if v else df.columns[-1]

def plot_scarcity_line(df_scarcity, name, color, dash_style, broadcast_x=None):
     if df_scarcity is not None and not df_scarcity.empty:
        val_col = get_chart_val_col(df_scarcity)
        
        y_vals = df_scarcity[val_col]
        x_vals = df_scarcity['Date']
        
        # Broadcasting logic: If 1D view (1 data point) vs Hourly X-axis
        if broadcast_x is not None and len(df_scarcity) == 1:
            val = y_vals.iloc[0]
            y_vals = [val] * len(broadcast_x)
            x_vals = broadcast_x
            
        return go.Scatter(
             x=x_vals, y=y_vals,
             name=name, mode='lines', line=dict(color=color, dash=dash_style, width=2)
        )
     return None

if selection == "Resumen":
    # Header
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem;">
            <h1 style="margin: 0; font-size: 1.5rem;">Indicadores MEM - Colombia</h1>
            <span style="background-color: {COLOR_ORANGE}20; color: {COLOR_ORANGE}; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;">En Vivo</span>
        </div>
        <p style="color: {TEXT_SUB}; font-size: 0.8rem; margin-top: 0.25rem;">Vista general del mercado para el periodo seleccionado.</p>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div style="text-align: right; color: {TEXT_SUB}; font-size: 0.8rem;">
            <span class="material-symbols-outlined" style="vertical-align: bottom; font-size: 1rem;">calendar_today</span>
            {dt.datetime.now().strftime('%d %B, %Y')}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. KPI CARDS
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
        <div style="width: 4px; height: 1.5rem; background-color: {COLOR_ORANGE}; border-radius: 2px;"></div>
        <h2 style="margin: 0; font-size: 1.25rem;">Indicadores Clave del Día</h2>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    # Data Fetching
    with st.spinner("Actualizando indicadores..."):
        # 1. Spot Price (Last Day - Average)
        df_bolsa = fetch_metric_data("PrecBolsNaci", "Sistema", start_date, end_date)
        val_bolsa = 0
        delta_bolsa = 0
        date_bolsa = ""
        
        if df_bolsa is not None and not df_bolsa.empty:
             val_col = get_chart_val_col(df_bolsa)
             # If hourly columns exist (melted or raw), we want daily average
             # fetch_metric_data returns unmelted for raw request usually? 
             # Actually fetch_metric_data returns raw DF. 
             # Let's handle raw hourly columns for last day
             last_row = df_bolsa.iloc[-1]
             date_bolsa = last_row['Date'].strftime('%Y-%m-%d')
             
             hour_cols = [c for c in df_bolsa.columns if 'Hour' in c]
             if hour_cols:
                 val_bolsa = last_row[hour_cols].mean()
                 
                 # Delta vs prev day
                 if len(df_bolsa) > 1:
                     prev_row = df_bolsa.iloc[-2]
                     prev_val = prev_row[hour_cols].mean()
                     delta_bolsa = ((val_bolsa - prev_val) / prev_val) * 100 if prev_val != 0 else 0
             else:
                 # Fallback if already aggregated or single value
                 val_bolsa = last_row[val_col] if val_col in last_row else 0
                 
        # Progress: Current Price vs Max of last 30 days
        prog_bolsa = 0.0
        if df_bolsa is not None and not df_bolsa.empty:
            val_b_col = get_chart_val_col(df_bolsa)
            max_30 = df_bolsa[val_b_col].max()
            if max_30 > 0:
                prog_bolsa = min(1.0, val_bolsa / max_30)

        # 2. Scarcity Price (Last Month) - Base
        # We need sup/inf for charts later, so fetch them here to avoid NameError
        df_escasez = fetch_metric_data("PrecEsca", "Sistema", start_date, end_date)
        df_escasez_sup = fetch_metric_data("PrecEscaSup", "Sistema", start_date, end_date)
        df_escasez_inf = fetch_metric_data("PrecEscaInf", "Sistema", start_date, end_date)
        
        # FIX: Directly access last valid non-zero value to avoid 0.0 or 0.xxxx issue
        if df_escasez is not None and not df_escasez.empty:
            val_col_esc = get_chart_val_col(df_escasez)
            # Filter for non-zero values
            valid_rows = df_escasez[df_escasez[val_col_esc] > 0.1] # Threshold to avoid near-zero noise
            if not valid_rows.empty:
                last_row_esc = valid_rows.iloc[-1]
                val_escasez = last_row_esc[val_col_esc]
            else:
                val_escasez = 0
            delta_escasez = 0 
        else:
            val_escasez = 0
            delta_escasez = 0
            
        # Progress: Base Scarcity vs Superior Scarcity (Ceiling)
        prog_escasez = 0.0
        if df_escasez_sup is not None and not df_escasez_sup.empty:
             try:
                val_sup_col = get_chart_val_col(df_escasez_sup)
                val_sup = df_escasez_sup.iloc[-1][val_sup_col]
                if val_sup > 0:
                    prog_escasez = min(1.0, val_escasez / val_sup)
             except: pass
        
        # 3. System Demand (Robust Check)
        df_demanda = fetch_metric_data("DemaCome", "Sistema", start_date, end_date)
        val_demanda = 0
        delta_demanda = 0
        date_demanda = ""
        
        if df_demanda is not None and not df_demanda.empty:
             hour_cols = [c for c in df_demanda.columns if 'Hour' in c]
             
             # Robust Loop: Find last valid day (Backtrack up to 5 days)
             target_idx = -1
             found_valid = False
             
             # Pre-calculate daily sums for the whole range
             daily_sums = df_demanda[hour_cols].sum(axis=1) if hour_cols else pd.Series([0]*len(df_demanda))
             
             # Iterate backwards from last day
             for i in range(1, min(6, len(df_demanda))):
                 idx = -1 * i
                 val_current = daily_sums.iloc[idx]
                 curr_date = df_demanda.iloc[idx]['Date']
                 
                 # Calculate 7-day average relative to THIS day (excluding this day)
                 # Filter strictly for valid history days (non-zero)
                 start_bound = max(0, len(df_demanda) + idx - 15) # Look back 15 days to find 7 valid
                 end_bound = len(df_demanda) + idx
                 
                 history_slice = daily_sums.iloc[start_bound:end_bound]
                 # Filter out zeros from history to get a TRUE average
                 valid_history = history_slice[history_slice > 10] # >10 GWh to be safe
                 
                 if not valid_history.empty:
                    avg_7 = valid_history.tail(7).mean() # Take avg of last 7 VALID days
                 else:
                    avg_7 = val_current # No history, assume current is valid if > 0
                 
                 # Heuristic: Valid if > 80% of its history, OR if history is 0
                 # Also check absolute threshold (e.g. > 100 GWh for system)
                 if val_current > 100 and (avg_7 == 0 or val_current > 0.8 * avg_7):
                     target_idx = idx
                     found_valid = True
                     break
             
             if not found_valid:
                 target_idx = -1 # Default to last if nothing passes

             row_dem = df_demanda.iloc[target_idx]
             val_demanda = row_dem[hour_cols].sum() if hour_cols else 0
             date_demanda = row_dem['Date'].strftime('%Y-%m-%d')
             
             # Delta
             if len(df_demanda) > abs(target_idx):
                 row_prev = df_demanda.iloc[target_idx - 1]
                 prev_dem = row_prev[hour_cols].sum() if hour_cols else 0
                 delta_demanda = ((val_demanda - prev_dem) / prev_dem) * 100 if prev_dem != 0 else 0
                 
        # Progress: Current Demand vs Max Demand of last 30 days
        prog_demanda = 0.0
        if df_demanda is not None and not df_demanda.empty and hour_cols:
             # Calculate daily sums just strictly for max check
            d_sums = df_demanda[hour_cols].sum(axis=1)
            max_dem_30 = d_sums.max()
            if max_dem_30 > 0:
                prog_demanda = min(1.0, val_demanda / max_dem_30)

        # 4. Max Offer (MaxPrecOferNal - Average of Day as Main)
        df_oferta = fetch_metric_data("MaxPrecOferNal", "Sistema", start_date, end_date)
        max_oferta, min_oferta, avg_oferta = 0, 0, 0
        date_oferta = ""
        
        if df_oferta is not None and not df_oferta.empty:
            last_row_off = df_oferta.iloc[-1]
            date_oferta = last_row_off['Date'].strftime('%Y-%m-%d')
            hour_cols = [c for c in df_oferta.columns if 'Hour' in c]
            if hour_cols:
                vals = last_row_off[hour_cols]
                max_oferta = vals.max()
                min_oferta = vals.min()
                avg_oferta = vals.mean()
        
        # Progress: Average Offer / Max Offer (Consensus Strength)
        prog_oferta = 0.0
        if max_oferta > 0:
            prog_oferta = min(1.0, avg_oferta / max_oferta)

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
            name='Precio Bolsa', mode='lines+markers', line=dict(color=COLOR_ORANGE)
        ))
        
        # Scarcity Lines - Pass broadcast_x for 1D view
        broadcast_x = df_bolsa_chart['Date'] if period_price == '1D' else None
        
        # 1. Base
        trace_base = plot_scarcity_line(df_escasez_chart, 'Precio Escasez', COLOR_BLUE_DARK, 'solid', broadcast_x)
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
        
        # User Feedback: Drop last day if it falls significantly (partial data)
        # Apply strict 7-day average check with loop
        if len(df_dem_chart) >= 8:
             # Loop backwards to find split point
             cut_idx = 0 # Default: Keep all
             
             # Check last few days
             for i in range(1, min(6, len(df_dem_chart))):
                 idx = -1 * i
                 val_current = df_dem_chart.iloc[idx][val_col_dem]
                 
                 # 7-day avg prior to this point (Robust check)
                 start_bound = max(0, len(df_dem_chart) + idx - 15)
                 end_bound = len(df_dem_chart) + idx
                 
                 history_slice = df_dem_chart.iloc[start_bound:end_bound][val_col_dem]
                 valid_history = history_slice[history_slice > 10]
                 
                 if valid_history.empty:
                     avg_7 = val_current # Assume valid
                 else:
                     avg_7 = valid_history.tail(7).mean()
                 
                 # Condition to DROP:
                 # If Avg is valid (>0) AND Current is < 80% of Avg
                 if avg_7 > 0 and val_current < 0.8 * avg_7:
                     cut_idx = idx - 1 # Cut this and subsequent
                 else:
                     # Found a good day, stop looking back
                     if cut_idx != 0: cut_idx = idx # Cut everything AFTER this good day
                     break
             
             if cut_idx < 0:
                 df_dem_chart = df_dem_chart.iloc[:cut_idx+1]
        elif len(df_dem_chart) > 2:
             last_val = df_dem_chart.iloc[-1][val_col_dem]
             prev_val = df_dem_chart.iloc[-2][val_col_dem]
             if last_val < 0.6 * prev_val:
                 df_dem_chart = df_dem_chart.iloc[:-1]
        
        # Scale to GWh (assuming raw is kWh)
        y_dem = df_dem_chart[val_col_dem] / 1e6
        
        fig_dg.add_trace(go.Scatter(
            x=df_dem_chart['Date'], y=y_dem,
            name='Demanda Comercial', mode='lines+markers', line=dict(color=COLOR_BLUE)
        ))
        
        if df_gen_chart is not None and not df_gen_chart.empty:
            val_col_gen = get_chart_val_col(df_gen_chart)
            y_gen = df_gen_chart[val_col_gen] / 1e6
            fig_dg.add_trace(go.Scatter(
                x=df_gen_chart['Date'], y=y_gen,
                name='Generación Real', mode='lines+markers', line=dict(color=COLOR_ORANGE)
            ))
            
        fig_dg = style_fig(fig_dg, "GWh")
        st.plotly_chart(fig_dg, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CHART 3: Capacity vs Volume ---
    st.markdown("### Capacidad Útil vs Volumen Útil (GWh)")
    col_c, col_cc = st.columns([1, 4])
    with col_c:
        period_cap = render_chart_controls("period_cap", options=["1M", "1Y"])
    
    # Note: Check IDs. 'CapaUtilDiarEner' vs 'VoluUtilDiarEner'
    df_cap = fetch_metric_data("CapaUtilDiarEner", "Sistema", start_date, end_date) # Check ID validity?
    df_vol = fetch_metric_data("VoluUtilDiarEner", "Sistema", start_date, end_date) # Check ID validity?

    # Aggregation: Mean is correct for capacity/volume (User Correction)
    df_cap_chart = calculate_periodicity(df_cap, period_cap, 'mean')
    df_vol_chart = calculate_periodicity(df_vol, period_cap, 'mean')
    
    if df_vol_chart is not None and not df_vol_chart.empty:
        fig_cv = go.Figure()
        
        val_col_vol = get_chart_val_col(df_vol_chart)
        # Volume
        y_vol = df_vol_chart[val_col_vol] / 1e6
        fig_cv.add_trace(go.Scatter(
            x=df_vol_chart['Date'], y=y_vol,
            name='Volumen Útil', mode='none', fill='tozeroy', fillcolor='rgba(19, 127, 236, 0.4)'
        ))

        # Capacity
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
        
    # Valid IDs: 'AporEner' (GWh). 'MediaHist' (Historical Mean).
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
            name='Aportes Hídricos', marker_color=COLOR_BLUE
        ))
        
        # Add Historical Mean Trace
        if df_media_chart is not None and not df_media_chart.empty:
             val_col_media = get_chart_val_col(df_media_chart)
             y_media = df_media_chart[val_col_media] / 1e6
             fig_ap.add_trace(go.Scatter(
                 x=df_media_chart['Date'], y=y_media,
                 name='Media Histórica', mode='lines', line=dict(color=COLOR_ORANGE, dash='solid', width=3)
             ))
             
        fig_ap = style_fig(fig_ap, "GWh")
        st.plotly_chart(fig_ap, use_container_width=True)
    else:
         st.info("Cargando datos de Aportes...")


# --- OTHER PAGES (Retaining logic but applying basic theme) ---

elif selection == "Precios":
    st.title("💰 Precios Detallados")
    metrics = {
        "Precio Bolsa Nacional": "PrecBolsNaci",
        "Oferta de Despacho": "PrecOferDesp",
        "Escasez": "PrecEscasez", 
        "Contratos Regulados": "PrecPromContRegu",
        "Contratos No Regulados": "PrecPromContNoRegu"
    }
    
    st.markdown("### Tendencia de Precios")
    
    for name, mid in metrics.items():
        st.subheader(name)
        df = fetch_metric_data(mid, "Sistema", start_date, end_date)
        if (df is None or df.empty) and mid == "PrecOferDesp": 
            df = fetch_metric_data(mid, "Recurso", start_date, end_date)
        
        if df is not None and not df.empty:
            if 'Recurso' in df.columns or 'Values_code' in df.columns:
                 # It's complex data, simplify to mean for chart
                 num = df.select_dtypes(include=['float', 'int'])
                 if not num.empty:
                     df['Promedio'] = num.mean(axis=1)
                     fig = px.line(df, x='Date', y='Promedio', color_discrete_sequence=[COLOR_BLUE])
                     st.plotly_chart(style_fig(fig, ""), use_container_width=True)
            else:
                 fig = px.line(df, x='Date', y=df.columns[-1], color_discrete_sequence=[COLOR_BLUE])
                 st.plotly_chart(style_fig(fig, ""), use_container_width=True)
        else:
             st.info(f"Sin datos para {name}")

elif selection == "Explorador":
    st.title("🔍 Explorador Avanzado")
    with st.spinner("Cargando catálogo..."):
        df_vars = get_catalog()
    
    df_vars['DisplayName'] = df_vars['MetricName'] + " (" + df_vars['Entity'] + ")"
    var_map = df_vars.set_index('DisplayName')[['MetricId', 'Entity']].to_dict('index')

    metric_option = st.selectbox("Seleccione Variable", options=sorted(list(var_map.keys())), index=0)

    if st.button("Consultar"):
        meta = var_map[metric_option]
        df = fetch_metric_data(meta['MetricId'], meta['Entity'], start_date, end_date)
        
        if df is not None:
            # Simple visualization
            cols = [c for c in df.columns if c not in ['Date', 'Id', 'Entity', 'MetricId', 'Values_code']]
            if cols:
                fig = px.line(df, x='Date', y=cols[0], title=metric_option, color_discrete_sequence=[COLOR_BLUE])
                st.plotly_chart(style_fig(fig, ""), use_container_width=True)
            st.dataframe(df)

# Footer for main content
st.markdown("---")
st.markdown(f"""
<div style="display: flex; justify-content: center; align-items: center; color: #94a3b8; font-size: 0.75rem;">
    <div>
        <b>POWERED BY STREAMLIT</b>
    </div>
</div>
""", unsafe_allow_html=True)
