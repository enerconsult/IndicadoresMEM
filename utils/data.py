import streamlit as st
import pandas as pd
import datetime as dt
from pydataxm.pydataxm import ReadDB

@st.cache_resource
def get_api():
    return ReadDB()

@st.cache_data(ttl=3600) 
def fetch_metric_data(metric_id, entity, start_date, end_date):
    """
    Fetches metric data from API with error handling and chunking.
    Splits requests > 365 days to avoid API limits.
    """
    api = get_api()
    
    # Initialize list to hold dataframes
    all_dfs = []
    
    # Ensure inputs are date objects
    if isinstance(start_date, dt.datetime): start_date = start_date.date()
    if isinstance(end_date, dt.datetime): end_date = end_date.date()
    
    curr_start = start_date
    
    try:
        while curr_start <= end_date:
            # Define chunk end (max 365 days from curr_start)
            # XM limit is often 365 days inclusive
            chunk_end = min(end_date, curr_start + dt.timedelta(days=364))
            
            # Fetch chunk
            # print(f"Fetching {metric_id} from {curr_start} to {chunk_end}")
            df_chunk = api.request_data(metric_id, entity, curr_start, chunk_end)
            
            if df_chunk is not None and not df_chunk.empty:
                all_dfs.append(df_chunk)
            
            # Move to next day
            curr_start = chunk_end + dt.timedelta(days=1)
            
        if not all_dfs:
            return None
            
        # Concatenate all chunks
        df_final = pd.concat(all_dfs, ignore_index=True)
        
        # Deduplicate just in case (overlapping boundaries?)
        if 'Date' in df_final.columns:
            df_final['Date'] = pd.to_datetime(df_final['Date'])
            df_final = df_final.drop_duplicates(subset=['Date', 'Entity'], keep='last')
            df_final = df_final.sort_values('Date')
            
        return df_final

    except Exception as e:
        # st.error(f"Error fetching {metric_id}: {e}")
        return None

@st.cache_data(ttl=86400) 
def get_catalog():
    api = get_api()
    try:
        return api.get_collections()
    except:
        return pd.DataFrame()

def get_chart_val_col(df):
    """Helper to find the value column in a dataframe."""
    if df is None: return None
    if 'Value' in df.columns: return 'Value'
    if 'DailyValue' in df.columns: return 'DailyValue'
    
    # Fallback: Exclude Date/Id/Metadata
    v = [c for c in df.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId', 'Name', 'To']]
    return v[0] if v else (df.columns[-1] if not df.empty else None)

def calculate_periodicity(df, period, agg_func='sum'):
    """
    Aggregates dataframe based on periodicity:
    1D: Last available day (hourly if available)
    1M: Last 30 days or Last Month (Daily aggregation)
    1Y: Last 365 days or Last Year (Monthly aggregation)
    """
    if df is None or df.empty: return df
    
    if 'Date' not in df.columns: return df
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Identify hourly columns
    hour_cols = [c for c in df.columns if 'Hour' in c]
    
    # Value columns (exclude metadata)
    val_cols = [c for c in df.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId', 'Name', 'To'] and c not in hour_cols]
    
    # 1D: Return last 24h
    if period == '1D':
        last_date = df['Date'].max().date()
        df_filtered = df[df['Date'].dt.date == last_date].copy()
        
        if hour_cols:
            df_melted = df_filtered.melt(id_vars=['Date'], value_vars=hour_cols, var_name='Hour', value_name='Value')
            df_melted['HourNum'] = df_melted['Hour'].str.extract(r'(\d+)').astype(int) - 1
            df_melted = df_melted.sort_values('HourNum')
            # Construct datetime for x-axis
            df_melted['DateTime'] = df_melted.apply(lambda x: x['Date'] + pd.Timedelta(hours=x['HourNum']), axis=1)
            return df_melted[['DateTime', 'Value']].rename(columns={'DateTime': 'Date'})
            
        return df_filtered
    
    # 1M: Daily Aggregation
    elif period == '1M':
        last_date = df['Date'].max()
        start_date = last_date - dt.timedelta(days=30)
        df_filtered = df[df['Date'] >= start_date].copy()
        
        if hour_cols:
            if agg_func == 'sum':
                df_filtered['DailyValue'] = df_filtered[hour_cols].sum(axis=1)
            else: 
                df_filtered['DailyValue'] = df_filtered[hour_cols].mean(axis=1)
            val_cols = ['DailyValue']
            
        df_agg = df_filtered.groupby(df_filtered['Date'].dt.date)[val_cols].agg(agg_func).reset_index()
        # Ensure Date is datetime again (groupby output might be date object)
        df_agg['Date'] = pd.to_datetime(df_agg['Date'])
        return df_agg

    # 1Y: Monthly Aggregation
    elif period == '1Y':
        last_date = df['Date'].max()
        start_date = last_date - dt.timedelta(days=365)
        df_filtered = df[df['Date'] >= start_date].copy()
        
        if hour_cols:
            if agg_func == 'sum':
                df_filtered['DailyValue'] = df_filtered[hour_cols].sum(axis=1)
            else:
                df_filtered['DailyValue'] = df_filtered[hour_cols].mean(axis=1)
            val_cols = ['DailyValue']

        df_filtered = df_filtered.set_index('Date')
        
        # Resample
        # Use 'ME' for Month End if pandas >= 2.2, else 'M'
        # Since we pinned pandas<2.2, we use 'M'
        try:
            df_agg = df_filtered[val_cols].resample('M').agg(agg_func).reset_index()
        except ValueError:
             # Fallback for newer pandas if pinned version fails
            df_agg = df_filtered[val_cols].resample('ME').agg(agg_func).reset_index()
            
        if not df_agg.empty:
            # Shift to start of month
            df_agg['Date'] = df_agg['Date'].dt.to_period('M').dt.to_timestamp()
            
        return df_agg
    
    return df

# --- METRIC HELPER FUNCTIONS ---

def get_spot_price_metric(df_bolsa):
    """Calculates Spot Price (PrecBolsNaci) metrics."""
    val_bolsa = 0
    delta_bolsa = 0
    date_bolsa = ""
    prog_bolsa = 0.0

    if df_bolsa is not None and not df_bolsa.empty:
        last_row = df_bolsa.iloc[-1]
        date_bolsa = last_row['Date'].strftime('%Y-%m-%d')
        
        hour_cols = [c for c in df_bolsa.columns if 'Hour' in c]
        if hour_cols:
            val_bolsa = last_row[hour_cols].mean()
            
            if len(df_bolsa) > 1:
                prev_row = df_bolsa.iloc[-2]
                prev_val = prev_row[hour_cols].mean()
                delta_bolsa = ((val_bolsa - prev_val) / prev_val) * 100 if prev_val != 0 else 0
        else:
            val_col = get_chart_val_col(df_bolsa)
            val_bolsa = last_row[val_col] if val_col in last_row else 0
        
        # Progress (vs Max 30d)
        val_b_col = get_chart_val_col(df_bolsa)
        if val_b_col:
            max_30 = df_bolsa[val_b_col].max() # This takes max of whatever aggregation
            # If hourly, we need max of averages? Or max hourly?
            # Keeping simple: Max of daily row values (if hourly, it's just one sample unless we compute daily means for all history)
            # Improving: Compute daily means history
            if hour_cols:
                daily_means = df_bolsa[hour_cols].mean(axis=1)
                max_30 = daily_means.max()
            
            if max_30 > 0:
                prog_bolsa = min(1.0, val_bolsa / max_30)

    return val_bolsa, delta_bolsa, date_bolsa, prog_bolsa

def get_scarcity_metric(df_escasez, df_escasez_sup):
    """Calculates Scarcity Price metrics."""
    val_escasez = 0
    delta_escasez = 0
    prog_escasez = 0.0
    
    if df_escasez is not None and not df_escasez.empty:
        val_col_esc = get_chart_val_col(df_escasez)
        valid_rows = df_escasez[df_escasez[val_col_esc] > 0.1]
        if not valid_rows.empty:
            last_row_esc = valid_rows.iloc[-1]
            val_escasez = last_row_esc[val_col_esc]
    
    if df_escasez_sup is not None and not df_escasez_sup.empty:
        try:
            val_sup_col = get_chart_val_col(df_escasez_sup)
            val_sup = df_escasez_sup.iloc[-1][val_sup_col]
            if val_sup > 0:
                prog_escasez = min(1.0, val_escasez / val_sup)
        except: pass

    return val_escasez, delta_escasez, prog_escasez

def get_robust_demand_metric(df_demanda):
    """
    Robustly determines the latest valid demand value, handling potential data drops.
    """
    val_demanda = 0
    delta_demanda = 0
    date_demanda = ""
    prog_demanda = 0.0
    
    if df_demanda is not None and not df_demanda.empty:
        hour_cols = [c for c in df_demanda.columns if 'Hour' in c]
        if not hour_cols: return 0, 0, "", 0.0

        daily_sums = df_demanda[hour_cols].sum(axis=1)
        
        target_idx = -1
        found_valid = False
        
        # Iterate backwards to find a "stable" day
        for i in range(1, min(6, len(df_demanda))):
            idx = -1 * i
            val_current = daily_sums.iloc[idx]
            
            # 7-day average relative to THIS day
            start_bound = max(0, len(df_demanda) + idx - 15)
            end_bound = len(df_demanda) + idx
            
            history_slice = daily_sums.iloc[start_bound:end_bound]
            valid_history = history_slice[history_slice > 10]
            
            if not valid_history.empty:
                avg_7 = valid_history.tail(7).mean()
            else:
                avg_7 = val_current
            
            # Heuristic: Valid if > 80% of its history, OR if history is 0 (new)
            # Also absolute threshold > 100
            if val_current > 100 and (avg_7 == 0 or val_current > 0.8 * avg_7):
                target_idx = idx
                found_valid = True
                break
        
        if not found_valid: target_idx = -1

        row_dem = df_demanda.iloc[target_idx]
        val_demanda = daily_sums.iloc[target_idx]
        date_demanda = row_dem['Date'].strftime('%Y-%m-%d')
        
        # Delta
        if len(df_demanda) > abs(target_idx):
            prev_dem = daily_sums.iloc[target_idx - 1]
            delta_demanda = ((val_demanda - prev_dem) / prev_dem) * 100 if prev_dem != 0 else 0
            
        # Progress (vs Max 30d)
        max_dem_30 = daily_sums.max()
        if max_dem_30 > 0:
            prog_demanda = min(1.0, val_demanda / max_dem_30)
            
    return val_demanda, delta_demanda, date_demanda, prog_demanda

def get_offer_metric(df_oferta):
    """Calculates Offer (MaxPrecOferNal) metrics."""
    max_oferta, min_oferta, avg_oferta = 0, 0, 0
    date_oferta = ""
    prog_oferta = 0.0

    if df_oferta is not None and not df_oferta.empty:
        last_row_off = df_oferta.iloc[-1]
        date_oferta = last_row_off['Date'].strftime('%Y-%m-%d')
        hour_cols = [c for c in df_oferta.columns if 'Hour' in c]
        if hour_cols:
            vals = last_row_off[hour_cols]
            max_oferta = vals.max()
            min_oferta = vals.min()
            avg_oferta = vals.mean()
            
            if max_oferta > 0:
                prog_oferta = min(1.0, avg_oferta / max_oferta)
                
    return max_oferta, min_oferta, avg_oferta, date_oferta, prog_oferta
