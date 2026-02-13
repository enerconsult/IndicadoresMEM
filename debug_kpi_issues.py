import datetime as dt
from pydataxm.pydataxm import ReadDB
import pandas as pd

api = ReadDB()
start_date = dt.datetime.now() - dt.timedelta(days=15)
end_date = dt.datetime.now()

print(f"Fetching from {start_date.date()} to {end_date.date()}...\n")

# 1. Debug Demand
print("--- DEMAND (DemaCome) ---")
df_dem = api.request_data("DemaCome", "Sistema", start_date.date(), end_date.date())
if df_dem is not None and not df_dem.empty:
    hour_cols = [c for c in df_dem.columns if 'Hour' in c]
    df_dem['DailySum'] = df_dem[hour_cols].sum(axis=1)
    
    # Calculate 7-day average of the FIRST 7 days in this window (stable config)
    # or just rolling average
    df_dem['Rolling7'] = df_dem['DailySum'].rolling(window=7).mean().shift(1)
    
    print(df_dem[['Date', 'DailySum', 'Rolling7']].tail(10))
    
    # Simulate Heuristic
    for i in range(1, 6):
        idx = -1 * i
        row = df_dem.iloc[idx]
        val = row['DailySum']
        avg = row['Rolling7']
        if pd.isna(avg): avg = 0
        
        ratio = val / avg if avg > 0 else 0
        status = "OK" if ratio > 0.8 else "PARTIAL/LOW"
        print(f"Index {idx} ({row['Date']}): Val={val:.2f}, RefAvg={avg:.2f}, Ratio={ratio:.2f} -> {status}")

# 2. Debug Scarcity
print("\n--- SCARCITY (PrecEsca) ---")
df_esc = api.request_data("PrecEsca", "Sistema", start_date.date(), end_date.date())
if df_esc is not None and not df_esc.empty:
    val_cols = [c for c in df_esc.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId']]
    print(df_esc[['Date'] + val_cols].tail(5))
else:
    print("PrecEsca is Empty")
