import datetime as dt
from pydataxm.pydataxm import ReadDB
import pandas as pd

api = ReadDB()
start_date = dt.datetime.now() - dt.timedelta(days=15)
end_date = dt.datetime.now()

print(f"Fetching data from {start_date.date()} to {end_date.date()}...\n")

# --- 1. SCARCITY DEEP DIVE ---
print(">>> SCARCITY (PrecEsca) <<<")
df_esc = api.request_data("PrecEsca", "Sistema", start_date.date(), end_date.date())

if df_esc is not None and not df_esc.empty:
    print("Columns:", df_esc.columns.tolist())
    print("Dtypes:\n", df_esc.dtypes)
    
    # Identify value column
    val_cols = [c for c in df_esc.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId']]
    print("Potential Value Columns:", val_cols)
    
    if val_cols:
        target_col = val_cols[0]
        print(f"Target Column: '{target_col}'")
        print("\nLast 10 Rows:")
        print(df_esc[['Date', target_col]].tail(10))
        
        # Check non-zero
        non_zeros = df_esc[df_esc[target_col] > 0]
        print(f"\nCount of non-zero rows: {len(non_zeros)}")
        if not non_zeros.empty:
            print("Last Non-Zero Row:")
            print(non_zeros.iloc[-1])
else:
    print("PrecEsca dataframe is None or empty.")

# --- 2. DEMAND DEEP DIVE ---
print("\n>>> DEMAND (DemaCome) <<<")
df_dem = api.request_data("DemaCome", "Sistema", start_date.date(), end_date.date())

if df_dem is not None and not df_dem.empty:
    hour_cols = [c for c in df_dem.columns if 'Hour' in c]
    df_dem['DailySum'] = df_dem[hour_cols].sum(axis=1)
    
    # Calculate rolling 7-day average (shifted by 1 to represent previous history)
    # We want to see what the heuristic sees
    
    print("\nLast 10 Days Analysis:")
    last_10 = df_dem.tail(10).copy().reset_index(drop=True)
    
    for i, row in last_10.iterrows():
        # Reconstruct 7-day history for this specific date from the FULL dataframe
        curr_date = row['Date']
        
        # Slice original df for 7 days strictly BEFORE curr_date
        history = df_dem[(df_dem['Date'] >= curr_date - dt.timedelta(days=7)) & (df_dem['Date'] < curr_date)]
        
        if not history.empty:
            avg_7 = history['DailySum'].mean()
        else:
            avg_7 = 0
            
        val = row['DailySum']
        ratio = val / avg_7 if avg_7 > 0 else 0
        status = "KEEP" if (avg_7 == 0 or val > 0.8 * avg_7) else "DROP (<80%)"
        
        print(f"Date: {curr_date.date()} | Val: {val:,.1f} | Avg7: {avg_7:,.1f} | Ratio: {ratio:.2f} -> {status}")

