import datetime as dt
from pydataxm.pydataxm import ReadDB
import pandas as pd

api = ReadDB()
start_date = dt.datetime.now() - dt.timedelta(days=10)
end_date = dt.datetime.now()

print(f"Fetching DemaCome from {start_date.date()} to {end_date.date()}...")

df = api.request_data("DemaCome", "Sistema", start_date.date(), end_date.date())

if df is not None and not df.empty:
    print("\nLast 5 rows:")
    # print(df.tail(5))
    
    hour_cols = [c for c in df.columns if 'Hour' in c]
    if hour_cols:
        print(f"\nFound {len(hour_cols)} hour columns.")
        
        # Calculate daily sums
        df['DailySum'] = df[hour_cols].sum(axis=1)
        
        print("\nDaily Sums (GWh):")
        print(df[['Date', 'DailySum']].tail(5))
        
        # Test Heuristic
        if len(df) > 1:
            v_last = df.iloc[-1]['DailySum']
            v_prev = df.iloc[-2]['DailySum']
            
            print(f"\nLast Value: {v_last}")
            print(f"Prev Value: {v_prev}")
            print(f"Ratio (Last/Prev): {v_last/v_prev:.2f}")
            
            if v_last < 0.6 * v_prev:
                print(">> HEURISTIC TRIGGERED: Last value is < 60% of previous.")
            else:
                print(">> HEURISTIC NOT TRIGGERED.")
                
            # Test 7-Day Average Heuristic
            if len(df) >= 8:
                # Exclude last day from average
                last_7 = df.iloc[-8:-1]['DailySum']
                avg_7 = last_7.mean()
                print(f"\n7-Day Average (excluding last): {avg_7:.2f}")
                print(f"Ratio (Last/Avg): {v_last/avg_7:.2f}")
                
                if v_last < 0.8 * avg_7: # Stricter threshold vs average?
                     print(">> AVG HEURISTIC TRIGGERED (< 80% of 7-day avg)")
    else:
        print("No hourly columns found.")
        print(df.columns)
else:
    print("DataFrame is empty.")
