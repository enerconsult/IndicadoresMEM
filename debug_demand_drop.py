from pydataxm.pydataxm import ReadDB
import datetime as dt
import pandas as pd

api = ReadDB()
end = dt.datetime.now().date()
start = end - dt.timedelta(days=10)

print("--- FETCHING DemaCome (Last 10 days) ---")
df = api.request_data("DemaCome", "Sistema", start, end)

if df is not None and not df.empty:
    hour_cols = [c for c in df.columns if 'Hour' in c]
    print("Hourly columns found:", len(hour_cols))
    print("Columns:", df.columns.tolist())
    
    # Check last 5 rows 
    print(df[['Date']].tail(5))

    # Check last row values
    last_row = df.iloc[-1]
    print(f"\nLast Date: {last_row['Date']}")
    print("Values:", last_row[hour_cols].values)
    print("Sum:", last_row[hour_cols].sum())
else:
    print("No data found.")
