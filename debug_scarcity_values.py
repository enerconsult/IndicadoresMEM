import datetime as dt
from pydataxm.pydataxm import ReadDB
import pandas as pd

api = ReadDB()
start_date = dt.datetime.now() - dt.timedelta(days=45)
end_date = dt.datetime.now()

print(f"Fetching PrecEsca from {start_date.date()} to {end_date.date()}...")

df = api.request_data("PrecEsca", "Sistema", start_date.date(), end_date.date())

if df is not None and not df.empty:
    val_cols = [c for c in df.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId']]
    print("Columns:", val_cols)
    print(df[['Date'] + val_cols].tail(15))
    
    # Check for non-zero values
    non_zeros = df[df[val_cols[0]] > 0]
    if not non_zeros.empty:
        print("\nLast and Non-Zero Value:")
        print(non_zeros[['Date'] + val_cols].iloc[-1])
    else:
        print("\nAll values are 0.")
else:
    print("DataFrame is empty.")
