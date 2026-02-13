import datetime as dt
from pydataxm.pydataxm import ReadDB
import pandas as pd

api = ReadDB()
start_date = dt.datetime.now() - dt.timedelta(days=30)
end_date = dt.datetime.now()

metrics = ["PrecEsca", "PrecEscaSup", "PrecEscaInf"]

print(f"Fetching metrics from {start_date.date()} to {end_date.date()}...")

for metric in metrics:
    print(f"\n--- {metric} ---")
    try:
        df = api.request_data(metric, "Sistema", start_date.date(), end_date.date())
        if df is not None and not df.empty:
            print("Columns:", df.columns.tolist())
            print(df.tail(3))
            
            # Check values
            val_cols = [c for c in df.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId']]
            if val_cols:
                print("Last values:", df[val_cols].iloc[-1].values)
            else:
                print("No value columns found.")
        else:
            print("DATAFRAME IS EMPTY OR NONE")
    except Exception as e:
        print(f"Error fetching {metric}: {e}")
