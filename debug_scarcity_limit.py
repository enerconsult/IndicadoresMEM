from pydataxm.pydataxm import ReadDB
import pandas as pd
import datetime as dt

api = ReadDB()

def test_query(name, start, end):
    days = (end - start).days + 1
    print(f"TEST: {name} ({days} days)")
    try:
        df = api.request_data("PrecEsca", "Sistema", start, end)
        if df is None:
            print(f"RESULT: None")
        elif df.empty:
            print(f"RESULT: Empty DataFrame")
        else:
            print(f"COLUMNS: {list(df.columns)}")
            
            # Simulate get_chart_val_col behavior
            v = [c for c in df.columns if c not in ['Date', 'Id', 'Values_code', 'Entity', 'MetricId', 'Name', 'To']]
            selected_col = v[0] if v else df.columns[-1]
            print(f"SELECTED COL: {selected_col}")
            
            max_val = df[selected_col].max()
            print(f"RESULT: Success. Rows={len(df)}. MaxVal={max_val} in {selected_col}")
    except Exception as e:
        print(f"RESULT: Error - {e}")
    print("-" * 20)

# 1. 366 Days (Fail?)
test_query("366 Days", dt.date(2025, 2, 13), dt.date(2026, 2, 13))
