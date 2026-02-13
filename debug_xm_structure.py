from pydataxm.pydataxm import ReadDB
import datetime as dt
import pandas as pd

api = ReadDB()
end = dt.datetime.now().date()
start = end - dt.timedelta(days=7)

print(f"Fetching metadata...")

# Get all collections
df_vars = api.get_collections("")

# Debug MaxPrecOferNal
start_date = dt.datetime.now() - dt.timedelta(days=30)
end_date = dt.datetime.now()
df_max = api.request_data("MaxPrecOferNal", "Sistema", start_date.date(), end_date.date())
if df_max is not None:
    print("MaxPrecOferNal columns:", df_max.columns.tolist())
    print(df_max.tail(2))
