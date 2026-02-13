from pydataxm.pydataxm import ReadDB
import pandas as pd

try:
    df = ReadDB().get_collections()
    with open("columns.txt", "w") as f:
        f.write(str(df.columns.tolist()))
    print("Columns written to columns.txt")
except Exception as e:
    print(e)
