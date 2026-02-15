"""
Offline data fetcher for XM metrics.

Runs via GitHub Actions on a daily cron schedule (or manually).
Downloads all summary metrics with retry logic and saves as Parquet files
in the data/ directory.  The Streamlit dashboard reads these files
for near-instant load times.
"""

import json
import datetime as dt
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Fix: pandas 2.2+ deprecated 'M' frequency, pydataxm still uses it
# Monkey-patch to convert 'M' to 'ME' in date_range and period_range
_orig_date_range = pd.date_range
_orig_period_range = pd.period_range

def _patched_date_range(*args, **kwargs):
    if 'freq' in kwargs and kwargs['freq'] == 'M':
        kwargs['freq'] = 'ME'
    return _orig_date_range(*args, **kwargs)

def _patched_period_range(*args, **kwargs):
    if 'freq' in kwargs and kwargs['freq'] == 'M':
        kwargs['freq'] = 'ME'
    return _orig_period_range(*args, **kwargs)

pd.date_range = _patched_date_range
pd.period_range = _patched_period_range

from pydataxm.pydataxm import ReadDB

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

METRICS = [
    ("PrecBolsNaci", "Sistema"),
    ("PrecEsca", "Sistema"),
    ("PrecEscaSup", "Sistema"),
    ("PrecEscaInf", "Sistema"),
    ("DemaCome", "Sistema"),
    ("MaxPrecOferNal", "Sistema"),
    ("Gene", "Sistema"),
    ("CapaUtilDiarEner", "Sistema"),
    ("VoluUtilDiarEner", "Sistema"),
    ("AporEner", "Sistema"),
    ("AporEnerMediHist", "Sistema"),
]

# 400 days covers the 1Y periodicity view (365 d) plus margin
LOOKBACK_DAYS = 400

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
CONCURRENT_WORKERS = 2  # Conservative to avoid overwhelming XM server
REQUEST_DELAY = 2  # Seconds between requests to same client


# ---------------------------------------------------------------------------
# Thread-safe XM client  (shares the inventory to skip redundant HTTP call)
# ---------------------------------------------------------------------------
class _Client(ReadDB):
    """Lightweight per-thread ReadDB clone."""

    def __new__(cls, *args, **kwargs):
        return super(ReadDB, cls).__new__(cls)

    def __init__(self, inventory, url):
        self.url = url
        self.connection = None
        self.request = ""
        self.inventario_metricas = inventory


def _safe_copy(df):
    """
    SOLUCIÓN DEFINITIVA: Crear una copia completamente independiente del DataFrame.
    Usa to_dict() que garantiza copia profunda de todos los datos.
    """
    if df is None or df.empty:
        return df
    
    # Método 1: to_dict garantiza copia completa
    try:
        return pd.DataFrame(df.to_dict('records'))
    except Exception:
        pass
    
    # Método 2: astype fuerza reconstrucción
    try:
        return df.astype(df.dtypes.to_dict())
    except Exception:
        pass
    
    # Método 3: pickle/unpickle (copia binaria completa)
    import pickle
    try:
        return pickle.loads(pickle.dumps(df))
    except Exception:
        pass
    
    # Método 4: json round-trip (más lento pero seguro)
    try:
        return pd.read_json(df.to_json(orient='records'), orient='records')
    except Exception:
        pass
    
    return df


# ---------------------------------------------------------------------------
# Fetch logic with retries
# ---------------------------------------------------------------------------
def fetch_all():
    DATA_DIR.mkdir(exist_ok=True)

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=LOOKBACK_DAYS)

    print(f"Fetching XM data  {start_date} -> {end_date}")
    print(f"Metrics: {len(METRICS)}")
    print(f"Config: {MAX_RETRIES} retries, {CONCURRENT_WORKERS} workers, {REQUEST_DELAY}s delay")
    print()

    # Build one base API to share the metric inventory across threads
    base_api = ReadDB()
    inventory = base_api.inventario_metricas
    url = base_api.url

    def _fetch_one(metric_id, entity):
        """Fetch a single metric with retry logic."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = _Client(inventory, url)
                df = client.request_data(metric_id, entity, start_date, end_date)
                
                if df is not None and not df.empty:
                    # SOLUCIÓN DEFINITIVA: Copia segura del DataFrame
                    df = _safe_copy(df)
                    
                    # Ahora podemos modificar sin problemas
                    if "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"])
                    if "Entity" not in df.columns:
                        df["Entity"] = entity
                    
                    path = DATA_DIR / f"{metric_id}.parquet"
                    df.to_parquet(path, index=False)
                    
                    if attempt > 1:
                        print(f"  OK    {metric_id:25s}  {len(df):>5} rows  (retry {attempt-1})")
                    else:
                        print(f"  OK    {metric_id:25s}  {len(df):>5} rows")
                    return metric_id, True
                
                print(f"  EMPTY {metric_id}")
                return metric_id, False
                
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "Cannot connect" in error_msg:
                    if attempt < MAX_RETRIES:
                        print(f"  RETRY {metric_id:25s}  (attempt {attempt}/{MAX_RETRIES}) - timeout")
                        time.sleep(RETRY_DELAY_SECONDS * attempt)
                        continue
                
                print(f"  ERR   {metric_id}: {e}")
                return metric_id, False
        
        return metric_id, False

    results = {}
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, mid, ent): mid
            for mid, ent in METRICS
        }
        for future in as_completed(futures):
            mid, ok = future.result()
            results[mid] = ok
            time.sleep(0.5)

    # Write metadata so the dashboard can check freshness
    meta = {
        "last_fetch": dt.datetime.utcnow().isoformat() + "Z",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "metrics": results,
    }
    meta_path = DATA_DIR / "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    ok_count = sum(v for v in results.values())
    print(f"\nDone: {ok_count}/{len(METRICS)} metrics saved to {DATA_DIR}")
    
    if ok_count < len(METRICS):
        failed = [m for m, ok in results.items() if not ok]
        print(f"Failed metrics: {', '.join(failed)}")
    
    return results


if __name__ == "__main__":
    fetch_all()
