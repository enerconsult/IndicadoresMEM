"""
Offline data fetcher for XM metrics.

Runs via GitHub Actions on a daily cron schedule (or manually).
Downloads all summary metrics with retry logic and saves as Parquet files
in the data/ directory.  The Streamlit dashboard reads these files
for near-instant load times.
"""

import asyncio
import json
import datetime as dt
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import pandas as pd
import requests

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
# In GitHub Actions, prefer fewer workers for XM stability.
CONCURRENT_WORKERS = 1 if os.getenv("GITHUB_ACTIONS") == "true" else 2
REQUEST_DELAY = 2  # Seconds between requests to same client
REQUEST_TIMEOUT_SECONDS = 45


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

    async def async_get_df(self, body, endpoint):
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, json=body, headers={"Connection": "close"}) as response:
                if response.status != 200:
                    text = (await response.text())[:200]
                    raise RuntimeError(f"HTTP {response.status} from XM: {text}")
                ctype = response.headers.get("content-type", "").lower()
                if "application/json" not in ctype:
                    text = (await response.text())[:200]
                    raise RuntimeError(f"Unexpected content-type '{ctype}': {text}")
                load = await response.json()
                return pd.json_normalize(load["Items"], endpoint, "Date", sep="_")

    async def run_async(self, list_bodies, endpoint):
        """
        Override pydataxm batch behavior:
        run requests sequentially per metric to avoid burst throttling in XM.
        """
        out = []
        for idx, body in enumerate(list_bodies):
            out.append(await self.async_get_df(body, endpoint))
            if idx < len(list_bodies) - 1:
                await asyncio.sleep(0.2)
        if not out:
            return pd.DataFrame()
        df = pd.concat(out)
        df.reset_index(drop=True, inplace=True)
        return df

    def request_data(self, coleccion, metrica, start_date, end_date, filtros=None):
        """
        Local patch for pydataxm compatibility with pandas/numpy where
        start_periods.values can be read-only.
        """
        if isinstance(filtros, list):
            self.filtros = filtros
        elif filtros is None:
            self.filtros = []
        else:
            self.filtros = []

        if coleccion not in self.inventario_metricas.MetricId.values:
            return pd.DataFrame()
        if metrica not in self.inventario_metricas.Entity.values:
            return pd.DataFrame()

        end_periods = pd.date_range(start_date, end_date, freq="M", inclusive="both")
        if not pd.Timestamp(end_date).is_month_end:
            end_periods = end_periods.append(pd.DatetimeIndex([end_date]))

        start_periods = (end_periods - pd.offsets.MonthBegin(1)).to_list()
        if (not pd.Timestamp(start_date).is_month_start) or (start_date == end_date):
            start_periods[0] = pd.Timestamp(start_date)

        list_periods = list(
            zip(
                [pd.Timestamp(x).date().isoformat() for x in start_periods],
                [pd.Timestamp(x).date().isoformat() for x in end_periods],
            )
        )

        period_dict = {
            "HourlyEntities": {"period_base": "hourly", "endpoint": "HourlyEntities"},
            "DailyEntities": {"period_base": "daily", "endpoint": "DailyEntities"},
            "MonthlyEntities": {"period_base": "monthly", "endpoint": "MonthlyEntities"},
            "AnnualEntities": {"period_base": "annual", "endpoint": "AnnualEntities"},
        }

        entity_type = self.inventario_metricas.query(
            "MetricId == @coleccion and Entity == @metrica"
        ).Type.values[0]

        if entity_type in period_dict:
            period_base = period_dict[entity_type]["period_base"]
            endpoint = period_dict[entity_type]["endpoint"]
            self.url = f"https://servapibi.xm.com.co/{period_base}"

            body_request = {
                "MetricId": coleccion,
                "StartDate": None,
                "EndDate": None,
                "Entity": metrica,
                "Filter": self.filtros,
            }

            list_bodies = []
            for _start, _end in list_periods:
                temp_body = body_request.copy()
                temp_body["StartDate"] = _start
                temp_body["EndDate"] = _end
                list_bodies.append(temp_body)

            try:
                loop = asyncio.get_event_loop()
                data = loop.run_until_complete(self.run_async(list_bodies, endpoint))
            except RuntimeError:
                data = asyncio.run(self.run_async(list_bodies, endpoint))
        elif (
            self.inventario_metricas.query("MetricId == @coleccion and Entity == @metrica")
            .Type.values
            == "ListsEntities"
        ):
            self.url = "https://servapibi.xm.com.co/lists"
            self.request = {"MetricId": coleccion, "Entity": metrica}
            self.connection = requests.post(self.url, json=self.request)
            data_json = json.loads(self.connection.content)
            data = pd.json_normalize(data_json["Items"], "ListEntities", "Date", sep="_")
        else:
            return pd.DataFrame()

        for col in data.columns:
            try:
                converted = pd.to_numeric(data[col], errors="coerce")
                if not converted.isna().all():
                    data[col] = converted
            except Exception:
                pass
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce", format="%Y-%m-%d")

        return data


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
                detail = str(e).strip() or repr(e)
                if attempt < MAX_RETRIES:
                    print(
                        f"  RETRY {metric_id:25s}  "
                        f"(attempt {attempt}/{MAX_RETRIES}) - {detail[:120]}"
                    )
                    time.sleep(RETRY_DELAY_SECONDS * attempt)
                    continue
                print(f"  ERR   {metric_id}: {detail}")
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
