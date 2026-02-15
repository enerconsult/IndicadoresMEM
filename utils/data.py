import json
from pathlib import Path

import streamlit as st
import pandas as pd
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydataxm.pydataxm import ReadDB

# ---------------------------------------------------------------------------
# Parquet data directory  (populated by scripts/fetch_data.py via GH Actions)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Thread-safe XM client (shares metric inventory to avoid redundant API calls)
# ---------------------------------------------------------------------------
class _ThreadSafeXMClient(ReadDB):
    """Lightweight ReadDB that reuses the metric inventory from the main instance."""

    def __new__(cls, *args, **kwargs):
        # Bypass ReadDB.__new__ which may call all_variables()
        return super(ReadDB, cls).__new__(cls)

    def __init__(self, inventory, url):
        self.url = url
        self.connection = None
        self.request = ""
        self.inventario_metricas = inventory


# ---------------------------------------------------------------------------
# Singleton API instance (cached across Streamlit reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_api():
    return ReadDB()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalize_df(df, entity):
    """Standard post-processing for any fetched DataFrame."""
    if df is None or df.empty:
        return None
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    if "Entity" not in df.columns:
        df["Entity"] = entity
    return df


# ---------------------------------------------------------------------------
# Parquet layer  (instant load from pre-fetched files)
# ---------------------------------------------------------------------------
def _parquet_is_fresh(max_age_hours=25):
    """Return True if local Parquet data was fetched less than *max_age_hours* ago."""
    meta_path = _DATA_DIR / "_metadata.json"
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        last = dt.datetime.fromisoformat(meta["last_fetch"].replace("Z", "+00:00"))
        age = dt.datetime.now(dt.timezone.utc) - last
        return age.total_seconds() < max_age_hours * 3600
    except Exception:
        return False


def _load_parquet(metric_id):
    """Load a single metric from its Parquet file, or return None."""
    path = _DATA_DIR / f"{metric_id}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception:
        return None


def _load_all_parquet(metrics_tuple, start_date_str, end_date_str):
    """
    Try to satisfy the request entirely from local Parquet files.
    Returns a dict {metric_id: DataFrame} on success, or None if any file is missing.
    """
    if not _parquet_is_fresh():
        return None

    start = pd.Timestamp(start_date_str)
    end = pd.Timestamp(end_date_str) + pd.Timedelta(days=1)  # inclusive end

    results = {}
    for mid, _ent in metrics_tuple:
        df = _load_parquet(mid)
        if df is None:
            return None  # file missing → fall back to API
        if "Date" in df.columns:
            df = df[(df["Date"] >= start) & (df["Date"] < end)]
        results[mid] = df if (df is not None and not df.empty) else None
    return results


# ---------------------------------------------------------------------------
# Parallel fetch  (ThreadPoolExecutor, explicit max_workers)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_metrics_parallel(metrics_tuple, start_date_str, end_date_str):
    """
    Fetch multiple metrics.

    Priority:
      1. Local Parquet files  (< 100 ms)
      2. Parallel API calls   (3-5 s, ThreadPoolExecutor)

    Args:
        metrics_tuple : tuple of (metric_id, entity) tuples  (hashable for cache)
        start_date_str: ISO date string  (hashable for cache)
        end_date_str  : ISO date string

    Returns:
        dict  {metric_id: DataFrame | None}
    """
    # --- 1. Try Parquet (instant) ---
    parquet_results = _load_all_parquet(metrics_tuple, start_date_str, end_date_str)
    if parquet_results is not None:
        return parquet_results

    # --- 2. Fallback: parallel API fetch ---
    start = dt.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date_str, "%Y-%m-%d").date()

    base = get_api()
    inventory = base.inventario_metricas
    url = base.url

    def _fetch_one(metric_id, entity):
        try:
            client = _ThreadSafeXMClient(inventory, url)
            df = client.request_data(metric_id, entity, start, end)
            return metric_id, _normalize_df(df, entity)
        except Exception:
            return metric_id, None

    results = {}
    workers = min(len(metrics_tuple), 12)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one, mid, ent): mid
            for mid, ent in metrics_tuple
        }
        for future in as_completed(futures):
            mid, df = future.result()
            results[mid] = df

    return results


# ---------------------------------------------------------------------------
# Single-metric fetch (for Explorer or ad-hoc queries)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_single_metric(metric_id, entity, start_date_str, end_date_str):
    start = dt.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    api = get_api()
    try:
        df = api.request_data(metric_id, entity, start, end)
        return _normalize_df(df, entity)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Catalog (cached 24 h)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_catalog():
    try:
        return get_api().get_collections()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
_META_COLS = frozenset({"Date", "Id", "Values_code", "Entity", "MetricId", "Name", "To"})


def get_value_col(df):
    """Return the primary numeric value column of *df*."""
    if df is None or df.empty:
        return None
    if "Value" in df.columns:
        return "Value"
    if "DailyValue" in df.columns:
        return "DailyValue"
    candidates = [c for c in df.columns if c not in _META_COLS]
    return candidates[0] if candidates else df.columns[-1]


def _hour_cols(df):
    return [c for c in df.columns if "Hour" in c]


# ---------------------------------------------------------------------------
# Periodicity aggregation
# ---------------------------------------------------------------------------
def calculate_periodicity(df, period, agg_func="sum"):
    """Aggregate *df* to 1D (hourly) / 1M (daily) / 1Y (monthly)."""
    if df is None or df.empty or "Date" not in df.columns:
        return df

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    hcols = _hour_cols(df)
    val_cols = [c for c in df.columns if c not in _META_COLS and c not in hcols]

    # --- 1D: last available day (hourly resolution) ---
    if period == "1D":
        last_date = df["Date"].max().date()
        day = df[df["Date"].dt.date == last_date].copy()
        if hcols:
            melted = day.melt(
                id_vars=["Date"], value_vars=hcols,
                var_name="Hour", value_name="Value",
            )
            melted["HourNum"] = melted["Hour"].str.extract(r"(\d+)").astype(int) - 1
            melted = melted.sort_values("HourNum")
            melted["Date"] = melted.apply(
                lambda r: r["Date"] + pd.Timedelta(hours=r["HourNum"]), axis=1,
            )
            return melted[["Date", "Value"]]
        return day

    # --- 1M: daily aggregation (last 30 days) ---
    if period == "1M":
        cutoff = df["Date"].max() - dt.timedelta(days=30)
        chunk = df[df["Date"] >= cutoff].copy()
        if hcols:
            chunk["DailyValue"] = (
                chunk[hcols].sum(axis=1) if agg_func == "sum"
                else chunk[hcols].mean(axis=1)
            )
            val_cols = ["DailyValue"]
        chunk["_day"] = chunk["Date"].dt.date
        agg = chunk.groupby("_day")[val_cols].agg(agg_func).reset_index()
        agg = agg.rename(columns={"_day": "Date"})
        agg["Date"] = pd.to_datetime(agg["Date"])
        return agg

    # --- 1Y: monthly aggregation (last 365 days) ---
    if period == "1Y":
        cutoff = df["Date"].max() - dt.timedelta(days=365)
        chunk = df[df["Date"] >= cutoff].copy()
        if hcols:
            chunk["DailyValue"] = (
                chunk[hcols].sum(axis=1) if agg_func == "sum"
                else chunk[hcols].mean(axis=1)
            )
            val_cols = ["DailyValue"]
        chunk = chunk.set_index("Date")
        try:
            agg = chunk[val_cols].resample("ME").agg(agg_func).reset_index()
        except ValueError:
            agg = chunk[val_cols].resample("M").agg(agg_func).reset_index()
        if not agg.empty:
            agg["Date"] = agg["Date"].dt.to_period("M").dt.to_timestamp()
        return agg

    return df


# ---------------------------------------------------------------------------
# KPI extractors
# ---------------------------------------------------------------------------
def extract_spot_price(df):
    """Return (value, delta%, date_str, progress) for PrecBolsNaci."""
    val, delta, date_str, progress = 0.0, 0.0, "", 0.0
    if df is None or df.empty:
        return val, delta, date_str, progress

    last = df.iloc[-1]
    date_str = last["Date"].strftime("%Y-%m-%d")
    hcols = _hour_cols(df)

    if hcols:
        val = float(last[hcols].mean())
        if len(df) > 1:
            prev = float(df.iloc[-2][hcols].mean())
            delta = ((val - prev) / prev) * 100 if prev else 0.0
        daily_means = df[hcols].mean(axis=1)
        mx = float(daily_means.max())
    else:
        col = get_value_col(df)
        val = float(last[col]) if col else 0.0
        mx = float(df[col].max()) if col else 0.0

    if mx > 0:
        progress = min(1.0, val / mx)
    return val, delta, date_str, progress


def extract_scarcity(df_base, df_sup):
    """Return (value, delta%, progress) for PrecEsca."""
    val, delta, progress = 0.0, 0.0, 0.0
    if df_base is not None and not df_base.empty:
        col = get_value_col(df_base)
        valid = df_base[df_base[col] > 0.1]
        if not valid.empty:
            val = float(valid.iloc[-1][col])
    if df_sup is not None and not df_sup.empty:
        col_s = get_value_col(df_sup)
        try:
            sup = float(df_sup.iloc[-1][col_s])
            if sup > 0:
                progress = min(1.0, val / sup)
        except Exception:
            pass
    return val, delta, progress


def extract_demand(df):
    """Return (value, delta%, date_str, progress) for DemaCome."""
    val, delta, date_str, progress = 0.0, 0.0, "", 0.0
    if df is None or df.empty:
        return val, delta, date_str, progress

    hcols = _hour_cols(df)
    if not hcols:
        return val, delta, date_str, progress

    daily_sums = df[hcols].sum(axis=1)
    target_idx = -1

    for i in range(1, min(6, len(df))):
        idx = -i
        current = float(daily_sums.iloc[idx])
        lo = max(0, len(df) + idx - 15)
        hi = len(df) + idx
        history = daily_sums.iloc[lo:hi]
        valid = history[history > 10]
        avg = float(valid.tail(7).mean()) if not valid.empty else current

        if current > 100 and (avg == 0 or current > 0.8 * avg):
            target_idx = idx
            break

    val = float(daily_sums.iloc[target_idx])
    date_str = df.iloc[target_idx]["Date"].strftime("%Y-%m-%d")

    if len(df) > abs(target_idx):
        prev = float(daily_sums.iloc[target_idx - 1])
        delta = ((val - prev) / prev) * 100 if prev else 0.0

    mx = float(daily_sums.max())
    if mx > 0:
        progress = min(1.0, val / mx)
    return val, delta, date_str, progress


def extract_offer(df):
    """Return (max, min, avg, date_str, progress) for MaxPrecOferNal."""
    mx, mn, avg, date_str, progress = 0.0, 0.0, 0.0, "", 0.0
    if df is None or df.empty:
        return mx, mn, avg, date_str, progress

    last = df.iloc[-1]
    date_str = last["Date"].strftime("%Y-%m-%d")
    hcols = _hour_cols(df)
    if hcols:
        vals = last[hcols]
        mx, mn, avg = float(vals.max()), float(vals.min()), float(vals.mean())
        if mx > 0:
            progress = min(1.0, avg / mx)
    return mx, mn, avg, date_str, progress


# ---------------------------------------------------------------------------
# Data tail trimmer (remove partial-data trailing days)
# ---------------------------------------------------------------------------
def trim_partial_tail(df, val_col):
    """Drop trailing rows whose value is suspiciously low (partial data)."""
    if df is None or df.empty or len(df) < 3:
        return df
    df = df.copy()

    if len(df) >= 8:
        cut = 0
        for i in range(1, min(6, len(df))):
            idx = -i
            current = float(df.iloc[idx][val_col])
            lo = max(0, len(df) + idx - 15)
            hi = len(df) + idx
            history = df.iloc[lo:hi][val_col]
            valid = history[history > 10]
            avg = float(valid.tail(7).mean()) if not valid.empty else current

            if avg > 0 and current < 0.8 * avg:
                cut = idx - 1
            else:
                if cut != 0:
                    cut = idx
                break
        if cut < 0:
            df = df.iloc[: cut + 1]
    else:
        last = float(df.iloc[-1][val_col])
        prev = float(df.iloc[-2][val_col])
        if prev > 0 and last < 0.6 * prev:
            df = df.iloc[:-1]
    return df
