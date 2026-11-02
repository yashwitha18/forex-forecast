"""Download and cache all raw data.   Owner: Member A.

Sources
  * FX rates and stock indexes: Yahoo Finance via yfinance
  * CPI, Fed funds, recession flag, federal debt: FRED CSV endpoint

Everything is cached under data/raw/. Re-runs use the cache, so results are
reproducible. Commit data/raw/ once you have the final download and write
the download date in the report (Yahoo occasionally revises history).
"""
from io import StringIO
from pathlib import Path

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _yahoo_close(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval="1d",
                     auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(
            f"Yahoo returned no rows for {ticker}. Check your connection, or "
            "export the CSV by hand from finance.yahoo.com/quote/<ticker>/history "
            f"and save it as data/raw/<name>.csv with columns date,close."
        )
    if isinstance(df.columns, pd.MultiIndex):        # yfinance >= 0.2.40
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"index": "date", "datetime": "date"})
    return df[["date", "close"]]


def _fred(series_id: str) -> pd.DataFrame:
    import urllib.request

    with urllib.request.urlopen(FRED_URL.format(sid=series_id), timeout=60) as r:
        text = r.read().decode()
    df = pd.read_csv(StringIO(text))
    df.columns = ["date", "value"]                   # DATE/observation_date, <id>
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." = missing
    return df


def download_all(cfg, force: bool = False) -> None:
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(f"fx_{k}", lambda t=t: _yahoo_close(t, cfg.START_DATE, cfg.END_DATE))
            for k, t in cfg.PAIRS.items()]
    jobs += [(f"idx_{k}", lambda t=t: _yahoo_close(t, cfg.START_DATE, cfg.END_DATE))
             for k, t in cfg.INDEXES.items()]
    jobs += [(f"fred_{k}", lambda s=s: _fred(s)) for k, (s, _, _) in cfg.FRED_SERIES.items()]

    for name, fn in jobs:
        path = cfg.RAW_DIR / f"{name}.csv"
        if path.exists() and not force:
            continue
        df = fn()
        df.to_csv(path, index=False)
        print(f"[data] {name}: {len(df)} rows -> {path.name}")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("*", "") for c in df.columns]
    if "adj_close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj_close": "close"})
    df = df.rename(columns={"observation_date": "date"})   # FRED's own CSV header
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    return df


def load_all(cfg) -> tuple[dict, dict, dict]:
    """Returns (fx, indexes, fred): dicts of name -> DataFrame."""
    def need(name):
        p = cfg.RAW_DIR / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run `python main.py --download` first.")
        return load(p)

    fx = {k: need(f"fx_{k}") for k in cfg.PAIRS}
    idx = {k: need(f"idx_{k}") for k in cfg.INDEXES}
    fred = {k: need(f"fred_{k}") for k in cfg.FRED_SERIES}
    return fx, idx, fred
