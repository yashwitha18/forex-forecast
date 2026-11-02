"""Fake data with the same shape as the real downloads, ONLY for testing the
code offline (`python main.py --synthetic`). Its results mean nothing and
must never appear in the report. Outputs go to separate *_synthetic folders.
"""
import numpy as np
import pandas as pd


def write(cfg, raw_dir, seed: int = 0):
    rng = np.random.default_rng(seed)
    raw_dir.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range("2003-12-01", "2018-11-30")
    start = {"GBP": 0.55, "EUR": 0.80, "JPY": 105.0, "EURINR": 60.0}
    for k in cfg.PAIRS:
        n = len(days)
        p = start.get(k, 1.0) * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
        df = pd.DataFrame({"date": days, "close": p}).sample(frac=0.99, random_state=1).sort_values("date")
        df.to_csv(raw_dir / f"fx_{k}.csv", index=False)
    for k in cfg.INDEXES:
        p = 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, len(days))))
        pd.DataFrame({"date": days, "close": p}).to_csv(raw_dir / f"idx_{k}.csv", index=False)
    months = pd.date_range("2000-01-01", "2018-12-01", freq="MS")
    quarters = pd.date_range("2000-01-01", "2018-10-01", freq="QS")
    cal = pd.date_range("2000-01-01", "2018-12-01", freq="D")
    fred = {
        "CPI_CHG": (months, 170 * np.exp(np.cumsum(rng.normal(0.002, 0.002, len(months))))),
        "FEDFUNDS": (cal, np.clip(2 + np.cumsum(rng.normal(0, 0.01, len(cal))), 0, 6)),
        "RECESSION": (months, ((months >= "2007-12-01") & (months <= "2009-06-01")).astype(int)),
        "DEBT_CHG": (quarters, 6e6 * np.exp(np.cumsum(rng.normal(0.02, 0.005, len(quarters))))),
    }
    for k, (d, v) in fred.items():
        pd.DataFrame({"observation_date": d, "value": v}).to_csv(raw_dir / f"fred_{k}.csv", index=False)
