"""Clean each price series and report what was found.   Owner: Member A.

Decisions (likely viva questions):
  * Only the daily close is used, as in the paper ("daily closed price").
  * Weekends/holidays are NOT filled. "Next day" = next available trading
    day for that pair, which is how the paper's backtest treats missing days.
  * Missing closes are dropped, never forward-filled: a filled price is a
    fake 0% return day.
  * Isolated bad prints are removed (dropped, not replaced). A price is a
    bad print if it sits more than BAD_PRINT_THRESHOLD (3%) away from BOTH
    neighbours in the same direction while the neighbours agree with each
    other: a one-day spike that snaps back. On the real Yahoo data this
    removes 5 EUR and 4 JPY rows, all on the 8th of a month in 2008 (a known
    Yahoo FX data error); genuine moves such as Brexit (GBP, June 2016) are
    level shifts that do not snap back, so they are kept.
  * Other large one-day moves are flagged in data_report.json, not deleted.
"""
import numpy as np
import pandas as pd


def find_bad_prints(close: pd.Series, threshold: float) -> pd.Series:
    """True where the price is a one-day spike-and-revert larger than threshold."""
    lp = np.log(close)
    prev, nxt = lp.shift(1), lp.shift(-1)
    dev = lp - (prev + nxt) / 2                  # distance from the neighbours' midpoint
    same_side = np.sign(lp - prev) == np.sign(lp - nxt)
    neighbours_agree = (nxt - prev).abs() < dev.abs()
    return (dev.abs() > threshold) & same_side & neighbours_agree


def clean_prices(df: pd.DataFrame, spike_threshold: float = 0.05,
                 bad_print_threshold: float | None = 0.03) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_raw": int(len(df))}
    out = df[["date", "close"]].copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    rep["missing_close"] = int(out["close"].isna().sum())
    rep["non_positive"] = int((out["close"] <= 0).sum())
    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0].sort_values("date")
    rep["duplicate_dates"] = int(out["date"].duplicated().sum())
    out = out.drop_duplicates("date", keep="last").reset_index(drop=True)

    if bad_print_threshold is not None:
        bad = find_bad_prints(out["close"], bad_print_threshold)
        rep["removed_bad_prints"] = [{"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 6)}
                                     for d, c in zip(out.loc[bad, "date"], out.loc[bad, "close"])]
        out = out[~bad].reset_index(drop=True)

    gaps = out["date"].diff().dt.days
    rep["gaps_over_5_days"] = [d.strftime("%Y-%m-%d") for d in out.loc[gaps > 5, "date"]]

    r = np.log(out["close"]).diff()
    sp = out.loc[r.abs() > spike_threshold, ["date", "close"]]
    rep["flagged_jumps"] = [{"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 6)}
                            for d, c in zip(sp["date"], sp["close"])]
    rep["rows_clean"] = int(len(out))
    rep["first_date"] = out["date"].min().strftime("%Y-%m-%d")
    rep["last_date"] = out["date"].max().strftime("%Y-%m-%d")
    return out, rep


def clean_all(fx: dict, idx: dict, cfg) -> tuple[dict, dict, dict]:
    report = {}
    fx_c, idx_c = {}, {}
    for k, df in fx.items():
        fx_c[k], report[f"fx_{k}"] = clean_prices(df, bad_print_threshold=cfg.BAD_PRINT_THRESHOLD)
    for k, df in idx.items():
        # No bad-print filter on stock indexes: they are far more volatile than
        # FX, so a 3% spike-and-revert there is usually a real market day
        # (e.g. Lehman, Sept 2008), and the Yahoo error affects only FX data.
        idx_c[k], report[f"idx_{k}"] = clean_prices(df, spike_threshold=0.08,
                                                    bad_print_threshold=None)

    if cfg.ALIGN_COMMON_START:
        start = max(d["date"].min() for d in fx_c.values())
        fx_c = {k: d[d["date"] >= start].reset_index(drop=True) for k, d in fx_c.items()}
        report["common_start"] = start.strftime("%Y-%m-%d")
    return fx_c, idx_c, report


def summary_stats(fx: dict) -> pd.DataFrame:
    rows = {}
    for k, df in fx.items():
        r = np.log(df["close"]).diff().dropna()
        rows[k] = {
            "days": len(df), "first": df["date"].min().date(), "last": df["date"].max().date(),
            "close_min": df["close"].min(), "close_max": df["close"].max(),
            "ret_mean_%": r.mean() * 100, "ret_std_%": r.std() * 100,
            "ret_skew": r.skew(), "ret_kurtosis": r.kurt(),
            "up_days_%": (r > 0).mean() * 100,
        }
    return pd.DataFrame(rows).T
