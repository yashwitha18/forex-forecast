"""Sanity tests on synthetic data. Run: python -m pytest -q

They guard the things that silently ruin forecasting results: future data
leaking into features, macro data used before it was published, a shuffled
split, and wrong metric / backtest arithmetic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src import backtest, evaluate, features, preprocess  # noqa: E402


def fx(n=800, seed=0, start="2010-01-01"):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"date": d, "close": 1.1 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))})


def test_fifteen_paper_features():
    t, sets = features.fx_features(fx(), C.RET_SMA_WINDOWS, C.LOGP_SMA_WINDOWS)
    assert len(sets["fx7"]) == 7 and len(sets["fx15"]) == 15


def test_targets_are_next_day():
    df = fx()
    t, _ = features.fx_features(df, C.RET_SMA_WINDOWS, C.LOGP_SMA_WINDOWS)
    i = 300
    assert np.isclose(t.loc[i, "next_ret"], np.log(df.loc[i + 1, "close"] / df.loc[i, "close"]))
    assert t.loc[i, "up"] == int(df.loc[i + 1, "close"] > df.loc[i, "close"])


def test_no_future_leakage_in_fx_features():
    df = fx()
    k = 500
    alt = df.copy()
    alt.loc[k + 1:, "close"] *= 1.3
    a, s = features.fx_features(df, C.RET_SMA_WINDOWS, C.LOGP_SMA_WINDOWS)
    b, _ = features.fx_features(alt, C.RET_SMA_WINDOWS, C.LOGP_SMA_WINDOWS)
    cols = s["fx15"]
    assert np.allclose(a.loc[:k, cols].fillna(0), b.loc[:k, cols].fillna(0))


def test_macro_respects_publication_lag():
    """A monthly value dated 1 March with a 45-day lag must not appear on
    an FX row before 15 April."""
    table = pd.DataFrame({"date": pd.bdate_range("2015-01-01", "2015-06-30")})
    monthly = pd.DataFrame({"date": pd.date_range("2014-12-01", "2015-06-01", freq="MS")})
    monthly["X"] = np.arange(len(monthly), dtype=float)
    monthly["avail"] = monthly["date"] + pd.Timedelta(days=45)
    out = features.attach_macro(table, [monthly[["avail", "X"]]])
    march_value = monthly.loc[monthly.date == "2015-03-01", "X"].item()
    first_seen = out.loc[out["X"] == march_value, "date"].min()
    assert first_seen >= pd.Timestamp("2015-04-15")


def test_split_80_10_10_chronological():
    t, _ = features.fx_features(fx(), C.RET_SMA_WINDOWS, C.LOGP_SMA_WINDOWS)
    t = t.dropna().reset_index(drop=True)
    tr, va, te = features.split(t, (0.8, 0.1, 0.1))
    assert tr.date.max() < va.date.min() and va.date.max() < te.date.min()
    assert len(tr) + len(va) + len(te) == len(t)
    assert abs(len(tr) / len(t) - 0.8) < 0.01


def test_clean_handles_dupes_and_nans():
    df = fx(60)
    dirty = pd.concat([df, df.iloc[[3]]]).sample(frac=1, random_state=1)
    dirty.iloc[0, 1] = np.nan
    out, rep = preprocess.clean_prices(dirty)
    assert out.date.is_monotonic_increasing and not out.date.duplicated().any()
    assert rep["missing_close"] == 1


def test_sign_metrics_match_paper_relationships():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    p = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 1])
    m = evaluate.sign_metrics(y, p)
    assert np.isclose(m["Acc"], 0.6)
    assert np.isclose(m["MAE"], 2 * (1 - m["Acc"]))       # holds for +/-1 labels
    assert np.isclose(m["DS"], m["Acc"])
    assert np.isclose(m["CU"], 4 / 6) and np.isclose(m["CD"], 2 / 4)


def test_backtest_matches_paper_example():
    """Paper: ratios GBP 1.0, EUR 1.02, JPY 0.95 -> buy JPY -> $105.26."""
    d = pd.Timestamp("2018-01-02")
    mk = lambda r: pd.DataFrame({"date": [d], "pred_ret": [np.log(r)], "next_ret": [np.log(r)]})
    frames = {"GBP": mk(1.0), "EUR": mk(1.02), "JPY": mk(0.95)}
    out = backtest.simulate(frames, ("GBP", "EUR", "JPY"), 100.0, 0.0)
    assert out["held"].item() == "JPY"
    assert np.isclose(out["cash"].item(), 100 / 0.95)
    # all predicted ratios >= 1 -> stay in USD
    frames = {"GBP": mk(1.01), "EUR": mk(1.02)}
    assert backtest.simulate(frames, ("GBP", "EUR"), 100.0, 0.0)["cash"].item() == 100.0


def test_bad_print_removed_but_level_shift_kept():
    c = pd.Series([1.00, 1.001, 1.10, 1.002, 1.003,    # spike that snaps back -> bad
                   1.004, 1.10, 1.101, 1.102])         # jump that stays -> real move
    bad = preprocess.find_bad_prints(c, 0.03)
    assert bad.tolist() == [False, False, True, False, False, False, False, False, False]
