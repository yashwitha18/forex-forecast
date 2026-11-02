"""Features, targets and the chronological 80/10/10 split.
Owner: Member A writes it, Member B reviews it for leakage.

Each row is one trading day t of one currency pair. Every feature uses only
information available at the close of day t. Targets describe day t+1, the
next available trading day for that pair.

Feature sets (paper section 3.2):
  fx7        ret_0 (today's log return), ret_1 (yesterday's),
             ret_sma_{5,10,20,60,120}
  fx15       fx7 + logp, logp^2, logp^3, logp_sma_{5,10,20,60,220}
  fx15_macro fx15 + 4 stock-index returns + CPI change, Fed funds,
             recession flag, federal-debt change

Targets:
  Experiment 1 (sign):   up = 1 if next-day log return > 0 else 0
  Experiment 2 (return): next_ret = log(rate_{t+1} / rate_t)
"""
import numpy as np
import pandas as pd

MACRO_COLS = ["SP500", "SP100", "NASDAQ", "DJIA", "CPI_CHG", "FEDFUNDS", "RECESSION", "DEBT_CHG"]


def fx_features(df: pd.DataFrame, ret_windows, logp_windows) -> tuple[pd.DataFrame, dict]:
    out = df[["date", "close"]].copy().reset_index(drop=True)
    logp = np.log(out["close"])
    r = logp.diff()

    out["ret_0"] = r
    out["ret_1"] = r.shift(1)
    for w in ret_windows:
        out[f"ret_sma_{w}"] = r.rolling(w).mean()          # window ends at t
    fx7 = ["ret_0", "ret_1"] + [f"ret_sma_{w}" for w in ret_windows]

    out["logp"] = logp
    out["logp2"] = logp ** 2
    out["logp3"] = logp ** 3
    for w in logp_windows:
        out[f"logp_sma_{w}"] = logp.rolling(w).mean()
    fx15 = fx7 + ["logp", "logp2", "logp3"] + [f"logp_sma_{w}" for w in logp_windows]

    out["next_date"] = out["date"].shift(-1)
    out["next_close"] = out["close"].shift(-1)
    out["next_ret"] = r.shift(-1)
    out["up"] = (out["next_ret"] > 0).astype(int)
    return out, {"fx7": fx7, "fx15": fx15}


def macro_daily(idx: dict, fred: dict, cfg) -> list:
    """List of small tables (available_from, factor value). A value is attached to
    an FX day only once it would actually have been public on that day."""
    parts = []
    for name, df in idx.items():
        s = df[["date", "close"]].copy()
        s[name] = np.log(s["close"]).diff()
        s["avail"] = s["date"] + pd.Timedelta(days=cfg.INDEX_LAG_DAYS)
        parts.append(s[["avail", name]].dropna())

    for name, (_, transform, lag) in cfg.FRED_SERIES.items():
        s = fred[name].dropna(subset=["value"]).sort_values("date").copy()
        s[name] = s["value"].pct_change() * 100 if transform == "pct_change" else s["value"]
        s["avail"] = s["date"] + pd.Timedelta(days=lag)
        parts.append(s[["avail", name]].dropna())
    return parts


def attach_macro(table: pd.DataFrame, parts: list) -> pd.DataFrame:
    out = table.sort_values("date")
    for p in parts:
        out = pd.merge_asof(out, p.sort_values("avail"), left_on="date",
                            right_on="avail", direction="backward").drop(columns="avail")
    return out


def build(fx_df: pd.DataFrame, macro_parts: list | None, cfg) -> tuple[pd.DataFrame, dict]:
    t, sets = fx_features(fx_df, cfg.RET_SMA_WINDOWS, cfg.LOGP_SMA_WINDOWS)
    if macro_parts is not None:
        t = attach_macro(t, macro_parts)
        sets["fx15_macro"] = sets["fx15"] + MACRO_COLS
    needed = sorted({c for v in sets.values() for c in v}) + ["next_ret", "next_close"]
    t = t.dropna(subset=needed).reset_index(drop=True)
    return t, sets


def split(df: pd.DataFrame, fracs=(0.8, 0.1, 0.1)):
    """Chronological train / validation / test. Never shuffled."""
    n = len(df)
    a = int(n * fracs[0])
    b = int(n * (fracs[0] + fracs[1]))
    tr, va, te = df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
    assert tr["date"].max() < va["date"].min() < te["date"].min()
    return tr, va, te
