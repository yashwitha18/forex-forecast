"""Profit simulation (paper section 5.2).   Owner: Member B.

Rates are "foreign currency per 1 USD". For pair c on day t:
    predicted ratio  R_hat = exp(predicted log return) ~ rate_{t+1} / rate_t
    actual ratio     R     = rate_{t+1} / rate_t

Rule (as in the paper), start with $100:
  * If every available predicted ratio is >= 1, stay in USD.
  * Otherwise convert all cash to the currency with the LOWEST predicted
    ratio for one day, then convert back. Cash is multiplied by 1 / R.
    Paper's example: ratios GBP 1.0, EUR 1.02, JPY 0.95 -> buy JPY ->
    $100 / 0.95 = $105.26.
  * Only currencies with data that day are considered; a day with none is
    skipped.

With a cost c per conversion, a trade multiplies cash by (1 - c)^2.

Combos: every non-empty subset of the currencies (the paper's Table 3:
GBP, JPY, EUR, GBP+JPY, JPY+EUR, GBP+EUR, GBP+JPY+EUR).
"""
from itertools import combinations

import numpy as np
import pandas as pd


def simulate(frames: dict, combo: tuple, start_cash: float, cost: float) -> pd.DataFrame:
    """frames: currency -> DataFrame[date, pred_ret, next_ret] for the test
    period. Returns the daily cash path."""
    rows = []
    for c in combo:
        f = frames[c][["date", "pred_ret", "next_ret"]].copy()
        f["ccy"] = c
        rows.append(f)
    allf = pd.concat(rows).sort_values(["date", "ccy"])

    cash, path = start_cash, []
    for date, day in allf.groupby("date", sort=True):
        r_hat = np.exp(day["pred_ret"].to_numpy())
        held = "USD"
        if (r_hat < 1).any():
            i = int(np.argmin(r_hat))
            held = day["ccy"].iloc[i]
            cash *= np.exp(-day["next_ret"].iloc[i]) * (1 - cost) ** 2
        path.append({"date": date, "cash": cash, "held": held})
    return pd.DataFrame(path)


def all_combos(ccys: list) -> list:
    return [c for k in range(1, len(ccys) + 1) for c in combinations(ccys, k)]


def run(pred_by_model: dict, ccys: list, start_cash: float, costs: list):
    """pred_by_model: model -> {ccy: frame}. Returns (summary table, paths)."""
    summary, paths = [], {}
    for model, frames in pred_by_model.items():
        for combo in all_combos(ccys):
            for cost in costs:
                p = simulate(frames, combo, start_cash, cost)
                key = (model, "+".join(combo), cost)
                paths[key] = p
                summary.append({
                    "model": model, "combo": "+".join(combo), "cost_per_side": cost,
                    "final_cash": p["cash"].iloc[-1],
                    "peak_cash": p["cash"].max(),
                    "max_drawdown_%": float(((p["cash"] / p["cash"].cummax()) - 1).min() * 100),
                    "days_in_foreign_%": float((p["held"] != "USD").mean() * 100),
                    "first_day": p["date"].min().date(), "last_day": p["date"].max().date(),
                })
    return pd.DataFrame(summary), paths
