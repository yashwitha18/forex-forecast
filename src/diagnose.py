"""Check whether a model's test-period directional accuracy also shows up on
the validation period.   Owner: Member B.

    python -m src.diagnose                       # SVM_poly on fx15_macro (default)
    python -m src.diagnose MLP fx15_macro
    python -m src.diagnose MLP fx15_macro --seeds   # repeat with 6 random seeds

A model with real skill should beat 50% on BOTH periods. If it beats 50%
only on the test period, the result was most likely luck in that period.
Run main.py first (needs results/ and data/processed/).
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone

import config as C
from src import features, models


def feature_list(name: str) -> list:
    fx7 = ["ret_0", "ret_1"] + [f"ret_sma_{w}" for w in C.RET_SMA_WINDOWS]
    fx15 = fx7 + ["logp", "logp2", "logp3"] + [f"logp_sma_{w}" for w in C.LOGP_SMA_WINDOWS]
    return {"fx7": fx7, "fx15": fx15, "fx15_macro": fx15 + features.MACRO_COLS}[name]


def seed_check(model: str, fs: str, seeds=(0, 1, 2, 3, 4, 42)):
    """Models with random initialisation (MLP) can look skilled with one seed
    and not another. Real skill should survive a change of seed."""
    by_pair = pd.read_csv(C.RESULTS_DIR / "exp2_return_by_pair.csv")
    cols = feature_list(fs)
    rows = []
    for seed in seeds:
        est, _ = models.regressors(C.RF_PARAMS, seed)[model]
        for k in C.PAIRS:
            t = pd.read_csv(C.PROCESSED_DIR / f"features_{k}.csv", parse_dates=["date"])
            tr, va, te = features.split(t, C.SPLIT)
            params = json.loads(by_pair.query("pair == @k and model == @model and feature_set == @fs")["params"].iloc[0])
            both = pd.concat([tr, va])
            p = clone(est).set_params(**params).fit(both[cols], both["next_ret"]).predict(te[cols])
            rows.append({"seed": seed, "pair": k, "test_dir_acc": np.mean(np.sign(p) == np.sign(te["next_ret"]))})
    out = pd.DataFrame(rows).pivot(index="seed", columns="pair", values="test_dir_acc")
    out["mean"] = out.mean(axis=1)
    out.to_csv(C.RESULTS_DIR / f"seeds_{model}_{fs}.csv")
    print(f"{model} on {fs}: test directional accuracy by random seed")
    print(out.round(3).to_string())


def main(model: str = "SVM_poly", fs: str = "fx15_macro"):
    by_pair = pd.read_csv(C.RESULTS_DIR / "exp2_return_by_pair.csv")
    est, _ = models.regressors(C.RF_PARAMS, C.RANDOM_STATE)[model]
    cols = feature_list(fs)
    rows = []
    for k in C.PAIRS:
        t = pd.read_csv(C.PROCESSED_DIR / f"features_{k}.csv", parse_dates=["date"])
        tr, va, te = features.split(t, C.SPLIT)
        params = json.loads(by_pair.query("pair == @k and model == @model and feature_set == @fs")["params"].iloc[0])

        m_val = clone(est).set_params(**params).fit(tr[cols], tr["next_ret"])
        pv = m_val.predict(va[cols])
        both = pd.concat([tr, va])
        m_test = clone(est).set_params(**params).fit(both[cols], both["next_ret"])
        pt = m_test.predict(te[cols])

        rows.append({
            "pair": k,
            "val_dir_acc": np.mean(np.sign(pv) == np.sign(va["next_ret"])),
            "test_dir_acc": np.mean(np.sign(pt) == np.sign(te["next_ret"])),
            "test_share_pred_up": np.mean(pt > 0),
            "test_share_actual_up": np.mean(te["next_ret"] > 0),
        })
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / f"diagnose_{model}_{fs}.csv", index=False)
    print(f"{model} on {fs}: directional accuracy, validation vs test")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--seeds"]
    if "--seeds" in sys.argv:
        seed_check(*(args or ["MLP", "fx15_macro"]))
    else:
        main(*args)
