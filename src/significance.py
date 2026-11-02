"""Is a model's directional accuracy better than a coin flip, or just luck?
Owner: Member B.   Run alone with:  python -m src.significance

For each model, the test-set days of all pairs are pooled. k = number of
days the direction was right, n = number of test days. Under "no skill",
k ~ Binomial(n, 0.5). The one-sided p-value is P(K >= k).

Because many models are tried, the smallest p-value is expected to look
good by chance. We therefore also report the Bonferroni-adjusted p-value
(p x number of models tried, capped at 1). A result counts as evidence of
skill only if the adjusted p < 0.05 AND the model would have been chosen
on the validation set (choosing the best model by test score is cheating).

Limitation: the binomial test assumes days are independent. Exp 2 uses
this test on directional accuracy only; a Diebold-Mariano test on squared
errors would be the stronger check (future work).
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

import config as C
from src import features


def test_sizes(processed_dir, pairs) -> dict:
    out = {}
    for k in pairs:
        t = pd.read_csv(processed_dir / f"features_{k}.csv", parse_dates=["date"])
        out[k] = len(features.split(t, C.SPLIT)[2])
    return out


def pooled(by_pair: pd.DataFrame, acc_col: str, val_col: str, n: dict) -> pd.DataFrame:
    df = by_pair[by_pair["feature_set"] != "baseline"].copy()
    df["n"] = df["pair"].map(n)
    df["k"] = (df[acc_col] * df["n"]).round().astype(int)
    g = df.groupby(["feature_set", "model"], sort=False).agg(
        k=("k", "sum"), n=("n", "sum"), val=(val_col, "mean")).reset_index()
    g["test_acc"] = g["k"] / g["n"]
    g["p_value"] = [binomtest(int(k), int(m), 0.5, alternative="greater").pvalue
                    for k, m in zip(g["k"], g["n"])]
    g["p_bonferroni"] = np.minimum(g["p_value"] * len(g), 1.0)
    best_val = g.groupby("feature_set")["val"].transform("max")
    g["chosen_by_validation"] = g["val"] == best_val
    return g.rename(columns={"val": f"mean_{val_col}"}).sort_values("test_acc", ascending=False)


def run(results_dir=None, processed_dir=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = results_dir or C.RESULTS_DIR
    processed_dir = processed_dir or C.PROCESSED_DIR
    n = test_sizes(processed_dir, C.PAIRS)
    e1 = pooled(pd.read_csv(results_dir / "exp1_sign_by_pair.csv"), "Acc", "val_Acc", n)
    e2 = pooled(pd.read_csv(results_dir / "exp2_return_by_pair.csv"), "DirAcc", "val_R2", n)
    e1.to_csv(results_dir / "significance_exp1.csv", index=False)
    e2.to_csv(results_dir / "significance_exp2_direction.csv", index=False)
    return e1, e2


if __name__ == "__main__":
    a, b = run()
    cols = ["feature_set", "model", "test_acc", "p_value", "p_bonferroni", "chosen_by_validation"]
    print("Exp 1 (sign) - top 5 by test accuracy:")
    print(a[cols].head(5).round(4).to_string(index=False))
    print("\nExp 2 (direction of predicted return) - top 5:")
    print(b[cols].head(5).round(4).to_string(index=False))
