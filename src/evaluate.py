"""Metrics.   Owner: Member B.

Experiment 1 (sign) - the paper's Table 1 columns: Acc, NMSE, MAE, DS, CU, CD.
Labels and predictions are mapped to +1 (up) / -1 (down) first; that is the
only reading under which the paper's numbers are consistent
(e.g. Acc 0.52 -> MAE = 2 * (1 - 0.52) = 0.96, their Lasso row shows 0.95).

  Acc  = share of days with the correct direction
  NMSE = mean((s - s_hat)^2) / var(s)
  MAE  = mean(|s - s_hat|)
  DS   = directional symmetry = share of days where predicted and actual
         direction agree. For a sign classifier this equals Acc.
  CU   = correct up-trend   = of the days that actually went up, the share
         predicted up
  CD   = correct down-trend = of the days that actually went down, the
         share predicted down

Experiment 2 (log return) - the paper uses R^2 (Tables 2 and 4). We add
MAE/RMSE (in basis points, 1 bp = 0.01%), directional accuracy, and
R^2 vs random walk, which is the out-of-sample R^2 used in return-
forecasting research (Campbell & Thompson, 2008):
  R2_vs_RW = 1 - SSE(model) / SSE(predict 0)
  > 0 means the model beats "tomorrow = today".
"""
import numpy as np
from sklearn.metrics import r2_score


def sign_metrics(y_up, p_up) -> dict:
    s = 2 * np.asarray(y_up) - 1
    sh = 2 * np.asarray(p_up) - 1
    up, down = s == 1, s == -1
    return {
        "Acc": float(np.mean(s == sh)),
        "NMSE": float(np.mean((s - sh) ** 2) / np.var(s)),
        "MAE": float(np.mean(np.abs(s - sh))),
        "DS": float(np.mean(s * sh > 0)),
        "CU": float(np.mean(sh[up] == 1)) if up.any() else np.nan,
        "CD": float(np.mean(sh[down] == -1)) if down.any() else np.nan,
    }


def return_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    nz = (np.sign(p) != 0) & (np.sign(y) != 0)
    return {
        "R2": float(r2_score(y, p)),
        "R2_vs_RW": float(1 - np.sum((y - p) ** 2) / np.sum(y ** 2)),
        "MAE_bp": float(np.mean(np.abs(y - p)) * 1e4),
        "RMSE_bp": float(np.sqrt(np.mean((y - p) ** 2)) * 1e4),
        "DirAcc": float(np.mean(np.sign(y[nz]) == np.sign(p[nz]))) if nz.any() else np.nan,
    }


def r2_score_fn(y, p) -> float:
    return float(r2_score(y, p))


def acc_score_fn(y, p) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(p)))
