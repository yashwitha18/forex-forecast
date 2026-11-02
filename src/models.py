"""Models for both experiments.   Owner: Member B.

Paper section 4: Random Forest, SVM, Multi-layer Perceptron, Lasso, Ridge,
and (for sign prediction) Logistic Regression. Models are trained per
currency pair. The validation set picks hyperparameters ("we use the
validation set to select our models"); the test set is touched once.

Additions to the paper (not replacements):
  * Baselines. Exp 1: majority class and "same sign as today".
    Exp 2: random walk (predict 0 return) and historical mean.
    The paper cites Meese & Rogoff (random walk is hard to beat) but never
    reports a baseline, so you cannot tell whether a model is any good.
  * Plain Linear Regression in Exp 2 (from the original proposal).
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

_NEW_LOGIT = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def _logit_plain():
    return LogisticRegression(C=np.inf, max_iter=2000) if _NEW_LOGIT else \
        LogisticRegression(penalty=None, max_iter=2000)


def _logit_l1():
    kw = dict(l1_ratio=1.0) if _NEW_LOGIT else dict(penalty="l1")
    return LogisticRegression(solver="saga", max_iter=5000, **kw)


def _scaled(est):
    return Pipeline([("scale", StandardScaler()), ("model", est)])


def _scaled_y(est):
    # Daily log returns are ~0.005 in size. SVR/MLP/Lasso defaults assume a
    # target of order 1, so we standardise y too and undo it on predict.
    return TransformedTargetRegressor(regressor=_scaled(est), transformer=StandardScaler())


def classifiers(rf_params: dict, seed: int) -> dict:
    """name -> (estimator, hyperparameter grid). Grid keys follow sklearn's
    nested naming (model__C = the C of the 'model' step)."""
    return {
        "Logistic": (_scaled(_logit_plain()), {}),
        "Lasso": (_scaled(_logit_l1()), {"model__C": [0.01, 0.1, 1.0]}),
        "Ridge": (_scaled(RidgeClassifier()), {"model__alpha": [0.1, 1, 10, 100]}),
        "SVM_rbf": (_scaled(SVC(kernel="rbf")), {"model__C": [0.1, 1, 10]}),
        "RandomForest": (RandomForestClassifier(**rf_params), {}),
        "MLP": (_scaled(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                      early_stopping=True, random_state=seed)),
                {"model__alpha": [1e-4, 1e-2]}),
    }


def regressors(rf_params: dict, seed: int) -> dict:
    return {
        "LinearRegression": (_scaled_y(LinearRegression()), {}),
        "Lasso": (_scaled_y(Lasso(max_iter=10000)), {"regressor__model__alpha": [0.001, 0.01, 0.1]}),
        "Ridge": (_scaled_y(Ridge()), {"regressor__model__alpha": [0.1, 1, 10, 100]}),
        "SVM_rbf": (_scaled_y(SVR(kernel="rbf")), {"regressor__model__C": [0.1, 1]}),
        "SVM_poly": (_scaled_y(SVR(kernel="poly", degree=3)), {"regressor__model__C": [0.1, 1]}),
        "RandomForest": (RandomForestRegressor(**rf_params), {}),
        "MLP": (_scaled_y(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                       early_stopping=True, random_state=seed)),
                {"regressor__model__alpha": [1e-4, 1e-2]}),
    }


def select_and_fit(est, grid: dict, train, val, feats, target, score_fn, refit: bool):
    """Try every grid point: fit on train, score on validation (higher is
    better). Keep the best, then refit it on train+val if refit=True.
    Returns (fitted model, best params, best validation score)."""
    best = (None, {}, -np.inf)
    for params in (ParameterGrid(grid) if grid else [{}]):
        m = clone(est).set_params(**params)
        m.fit(train[feats], train[target])
        s = score_fn(val[target].to_numpy(), m.predict(val[feats]))
        if s > best[2]:
            best = (m, params, s)
    model, params, score = best
    if refit:
        model = clone(est).set_params(**params)
        both = pd.concat([train, val])
        model.fit(both[feats], both[target])
    return model, params, score


# ---- baselines -------------------------------------------------------------
def baseline_sign(name: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if name == "Majority":
        return np.full(len(test), int(train["up"].mean() >= 0.5))
    if name == "SameAsToday":
        return (test["ret_0"] > 0).astype(int).to_numpy()
    raise ValueError(name)


def baseline_return(name: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if name == "RandomWalk":
        return np.zeros(len(test))             # tomorrow's rate = today's
    if name == "HistMean":
        return np.full(len(test), train["next_ret"].mean())
    if name == "Momentum":
        return test["ret_0"].to_numpy()        # tomorrow's return = today's
    raise ValueError(name)
