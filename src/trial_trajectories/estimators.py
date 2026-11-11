"""Estimators of the average treatment effect E[Y(1) - Y(0)] from outcome y, treatment t (0/1) and baseline
covariates x.

- diff_in_means: the unadjusted contrast (right only when treatment is randomised);
- ancova: ordinary least squares of y on t and x, robust (HC1) standard error;
- ipw: inverse probability weighting (Hajek) with a logistic propensity model;
- aipw: augmented IPW (doubly robust), outcome models per arm and propensity fitted by cross-fitting
  (Chernozhukov et al., 2018), with linear or gradient-boosted nuisance models. Boosting stops early on a
  held-out 20% of the training rows (validation loss), so the number of trees is chosen by the data.
Estimated propensities are clipped to [clip, 1 - clip].
"""
from typing import Callable, NamedTuple

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Estimate(NamedTuple):
    est: float
    se: float

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        z = norm.ppf(0.5 + level / 2)
        return self.est - z * self.se, self.est + z * self.se


def folds_for(n: int, n_folds: int, seed: int) -> np.ndarray:
    """A random fold number (0 .. n_folds - 1) per row, folds of equal size up to one row."""
    folds = np.empty(n, dtype=int)
    folds[np.random.default_rng(seed).permutation(n)] = np.arange(n) % n_folds
    return folds


def _predict(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)


def cross_fit_predict(make_model: Callable, x: np.ndarray, y: np.ndarray, folds: np.ndarray,
                      rows: np.ndarray | None = None) -> np.ndarray:
    """Predictions for every row from a model fitted on the other folds (restricted to `rows` if given)."""
    rows = np.ones(len(y), dtype=bool) if rows is None else rows
    pred = np.empty(len(y))
    for k in np.unique(folds):
        train = (folds != k) & rows
        pred[folds == k] = _predict(make_model().fit(x[train], y[train]), x[folds == k])
    return pred


BOOSTING = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=20, early_stopping=True,
                validation_fraction=0.2, n_iter_no_change=10)


def outcome_model(nuisance: str, seed: int = 0):
    if nuisance == "linear":
        return LinearRegression()
    return HistGradientBoostingRegressor(**BOOSTING, random_state=seed)


def propensity_model(nuisance: str, seed: int = 0):
    if nuisance == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return HistGradientBoostingClassifier(**BOOSTING, random_state=seed)


def clipped_propensity(e: np.ndarray, clip: float) -> np.ndarray:
    return np.clip(e, clip, 1 - clip)


def diff_in_means(y: np.ndarray, t: np.ndarray) -> Estimate:
    y1, y0 = y[t == 1], y[t == 0]
    return Estimate(float(y1.mean() - y0.mean()), float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))))


def _ols(y: np.ndarray, t: np.ndarray, x: np.ndarray):
    design = np.column_stack([np.ones(len(y)), t, x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return design, beta, y - design @ beta


def ancova(y: np.ndarray, t: np.ndarray, x: np.ndarray) -> Estimate:
    design, beta, resid = _ols(y, t, x)
    n, k = design.shape
    bread = np.linalg.pinv(design.T @ design)
    meat = design.T @ (design * resid[:, None] ** 2)
    cov = n / (n - k) * bread @ meat @ bread
    return Estimate(float(beta[1]), float(np.sqrt(cov[1, 1])))


def ols_treatment_t(y: np.ndarray, t: np.ndarray, x: np.ndarray) -> tuple[float, int]:
    """Classical (non-robust) t-statistic of t in the regression of y on t and x, and its degrees of freedom;
    the inputs of the sensitivity analysis (Cinelli and Hazlett, 2020)."""
    design, beta, resid = _ols(y, t, x)
    n, k = design.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(sigma2 * np.linalg.pinv(design.T @ design)[1, 1])
    return float(beta[1] / se), n - k


def ipw(y: np.ndarray, t: np.ndarray, x: np.ndarray, clip: float = 0.01) -> Estimate:
    e = clipped_propensity(_predict(propensity_model("linear").fit(x, t), x), clip)
    w1, w0 = t / e, (1 - t) / (1 - e)
    mu1, mu0 = (w1 @ y) / w1.sum(), (w0 @ y) / w0.sum()
    influence = w1 * (y - mu1) / w1.mean() - w0 * (y - mu0) / w0.mean()
    return Estimate(float(mu1 - mu0), float(influence.std(ddof=1) / np.sqrt(len(y))))


def aipw(y: np.ndarray, t: np.ndarray, x: np.ndarray, nuisance: str = "linear", n_folds: int = 5, seed: int = 0,
         clip: float = 0.01) -> Estimate:
    folds = folds_for(len(y), n_folds, seed)
    mu1 = cross_fit_predict(lambda: outcome_model(nuisance, seed), x, y, folds, rows=t == 1)
    mu0 = cross_fit_predict(lambda: outcome_model(nuisance, seed), x, y, folds, rows=t == 0)
    e = clipped_propensity(cross_fit_predict(lambda: propensity_model(nuisance, seed), x, t, folds), clip)
    psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    return Estimate(float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(y))))
