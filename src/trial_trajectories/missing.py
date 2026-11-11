"""Means and effects when the outcome is missing for some patients (dropout before week 96).

Within one arm, the mean of y over all patients is estimated from the observed ones by
- complete: the mean of the observed (right only if dropout is unrelated to y);
- ipcw: inverse probability of observation weighting, P(observed | h) from a logistic model (Robins et al., 1995);
- aipw: a linear outcome model on h fitted on the observed, plus inverse-weighted residuals (doubly robust).
Both corrections assume dropout depends only on the history h observed before it (missing at random).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trial_trajectories.data import COVARIATES

HISTORY96 = COVARIATES + ["cd420"]  # what is known about a patient before the week-96 visit


def week96_history(df: pd.DataFrame) -> np.ndarray:
    return df[HISTORY96].to_numpy(dtype=float)


def dropout_model(h: np.ndarray, observed: np.ndarray):
    """Logistic model of P(observed | h) on standardised inputs."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(h, observed.astype(int))


def arm_mean(y: np.ndarray, observed: np.ndarray, h: np.ndarray, method: str = "complete",
             clip: float = 0.01) -> float:
    if method == "complete":
        return float(y[observed].mean())
    pi = np.clip(dropout_model(h, observed).predict_proba(h)[:, 1], clip, 1)
    w = observed / pi
    if method == "ipcw":
        return float((w[observed] @ y[observed]) / w.sum())
    if method == "aipw":
        m = LinearRegression().fit(h[observed], y[observed]).predict(h)
        resid = np.where(observed, y - m, 0.0)
        return float(np.mean(m + w * resid))
    raise ValueError(f"unknown method {method}")


def effect_in_group(y: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
    """Difference in mean y between treated and controls within mask; NaN if an arm is empty there."""
    y1, y0 = y[mask & (t == 1)], y[mask & (t == 0)]
    if len(y1) == 0 or len(y0) == 0:
        return float("nan")
    return float(y1.mean() - y0.mean())
