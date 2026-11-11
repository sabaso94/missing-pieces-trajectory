"""Sensitivity of a regression estimate to an unmeasured confounder (Cinelli and Hazlett, 2020).

A confounder z explaining a share R2_yz of the outcome's residual variance (given treatment and covariates)
and R2_tz of the treatment's (given covariates) changes the treatment coefficient by exactly
se * sqrt(dof) * sqrt(R2_yz * R2_tz / (1 - R2_tz)). The robustness value is the share that a confounder
would need to explain of both to remove the estimate (or its significance at level alpha).
"""
import numpy as np
from scipy.stats import t as student_t


def partial_r2(t_stat: float, dof: int) -> float:
    return t_stat ** 2 / (t_stat ** 2 + dof)


def robustness_value(t_stat: float, dof: int, q: float = 1.0, alpha: float | None = None) -> float:
    f = q * abs(t_stat) / np.sqrt(dof)
    if alpha is not None:
        f -= abs(student_t.ppf(alpha / 2, dof - 1)) / np.sqrt(dof - 1)
        if f <= 0:
            return 0.0
    return float(0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2))


def bias_bound(r2_yz: float, r2_tz: float, se: float, dof: int) -> float:
    return float(se * np.sqrt(dof) * np.sqrt(r2_yz * r2_tz / (1 - r2_tz)))


def partial_r2_of(y: np.ndarray, z: np.ndarray, others: np.ndarray) -> float:
    """Partial R2 of z in the least-squares regression of y on (1, others, z)."""
    design = np.column_stack([np.ones(len(y)), others, z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = len(y) - design.shape[1]
    se = np.sqrt(resid @ resid / dof * np.linalg.pinv(design.T @ design)[-1, -1])
    return partial_r2(beta[-1] / se, dof)
