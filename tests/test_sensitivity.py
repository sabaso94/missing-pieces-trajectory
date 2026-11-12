import numpy as np

from trial_trajectories.estimators import ols_treatment_t
from trial_trajectories.sensitivity import bias_bound, partial_r2, partial_r2_of, robustness_value


def test_robustness_value_matches_cinelli_hazlett():
    # Darfur example, Cinelli and Hazlett (2020): t = 4.18 on 783 degrees of freedom
    assert abs(partial_r2(4.18, 783) - 0.022) < 0.001
    assert abs(robustness_value(4.18, 783) - 0.139) < 0.002
    assert abs(robustness_value(4.18, 783, alpha=0.05) - 0.076) < 0.002


def test_bias_bound_is_the_omitted_variable_bias_of_a_known_confounder():
    rng = np.random.default_rng(0)
    n = 3000
    x, z = rng.normal(size=(n, 2)), rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    y = 1.0 * t + 2 * z + x[:, 0] + rng.normal(size=n)
    full = np.linalg.lstsq(np.column_stack([np.ones(n), t, x, z]), y, rcond=None)[0][1]
    short = np.linalg.lstsq(np.column_stack([np.ones(n), t, x]), y, rcond=None)[0][1]
    t_stat, dof = ols_treatment_t(y, t, x)
    r2_yz = partial_r2_of(y, z, np.column_stack([t, x]))
    r2_tz = partial_r2_of(t, z, x)
    assert abs(bias_bound(r2_yz, r2_tz, short / t_stat, dof) - abs(short - full)) < 1e-6


def test_robustness_value_uses_the_extreme_formula_for_very_large_t():
    # sensemakr: when q|t|/sqrt(dof) exceeds 1/f_crit, the bound is reached at R2 = 1 for the treatment
    from scipy.stats import t as student_t
    t_stat, dof = 50.0, 10
    fq2 = t_stat ** 2 / dof
    fc2 = (abs(student_t.ppf(0.025, dof - 1)) / np.sqrt(dof - 1)) ** 2
    assert abs(robustness_value(t_stat, dof, alpha=0.05) - (fq2 - fc2) / (1 + fq2)) < 1e-12
