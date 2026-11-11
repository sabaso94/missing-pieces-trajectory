"""E2: do causal estimators recover the trial's answer from confounded data?

    python scripts/run_e2.py [--draws 500] [--jobs 8]

For each confounding strength gamma, the trial is subsampled so that sicker patients are more likely to be
treated (P(T = 1 | x) = expit(gamma * severity)), keeping the covariate distribution; see confounding.py.
Five estimators of the week-20 effect, adjusting for all baseline covariates ("measured"), for all but the
baseline CD4 count ("cd40 hidden"), or for all but cd40, Karnofsky score and symptoms, are scored against the trial's own estimate (ANCOVA on all 2,139 patients).
For the hidden setting, the sensitivity analysis of Cinelli and Hazlett (2020) is run on the ANCOVA estimate.
Coverage is of the trial's estimate by the 95% interval, corrected for the overlap between each sample and the
trial (see summarise).
Writes results/e2_draws.csv, results/e2_summary.csv and results/e2_sensitivity.csv.
"""
import os

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")  # one thread per worker: identical results however many workers

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from trial_trajectories.confounding import confounded_sample, severity_score
from trial_trajectories.data import COVARIATES, ROOT, covariate_matrix, load_trial
from trial_trajectories.estimators import aipw, ancova, diff_in_means, ipw, ols_treatment_t
from trial_trajectories.metrics import coverage_of_truth
from trial_trajectories.sensitivity import bias_bound, partial_r2_of, robustness_value

GAMMAS = [0.0, 0.5, 1.0, 2.0]
HIDDEN = "cd40"
# Karnofsky score and symptoms drive treatment in the samples (they are part of the severity score) but barely the
# outcome; adjusting for them while cd40 is hidden amplifies its bias (Pearl, 2010).
TREATMENT_ONLY = ["karnof", "symptom"]


def estimators(y, t, x, seed):
    return {"difference in means": diff_in_means(y, t), "ANCOVA": ancova(y, t, x), "IPW": ipw(y, t, x),
            "AIPW, linear": aipw(y, t, x, "linear", seed=seed), "AIPW, boosting": aipw(y, t, x, "boosting", seed=seed)}


def one_draw(gamma, draw, y, t, x, score, seed):
    idx = confounded_sample(t, score, gamma, np.random.default_rng(seed))
    yi, ti, xi = y[idx], t[idx], x[idx]
    hidden_col = COVARIATES.index(HIDDEN)
    xh = np.delete(xi, hidden_col, axis=1)
    xhs = np.delete(xi, [COVARIATES.index(c) for c in [HIDDEN, *TREATMENT_ONLY]], axis=1)
    rows = []
    for adjustment, xa in [("measured", xi), ("cd40 hidden", xh), ("cd40 hidden, karnof and symptom left out", xhs)]:
        for name, est in estimators(yi, ti, xa, seed).items():
            low, high = est.ci()
            rows.append({"gamma": gamma, "draw": draw, "adjustment": adjustment, "estimator": name, "est": est.est,
                         "se": est.se, "low": low, "high": high, "n": len(idx), "treated_share": ti.mean()})
    t_stat, dof = ols_treatment_t(yi, ti, xh)
    z = xi[:, hidden_col]
    r2_yz, r2_tz = partial_r2_of(yi, z, np.column_stack([ti, xh])), partial_r2_of(ti.astype(float), z, xh)
    est_hidden = ancova(yi, ti, xh).est
    sens = {"gamma": gamma, "draw": draw, "est_hidden": est_hidden, "t_stat": t_stat, "dof": dof,
            "rv": robustness_value(t_stat, dof), "rv_alpha05": robustness_value(t_stat, dof, alpha=0.05),
            "r2_y_cd40": r2_yz, "r2_t_cd40": r2_tz,
            "bias_bound_cd40": bias_bound(r2_yz, r2_tz, est_hidden / t_stat, dof)}
    return rows, sens


def summarise(draws: pd.DataFrame, truth: float, se_truth: float) -> pd.DataFrame:
    """Bias, spread and RMSE against the trial's estimate, and coverage of that estimate by the 95% interval.

    Every sample is drawn from the 2,139 trial patients that also give the truth, so a sample estimate varies
    around the trial's estimate by less than its standard error says; coverage uses the variance
    se^2 - se_truth^2 (see metrics.coverage_of_truth)."""
    covered = coverage_of_truth(draws["est"].to_numpy(), draws["se"].to_numpy(), truth, se_truth)
    g = draws.assign(err=draws["est"] - truth, covered=covered)
    out = g.groupby(["gamma", "adjustment", "estimator"], sort=False).agg(
        bias=("err", "mean"), sd=("est", "std"), rmse=("err", lambda e: np.sqrt((e ** 2).mean())),
        coverage=("covered", "mean"), mean_se=("se", "mean"), n=("n", "mean"), draws=("est", "size"))
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    df = load_trial()
    y, t, x, score = df["y20"].to_numpy(float), df["treat"].to_numpy(), covariate_matrix(df), severity_score(df)
    trial = ancova(y, t, x)
    truth = trial.est
    tasks = [(gamma, d, 10_000 * (g + 1) + d) for g, gamma in enumerate(GAMMAS) for d in range(args.draws)]
    results = Parallel(n_jobs=args.jobs)(delayed(one_draw)(gamma, d, y, t, x, score, seed) for gamma, d, seed in tasks)
    draws = pd.DataFrame([row for rows, _ in results for row in rows])
    sens = pd.DataFrame([s for _, s in results])
    out = ROOT / "results"
    draws.to_csv(out / "e2_draws.csv", index=False, float_format="%.4f")
    summary = summarise(draws, truth, trial.se).assign(truth=truth)
    summary.to_csv(out / "e2_summary.csv", index=False, float_format="%.4f")
    sens = sens.assign(abs_error=(sens["est_hidden"] - truth).abs(),
                       cd40_explains_away=sens["bias_bound_cd40"] >= sens["est_hidden"].abs())
    sens_summary = sens.groupby("gamma").agg(
        est_hidden=("est_hidden", "mean"), rv=("rv", "mean"), rv_alpha05=("rv_alpha05", "mean"),
        r2_y_cd40=("r2_y_cd40", "mean"), r2_t_cd40=("r2_t_cd40", "mean"),
        bias_bound_cd40=("bias_bound_cd40", "mean"), abs_error=("abs_error", "mean"),
        share_explained_away=("cd40_explains_away", "mean")).reset_index().assign(truth=truth)
    sens_summary.to_csv(out / "e2_sensitivity.csv", index=False, float_format="%.4f")
    print(f"truth {truth:.2f}")
    print(summary.round(3).to_string(index=False))
    print(sens_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
