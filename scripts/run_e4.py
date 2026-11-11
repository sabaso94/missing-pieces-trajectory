"""E4: who benefits most from the newer regimens?

    python scripts/run_e4.py

On the full trial (randomised, so every group contrast is unconfounded), the week-20 effect:
- in groups fixed before looking at the data: prior antiretroviral therapy, baseline CD4 tertile, symptoms;
- by predicted benefit: out-of-fold predictions from a T-learner (linear, boosting) and a DR-learner, cut into
  quartiles; within each quartile the randomised effect checks whether predicted benefit is real benefit.
Writes results/e4_subgroups.csv and results/e4_calibration.csv.
"""
import os

for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")  # small models: one thread is fastest and keeps results identical

import numpy as np
import pandas as pd

from trial_trajectories.data import ROOT, covariate_matrix, load_trial
from trial_trajectories.estimators import folds_for
from trial_trajectories.hte import dr_learner, group_effect_difference, group_effects, t_learner


def subgroups(df: pd.DataFrame, y: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    tertile = pd.qcut(df["cd40"], 3, labels=False).to_numpy()
    edges = df.groupby(tertile)["cd40"].agg(["min", "max"]).astype(int)
    factors = {
        "prior antiretroviral therapy": np.where(df["str2"] == 1, "yes", "no"),
        "baseline CD4": np.array([f"{edges.loc[k, 'min']}-{edges.loc[k, 'max']}" for k in tertile]),
        "symptomatic": np.where(df["symptom"] == 1, "yes", "no"),
    }
    order = {"prior antiretroviral therapy": ["no", "yes"], "symptomatic": ["no", "yes"],
             "baseline CD4": [f"{edges.loc[k, 'min']}-{edges.loc[k, 'max']}" for k in range(3)]}
    rows = []
    for f, (factor, groups) in enumerate(factors.items()):
        res = group_effects(y, t, groups, seed=f).set_index("group").loc[order[factor]].reset_index()
        diff, low, high = group_effect_difference(y, t, groups, order[factor][0], order[factor][-1], seed=f)
        rows.append(res.assign(factor=factor, diff_last_minus_first=diff, diff_low=low, diff_high=high))
    return pd.concat(rows, ignore_index=True)


def calibration(y: np.ndarray, t: np.ndarray, x: np.ndarray) -> pd.DataFrame:
    folds = folds_for(len(y), 5, seed=0)
    preds = {"T-learner, linear": t_learner(y, t, x, "linear", folds),
             "T-learner, boosting": t_learner(y, t, x, "boosting", folds),
             "DR-learner": dr_learner(y, t, x, folds, p_treat=t.mean())}
    rows = []
    for k, (learner, pred) in enumerate(preds.items()):
        quartile = pd.qcut(pred, 4, labels=False) + 1
        res = group_effects(y, t, quartile, seed=10 + k).sort_values("group")
        res["predicted"] = [pred[quartile == q].mean() for q in res["group"]]
        diff, low, high = group_effect_difference(y, t, quartile, 1, 4, seed=10 + k)
        rows.append(res.assign(learner=learner, top_minus_bottom=diff, diff_low=low, diff_high=high,
                               pred_sd=pred.std()))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    df = load_trial()
    y, t, x = df["y20"].to_numpy(float), df["treat"].to_numpy(), covariate_matrix(df)
    out = ROOT / "results"
    sub = subgroups(df, y, t)
    sub.to_csv(out / "e4_subgroups.csv", index=False, float_format="%.3f")
    cal = calibration(y, t, x)
    cal.to_csv(out / "e4_calibration.csv", index=False, float_format="%.3f")
    print(sub.round(1).to_string(index=False))
    print(cal.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
