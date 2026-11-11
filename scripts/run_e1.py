"""E1: the natural course of CD4 counts and the trial's treatment effect.

    python scripts/run_e1.py

The control arm (zidovudine alone) shows what happens to CD4 counts under the old standard; the change in
treated patients mixes that course with the effect of the newer regimens; the randomised contrast isolates the
effect. Week 96 uses the patients with a week-96 count (E3 looks at who is missing).
Writes results/e1_trajectory.csv and results/e1_effects.csv.
"""
import numpy as np
import pandas as pd

from trial_trajectories.data import ARM_NAMES, ROOT, covariate_matrix, load_trial
from trial_trajectories.estimators import ancova
from trial_trajectories.metrics import bootstrap_ci

WEEKS = {0: "cd40", 20: "cd420", 96: "cd496"}
OUTCOME = {20: "y20", 96: "y96"}


def trajectory(df: pd.DataFrame) -> pd.DataFrame:
    groups = {"zidovudine alone": df["treat"] == 0, "newer regimens": df["treat"] == 1}
    groups.update({ARM_NAMES[a]: df["arms"] == a for a in (1, 2, 3)})
    rows = []
    for name, mask in groups.items():
        for week, col in WEEKS.items():
            values = df.loc[mask, col].dropna().to_numpy()
            low, high = bootstrap_ci(lambda idx: values[idx].mean(), len(values), seed=week)
            rows.append({"group": name, "week": week, "n": len(values), "mean_cd4": values.mean(),
                         "low": low, "high": high})
    return pd.DataFrame(rows)


def effects(df: pd.DataFrame, name: str, week: int) -> list[dict]:
    """Natural course, before-after change in the treated, and the randomised effect (unadjusted and ANCOVA)."""
    sub = df[df[OUTCOME[week]].notna()]
    y, t, x = sub[OUTCOME[week]].to_numpy(), sub["treat_vs"].to_numpy(), covariate_matrix(sub)

    def stats(idx):
        yi, ti = y[idx], t[idx]
        return np.array([yi[ti == 0].mean(), yi[ti == 1].mean(), yi[ti == 1].mean() - yi[ti == 0].mean(),
                         ancova(yi, ti, x[idx]).est])

    point = stats(np.arange(len(y)))
    low, high = bootstrap_ci(stats, len(y), seed=week)
    labels = ["natural course (change on zidovudine alone)", "before-after change in the treated",
              "effect, difference in means", "effect, ANCOVA"]
    return [{"comparison": name, "week": week, "quantity": q, "est": p, "low": lo, "high": hi, "n": len(y)}
            for q, p, lo, hi in zip(labels, point, low, high)]


def main() -> None:
    df = load_trial()
    out = ROOT / "results"
    traj = trajectory(df)
    traj.to_csv(out / "e1_trajectory.csv", index=False, float_format="%.3f")
    comparisons = {"newer regimens vs zidovudine alone": df.assign(treat_vs=df["treat"])}
    for a in (1, 2, 3):
        sub = df[df["arms"].isin([0, a])]
        comparisons[f"{ARM_NAMES[a]} vs zidovudine alone"] = sub.assign(treat_vs=(sub["arms"] == a).astype(int))
    rows = [r for name, sub in comparisons.items() for week in (20, 96) for r in effects(sub, name, week)]
    eff = pd.DataFrame(rows)
    eff.to_csv(out / "e1_effects.csv", index=False, float_format="%.3f")
    print(traj.round(1).to_string(index=False))
    print(eff.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
