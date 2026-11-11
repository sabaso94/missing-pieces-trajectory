"""Cohort report: patients, baseline covariates and CD4 counts per arm and per treatment group.

    python scripts/cohort_report.py

Writes results/cohort.csv (one row per group; means, with SDs for continuous covariates).
"""
import pandas as pd

from trial_trajectories.data import ARM_NAMES, COVARIATES, ROOT, load_trial

CONTINUOUS = ["age", "wtkg", "karnof", "cd40", "cd80", "preanti"]


def describe(group: pd.DataFrame, name: str) -> dict:
    row = {"group": name, "n": len(group)}
    for col in COVARIATES:
        row[col] = group[col].mean()
        if col in CONTINUOUS:
            row[f"{col}_sd"] = group[col].std()
    for week, col in [(0, "cd40"), (20, "cd420"), (96, "cd496")]:
        row[f"cd4_week{week}"] = group[col].mean()
    row["missing_week96"] = 1 - group["observed96"].mean()
    return row


def main() -> None:
    df = load_trial()
    rows = [describe(df, "all")]
    rows += [describe(df[df["treat"] == t], name) for t, name in [(0, "zidovudine alone"), (1, "newer regimens")]]
    rows += [describe(df[df["arms"] == a], ARM_NAMES[a]) for a in sorted(ARM_NAMES)]
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "cohort.csv", index=False, float_format="%.4f")
    print(out[["group", "n", "cd4_week0", "cd4_week20", "cd4_week96", "missing_week96"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
