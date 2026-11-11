"""The ACTG 175 trial (Hammer et al., 1996) as shipped in the CRAN package speff2trial.

One row per patient. Treatment `treat`: 0 = zidovudine alone, 1 = one of the three newer regimens. CD4 counts
at baseline (`cd40`), week 20 (`cd420`, complete) and week 96 (`cd496`, missing for about 37%).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "ACTG175.txt"
N_PATIENTS = 2139

# Baseline covariates only. `zprior` is the same for every patient; `strat` repeats `str2` and `preanti`.
COVARIATES = ["age", "wtkg", "karnof", "cd40", "cd80", "preanti", "hemo", "homo", "drugs", "oprior", "z30",
              "race", "gender", "str2", "symptom"]
ARM_NAMES = {0: "zidovudine", 1: "zidovudine + didanosine", 2: "zidovudine + zalcitabine", 3: "didanosine"}


def load_trial(path: Path | None = None) -> pd.DataFrame:
    """The trial with the outcomes y20 = cd420 - cd40 and y96 = cd496 - cd40 (NaN if week 96 is missing)."""
    df = pd.read_csv(path or DATA_FILE, sep=r"\s+")
    if path is None and len(df) != N_PATIENTS:
        raise ValueError(f"expected {N_PATIENTS} patients, found {len(df)}")
    df["y20"] = df["cd420"] - df["cd40"]
    df["y96"] = df["cd496"] - df["cd40"]
    df["observed96"] = df["cd496"].notna()
    return df


def covariate_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[COVARIATES].to_numpy(dtype=float)
