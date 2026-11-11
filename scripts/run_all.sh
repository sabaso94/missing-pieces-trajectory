#!/usr/bin/env bash
# Reproduce every result and figure: download -> tests -> cohort report -> E1 -> E2 -> E3 -> E4 -> figures.
# Usage: bash scripts/run_all.sh   (set JOBS to the number of parallel workers for E2; default 8)
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
JOBS="${JOBS:-8}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

bash scripts/download_data.sh
"$PYTHON" -m pytest -q
"$PYTHON" scripts/cohort_report.py
"$PYTHON" scripts/run_e1.py
"$PYTHON" scripts/run_e2.py --jobs "$JOBS"
"$PYTHON" scripts/run_e3.py
"$PYTHON" scripts/run_e4.py
"$PYTHON" scripts/make_figures.py
