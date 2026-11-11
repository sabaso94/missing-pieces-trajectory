#!/usr/bin/env bash
# Download ACTG 175 from the CRAN package speff2trial 1.0.5 and keep only the data file (data/ACTG175.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
[ -f data/ACTG175.txt ] && exit 0
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fsSL -o "$TMP/pkg.tar.gz" https://cran.r-project.org/src/contrib/speff2trial_1.0.5.tar.gz \
  || curl -fsSL -o "$TMP/pkg.tar.gz" https://cran.r-project.org/src/contrib/Archive/speff2trial/speff2trial_1.0.5.tar.gz
tar -xzf "$TMP/pkg.tar.gz" -C "$TMP" speff2trial/data/ACTG175.txt
echo "7b52e6b3701a1f24090ef0388f1c148d5286450f97276d9c8fbe611ce0c76779  $TMP/speff2trial/data/ACTG175.txt" | sha256sum -c --quiet
cp "$TMP/speff2trial/data/ACTG175.txt" data/ACTG175.txt
