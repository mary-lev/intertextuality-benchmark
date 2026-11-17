#!/usr/bin/env bash
# Reproduce the paper's figures and result tables.
#   ./reproduce.sh          figures from the shipped result tables (no corpus needed)
#   ./reproduce.sh --full   recompute the result tables from the corpus first (several hours)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"

if [[ "${1:-}" == "--full" ]]; then
  for f in data/poetree/poetree_full_poems.csv cache/lemmas/poetree_lemmas.csv; do
    [[ -e "$f" ]] || { echo "missing $f: see README, Reproducing the results"; exit 1; }
  done
  for s in factorial prior_ladder prior_ladder_control granularity invariance dante_factorial; do
    echo "== $s"
    "$PY" "scripts/$s.py"
  done
  echo "== dante_factorial, open search space"
  CORPUS=open "$PY" scripts/dante_factorial.py
  echo "== audit of the numbers quoted in sections 5 to 8"
  "$PY" scripts/audit_s5.py
  "$PY" scripts/audit_s6_s7.py
fi

echo "== figures"
"$PY" scripts/make_figures.py
echo "figures written to figures/"
