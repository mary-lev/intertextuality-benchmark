#!/usr/bin/env bash
# Crime and Punishment (paper section 6): rebuild the method outputs and the gold reach table.
# Run from the release root. Steps 4 and 14 load local sentence encoders (about 25 minutes
# each on a CPU); the rest take minutes. No paid API calls.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
R=scripts
echo "[ 1] segment";        $PY $R/cnp_segment.py
echo "[ 2] lemmatize";      $PY $R/cnp_lemmatize.py
echo "[ 3] tokenize";       $PY $R/cnp_tokenize.py
echo "[ 4] dense";          $PY $R/cnp_dense.py
echo "[ 5] alignment";      $PY $R/cnp_match.py
echo "[ 6] features";       $PY $R/cnp_features.py
echo "[ 7] BM25, Tesserae"; $PY $R/cnp_baselines.py
echo "[ 8] char 4-grams";   $PY $R/cnp_char.py
echo "[ 9] surface (1823)"; $PY $R/cnp_nt_surface.py
echo "[10] weighted char";  $PY $R/cnp_wchar.py
echo "[11] word pairs";     $PY $R/cnp_wpair.py
echo "[12] alignment (CS)"; $PY $R/cnp_match_cs.py
echo "[13] surface (CS)";   $PY $R/cnp_match_cs_surface.py
echo "[14] reranker";       $PY $R/cnp_rerank.py
echo "[15] rank fusion";    $PY $R/cnp_fusion.py
echo "[16] gold reach";     $PY $R/cnp_gold_reach.py
