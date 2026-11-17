#!/usr/bin/env bash
# Rebuild the CHEAP (no-embedding) CnP->NT lexical pipeline + simple retrievers in
# dependency order after an nt1823 corpus change, then re-consolidate the apparatus.
#
# Step 1 (lemmatize) regenerates the NT+segment lemma cache from the CURRENT
# data/crime_and_punishment/nt1823.jsonl, so a corpus edit propagates correctly (no stale cache).
#
# DEFERRED — expensive, embedding-based (run separately, ~25 min CPU each, no API):
#   cnp_dense.py, cnp_rerank.py, cnp_bigram_embed.py
# cnp_fusion.py depends only on cnp_dense's OUTPUT (results/crime_and_punishment/cnp_dense_top5.jsonl), not on
# re-running the embedding itself -- it IS run here (fast, deterministic RRF, no
# embedding call) so the fusion store never goes stale relative to the lexical layer.
set -euo pipefail
cd "$(dirname "$0")/.."          # release root
PY="${PYTHON:-.venv/bin/python}"
R=scripts

echo "[ 1] lemmatize    (Mystem; regen NT+seg lemma cache)"; $PY $R/cnp_lemmatize.py
echo "[ 2] tokenize";                                        $PY $R/cnp_tokenize.py
echo "[ 3] match         -> results/crime_and_punishment/cnp_match_raw.jsonl";            $PY $R/cnp_match.py
echo "[ 4] features      -> results/crime_and_punishment/cnp_match_features.jsonl";       $PY $R/cnp_features.py
echo "[ 5] background    (CnP self-frequency)";              $PY $R/cnp_background.py
echo "[ 6] distinctive   -> results/crime_and_punishment/cnp_match_distinct.jsonl";       $PY $R/cnp_distinctiveness.py
echo "[ 7] report_v2     -> cnp_bible_refs.{csv,jsonl}";     $PY $R/cnp_report_v2.py
echo "[ 8] tier          -> cnp_bible_tiered.{csv,jsonl}";   $PY $R/cnp_tier.py
echo "[ 9] groups        -> results/crime_and_punishment/cnp_bible_groups.md";            $PY $R/cnp_groups.py
echo "[10] baselines     -> bm25 + tesserae";                $PY $R/cnp_baselines.py
echo "[11] lexical_extra -> char + overlap";                 $PY $R/cnp_lexical_extra.py
echo "[12] synonym       -> results/crime_and_punishment/cnp_synonym_top5.jsonl";         $PY $R/cnp_synonym.py
echo "[13] nt_surface    -> results/crime_and_punishment/cnp_nt_surface.jsonl";           $PY $R/cnp_nt_surface.py
echo "[14] wchar         -> results/crime_and_punishment/cnp_wchar_top5.jsonl";           $PY $R/cnp_wchar.py
echo "[15] wpair         -> results/crime_and_punishment/cnp_wpair_top5.jsonl";           $PY $R/cnp_wpair.py
echo "[16] match_cs      -> results/crime_and_punishment/cnp_match_cs_raw.jsonl";         $PY $R/cnp_match_cs.py
echo "[17] cs_surface    -> results/crime_and_punishment/cnp_cs_surface.jsonl";           $PY $R/cnp_match_cs_surface.py
echo "[18] consolidate   -> cnp_connections.*";              $PY $R/cnp_consolidate.py
echo "[19] fusion        -> results/crime_and_punishment/cnp_fusion_top.jsonl (RRF, needs an existing results/crime_and_punishment/cnp_dense_top5.jsonl)"
                                                             $PY $R/cnp_fusion.py
echo "[20] apparatus     -> results/crime_and_punishment/cnp_apparatus.html";             $PY $R/cnp_apparatus.py
echo "DONE — cheap lexical rebuild + fusion complete (dense/rerank re-embedding deferred)."
