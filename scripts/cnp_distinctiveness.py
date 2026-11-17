"""Phrase-distinctiveness rescoring of CnP->NT candidate links into evidence tiers.

NOTE (partly superseded): the Poetree `phrase_bg_idf` term here is GENRE-MISMATCHED (poetry, not
prose) and is NO LONGER USED for tiering — final tiers are assigned in `cnp_tier.py` using the
prose background we actually have (CnP self-frequency) + shared-run length. The useful columns this
script still produces are `run` and `cnp_self_freq`. Kept for provenance; the file is regenerable.

Adds phrase-level background frequency (Poetree document-frequency + CnP self raw
frequency) to the existing aligned-span features, so distinctive biblical phrases
(«да приидет царствие») separate from ordinary Russian collocations («изъявил желание»,
«пристально посмотрел»).

Pure helpers (TDD) first, then the driver.

    python scripts/cnp_distinctiveness.py    # -> results/crime_and_punishment/cnp_match_distinct.jsonl
"""
import difflib
import json
import math
import sys

import ngram_baseline as nb
from cnp_align import sw_span
from cnp_match import STOP, _content

CACHE = nb.HERE + "cache/lemmas/"

# --- PROVISIONAL, TUNABLE tier thresholds (controller tunes after the evidence) ---
IDF_HI = 8.0    # phrase_bg_idf above this = distinctive (rare in Poetree)
IDF_MID = 6.0   # phrase_bg_idf above this = moderately distinctive
SELF_LO = 3     # cnp_self_freq at/below this = not an authorial recurring tic


def longest_common_run(a_lems, b_lems):
    """Longest contiguous shared lemma run between two lemma token lists.

    Returns the actual tokens (from a_lems) as a list, e.g.
    ["приидти", "царствие", "твой"]. Empty list if nothing shared.
    """
    sm = difflib.SequenceMatcher(None, a_lems, b_lems)
    m = sm.find_longest_match(0, len(a_lems), 0, len(b_lems))
    return list(a_lems[m.a:m.a + m.size])


def _run_bigrams(run):
    return [run[i] + " " + run[i + 1] for i in range(len(run) - 1)]


def distinctiveness(run, poetree_df, cnp_freq):
    """Three phrase-distinctiveness features for a shared lemma run.

    poetree_df: {"N": N_poems, "bi": {bigram: df}, ...}
    cnp_freq:   {"bi": {bigram: count}}

    For the run's consecutive-pair bigrams (if run has <2 tokens, no bigrams ->
    features default to 0):
      phrase_bg_idf = MIN over bigrams of log(N / (1+df))   (weakest link; higher=distinctive)
      bg_max_df     = MAX Poetree bigram df in the run      (high=common collocation)
      cnp_self_freq = MAX CnP bigram raw count in the run   (high=authorial recurring phrase)
    """
    N = poetree_df.get("N", 1)
    bi_df = poetree_df.get("bi", {})
    cnp_bi = cnp_freq.get("bi", {})
    grams = _run_bigrams(run)
    if not grams:
        return {"phrase_bg_idf": 0.0, "bg_max_df": 0, "cnp_self_freq": 0}
    # N=0 = the Poetree cache was absent (superseded signal, skip-if-absent): idf 0, not log(0)
    idfs = [math.log(N / (1 + bi_df.get(g, 0))) if N > 0 else 0.0 for g in grams]
    dfs = [bi_df.get(g, 0) for g in grams]
    selfs = [cnp_bi.get(g, 0) for g in grams]
    return {
        "phrase_bg_idf": round(min(idfs), 4),
        "bg_max_df": max(dfs),
        "cnp_self_freq": max(selfs),
    }


def assign_tier(feat, dist):
    """Evidence tier from reused aligned-span features + distinctiveness.

    feat: dict with aligned_span_surface_similarity, longest_shared_surface_ngram,
          longest_shared_lemma_ngram
    dist: dict from distinctiveness()
    """
    surf_sim = feat.get("aligned_span_surface_similarity", 0.0)
    surf_ng = feat.get("longest_shared_surface_ngram", 0)
    lem_ng = feat.get("longest_shared_lemma_ngram", 0)
    idf = dist["phrase_bg_idf"]
    self_freq = dist["cnp_self_freq"]

    if surf_sim >= 0.90 and surf_ng >= 4:
        return "B"
    if lem_ng >= 3 and idf >= IDF_HI and self_freq <= SELF_LO:
        return "C"
    if lem_ng >= 2 and idf >= IDF_MID and self_freq <= SELF_LO:
        return "D"
    return "E"


# ------------------------------------------------------------------ driver

def _load_tokens(path):
    return {o["id"]: o for o in (json.loads(x) for x in open(path, encoding="utf-8"))}


def _load_features(path):
    feats = {}
    for line in open(path, encoding="utf-8"):
        o = json.loads(line)
        feats[(o["seg_id"], o["nt_ref"])] = o
    return feats


def main():
    seg_tok = _load_tokens(CACHE + "cnp_segments_tokens.jsonl")
    nt_tok = _load_tokens(CACHE + "nt1823_tokens.jsonl")
    feats = _load_features(nb.HERE + "results/crime_and_punishment/cnp_match_features.jsonl")
    try:
        poetree_df = json.load(open(CACHE + "poetree_phrase_df.json", encoding="utf-8"))
    except FileNotFoundError:
        print(f"  [skip] {CACHE}poetree_phrase_df.json not found (superseded signal); "
              "phrase_bg_idf will be 0 for all candidates", file=sys.stderr)
        poetree_df = {"N": 0, "bi": {}, "tri": {}}
    cnp_freq = json.load(open(CACHE + "cnp_phrase_freq.json", encoding="utf-8"))

    # idf for sw_span: rebuild lemma->idf from NT verse token caches (content lemmas)
    from collections import defaultdict
    df = defaultdict(int)
    verses = list(nt_tok.values())
    for v in verses:
        for w in set(_content_lems(v["lem"])):
            df[w] += 1
    Nv = len(verses)
    idf = {w: math.log(Nv / d) for w, d in df.items()}

    out_path = nb.HERE + "results/crime_and_punishment/cnp_match_distinct.jsonl"
    tier_counts = {"B": 0, "C": 0, "D": 0, "E": 0}
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for line in open(nb.HERE + "results/crime_and_punishment/cnp_match_raw.jsonl", encoding="utf-8"):
            cand = json.loads(line)
            seg_id, nt_ref = cand["seg_id"], cand["nt_ref"]
            feat = feats.get((seg_id, nt_ref), {})
            st = seg_tok.get(seg_id)
            vt = nt_tok.get(nt_ref)
            if st is None or vt is None:
                continue
            q_lems = _content_lems(st["lem"])
            v_lems = _content_lems(vt["lem"])
            # aligned span on content lemmas, then the longest contiguous shared run
            _, qi, qj, pi, pj = sw_span(q_lems, v_lems, idf)
            run = longest_common_run(q_lems[qi:qj], v_lems[pi:pj])
            dist = distinctiveness(run, poetree_df, cnp_freq)
            tier = assign_tier(feat, dist)
            tier_counts[tier] += 1
            n += 1
            out.write(json.dumps({
                "seg_id": seg_id,
                "nt_ref": nt_ref,
                "tier": tier,
                "run": " ".join(run),
                "phrase_bg_idf": dist["phrase_bg_idf"],
                "bg_max_df": dist["bg_max_df"],
                "cnp_self_freq": dist["cnp_self_freq"],
                "longest_shared_lemma_ngram": feat.get("longest_shared_lemma_ngram", 0),
                "longest_shared_surface_ngram": feat.get("longest_shared_surface_ngram", 0),
                "aligned_span_surface_similarity": feat.get("aligned_span_surface_similarity", 0.0),
                "number_of_shared_content_lemmas": feat.get("number_of_shared_content_lemmas", 0),
            }, ensure_ascii=False) + "\n")

    print(f"{n} candidates -> {out_path}", file=sys.stderr)
    print("tiers: " + "  ".join(f"{k}={tier_counts[k]}" for k in ("B", "C", "D", "E")),
          file=sys.stderr)


def _content_lems(lem_list):
    """Content-lemma filter (cnp_match._content rule) applied to a lemma token list."""
    return [w for w in lem_list if w not in STOP and len(w) > 1 and w.isalpha()]


if __name__ == "__main__":
    main()
