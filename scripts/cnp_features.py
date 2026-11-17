"""Aligned-span features for CnP↔NT candidate links.

The old band used a SequenceMatcher ratio between the WHOLE segment and the WHOLE verse,
which penalizes a short quotation embedded in a long sentence. Instead, align segment and
verse on lemmas (Smith–Waterman, cnp_align.sw_span), then compute similarity / shared-run /
idf features on the **aligned span** only. Surface words are recovered from the parallel
token lists produced by cnp_tokenize.py.

7 features per pair:
  1. aligned_span_surface_similarity
  2. aligned_span_lemma_similarity
  3. longest_shared_surface_ngram   (tokens)
  4. longest_shared_lemma_ngram     (tokens)
  5. number_of_shared_content_lemmas
  6. sum_idf_shared_content_lemmas
  7. max_idf_shared_lemma

    python scripts/cnp_features.py    # -> results/crime_and_punishment/cnp_match_features.jsonl
"""
import json, math, re, sys
from collections import defaultdict
from difflib import SequenceMatcher
import ngram_baseline as nb
from cnp_align import sw_span

CACHE = nb.HERE + "cache/lemmas/"


def nrm(w):
    return re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))


def compute_features(seg_surf, seg_lem, ver_surf, ver_lem, idf):
    score, qi, qj, pi, pj = sw_span(seg_lem, ver_lem, idf)
    qs, ql = seg_surf[qi:qj], seg_lem[qi:qj]
    ps, pl = ver_surf[pi:pj], ver_lem[pi:pj]

    qs_n = [nrm(w) for w in qs]
    ps_n = [nrm(w) for w in ps]

    surf_sim = SequenceMatcher(None, " ".join(qs_n), " ".join(ps_n)).ratio()
    lem_sim = SequenceMatcher(None, " ".join(ql), " ".join(pl)).ratio()

    lcs_surf = SequenceMatcher(None, qs_n, ps_n).find_longest_match(
        0, len(qs_n), 0, len(ps_n)).size
    lcs_lem = SequenceMatcher(None, ql, pl).find_longest_match(
        0, len(ql), 0, len(pl)).size

    shared = set(ql) & set(pl)

    return {
        "aligned_span_surface_similarity": round(surf_sim, 4),
        "aligned_span_lemma_similarity": round(lem_sim, 4),
        "longest_shared_surface_ngram": lcs_surf,
        "longest_shared_lemma_ngram": lcs_lem,
        "number_of_shared_content_lemmas": len(shared),
        "sum_idf_shared_content_lemmas": round(sum(idf.get(w, 0.0) for w in shared), 3),
        "max_idf_shared_lemma": round(max((idf.get(w, 0.0) for w in shared), default=0.0), 3),
        "q_span": [qi, qj],
        "p_span": [pi, pj],
        "score": round(score, 3),
    }


# ------------------------------------------------------------------ driver
def _load_tokens(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        d[r["id"]] = (r["surf"], r["lem"])
    return d


def _build_idf(verse_tokens):
    """idf over the token-cache verse lem lists, consistent with cnp_match.load_lemmas."""
    df = defaultdict(int)
    for _surf, lem in verse_tokens.values():
        for w in set(lem):
            df[w] += 1
    N = len(verse_tokens)
    return {w: math.log(N / d) for w, d in df.items()}


def main():
    seg_tok = _load_tokens(CACHE + "cnp_segments_tokens.jsonl")
    ver_tok = _load_tokens(CACHE + "nt1823_tokens.jsonl")
    idf = _build_idf(ver_tok)

    out_path = nb.HERE + "results/crime_and_punishment/cnp_match_features.jsonl"
    raw = nb.HERE + "results/crime_and_punishment/cnp_match_raw.jsonl"

    written = skipped = 0
    checks = {
        ("ч4.гл4:p2440", "Ин 11:25"): None,
        ("ч1.гл2:p83", "Лк 7:47"): None,
        ("Деян 3:4",): None,           # NOISE «пристально посмотрел», any seg
        ("ч6.гл4:p3338", "2Тим 3:7"): None,
    }

    with open(out_path, "w", encoding="utf-8") as out:
        for line in open(raw, encoding="utf-8"):
            r = json.loads(line)
            sid, ref = r["seg_id"], r["nt_ref"]
            seg = seg_tok.get(sid)
            ver = ver_tok.get(ref)
            if seg is None or ver is None:
                skipped += 1
                continue
            f = compute_features(seg[0], seg[1], ver[0], ver[1], idf)
            row = {"seg_id": sid, "nt_ref": ref, "score": r["score"],
                   "aligned_span_surface_similarity": f["aligned_span_surface_similarity"],
                   "aligned_span_lemma_similarity": f["aligned_span_lemma_similarity"],
                   "longest_shared_surface_ngram": f["longest_shared_surface_ngram"],
                   "longest_shared_lemma_ngram": f["longest_shared_lemma_ngram"],
                   "number_of_shared_content_lemmas": f["number_of_shared_content_lemmas"],
                   "sum_idf_shared_content_lemmas": f["sum_idf_shared_content_lemmas"],
                   "max_idf_shared_lemma": f["max_idf_shared_lemma"]}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            # capture check rows (seg_id starts with the paragraph prefix)
            para = sid.split("/")[0]
            if (para, ref) in checks and checks[(para, ref)] is None:
                checks[(para, ref)] = (sid, row)
            if ref == "Деян 3:4" and checks[("Деян 3:4",)] is None:
                checks[("Деян 3:4",)] = (sid, row)

    print(f"rows written: {written}", file=sys.stderr)
    print(f"rows skipped: {skipped}", file=sys.stderr)
    print("\nCHECK ROWS:", file=sys.stderr)
    for key, val in checks.items():
        if val is None:
            print(f"  {key}: NOT FOUND", file=sys.stderr)
        else:
            sid, row = val
            print(f"  {key} seg={sid}", file=sys.stderr)
            for k in ("score", "aligned_span_surface_similarity", "aligned_span_lemma_similarity",
                      "longest_shared_surface_ngram", "longest_shared_lemma_ngram",
                      "number_of_shared_content_lemmas", "sum_idf_shared_content_lemmas",
                      "max_idf_shared_lemma"):
                print(f"      {k}: {row[k]}", file=sys.stderr)


if __name__ == "__main__":
    main()
