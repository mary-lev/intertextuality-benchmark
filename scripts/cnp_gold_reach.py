"""Which retrieval methods reach each relation of the Crime and Punishment gold (paper §6, Table 8).

A relation is reached by a method when the method's own output pairs a sentence of the quoted
passage with one of the cited verses: a sentence within three paragraphs of the paragraph the
note is attached to, and a verse reference that overlaps the cited range. Every method is judged
on its own output; nothing is combined.

    python scripts/cnp_gold_reach.py     # from the release root -> results/crime_and_punishment/cnp_gold_reach.json

Method names (Table 3 family in brackets): sw [alignment], cs_sw [alignment, Church Slavonic
text], nt_surface and cs_surface [surface, words; against the 1823 / Church Slavonic text],
char and wchar [surface, character 4-grams], bm25 [BM25], tesserae and wpair [collocation],
dense [dense], rerank [reranker], fusion [rank fusion].
"""
from __future__ import annotations
import json
from collections import defaultdict

import ngram_baseline as nb
from cnp_eval_commentary import load_gold, matches, pidx

WINDOW = 3            # paragraphs either side of the note's paragraph
WCHAR_FLOOR = 55.0    # wchar's own threshold (the score below which it returns nothing)

# method -> (file, how to read the verse references of one output row)
SOURCES = {
    "sw": ("results/crime_and_punishment/cnp_match_features.jsonl", lambda r: [r["nt_ref"]]),
    "cs_sw": ("results/crime_and_punishment/cnp_match_cs_raw.jsonl", lambda r: [r["nt_ref"]]),
    "nt_surface": ("results/crime_and_punishment/cnp_nt_surface.jsonl", lambda r: [r["nt_ref"]]),
    "cs_surface": ("results/crime_and_punishment/cnp_cs_surface.jsonl", lambda r: [r["cs_ref"]]),
    "char": ("results/crime_and_punishment/cnp_char_top5.jsonl", lambda r: r["top"]),
    "wchar": ("results/crime_and_punishment/cnp_wchar_top5.jsonl", lambda r: [h["nt_ref"] for h in r["top"] if h["score"] >= WCHAR_FLOOR]),
    "bm25": ("results/crime_and_punishment/cnp_bm25_top5.jsonl", lambda r: r["top"]),
    "tesserae": ("results/crime_and_punishment/cnp_tesserae_top5.jsonl", lambda r: r["top"]),
    "wpair": ("results/crime_and_punishment/cnp_wpair_top5.jsonl", lambda r: r["top"]),
    "dense": ("results/crime_and_punishment/cnp_dense_top5.jsonl", lambda r: r["top"]),
    "rerank": ("results/crime_and_punishment/cnp_rerank_top.jsonl", lambda r: r["reranked"]),
    "fusion": ("results/crime_and_punishment/cnp_fusion_top.jsonl", lambda r: r["fused"]),
}


def ref_of(x):
    return x if isinstance(x, str) else x.get("nt_ref")


def main():
    by_para = {}                                  # method -> paragraph index -> [verse refs]
    for m, (fn, refs) in SOURCES.items():
        d = defaultdict(list)
        for l in open(nb.HERE + fn, encoding="utf-8"):
            r = json.loads(l)
            d[pidx(r["seg_id"])].extend(ref_of(x) for x in refs(r))
        by_para[m] = d
    out = []
    for g in load_gold():
        pi = pidx(g["paragraph_ref"])
        reached = sorted(m for m, d in by_para.items()
                         if g["bk"] and any(ref and matches(ref, g["bk"], g["ch"], g["gv"])
                                            for p in range(pi - WINDOW, pi + WINDOW + 1)
                                            for ref in d.get(p, ())))
        out.append({"id": g["id"], "type": g["type"], "nt_ref": g["nt_ref"],
                    "paragraph_ref": g["paragraph_ref"],
                    "in_scope": g["type"] in ("quotation", "allusion"), "methods": reached})
    json.dump(out, open(nb.HERE + "results/crime_and_punishment/cnp_gold_reach.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    scope = [r for r in out if r["in_scope"]]
    print(f"in scope {len(scope)}; reached {sum(bool(r['methods']) for r in scope)}; "
          f"by rank fusion {sum('fusion' in r['methods'] for r in scope)}")
    for r in scope:
        print(f"  {r['id']:5s} {r['nt_ref']:12s} {', '.join(r['methods']) or '—'}")


if __name__ == "__main__":
    main()
