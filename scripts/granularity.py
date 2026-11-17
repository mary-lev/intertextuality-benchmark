"""Query-granularity sensitivity: exact fragment -> containing stanza -> whole poem.

The localization analog of the prior ladder: hold the ranker fixed, widen the QUERY from the marked
span to its enclosing stanza(s) to the whole commented poem, and measure the dilution.
Two fixed rankers expose the mechanism:

  bm25    surface bag -- predicts dilution (added context floods the query)
  passim  Smith-Waterman lemma alignment (gated candidates) -- LOCAL alignment,
          predicts partial robustness to added context

STANZA PROVENANCE: the local Poetree cache has no stanza structure -- the upstream
PoeTree API serves per-line `id_stanza`, but poetree_corpus_downloader.py (neighbour
project, lines 163-174) kept only each line's text and joined with a single newline
(its `stanza_count` column is really a line count). The true stanza level here uses
the `source_stanza_lines` field of data/russian_poetry/line_pairs.jsonl: the lines of the stanza(s) containing
the fragment, derived from stanza ids re-fetched from the API for the commented poems of
this experiment's pool. A CONTEXT WINDOW level (resolved fragment
lines +-2) is kept as a stanza-independent intermediate. Eval subset: S1 connections
with resolved source lines AND a stanza map (same subset at every granularity).
Reports ALL + bands, R@k with bootstrap CIs, and the median query token length per
level (the dilution covariate).

    python scripts/granularity.py         # from the release root; needs the lemma cache

Deterministic. Writes results/russian_poetry/granularity.csv + prints the table.
"""
from __future__ import annotations
import csv, json, math, sys
from collections import defaultdict

import ngram_baseline as nb
import bm25_baseline as bm
from passim_baseline import sw
from tesserae_baseline import content_lemmas, DF_CAP, MIN_SHARED
from rank_metrics import recall_at_k, mrr, bootstrap_ci

csv.field_size_limit(10 ** 8)
KS = (1, 5, 10, 50)
LEMMAS = nb.HERE + "cache/lemmas/poetree_lemmas.csv"
LEVELS = ("fragment", "stanza", "context2", "poem")
CTX = 2


def stanza_query(commented_text: str, stanza_line_idxs: list) -> str:
    """All lines of the commented poem in the stanza(s) that contain a resolved
    fragment line (`source_stanza_lines` of data/russian_poetry/line_pairs.jsonl)."""
    lines = [ln.strip() for ln in commented_text.split("\n") if ln.strip()]
    return " ".join(lines[i] for i in stanza_line_idxs)


def context_of_lines(commented_text: str, line_idxs: list, ctx: int = CTX) -> str:
    """The resolved fragment lines expanded by +-ctx canonical (non-empty) lines of
    the commented poem — the intermediate granularity (no stanza structure exists in
    the data; see module docstring)."""
    lines = [ln.strip() for ln in commented_text.split("\n") if ln.strip()]
    keep = sorted({j for i in line_idxs for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1))})
    return " ".join(lines[j] for j in keep)


def main():
    from pymystem3 import Mystem
    print("loading corpus…", file=sys.stderr)
    corpus = nb.load_corpus()
    dups = nb.dup_groups(corpus)
    gold = {g["conn_id"]: g for g in nb.load_gold(dups)}
    postings, doclen, idf_bm, avgdl, N, _ = bm.build_bm25(corpus)

    seqs, df_lem = {}, defaultdict(int)
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        lines = [content_lemmas(ln) for ln in r["lemma_text"].split("\n")]
        seqs[r["poem_id"]] = [w for l in lines if l for w in l]
        for l in lines:
            for w in set(l):
                df_lem[w] += 1
    idf_lem = {w: math.log(len(seqs) / d) for w, d in df_lem.items()}
    inv = defaultdict(set)
    for pid, s in seqs.items():
        for w in set(s):
            inv[w].add(pid)
    mys = Mystem()

    # source-line resolutions (committed artifact) + poem texts for stanza queries
    src_lines, stanza_lines = {}, {}
    for l in open(nb.HERE + "data/russian_poetry/line_pairs.jsonl", encoding="utf-8"):
        o = json.loads(l)
        if o["stratum"] == "S1" and o["source_lines"]:
            src_lines[o["conn_id"]] = o["source_lines"]
            if o.get("source_stanza_lines"):
                stanza_lines[o["conn_id"]] = o["source_stanza_lines"]
    poem_text = {}
    for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/gold.csv", encoding="utf-8")):
        poem_text[r["conn_id"]] = r["commented_text"]

    sub = []
    for cid, g in gold.items():
        if g["stratum"] != "S1" or cid not in stanza_lines:
            continue
        sub.append(dict(g, _stanza_q=stanza_query(poem_text[cid], stanza_lines[cid])))
    print(f"eval subset: {len(sub)} S1 connections with resolved source lines "
          f"and stanza lines", file=sys.stderr)

    def bm25_rank_of(q, g):
        scores = bm.bm25_rank(q.split(), postings, doclen, idf_bm, avgdl)
        for s in g["self"]:
            scores.pop(s, None)
        return nb.best_rank(scores, g["hit"])

    def _gated_cands(qset, selfids):
        cand = defaultdict(int)
        for w in qset:
            if w in inv and len(inv[w]) <= DF_CAP:
                for pid in inv[w]:
                    cand[pid] += 1
        return [pid for pid, c in cand.items() if c >= MIN_SHARED and pid not in selfids]

    # Pure-Python SW is O(|Q|*|P|) per candidate; an epic-length poem query (>1000
    # tokens; conns 147/148/151/152 share a 190-line poem) explodes both the candidate
    # set and the per-candidate cost (observed: >2.5h on one connection). Since SW is
    # LOCAL, oversized queries are evaluated in overlapping lemma windows (60, stride
    # 30) with per-window candidate gating; a candidate's score = max over windows.
    # Queries <=150 content lemmas (93/97 connections) take the exact original path.
    def passim_rank_of(q, g):
        Q = content_lemmas("".join(mys.lemmatize(q)))
        if len(Q) <= 150:
            cands = _gated_cands(set(Q), g["self"])
            return nb.best_rank({pid: sw(Q, seqs[pid], idf_lem) for pid in cands}, g["hit"])
        best = {}
        for i in range(0, len(Q), 30):
            chunk = Q[i:i + 60]
            if len(chunk) < 5:
                break
            for pid in _gated_cands(set(chunk), g["self"]):
                s = sw(chunk, seqs[pid], idf_lem)
                if s > best.get(pid, 0.0):
                    best[pid] = s
        return nb.best_rank(best, g["hit"])

    rows, qlens = [], defaultdict(list)
    for gi, g in enumerate(sub):
        cid = g["conn_id"]
        queries = {
            "fragment": g["q_frag"],
            "stanza": nb.norm(g["_stanza_q"]),
            "context2": nb.norm(context_of_lines(poem_text[cid], src_lines[cid])),
            "poem": nb.norm(poem_text[cid]),
        }
        rec = {"conn_id": cid, "band": g["band"]}
        for lvl, q in queries.items():
            qlens[lvl].append(len(q.split()))
            rec[f"bm25_{lvl}"] = bm25_rank_of(q, g) or ""
            rec[f"passim_{lvl}"] = passim_rank_of(q, g) or ""
        rows.append(rec)
        print(f"  {gi+1}/{len(sub)} {cid} ({len(queries['poem'].split())} poem tokens)",
              file=sys.stderr)

    with open(nb.HERE + "results/russian_poetry/granularity.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import statistics
    print(f"\n### query-granularity sensitivity (S1, n={len(sub)}; "
          "same connections at every level)\n")
    print("| ranker | granularity | median q tokens | R@1 [CI] | R@10 [CI] | R@50 | MRR |")
    print("|---|---|---:|---|---|---:|---:|")
    for ranker in ("bm25", "passim"):
        for lvl in LEVELS:
            ranks = [int(r[f"{ranker}_{lvl}"]) if str(r[f"{ranker}_{lvl}"]).strip() else None
                     for r in rows]
            p1, l1, h1 = bootstrap_ci(ranks, lambda rs: recall_at_k(rs, 1), seed=0)
            p10, l10, h10 = bootstrap_ci(ranks, lambda rs: recall_at_k(rs, 10), seed=0)
            print(f"| {ranker} | {lvl} | {int(statistics.median(qlens[lvl]))} "
                  f"| {p1:.2f} [{l1:.2f},{h1:.2f}] | {p10:.2f} [{l10:.2f},{h10:.2f}] "
                  f"| {recall_at_k(ranks, 50):.2f} | {mrr(ranks):.3f} |")
    # per-band R@10 at each level (bm25) for the dilution-by-band view
    print("\n| ranker | granularity | band | n | R@10 |")
    print("|---|---|---|---:|---:|")
    for ranker in ("bm25", "passim"):
        for lvl in LEVELS:
            for b in ("verbatim", "close", "loose"):
                ranks = [int(r[f"{ranker}_{lvl}"]) if str(r[f"{ranker}_{lvl}"]).strip() else None
                         for r in rows if r["band"] == b]
                if ranks:
                    print(f"| {ranker} | {lvl} | {b} | {len(ranks)} | {recall_at_k(ranks, 10):.2f} |")
    print("\nwrote results/russian_poetry/granularity.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
