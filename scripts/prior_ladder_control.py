"""Cardinality-matched control for the knowledge-prior ladder (reviewer fix, 2026-08-14).

The ladder's rungs shrink the candidate pool as well as inform it — so monotone gains
could be generic search-space reduction rather than historical information. Control:
for every connection and every informed rung, draw a RANDOM pool of exactly the same
size that (a) contains the connection's in-corpus gold hits and (b) excludes self,
rank with the same BM25 harness, and compare informed vs size-matched null.

5 seeded draws per connection per rung; we report the null's mean R@k across seeds
(per-seed values to show spread) beside the informed value. The informed-minus-null
gap is the information content of the prior.

    python scripts/prior_ladder_control.py   # from the release root; CPU

Exact control (2026-11-12). BM25 scores are fixed per query, so the random pool needs no
sampling: with M eligible non-target candidates, B of them ahead of the target in the
full ordering, and n non-target slots in the pool, the number of survivors ahead of the
target is Hypergeometric(M, B, n), and P(rank <= k) = P(X <= k-1). Columns exact_R@k give
this expectation; comp_* compare the competitors that survive the informed restriction
with that expectation (queries with B > 0 only). A relation whose referent is the
query's own poem (conn 222) is unrecoverable in both arms.

Deterministic (seeds 0-4). Writes results/russian_poetry/prior_ladder_control.csv.
"""
from __future__ import annotations
import csv, random, statistics, sys
from collections import defaultdict

import ngram_baseline as nb
import bm25_baseline as bm
from rank_metrics import recall_at_k, mrr
from scipy.stats import hypergeom

csv.field_size_limit(10 ** 8)
WINDOWS = (50, 25, 10)
SEEDS = (0, 1, 2, 3, 4)
KS = (1, 10, 50)


def main():
    print("loading corpus…", file=sys.stderr)
    corpus = nb.load_corpus()
    dups = nb.dup_groups(corpus)
    gold = nb.load_gold(dups)
    postings, doclen, idf, avgdl, N, _ = bm.build_bm25(corpus)

    author, year = {}, {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        pid = p["poetree_poem_id"]
        if pid not in corpus:
            continue
        author[pid] = (p["author_name"] or "").strip()
        y = (p["year_created_from"] or "").strip()
        try:
            year[pid] = int(float(y))
        except ValueError:
            pass
    by_author = defaultdict(set)
    for pid, a in author.items():
        by_author[a].add(pid)
    all_ids = sorted(corpus)          # sorted -> deterministic sampling

    sub = []
    for g in gold:
        hits_meta = [h for h in g["hit"] if h in author and h in year]
        if hits_meta:
            sub.append(dict(g, ref_authors={author[h] for h in hits_meta},
                            ref_years=[year[h] for h in hits_meta],
                            hits_in={h for h in g["hit"] if h in corpus}))
    print(f"eval subset: {len(sub)} (same as prior_ladder.py)", file=sys.stderr)

    def informed_pool(g, spec):
        if spec == "author":
            pool = set()
            for a in g["ref_authors"]:
                pool |= by_author[a]
            return pool - g["self"]
        ymin, ymax = min(g["ref_years"]) - spec, max(g["ref_years"]) + spec
        pool = {pid for pid, y in year.items() if ymin <= y <= ymax} - g["self"]
        return pool | (g["hits_in"] - g["self"])

    def rank_in_pool(g, pool, scores):
        sc = {pid: s for pid, s in scores.items() if pid in pool}
        order = sorted(pool, key=lambda pid: (-sc.get(pid, 0.0), pid))
        return next((i for i, pid in enumerate(order, 1) if pid in g["hit"]), None)

    rungs = [(f"period±{w}", w) for w in WINDOWS] + [("author", "author")]
    rows = []
    # precompute BM25 scores once per connection (pool filtering happens after)
    for rname, spec in rungs:
        inf_ranks, null_ranks = [], {s: [] for s in SEEDS}
        exact = {k: [] for k in KS}
        comp = []                     # (survivors ahead of target, expectation) per query, B > 0
        for gi, g in enumerate(sub):
            scores = bm.bm25_rank(g["q_frag"].split(), postings, doclen, idf, avgdl)
            pool = informed_pool(g, spec)
            n_pool = len(pool)
            inf_ranks.append(rank_in_pool(g, pool, scores))
            must = g["hits_in"] - g["self"]
            eligible = [pid for pid in all_ids if pid not in g["self"] and pid not in must]
            n_other = max(0, n_pool - len(must))
            if must:
                key = lambda pid: (-scores.get(pid, 0.0), pid)
                tkey = min(key(pid) for pid in must)
                above = {pid for pid in eligible if key(pid) < tkey}
                for k in KS:
                    exact[k].append(float(hypergeom.cdf(k - 1, len(eligible), len(above), n_other)))
                if above:
                    comp.append((len(above & pool), len(above) * n_other / len(eligible)))
            else:
                for k in KS:
                    exact[k].append(0.0)
            for s in SEEDS:
                rng = random.Random(f"{rname}:{g['conn_id']}:{s}")
                k = max(0, n_pool - len(must))
                rnd = set(rng.sample(eligible, min(k, len(eligible)))) | must
                null_ranks[s].append(rank_in_pool(g, rnd, scores))
        row = {"rung": rname, "n": len(sub)}
        for k in KS:
            row[f"informed_R@{k}"] = round(recall_at_k(inf_ranks, k), 3)
            per_seed = [recall_at_k(null_ranks[s], k) for s in SEEDS]
            row[f"null_R@{k}_mean"] = round(statistics.mean(per_seed), 3)
            row[f"null_R@{k}_min"] = round(min(per_seed), 3)
            row[f"null_R@{k}_max"] = round(max(per_seed), 3)
            row[f"gap_R@{k}"] = round(row[f"informed_R@{k}"] - row[f"null_R@{k}_mean"], 3)
            row[f"exact_R@{k}"] = round(statistics.mean(exact[k]), 3)
            row[f"exact_gap_R@{k}"] = round(recall_at_k(inf_ranks, k) - statistics.mean(exact[k]), 3)
        row["informed_MRR"] = round(mrr(inf_ranks), 3)
        row["null_MRR_mean"] = round(statistics.mean([mrr(null_ranks[s]) for s in SEEDS]), 3)
        row["comp_n"] = len(comp)
        row["comp_more"] = sum(x > e for x, e in comp)
        row["comp_fewer"] = sum(x < e for x, e in comp)
        row["comp_median_ratio"] = round(statistics.median(x / e for x, e in comp if e > 0), 2)
        rows.append(row)
        print(f"  {rname}: exact null R@1/10/50 {[row[f'exact_R@{k}'] for k in KS]}; competitors "
              f"above chance in {row['comp_more']}, below in {row['comp_fewer']} of {row['comp_n']}; "
              f"median ratio {row['comp_median_ratio']}", file=sys.stderr)
        print(f"  {rname}: informed R@10 {row['informed_R@10']} vs null "
              f"{row['null_R@10_mean']} [{row['null_R@10_min']},{row['null_R@10_max']}] "
              f"-> gap {row['gap_R@10']}", file=sys.stderr)

    with open(nb.HERE + "results/russian_poetry/prior_ladder_control.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\n| rung | informed R@10 | null R@10 (mean [min,max]) | gap |")
    print("|---|---:|---|---:|")
    for r in rows:
        print(f"| {r['rung']} | {r['informed_R@10']} | {r['null_R@10_mean']} "
              f"[{r['null_R@10_min']},{r['null_R@10_max']}] | {r['gap_R@10']} |")
    print("wrote results/russian_poetry/prior_ladder_control.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
