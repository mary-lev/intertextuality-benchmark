"""The knowledge-prior ladder: none -> period(+-50) -> period(+-25) -> period(+-10) -> author.

How much of "detection" is textual signal vs the scholar's contextual prior about the
SOURCE? Each rung restricts the BM25 candidate pool by increasingly specific knowledge
of the referent (Poetree metadata), holding the ranker fixed (BM25, poem_id tiebreak —
same harness as author_oracle.py, which is the top rung).

DESIGN NOTE: a "genre/meter" rung
is NOT implementable — the Poetree extract carries no meter/genre field (only
author_name, year_created_from, line/stanza counts). The intermediate gradation is
instead the PERIOD WINDOW WIDTH around the referent poem's creation year (+-50/25/10
years) — a judgment call; canonical literary-period buckets or author birth-year
windows would be alternatives.

Eval basis: connections whose referent is a Poetree poem with BOTH author and year
metadata — the SAME subset on every rung, so rungs are comparable. Reports ALL /
strata / S1 bands per rung with median pool size, plus bootstrap CIs on R@1/R@10.

    python scripts/prior_ladder.py        # from the release root; stdlib + lemma-free

Deterministic. Writes results/russian_poetry/prior_ladder.csv + prints the table.
"""
from __future__ import annotations
import csv, statistics, sys
from collections import defaultdict

import ngram_baseline as nb
import bm25_baseline as bm
from rank_metrics import recall_at_k, mrr, bootstrap_ci

csv.field_size_limit(10 ** 8)
WINDOWS = (50, 25, 10)          # period rungs: referent year +- W
KS = (1, 5, 10, 50)


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
    all_ids = set(corpus)

    # eval subset: referent in-corpus with author AND year (same subset on every rung)
    sub = []
    for g in gold:
        hits_meta = [h for h in g["hit"] if h in author and h in year]
        if hits_meta:
            g = dict(g, ref_authors={author[h] for h in hits_meta},
                     ref_years=[year[h] for h in hits_meta])
            sub.append(g)
    print(f"eval subset: {len(sub)}/{len(gold)} connections "
          f"(referent has author+year metadata)", file=sys.stderr)

    rungs = [("none", None)] + [(f"period±{w}", w) for w in WINDOWS] + [("author", "author")]
    rows, table = [], []
    for rname, spec in rungs:
        strat, band, allr, pools = defaultdict(list), defaultdict(list), [], []
        for g in sub:
            if spec is None:
                pool = all_ids - g["self"]
            elif spec == "author":
                pool = set()
                for a in g["ref_authors"]:
                    pool |= by_author[a]
                pool -= g["self"]
            else:
                ymin = min(g["ref_years"]) - spec
                ymax = max(g["ref_years"]) + spec
                pool = {pid for pid, y in year.items() if ymin <= y <= ymax} - g["self"]
                pool |= (g["hit"] & set(author)) - g["self"]   # referent always eligible
            pools.append(len(pool))
            scores = bm.bm25_rank(g["q_frag"].split(), postings, doclen, idf, avgdl)
            scores = {pid: s for pid, s in scores.items() if pid in pool}
            order = sorted(pool, key=lambda pid: (-scores.get(pid, 0.0), pid))
            r = next((i for i, pid in enumerate(order, 1) if pid in g["hit"]), None)
            allr.append(r); strat[g["stratum"]].append(r)
            if g["band"]:
                band[g["band"]].append(r)

        def rec(cell, ranks):
            n = len(ranks)
            e = {"rung": rname, "cell": cell, "n": n,
                 "median_pool": int(statistics.median(pools)) if cell == "ALL" else "",
                 "MRR": round(mrr(ranks), 3)}
            for k in KS:
                pt, lo, hi = bootstrap_ci(ranks, lambda rs, k=k: recall_at_k(rs, k), seed=0)
                e[f"R@{k}"] = round(pt, 3)
                e[f"R@{k}_lo"] = round(lo, 3); e[f"R@{k}_hi"] = round(hi, 3)
            rows.append(e)
            return e

        e = rec("ALL", allr)
        table.append(f"| {rname} | {e['median_pool']} | {e['n']} "
                     f"| {e['R@1']:.2f} [{e['R@1_lo']:.2f},{e['R@1_hi']:.2f}] "
                     f"| {e['R@10']:.2f} [{e['R@10_lo']:.2f},{e['R@10_hi']:.2f}] "
                     f"| {e['R@50']:.2f} | {e['MRR']:.3f} |")
        for s in ("S1", "S2", "S3", "S4"):
            if strat[s]:
                rec(s, strat[s])
        for b in ("verbatim", "close", "loose"):
            if band[b]:
                rec(f"S1/{b}", band[b])

    with open(nb.HERE + "results/russian_poetry/prior_ladder.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n### knowledge-prior ladder (BM25 within pool; "
          f"n={len(sub)} referents with author+year)\n")
    print("| rung | median pool | n | R@1 [CI] | R@10 [CI] | R@50 | MRR |")
    print("|---|---:|---:|---|---|---:|---:|")
    print("\n".join(table))
    print("\nwrote results/russian_poetry/prior_ladder.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
