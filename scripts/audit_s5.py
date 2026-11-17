"""Pre-freeze audit for paper §5 (reviewer request, 2026-08-15): one reproducible
script that recomputes the cohort accounting, corpus counts, control-seed summary,
which relations each probe finds, and the rerank-vs-author-baseline comparison on a
MATCHED population. Prints; changes nothing.

    python scripts/audit_s5.py     # from the release root
"""
from __future__ import annotations
import csv
from collections import defaultdict

import ngram_baseline as nb
import bm25_baseline as bm
from rank_metrics import recall_at_k

csv.field_size_limit(10 ** 8)


def main():
    print("== 1. cohort accounting (from data, not prose) ==")
    corpus = nb.load_corpus()
    dups = nb.dup_groups(corpus)
    gold = nb.load_gold(dups)
    n_tot = len(gold)
    strat = defaultdict(list)
    for g in gold:
        strat[g["stratum"]].append(g)
    print(f"  evaluable connections: {n_tot}")
    print(f"  stratified profile: S1 {len(strat['S1'])} + S4 {len(strat['S4'])} = "
          f"{len(strat['S1']) + len(strat['S4'])}  (excluded: S2 {len(strat['S2'])} "
          f"fragment-only + S3 {len(strat['S3'])} citation-only = "
          f"{len(strat['S2']) + len(strat['S3'])})")
    author, year = {}, {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        pid = p["poetree_poem_id"]
        if pid in corpus:
            author[pid] = (p["author_name"] or "").strip()
            try:
                year[pid] = int(float(p["year_created_from"]))
            except ValueError:
                pass
    ladder = [g for g in gold if any(h in author and h in year for h in g["hit"])]
    print(f"  prior-restriction cohort: {len(ladder)}  (excluded "
          f"{n_tot - len(ladder)}: referent lacks author and/or year metadata)")
    src = {r["conn_id"] for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/gold.csv", encoding="utf-8"))
           if r["commented_stanza_lines"]}
    pid1 = {r["conn_id"]: (r["commented_text_id"] or "").split(";")[0]
            for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/gold.csv", encoding="utf-8"))}
    gran, gran_excl = [], []
    for g in strat["S1"]:
        cid = g["conn_id"]
        if cid in src:
            gran.append(cid)
        else:
            gran_excl.append((cid, pid1.get(cid, "")))
    print(f"  query-context cohort: {len(gran)} of S1 {len(strat['S1'])}  "
          f"(excluded {len(gran_excl)}: {[p for _, p in gran_excl]})")

    print("\n== 2. corpus counts ==")
    print(f"  corpus poems (post-exclusion): {len(corpus)}  "
          f"(expect 45,452 incl. extra-* supplements? ->", len(corpus), ")")
    g0 = ladder[0]
    print(f"  unrestricted candidates for a sample conn (minus self): "
          f"{len(set(corpus)) - len(g0['self'] & set(corpus))}")

    print("\n== 3. random-control aggregation (from prior_ladder_control.csv) ==")
    for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/prior_ladder_control.csv", encoding="utf-8")):
        print(f"  {r['rung']:10s} informed {r['informed_R@10']}  null mean "
              f"{r['null_R@10_mean']} range [{r['null_R@10_min']},{r['null_R@10_max']}]"
              f"  gap {r['gap_R@10']}   (5 seeded pools per query x rung)")
        print(f"    exact expectation {r['exact_R@10']}  gap {r['exact_gap_R@10']};  competitors above chance in "
              f"{r['comp_more']}, below in {r['comp_fewer']} of {r['comp_n']} (median ratio {r['comp_median_ratio']})")

    print("\n== 4. which relations each probe finds (S1, top ten) ==")
    def top10(v):
        v = (v or "").strip()
        return v not in ("", "NA", "None") and float(v) <= 10
    fr = {r["conn_id"]: r for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/factorial_ranks.csv", encoding="utf-8"))}
    e5 = {r["conn_id"]: r["dense_e5"] for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/dense_e5_ranks.csv", encoding="utf-8"))}
    probes = ["ngram", "passim", "bm25", "wpair", "dense"]
    s1 = [k for k, r in fr.items() if r["stratum"] == "S1"]
    found = {k: {p for p in probes if top10(fr[k][p])} | ({"e5"} if top10(e5.get(k)) else set()) for k in s1}
    loose = [k for k in s1 if fr[k]["band"] == "loose"]
    single = defaultdict(int)
    for v in found.values():
        if len(v) == 1:
            single[next(iter(v))] += 1
    print(f"  alignment {sum('passim' in v for v in found.values())}; at least one probe, E5 included, "
          f"{sum(bool(v) for v in found.values())} of {len(s1)}")
    print(f"  loose: union {sum(bool(found[k]) for k in loose)} vs alignment "
          f"{sum('passim' in found[k] for k in loose)} of {len(loose)}")
    print(f"  found by one probe only: {sum(single.values())} {dict(single)}")
    sl = {r["conn_id"]: r for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/hybrid_shortlist.csv", encoding="utf-8"))}
    print(f"  fusion top ten: {sum(1 for k in s1 if k in sl and top10(sl[k]['gold_rank_fused']))}")
    print(f"  loose, E5 not BM25 / BM25 not E5: {sum(1 for k in loose if top10(e5[k]) and not top10(fr[k]['bm25']))} / "
          f"{sum(1 for k in loose if top10(fr[k]['bm25']) and not top10(e5[k]))}")
    prof = [k for k, r in fr.items() if r["stratum"] in ("S1", "S4")]
    b = sum(1 for k in prof if top10(e5[k]) and not top10(fr[k]["dense"]))
    c = sum(1 for k in prof if top10(fr[k]["dense"]) and not top10(e5[k]))
    from math import comb
    p = min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))
    print(f"  E5 not MiniLM / MiniLM not E5 (profile, n={len(prof)}): {b} / {c}, exact McNemar p={p:.3f}")
    lx = {r["conn_id"]: r["wording_class"] for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/lexical_evidence.csv", encoding="utf-8"))}
    one = [k for k, w in lx.items() if w == "loose"]   # the paper's "at most one shared content word"
    print(f"  at most one shared content word: {len(one)}; in a top ten: "
          f"{ {p: sum(1 for k in one if p in found.get(k, set())) for p in probes + ['e5']} }, fusion "
          f"{sum(1 for k in one if k in sl and top10(sl[k]['gold_rank_fused']))}")

    print("\n== 5. rerank vs author-baseline on a MATCHED population ==")
    rr = {}
    for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/hybrid_rerank_results.csv", encoding="utf-8")):
        v = (r["rerank_rank"] or "").strip()
        rr[r["conn_id"]] = int(v) if v.isdigit() else None
    postings, doclen, idf, avgdl, N, _ = bm.build_bm25(corpus)
    by_author = defaultdict(set)
    for pid, a in author.items():
        by_author[a].add(pid)
    matched = [g for g in ladder if g["stratum"] == "S1" and g["conn_id"] in rr]
    orc, rrk = [], []
    for g in matched:
        pool = set()
        for h in g["hit"]:
            if h in author:
                pool |= by_author[author[h]]
        pool -= g["self"]
        scores = bm.bm25_rank(g["q_frag"].split(), postings, doclen, idf, avgdl)
        sc = {pid: s for pid, s in scores.items() if pid in pool}
        order = sorted(pool, key=lambda pid: (-sc.get(pid, 0.0), pid))
        orc.append(next((i for i, pid in enumerate(order, 1) if pid in g["hit"]), None))
        rrk.append(rr[g["conn_id"]])
    print(f"  matched population: S1 & author-metadata & rerank-pool  n={len(matched)}")
    print(f"  author-baseline (BM25 in oeuvre): R@1 {recall_at_k(orc, 1):.3f}  "
          f"R@10 {recall_at_k(orc, 10):.3f}")
    print(f"  fused->rerank pipeline:           R@1 {recall_at_k(rrk, 1):.3f}  "
          f"R@10 {recall_at_k(rrk, 10):.3f}")


if __name__ == "__main__":
    main()
