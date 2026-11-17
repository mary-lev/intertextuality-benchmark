"""The paper's factorial: signal-class x intertext-type detectability profiles.

One CANONICAL probe per signal class (wiki/paper-chr2027-plan.md "The experimental
design"), run over ALL 158 grounded poetry connections (S1-S4; S3/S4 queries fall back
to the whole commented poem, per ngram_baseline.load_gold):

  signal class              probe        implementation
  exact surface overlap     ngram        word-2gram idf overlap (ngram_baseline)
  lemmatized alignment      passim       Smith-Waterman over Mystem lemmas, gated
                                         candidates (DF_CAP/MIN_SHARED, passim_baseline.sw)
  rarity-weighted bag       bm25         BM25 over surface tokens (bm25_baseline)
  order-free collocation    wpair        distinctive unordered lemma pairs (wpair_baseline)
  distributional semantics  dense        MiniLM line-to-line max-sim (line_dense cache)
  LLM judgment              rerank       claude-opus-4-8 over the fused top-10
                                         (hybrid_rerank_results.csv; end-to-end rank,
                                         None when the gold missed the shortlist;
                                         S1/S2 fragment pool only)

Ranks are computed from FULL score dicts via ngram_baseline.best_rank (poem_id
tiebreak -> deterministic; no top-N truncation, so R@50 is exact). Cells = stratum
(S1/S2/S3/S4) x S1 closeness band (verbatim/close/loose); out-of-pool categories are
not cells (P1). Metrics per cell: n, R@1/5/10/50 + MRR, with 95% bootstrap CIs
(rank_metrics.bootstrap_ci, seed 0).

Outputs (results/russian_poetry/):
  factorial_ranks.csv    one row per connection: stratum, band, per-probe gold rank
  factorial_profile.csv  one row per probe x cell x metric with CI bounds
  factorial_profile.md   the paper-facing table

    python scripts/factorial.py          # from the release root; needs lemma + MiniLM caches

    CHRONO=1 python scripts/factorial.py # the five computed probes with the chronology
                                           # filter (ngram_baseline.not_later); writes only
                                           # factorial_ranks_chrono.csv

Deterministic; no API calls (the rerank column is read from the persisted run).
"""
from __future__ import annotations
import csv, math, os, sys
from collections import defaultdict

import numpy as np

import ngram_baseline as nb
import bm25_baseline as bm
import wpair_baseline as wp
import line_dense as ld
import passage_baseline as pb
from passim_baseline import sw
from tesserae_baseline import content_lemmas, DF_CAP, MIN_SHARED
from rank_metrics import recall_at_k, mrr, bootstrap_ci

csv.field_size_limit(10 ** 8)
PROBES = ("ngram", "passim", "bm25", "wpair", "dense", "rerank")
KS = (1, 5, 10, 50)
LEMMAS = nb.HERE + "cache/lemmas/poetree_lemmas.csv"
CHRONO = os.environ.get("CHRONO", "0") == "1"


def main():
    from sentence_transformers import SentenceTransformer
    from pymystem3 import Mystem
    import hashlib, os

    print("loading corpora…", file=sys.stderr)
    surf = nb.load_corpus()
    dups = nb.dup_groups(surf)
    gold = nb.load_gold(dups)

    # --- ngram (word-2gram) index ---
    ng_index, ng_idf = nb.build_index(surf, lambda t: nb.word_ngrams(t, 2))
    # --- BM25 ---
    postings, doclen, idf_bm, avgdl, Nbm, _ = bm.build_bm25(surf)
    # --- lemma corpus (Passim + wpair) ---
    seqs, lem_lines, df_lem = {}, {}, defaultdict(int)
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        lines = [content_lemmas(ln) for ln in r["lemma_text"].split("\n")]
        lines = [l for l in lines if l]
        lem_lines[r["poem_id"]] = lines
        seqs[r["poem_id"]] = [w for l in lines for w in l]
        for l in lines:
            for w in set(l):
                df_lem[w] += 1
    Nlem = len(seqs)
    idf_lem = {w: math.log(Nlem / d) for w, d in df_lem.items()}
    inv = defaultdict(set)
    for pid, s in seqs.items():
        for w in set(s):
            inv[w].add(pid)
    print("building wpair index…", file=sys.stderr)
    wp_keep, wp_pidf, wp_inv = wp.build_index(seqs)

    # --- dense (cached MiniLM single-line embeddings; identical to hybrid_baseline) ---
    print("loading dense line cache + model…", file=sys.stderr)
    lines_by_id = pb.corpus_lines()
    pids_d, starts_d, dtexts = [], [], []
    for pid, lns in lines_by_id.items():
        starts_d.append(len(dtexts)); pids_d.append(pid)
        dtexts.extend(lns)
    starts_d = np.array(starts_d, dtype=np.int64)
    sig = hashlib.md5((ld.MODEL + "|line1|" + str(len(dtexts)) + "|" +
                       hashlib.md5("".join(dtexts[::997]).encode()).hexdigest()).encode()).hexdigest()[:16]
    vpath = ld.CACHE + f"line1_{sig}.npy"
    if not os.path.exists(vpath):
        sys.exit("MiniLM line cache missing — run line_dense.py first")
    dvecs = np.asarray(np.load(vpath, mmap_mode="r"), dtype=np.float32)
    pidx_d = {p: i for i, p in enumerate(pids_d)}
    model = SentenceTransformer(ld.MODEL)
    cpos = ld.commented_poem_lines()
    mys = Mystem()

    # --- rerank ranks from the persisted paid run (end-to-end: None if not in shortlist) ---
    rr = {}
    try:
        for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/hybrid_rerank_results.csv", encoding="utf-8")):
            v = (r["rerank_rank"] or "").strip()
            rr[r["conn_id"]] = int(v) if v.isdigit() else None
    except FileNotFoundError:
        print("  [warn] hybrid_rerank_results.csv missing — rerank column empty", file=sys.stderr)

    print(f"scoring {len(gold)} connections x {len(PROBES)} probes…", file=sys.stderr)
    rows = []
    years = nb.load_years() if CHRONO else {}
    for gi, g in enumerate(gold):
        ranks = {}
        ok = nb.not_later(years, g["self"])                    # always true without CHRONO
        keep = (lambda sc: {p: v for p, v in sc.items() if ok(p)}) if CHRONO else (lambda sc: sc)
        # ngram
        scores = nb.rank(nb.word_ngrams(g["q_frag"], 2), ng_index, ng_idf)
        for s in g["self"]:
            scores.pop(s, None)
        ranks["ngram"] = nb.best_rank(keep(scores), g["hit"])
        # bm25
        bms = bm.bm25_rank(g["q_frag"].split(), postings, doclen, idf_bm, avgdl)
        for s in g["self"]:
            bms.pop(s, None)
        ranks["bm25"] = nb.best_rank(keep(bms), g["hit"])
        # passim (gated candidates, SW over lemma seqs)
        Q = content_lemmas("".join(mys.lemmatize(g["q_frag"])))
        qset = set(Q)
        cand = defaultdict(int)
        for w in qset:
            if w in inv and len(inv[w]) <= DF_CAP:
                for pid in inv[w]:
                    cand[pid] += 1
        cands = [pid for pid, c in cand.items() if c >= MIN_SHARED and pid not in g["self"] and ok(pid)]
        ranks["passim"] = nb.best_rank({pid: sw(Q, seqs[pid], idf_lem) for pid in cands}, g["hit"])
        # wpair
        ranks["wpair"] = nb.best_rank(keep(wp.score(Q, wp_keep, wp_pidf, wp_inv, g["self"])), g["hit"])
        # dense (line-to-line max-sim; recover verse lines of the fragment when possible)
        frag_lines = None
        for sid in g["self"]:
            if sid in cpos:
                rec = ld.recover_query_lines(g["q_frag"], cpos[sid])
                if rec:
                    frag_lines = rec
                    break
        if frag_lines is None:
            frag_lines = [g["q_frag"]]
        Qv = model.encode([ld.QPFX + l for l in frag_lines], normalize_embeddings=True,
                          convert_to_numpy=True).astype(np.float32)
        sims = Qv @ dvecs.T
        poem_max = np.maximum.reduceat(sims, starts_d, axis=1).max(axis=0)
        for s in g["self"]:
            if s in pidx_d:
                poem_max[pidx_d[s]] = -1e9
        ranks["dense"] = nb.best_rank(
            keep({pids_d[i]: float(poem_max[i]) for i in range(len(pids_d)) if poem_max[i] > -1e8}),
            g["hit"])
        # rerank (persisted; S1/S2 fragment pool only)
        ranks["rerank"] = rr.get(g["conn_id"])
        rows.append({"conn_id": g["conn_id"], "stratum": g["stratum"], "band": g["band"],
                     "in_rerank_pool": int(g["conn_id"] in rr),
                     **{p: (ranks[p] if ranks[p] is not None else "") for p in PROBES}})
        if (gi + 1) % 20 == 0:
            print(f"  {gi+1}/{len(gold)}", file=sys.stderr)

    if CHRONO:        # separate file; the profile files of the paper are not rewritten
        with open(nb.HERE + "results/russian_poetry/factorial_ranks_chrono.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["conn_id", "stratum", "band", *PROBES[:-1]])
            w.writeheader()
            w.writerows({k: r[k] for k in w.fieldnames} for r in rows)
        print("wrote factorial_ranks_chrono.csv", file=sys.stderr)
        return
    with open(nb.HERE + "results/russian_poetry/factorial_ranks.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conn_id", "stratum", "band", "in_rerank_pool", *PROBES])
        w.writeheader(); w.writerows(rows)

    # ---- aggregate: probe x cell -> metrics + CIs ----
    def cell_rows(sel):
        return [r for r in rows if sel(r)]

    cells = [("ALL", lambda r: True)]
    cells += [(s, lambda r, s=s: r["stratum"] == s) for s in ("S1", "S2", "S3", "S4")]
    cells += [(f"S1/{b}", lambda r, b=b: r["band"] == b) for b in ("verbatim", "close", "loose")]

    prof, md = [], []
    md.append("| probe | cell | n | R@1 [CI] | R@5 | R@10 [CI] | R@50 [CI] | MRR |")
    md.append("|---|---|---:|---|---:|---|---|---:|")
    for probe in PROBES:
        for cname, sel in cells:
            sub = cell_rows(sel)
            if probe == "rerank":
                sub = [r for r in sub if r["in_rerank_pool"]]
            if not sub:
                continue
            ranks = [int(r[probe]) if str(r[probe]).strip() else None for r in sub]
            n = len(ranks)
            entry = {"probe": probe, "cell": cname, "n": n, "MRR": round(mrr(ranks), 3)}
            mdvals = {}
            for k in KS:
                point, lo, hi = bootstrap_ci(ranks, lambda rs, k=k: recall_at_k(rs, k), seed=0)
                entry[f"R@{k}"] = round(point, 3)
                entry[f"R@{k}_lo"] = round(lo, 3)
                entry[f"R@{k}_hi"] = round(hi, 3)
                mdvals[k] = (point, lo, hi)
            prof.append(entry)
            md.append(f"| {probe} | {cname} | {n} "
                      f"| {mdvals[1][0]:.2f} [{mdvals[1][1]:.2f},{mdvals[1][2]:.2f}] "
                      f"| {mdvals[5][0]:.2f} "
                      f"| {mdvals[10][0]:.2f} [{mdvals[10][1]:.2f},{mdvals[10][2]:.2f}] "
                      f"| {mdvals[50][0]:.2f} [{mdvals[50][1]:.2f},{mdvals[50][2]:.2f}] "
                      f"| {entry['MRR']:.3f} |")

    with open(nb.HERE + "results/russian_poetry/factorial_profile.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prof[0].keys()))
        w.writeheader(); w.writerows(prof)
    with open(nb.HERE + "results/russian_poetry/factorial_profile.md", "w", encoding="utf-8") as f:
        f.write("# Factorial: signal-class x type detectability profile\n\n"
                "Canonical probes over the 158 grounded poetry connections; ranks in\n"
                "`factorial_ranks.csv`; 95% percentile bootstrap CIs (seed 0). The rerank\n"
                "probe is end-to-end over the fused top-10 shortlist (S1/S2 pool only).\n\n")
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote factorial_ranks.csv, factorial_profile.csv, factorial_profile.md",
          file=sys.stderr)


if __name__ == "__main__":
    main()
