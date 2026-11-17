"""Phase 4b — hybrid first-stage retriever (Reciprocal Rank Fusion).

Rerank can't add recall (Phase 4): it only reorders the shortlist. So to give the Claude reranker
a better shortlist we fuse our complementary retrievers into one higher-recall ranking. RRF needs
no score normalization — each method contributes 1/(C + rank) per candidate:

    rrf(d) = sum over methods m of  1 / (C + rank_m(d))           # C = 60 (standard)

Fused methods (each contributes its top-N per query):
  * BM25            — deep bag recall
  * Tesserae(lemma) — inflection-proof line overlap
  * Passim(lemma SW)— best overall, local alignment
  * dense line-line — MiniLM, catches no-shared-word cases
  * wpair           — distinctive lemma-pairs (USE_WPAIR=1, default)
  * lc              — ungated idf line-containment (USE_LC=1, default): recovers
                      gate-excluded golds; lifts fused R@5 0.44->0.48, R@10 0.51->0.53
                      (close band R@10 0.68->0.74). NB: the multi-line window (lcb WIN=2)
                      is *redundant* inside RRF — fold in WIN=1 only.

Scope: the 105 fragment-bearing connections (S1/S2), matching the reranker. Reports fused
Recall@k / MRR vs each component, and writes hybrid_shortlist.csv (top-K fused ids per conn) for
the reranker to consume.

    python scripts/hybrid_baseline.py        # uses lemma cache + cached MiniLM line embeddings
"""
from __future__ import annotations
import os, sys, csv, math, hashlib
from collections import defaultdict
import numpy as np
import ngram_baseline as nb
import bm25_baseline as bm
from tesserae_baseline import content_lemmas, LEMMAS, DF_CAP, MIN_SHARED
from passim_baseline import sw
import passage_baseline as pb
import line_dense as ld
import wpair_baseline as wp
import lc_baseline as lcb

csv.field_size_limit(10**8)
TOPN = 100          # per-method depth fed into the fusion
C = 60              # RRF constant
SHORT_K = 10        # shortlist size written for the reranker
USE_WPAIR = os.environ.get("USE_WPAIR", "1") == "1"   # +wpair fusion (default)
USE_LC = os.environ.get("USE_LC", "1") == "1"         # +lc (ungated line-containment) fusion (default)
# SCOPE=whole runs the same fusion on the relations WITHOUT a query-side fragment (S3/S4: the
# query is the whole commented poem) and writes *_wholepoem.csv; the default files are untouched.
WHOLE = os.environ.get("SCOPE", "fragment") == "whole"


def topn(scores, n=TOPN):
    return [pid for pid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]   # id tie-break


def main():
    from sentence_transformers import SentenceTransformer
    print("loading corpora…", file=sys.stderr)
    # --- lexical (surface) for BM25 ---
    surf = nb.load_corpus()
    dups = nb.dup_groups(surf)
    gold_all = nb.load_gold(dups)
    gold = [g for g in gold_all if g["stratum"] in (("S3", "S4") if WHOLE else ("S1", "S2"))]
    postings, doclen, idf_bm, avgdl, Nbm, df_bm = bm.build_bm25(surf)

    # --- lemma corpus for Tesserae + Passim ---
    lem_lines, df_lem = defaultdict(list), defaultdict(int)
    seqs = {}
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
    freq_lem = {w: d for w, d in df_lem.items()}
    inv = defaultdict(set)
    for pid, s in seqs.items():
        for w in set(s):
            inv[w].add(pid)

    # --- wpair distinctive-pair index (experimental fusion add-on) ---
    print("building wpair pair index…", file=sys.stderr)
    wp_keep, wp_pidf, wp_inv = wp.build_index(seqs)

    # --- dense (cached MiniLM single-line embeddings) ---
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

    from pymystem3 import Mystem
    mys = Mystem()
    print(f"corpus ~{Nbm} | fragment connections {len(gold)}", file=sys.stderr)

    methods = ("bm25", "tess", "passim", "dense")
    methods5 = methods + ("wpair",)                 # 4-method fusion + wpair
    methodsD = methods5 + (("lc",) if USE_LC else ())   # deployable: + lc (ungated line-containment)
    per = {m: [] for m in methodsD}
    fused_ranks, fused5_ranks, fusedD_ranks, rows = [], [], [], []
    varrows = []          # per-conn variant ranks (invariance verification; output-only)
    fband = defaultdict(list)                        # band -> [(fused4_rank, fused5_rank, fusedD_rank)]
    for gi, g in enumerate(gold):
        # ---- BM25 ----
        bms = bm.bm25_rank(g["q_frag"].split(), postings, doclen, idf_bm, avgdl)
        for s in g["self"]:
            bms.pop(s, None)
        L = {"bm25": topn(bms)}

        # ---- lemma candidate set (shared >=2 rare lemmas) for Tesserae + Passim ----
        Q = content_lemmas("".join(mys.lemmatize(g["q_frag"])))
        qset = set(Q)
        cand = defaultdict(int)
        for w in qset:
            if w in inv and len(inv[w]) <= DF_CAP:
                for pid in inv[w]:
                    cand[pid] += 1
        cands = [pid for pid, c in cand.items() if c >= MIN_SHARED and pid not in g["self"]]
        tess = {}
        for pid in cands:
            best = 0.0
            for line in lem_lines[pid]:
                sh = qset & set(line)
                if len(sh) >= MIN_SHARED:
                    sc = sum(1.0 / freq_lem[w] for w in sh)
                    if sc > best:
                        best = sc
            if best > 0:
                tess[pid] = best
        L["tess"] = topn(tess)
        L["passim"] = topn({pid: sw(Q, seqs[pid], idf_lem) for pid in cands})

        # ---- dense line-to-line: recover query verse lines via the commented poem (self ids) ----
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
        order = np.argsort(-poem_max)[:TOPN]
        L["dense"] = [pids_d[i] for i in order]

        # ---- wpair distinctive-pair ranking ----
        L["wpair"] = topn(wp.score(Q, wp_keep, wp_pidf, wp_inv, g["self"]))

        # ---- lc: ungated idf line-containment (recovers gate-excluded golds) ----
        if USE_LC:
            L["lc"] = topn(lcb.lc_rank(Q, lem_lines, inv, idf_lem, g["self"]))

        # ---- per-method gold ranks (for the comparison table) ----
        for mname in methodsD:
            r = next((i for i, pid in enumerate(L[mname], 1) if pid in g["hit"]), None)
            per[mname].append(r)

        # ---- RRF fusion (4-method deployable, and +wpair experiment) ----
        def fuse(mlist):
            rrf = defaultdict(float)
            for mname in mlist:
                for rank, pid in enumerate(L[mname], 1):
                    rrf[pid] += 1.0 / (C + rank)
            return [pid for pid, _ in sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))]   # id tie-break
        fused = fuse(methods)
        fused5 = fuse(methods5)
        fusedD = fuse(methodsD)                       # deployable set (+wpair +lc)
        fr = next((i for i, pid in enumerate(fused, 1) if pid in g["hit"]), None)
        fr5 = next((i for i, pid in enumerate(fused5, 1) if pid in g["hit"]), None)
        frD = next((i for i, pid in enumerate(fusedD, 1) if pid in g["hit"]), None)
        fused_ranks.append(fr); fused5_ranks.append(fr5); fusedD_ranks.append(frD)
        if g["band"]:
            fband[g["band"]].append((fr, fr5, frD))
        deploy = fusedD                              # the shortlist written for the reranker
        rows.append({"conn_id": g["conn_id"], "band": g["band"],
                     "shortlist": ";".join(deploy[:SHORT_K]),
                     "gold_rank_fused": frD or ""})
        varrows.append({"conn_id": g["conn_id"], "stratum": g["stratum"], "band": g["band"],
                        **{f"m_{m}": per[m][-1] or "" for m in methodsD},
                        "fused4": fr or "", "fused5": fr5 or "", "fusedD": frD or ""})
        if (gi + 1) % 20 == 0:
            print(f"  {gi+1}/{len(gold)}", file=sys.stderr)

    def rk(rs, k):
        return sum(1 for r in rs if r and r <= k) / (len(rs) or 1)
    def mrr(rs):
        return sum(1 / r for r in rs if r) / (len(rs) or 1)
    print(f"\n### hybrid front-end — RRF fusion (n={len(gold)}, S1/S2)")
    print(f"  {'method':10s} R@1    R@5    R@10   R@50   MRR")
    for mname in methodsD:
        rs = per[mname]
        print(f"  {mname:10s} {rk(rs,1):.2f}   {rk(rs,5):.2f}   {rk(rs,10):.2f}   {rk(rs,50):.2f}   {mrr(rs):.3f}")
    print(f"  {'FUSED(4)':10s} {rk(fused_ranks,1):.2f}   {rk(fused_ranks,5):.2f}   "
          f"{rk(fused_ranks,10):.2f}   {rk(fused_ranks,50):.2f}   {mrr(fused_ranks):.3f}")
    print(f"  {'FUSED+wp':10s} {rk(fused5_ranks,1):.2f}   {rk(fused5_ranks,5):.2f}   "
          f"{rk(fused5_ranks,10):.2f}   {rk(fused5_ranks,50):.2f}   {mrr(fused5_ranks):.3f}")
    print(f"  {'+wp+lc*':10s} {rk(fusedD_ranks,1):.2f}   {rk(fusedD_ranks,5):.2f}   "
          f"{rk(fusedD_ranks,10):.2f}   {rk(fusedD_ranks,50):.2f}   {mrr(fusedD_ranks):.3f}   (* deployable, written)")
    print("  -- per band: FUSED(4) -> +wpair -> +wpair+lc (R@5 / R@10) --")
    for b in ("verbatim", "close", "loose"):
        rs = fband[b]
        if not rs:
            continue
        f4, f5, fd = [x[0] for x in rs], [x[1] for x in rs], [x[2] for x in rs]
        print(f"  {b:10s} n={len(rs):3d}  R@5 {rk(f4,5):.2f}->{rk(f5,5):.2f}->{rk(fd,5):.2f}   "
              f"R@10 {rk(f4,10):.2f}->{rk(f5,10):.2f}->{rk(fd,10):.2f}")

    tag = ("" if "MiniLM" in ld.MODEL else "_" + ld.MODEL.split("/")[-1]) + ("_wholepoem" if WHOLE else "")
    with open(nb.HERE + f"results/russian_poetry/hybrid_shortlist{tag}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conn_id", "band", "shortlist", "gold_rank_fused"])
        w.writeheader(); w.writerows(rows)
    with open(nb.HERE + f"results/russian_poetry/fusion_variants{tag}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(varrows[0].keys()))
        w.writeheader(); w.writerows(varrows)
    print("wrote results/russian_poetry/hybrid_shortlist.csv + fusion_variants.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
