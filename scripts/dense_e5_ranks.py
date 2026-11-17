"""The dense probe with multilingual-e5-base on ALL scored Russian relations.

line_dense.py covers only the relations with a query-side fragment (S1/S2). This script
applies the factorial's dense rule (factorial.py: one vector per verse line, poem score =
best query line x poem line, the commented poem excluded) with the E5 encoder to every
relation the factorial scores, so that the whole-poem relations (S3/S4) get an E5 rank too.
Needs the E5 single-line cache written by `EMB_MODEL=intfloat/multilingual-e5-base
python scripts/line_dense.py`.

    python scripts/dense_e5_ranks.py     # from the release root; CPU, minutes

Writes results/russian_poetry/dense_e5_ranks.csv (conn_id, stratum, band, dense_e5).
"""
from __future__ import annotations
import csv, hashlib, os, sys

os.environ["EMB_MODEL"] = "intfloat/multilingual-e5-base"   # read by line_dense at import

import numpy as np
import ngram_baseline as nb
import passage_baseline as pb
import line_dense as ld
from rank_metrics import recall_at_k

QBATCH = 32      # query lines per matrix product (a whole poem can have hundreds)


def main():
    from sentence_transformers import SentenceTransformer
    corpus = nb.load_corpus()
    gold = nb.load_gold(nb.dup_groups(corpus))

    lines_by_id = pb.corpus_lines()
    pids, starts, texts = [], [], []
    for pid, lns in lines_by_id.items():
        starts.append(len(texts)); pids.append(pid)
        texts.extend(lns)
    starts = np.array(starts, dtype=np.int64)
    sig = hashlib.md5((ld.MODEL + "|line1|" + str(len(texts)) + "|" +
                       hashlib.md5("".join(texts[::997]).encode()).hexdigest()).encode()).hexdigest()[:16]
    vpath = ld.CACHE + f"line1_{sig}.npy"
    if not os.path.exists(vpath):
        sys.exit("E5 line cache missing — run line_dense.py with EMB_MODEL set first")
    vecs = np.asarray(np.load(vpath, mmap_mode="r"), dtype=np.float32)
    pidx = {p: i for i, p in enumerate(pids)}
    model = SentenceTransformer(ld.MODEL)
    cpos = ld.commented_poem_lines()

    rows = []
    for gi, g in enumerate(gold):
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
        poem_max = np.full(len(pids), -np.inf, dtype=np.float32)
        for a in range(0, len(Qv), QBATCH):
            sims = Qv[a:a + QBATCH] @ vecs.T
            poem_max = np.maximum(poem_max, np.maximum.reduceat(sims, starts, axis=1).max(axis=0))
        for s in g["self"]:
            if s in pidx:
                poem_max[pidx[s]] = -1e9
        rank = nb.best_rank({pids[i]: float(poem_max[i]) for i in range(len(pids))
                             if poem_max[i] > -1e8}, g["hit"])
        rows.append({"conn_id": g["conn_id"], "stratum": g["stratum"], "band": g["band"],
                     "n_query_lines": len(frag_lines), "dense_e5": rank or ""})
        if (gi + 1) % 20 == 0:
            print(f"  {gi+1}/{len(gold)}", file=sys.stderr)

    with open(nb.HERE + "results/russian_poetry/dense_e5_ranks.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for s in ("S1", "S2", "S3", "S4"):
        rk = [r["dense_e5"] or None for r in rows if r["stratum"] == s]
        if rk:
            print(f"{s}  n={len(rk):3d}  R@1 {recall_at_k(rk, 1):.2f}  R@10 {recall_at_k(rk, 10):.2f}  "
                  f"R@50 {recall_at_k(rk, 50):.2f}")
    print("wrote results/russian_poetry/dense_e5_ranks.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
