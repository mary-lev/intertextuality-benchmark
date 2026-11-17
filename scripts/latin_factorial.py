"""External check: the signal-class probes on a third-party gold of Latin epic parallels.

Valerius Flaccus, Argonautica 1 -> Vergil (Aeneid), Lucan (Bellum Civile), Ovid
(Metamorphoses), Statius (Thebaid): the 945 parallels of Burns et al. 2021, recorded from
three commentaries (Kleywegt, Zissos, Spaltenstein). This gold was not built by us and not
under the paper's construction principles; the run was added after the Russian and Dante
results were seen. Data: fetch_latin_epic.py.

Mirrors dante_factorial.py. Differences, all forced by the material:
  unit        one row per recorded parallel
  query       the Valerius Flaccus verse line(s) that hold the commentary lemma
  candidates  39,675 single verse lines; a parallel is recovered at rank k when one of its
              target lines (or a text-identical line) is among the first k
  tokens      surface forms throughout — no Latin lemmatizer in this stack
  wpair       df caps scaled from the Russian ones to the number of lines (pair 87,
              lemma 7000 of 39,675 — judgment call, as for Dante)
  dense       three encoders, one cosine per (query, line), the query encoded as one string:
              dense        MiniLM (the Russian probe; its card does not list Latin)
              dense_e5     multilingual-e5-base (lists Latin)
              dense_sphil  bowphs/SPhilBerta (Riemenschneider & Frank 2023; trained on
                           Latin and Ancient Greek)
  bands       Russian thresholds (0.85 / 0.5) on the SequenceMatcher ratio between the query
              line(s) and the target line(s) — line to line: the upstream phrase columns are
              partly lemmatized and are not used

    python scripts/latin_factorial.py     # from the release root; CPU, ~15 minutes

Deterministic. Writes results/latin_epic/latin_factorial_ranks.csv + prints the profile. Intervals
resample queries (the parallels of one query are not independent).
"""
from __future__ import annotations
import csv, math, os, random, sys
from collections import defaultdict
from difflib import SequenceMatcher
from multiprocessing import Pool

import ngram_baseline as nb
import bm25_baseline as bm
import wpair_baseline as wp
from passim_baseline import sw
from rank_metrics import recall_at_k, mrr

SRC = nb.HERE + "data/latin_epic/"
PROBES = ("ngram", "char4", "sw", "bm25", "wpair", "dense", "dense_e5", "dense_sphil")
# probe -> (checkpoint, query prefix, passage prefix)
ENCODERS = {"dense": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
            "dense_e5": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
            "dense_sphil": ("bowphs/SPhilBerta", "", "")}
COMMENTARIES = ("Kleywegt Ref.", "Zissos Ref.", "Spaltenstein Ref.")
WPAIR_PAIR_CAP, WPAIR_LEMMA_CAP, WPAIR_W = 87, 7000, 4

_SEQS, _IDF = [], {}


def _sw_one(Q):
    return [sw(Q, P, _IDF) for P in _SEQS]


def load():
    """-> (corpus id -> line, query id -> text, relations)."""
    import pandas as pd
    c = pd.read_parquet(SRC + "corpus.parquet")
    q = pd.read_parquet(SRC + "queries.parquet")
    corpus, qtext = dict(zip(c["id"], c["text"])), dict(zip(q["id"], q["text"]))
    norm_corpus = {cid: nb.norm(t) for cid, t in corpus.items()}
    dups = nb.dup_groups(norm_corpus)
    rels = []
    for i, r in enumerate(csv.DictReader(open(SRC + "vf_intertext_dataset_1_0.csv",
                                              encoding="utf-8-sig")), 1):
        qid = f'vf-1-{r["VF: Line Start"]}-{r["VF: Line End"]}'
        key = r["Intertext: Author"].strip().lower()
        ids = [f'{key}-{r["Intertext: Book"]}-{n}'
               for n in range(int(r["Intertext: Line Start"]), int(r["Intertext: Line End"]) + 1)]
        if qid not in qtext or any(x not in corpus for x in ids):
            sys.exit(f"row {i}: query or target line not in the packaged texts")
        hit = set()
        for x in ids:
            hit |= dups[x]
        ratio = SequenceMatcher(None, nb.norm(qtext[qid]),
                                " ".join(norm_corpus[x] for x in ids)).ratio()
        rels.append({"conn_id": i, "query_id": qid, "referent_id": ";".join(ids),
                     "author": r["Intertext: Author"].strip(),
                     "n_commentaries": sum(bool(r[k].strip()) for k in COMMENTARIES),
                     "band": "verbatim" if ratio >= .85 else "close" if ratio >= .5 else "loose",
                     "_hit": hit})
    return corpus, norm_corpus, qtext, rels


def main():
    global _SEQS, _IDF
    import numpy as np
    from sentence_transformers import SentenceTransformer

    corpus, norm_corpus, qtext, rels = load()
    ids = list(norm_corpus)
    uq = sorted({r["query_id"] for r in rels})
    qn = {u: nb.norm(qtext[u]) for u in uq}
    print(f"{len(rels)} parallels / {len(uq)} queries / {len(ids)} lines", file=sys.stderr)

    ng_index, ng_idf = nb.build_index(norm_corpus, lambda t: nb.word_ngrams(t, 2))
    c4_index, c4_idf = nb.build_index(norm_corpus, lambda t: nb.char_ngrams(t, 4))
    postings, doclen, idf_bm, avgdl, N, _ = bm.build_bm25(norm_corpus)
    seqs = {cid: t.split() for cid, t in norm_corpus.items()}
    df = defaultdict(int)
    for s in seqs.values():
        for w_ in set(s):
            df[w_] += 1
    _IDF = {w_: math.log(len(seqs) / d) for w_, d in df.items()}
    _SEQS = [seqs[cid] for cid in ids]
    wp_keep, wp_pidf, wp_inv = wp.build_index(seqs, WPAIR_PAIR_CAP, WPAIR_LEMMA_CAP, WPAIR_W)

    print("alignment…", file=sys.stderr)
    with Pool(min(16, os.cpu_count() or 1)) as pool:      # ordered map: deterministic
        sw_all = pool.map(_sw_one, [qn[u].split() for u in uq], chunksize=4)

    sims = {}
    for probe, (name, qp, pp) in ENCODERS.items():
        print(f"encoding with {name}…", file=sys.stderr)
        m = SentenceTransformer(name)
        cv = m.encode([pp + corpus[cid] for cid in ids], normalize_embeddings=True,
                      convert_to_numpy=True, batch_size=128).astype(np.float32)
        qv = m.encode([qp + qtext[u] for u in uq], normalize_embeddings=True,
                      convert_to_numpy=True).astype(np.float32)
        sims[probe] = qv @ cv.T

    S = {}
    for k, u in enumerate(uq):
        Q = qn[u].split()
        S[u] = {"ngram": nb.rank(nb.word_ngrams(qn[u], 2), ng_index, ng_idf),
                "char4": nb.rank(nb.char_ngrams(qn[u], 4), c4_index, c4_idf),
                "bm25": bm.bm25_rank(Q, postings, doclen, idf_bm, avgdl),
                "sw": {cid: v for cid, v in zip(ids, sw_all[k]) if v > 0},
                "wpair": wp.score(Q, wp_keep, wp_pidf, wp_inv, (), WPAIR_W)}
        for probe in ENCODERS:
            S[u][probe] = dict(zip(ids, sims[probe][k].tolist()))
    rows = []
    for r in rels:
        hit = r.pop("_hit")
        for probe in PROBES:
            r[probe] = nb.best_rank(S[r["query_id"]][probe], hit) or ""
        rows.append(r)

    with open(nb.HERE + "results/latin_epic/latin_factorial_ranks.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    report(rows)
    print("\nwrote results/latin_epic/latin_factorial_ranks.csv", file=sys.stderr)


def report(rows):
    val = lambda r, p: int(r[p]) if str(r[p]).strip() else None

    def groups(sel):
        g = defaultdict(list)
        for r in sel:
            g[r["query_id"]].append(r)
        return list(g.values())

    def boot(sel, fn, n_boot=10000):
        gs, rng = groups(sel), random.Random(0)
        b = sorted(fn([r for g in (gs[rng.randrange(len(gs))] for _ in gs) for r in g])
                   for _ in range(n_boot))
        return b[int(n_boot * .025)], b[int(n_boot * .975)]

    r10 = lambda p: (lambda rs: recall_at_k([val(r, p) for r in rs], 10))

    def show(sel, label):
        print(f"\n### {label} (n={len(sel)} parallels, {len(groups(sel))} queries)")
        print("| probe | R@1 | R@5 | R@10 [CI] | R@50 | MRR |")
        print("|---|---:|---:|---|---:|---:|")
        for p in PROBES:
            rk = [val(r, p) for r in sel]
            lo, hi = boot(sel, r10(p))
            print(f"| {p} | {recall_at_k(rk, 1):.2f} | {recall_at_k(rk, 5):.2f} | "
                  f"{recall_at_k(rk, 10):.2f} [{lo:.2f},{hi:.2f}] | {recall_at_k(rk, 50):.2f} "
                  f"| {mrr(rk):.3f} |")

    show(rows, "Latin epic — all")
    print("\npaired differences at R@10 [95%]:")
    for a, b in (("char4", "sw"), ("char4", "bm25"), ("char4", "ngram"), ("sw", "bm25"),
                 ("sw", "dense_e5"), ("bm25", "dense_e5"), ("dense_e5", "dense_sphil"),
                 ("dense_e5", "dense")):
        f = lambda rs: r10(a)(rs) - r10(b)(rs)
        lo, hi = boot(rows, f)
        print(f"  {a} - {b}: {f(rows):+.2f} [{lo:+.2f},{hi:+.2f}]")
    for b in ("verbatim", "close", "loose"):
        sel = [r for r in rows if r["band"] == b]
        if sel:
            show(sel, f"Latin epic — band {b}")
    # recovery by number of commentaries that record the parallel, within band
    print("\n### R@10 by number of commentaries recording the parallel, within band")
    print("| band | commentaries | n | " + " | ".join(PROBES) + " |")
    print("|---|---:|---:|" + "---:|" * len(PROBES))
    for b in ("all", "close", "loose"):
        for n in (1, 2, 3):
            sel = [r for r in rows if r["n_commentaries"] == n and b in ("all", r["band"])]
            print(f"| {b} | {n} | {len(sel)} | "
                  + " | ".join(f"{r10(p)(sel):.2f}" for p in PROBES) + " |")
    one = [r for r in rows if r["n_commentaries"] == 1]
    more = [r for r in rows if r["n_commentaries"] > 1]
    print(f"\nshare in the close band: one commentary {sum(r['band']=='close' for r in one)/len(one):.2f}, "
          f"two or three {sum(r['band']=='close' for r in more)/len(more):.2f}")


if __name__ == "__main__":
    main()
