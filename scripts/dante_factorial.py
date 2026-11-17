"""Dante transfer test: signal-class probes on the per-citation Dante↔Duecento gold.

Does the detectability profile transfer to medieval Italian? Probes mirror the Russian
factorial's signal classes (factorial.py) over data/dante/gold.csv (from the
DanteSources citations: one row per citation, a Dante
lemma -> cited verses of a Duecento poem; 431-chunk corpus, 8 authors):

  exact surface    ngram   word-2gram idf overlap (language-agnostic)
                   char4   the same score over character 4-grams — the surface variant of
                           the Russian robustness table (invariance.py)
  alignment        sw      Smith-Waterman over SURFACE tokens — no Duecento lemmatizer
                           exists in this stack; the Russian probe ran on Mystem lemmas
  rarity bag       bm25    BM25 over surface tokens
  collocation      wpair   distinctive unordered surface-token pairs; df caps scaled to
                           corpus size (Russian caps were ~0.18 / ~0.0022 of 45k -> here
                           lemma_cap 80, pair_cap 5 of 431 — judgment call, disclosed)
  dense            dense   MiniLM line-to-line max-sim (the Russian probe's model)
                   dense_e5 the same rule with intfloat/multilingual-e5-base and its
                           "query: " / "passage: " prefixes
  (LLM rerank      —       excluded by design, see the paper)

Query, three definitions, each column prefixed accordingly:
  wide    HEADLINE: the lemma widened to the verse line(s) or prose sentence(s) it falls in
          — the analogue of the Russian fragment query
  lemma   the located lemma alone (median 3 words)
  whole   the whole Dante passage (the converted benchmark's query)
`wide` and `lemma` exist for the relations whose lemma is located in the passage.

Unit: one row per relation. Gold = the chunk(s) covering the cited verses; where they span
two chunks the best-ranked one counts. A query hit rate (best relation per passage) is
reported secondarily; intervals resample passages. Bands as for the Russian set, span to
span: SequenceMatcher ratio between the widened lemma and the cited verses (0.85 / 0.5).

    python scripts/dante_factorial.py     # from the release root; CPU, ~minutes

    CORPUS=open python scripts/dante_factorial.py
        the same run against an OPEN candidate space: the 431 chunks of the 60 cited poems
        plus every other Duecento poem of Biblioteca Italiana (poems_index.json, 929 poems)
        cut into chunks of four lines, as the cited poems are. Poems whose incipit matches a
        cited poem are left out (other editions of the same poem), and text-identical poems
        are kept once. wpair caps are scaled to the number of chunks.
        Writes results/dante/dante_factorial_ranks_open.csv; the default file is not touched.

Gold: data/dante/gold.csv has two kinds of relation (referent_corpus column). `verified`
relations cite a poem of the 431-chunk corpus; `biblioteca_italiana` relations cite one of
the further poems, so they can only be scored in the open space. The default run scores the
verified relations alone; the open run scores all, and prints the verified subset too. A
further printing of a cited poem (same incipit) is never a candidate.

Deterministic. Writes results/dante/dante_factorial_ranks.csv + prints the profile.
"""
from __future__ import annotations
import csv, json, math, os, random, re, sys
from collections import defaultdict
from difflib import SequenceMatcher

import ngram_baseline as nb
import bm25_baseline as bm
import wpair_baseline as wp
from passim_baseline import sw
from rank_metrics import recall_at_k, mrr

csv.field_size_limit(10 ** 8)
PROBES = ("ngram", "char4", "sw", "bm25", "wpair", "dense", "dense_e5")
CONDS = ("wide", "lemma", "whole")
# probe -> (checkpoint, query prefix, passage prefix)
ENCODERS = {"dense": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
            "dense_e5": ("intfloat/multilingual-e5-base", "query: ", "passage: ")}
KS = (1, 5, 10, 50)
WPAIR_LEMMA_CAP, WPAIR_PAIR_CAP, WPAIR_W = 80, 5, 4
OPEN = os.environ.get("CORPUS", "cited") == "open"


def widen(text: str, frag: str) -> str:
    """The verse line(s) (text with line breaks) or sentence(s) (prose) the lemma falls in."""
    a = text.find(frag) if frag else -1
    if a < 0:
        return ""
    e = a + len(frag)
    pat = r"[^\n]+" if "\n" in text.strip() else r"[^.;:!?]+[.;:!?]*"
    return "\n".join(text[m.start():m.end()].strip() for m in re.finditer(pat, text)
                     if m.start() < e and m.end() > a)


def band_of(query_span: str, referent_span: str) -> str:
    r = SequenceMatcher(None, nb.norm(query_span), nb.norm(referent_span)).ratio()
    return "verbatim" if r >= 0.85 else "close" if r >= 0.5 else "loose"


def main():
    from sentence_transformers import SentenceTransformer
    import numpy as np

    rels = list(csv.DictReader(open(nb.HERE + "data/dante/gold.csv", encoding="utf-8")))
    if not OPEN:
        rels = [r for r in rels if r["referent_corpus"] == "verified"]
    gt = json.load(open(nb.HERE + "data/dante/"
                        "retrieval_corpus.json", encoding="utf-8"))
    def slug(t, n=4):
        return "-".join(re.findall(r"[a-zàèéìòóù]+", (t or "").lower())[:n])
    corpus = {}
    for c in gt:
        corpus[f'{c["chunk_id"]}@{slug(c["poem_incipit"])}'] = c["chunk_text"]
    lemma_cap, pair_cap = WPAIR_LEMMA_CAP, WPAIR_PAIR_CAP
    if OPEN:
        n_cited = len(corpus)
        cited = {slug(c["poem_incipit"]) for c in gt}
        # cited poems that are not in the 431-chunk corpus: position in poems_index -> slug
        gold_open = {int(x.split("_")[0][4:]): x.split("@")[1] for r in rels
                     if r["referent_corpus"] != "verified" for x in r["referent_text_id"].split(";")}
        seen = set()
        for i, p in enumerate(json.load(open(nb.HERE + "data/dante/"
                                             "poems_index.json", encoding="utf-8"))):
            key = nb.norm(p["full_text"])
            if i not in gold_open and (slug(p["incipit"]) in cited or key in seen or not key
                                       or slug(p["incipit"]) in gold_open.values()):
                continue
            seen.add(key)
            lns = [l for l in p["full_text"].split("\n") if l.strip()]
            for a in range(0, len(lns), 4):
                corpus[f'open{i}_chunk_{a // 4 + 1}@{slug(p["incipit"])}'] = "\n".join(lns[a:a + 4])
        print(f"open space: {len(seen)} further poems, {len(corpus) - n_cited} further chunks", file=sys.stderr)
        lemma_cap = round(WPAIR_LEMMA_CAP * len(corpus) / n_cited)
        pair_cap = round(WPAIR_PAIR_CAP * len(corpus) / n_cited)
    missing = {x for r in rels for x in r["referent_text_id"].split(";")} - set(corpus)
    assert not missing, f"cited chunks not in the search space: {sorted(missing)[:5]}"
    print(f"{len(rels)} relations / {len({r['commented_text_id'] for r in rels})} passages / "
          f"{len(corpus)} chunks", file=sys.stderr)

    norm_corpus = {cid: nb.norm(t) for cid, t in corpus.items()}

    # --- indexes ---
    ng_index, ng_idf = nb.build_index(norm_corpus, lambda t: nb.word_ngrams(t, 2))
    c4_index, c4_idf = nb.build_index(norm_corpus, lambda t: nb.char_ngrams(t, 4))
    postings, doclen, idf_bm, avgdl, N, _ = bm.build_bm25(norm_corpus)
    seqs = {cid: t.split() for cid, t in norm_corpus.items()}
    df = defaultdict(int)
    for s in seqs.values():
        for w_ in set(s):
            df[w_] += 1
    idf_sw = {w_: math.log(len(seqs) / d) for w_, d in df.items()}
    wp_keep, wp_pidf, wp_inv = wp.build_index(seqs, pair_cap, lemma_cap, WPAIR_W)

    cid_list, starts, all_lines = [], [], []
    for cid, t in corpus.items():
        lns = [l.strip() for l in t.split("\n") if l.strip()]
        starts.append(len(all_lines)); cid_list.append(cid)
        all_lines.extend(lns if lns else [t])
    starts = np.array(starts, dtype=np.int64)
    print(f"encoding {len(all_lines)} chunk lines…", file=sys.stderr)
    models, cvecs = {}, {}
    for probe, (name, _, pp) in ENCODERS.items():
        models[probe] = SentenceTransformer(name)
        cvecs[probe] = models[probe].encode([pp + l for l in all_lines], normalize_embeddings=True,
                                            convert_to_numpy=True, batch_size=64).astype(np.float32)

    cache = {}

    def scores(qtext):
        if qtext in cache:
            return cache[qtext]
        qn = nb.norm(qtext)
        Q = qn.split()
        S = {}
        S["ngram"] = nb.rank(nb.word_ngrams(qn, 2), ng_index, ng_idf)
        S["char4"] = nb.rank(nb.char_ngrams(qn, 4), c4_index, c4_idf)
        S["bm25"] = bm.bm25_rank(Q, postings, doclen, idf_bm, avgdl)
        S["sw"] = {cid: sw(Q, seqs[cid], idf_sw) for cid in seqs}
        S["wpair"] = wp.score(Q, wp_keep, wp_pidf, wp_inv, (), WPAIR_W)
        qlines = [l.strip() for l in qtext.split("\n") if l.strip()] or [qtext]
        for probe, (_, qp, _) in ENCODERS.items():
            Qv = models[probe].encode([qp + l for l in qlines], normalize_embeddings=True,
                                      convert_to_numpy=True).astype(np.float32)
            sims = Qv @ cvecs[probe].T
            cmax = np.maximum.reduceat(sims, starts, axis=1).max(axis=0)
            S[probe] = {cid_list[i]: float(cmax[i]) for i in range(len(cid_list))}
        cache[qtext] = S
        return S

    rows = []
    for ri, r in enumerate(rels):
        hit = set(r["referent_text_id"].split(";"))
        frag = r["commented_fragment"]
        wide = widen(r["commented_text"], frag)
        rec = {"conn_id": r["conn_id"], "query_id": r["commented_text_id"],
               "referent_id": r["referent_text_id"], "referent_corpus": r["referent_corpus"],
               "band": band_of(wide, r["referent_span"]) if wide else ""}
        for cond, q in (("wide", wide), ("lemma", frag if wide else ""), ("whole", r["commented_text"])):
            S = scores(q) if q else None
            for probe in PROBES:
                rec[f"{cond}_{probe}"] = (nb.best_rank(S[probe], hit) or "") if S else ""
        rows.append(rec)
        if (ri + 1) % 100 == 0:
            print(f"  {ri+1}/{len(rels)} relations", file=sys.stderr)

    out = "results/dante/dante_factorial_ranks_open.csv" if OPEN else "results/dante/dante_factorial_ranks.csv"
    with open(nb.HERE + out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def show(rows_sel, label, cond):
        # Intervals resample PASSAGES, not relations: the relations of one passage are not
        # independent observations.
        print(f"\n### {label}")
        print("| probe | n | R@1 [CI] | R@5 | R@10 [CI] | R@50 | MRR |")
        print("|---|---:|---|---:|---|---:|---:|")
        byq = defaultdict(list)
        for r in rows_sel:
            byq[r["query_id"]].append(r)
        groups = list(byq.values())
        val = lambda r, probe: int(r[f"{cond}_{probe}"]) if str(r[f"{cond}_{probe}"]).strip() else None

        def ci(probe, k, n_boot=10000):
            rng = random.Random(0)
            boots = []
            for _ in range(n_boot):
                rs = [val(r, probe) for g in (groups[rng.randrange(len(groups))] for _ in groups) for r in g]
                boots.append(recall_at_k(rs, k))
            boots.sort()
            return boots[int(n_boot * .025)], boots[int(n_boot * .975)]

        for probe in PROBES:
            ranks = [val(r, probe) for r in rows_sel]
            l1, h1 = ci(probe, 1)
            l10, h10 = ci(probe, 10)
            print(f"| {probe} | {len(ranks)} | {recall_at_k(ranks, 1):.2f} [{l1:.2f},{h1:.2f}] "
                  f"| {recall_at_k(ranks, 5):.2f} | {recall_at_k(ranks, 10):.2f} [{l10:.2f},{h10:.2f}] "
                  f"| {recall_at_k(ranks, 50):.2f} | {mrr(ranks):.3f} |")

    loc = [r for r in rows if r["band"]]
    npass = lambda rs: len({r["query_id"] for r in rs})
    show(loc, f"Dante — HEADLINE, widened lemma as query (n={len(loc)} relations, {npass(loc)} passages)", "wide")
    show(loc, f"Dante — lemma alone as query (n={len(loc)})", "lemma")
    show(loc, f"Dante — whole passage as query, same relations (n={len(loc)})", "whole")
    show(rows, f"Dante — whole passage as query, all relations (n={len(rows)}, {npass(rows)} passages)", "whole")
    if OPEN:
        ver = [r for r in loc if r["referent_corpus"] == "verified"]
        show(ver, f"Dante — widened anchor, 60-poem subset only (n={len(ver)})", "wide")
        oth = [r for r in loc if r["referent_corpus"] != "verified"]
        show(oth, f"Dante — widened lemma, relations added from Biblioteca Italiana (n={len(oth)})", "wide")
    for b in ("verbatim", "close", "loose"):
        sel = [r for r in loc if r["band"] == b]
        if sel:
            show(sel, f"Dante — band {b}, widened lemma (n={len(sel)})", "wide")
    # query hit rate: a passage counts when its best-ranked relation is in the top k
    bq = []
    for qid in sorted({r["query_id"] for r in loc}):
        qrows = [r for r in loc if r["query_id"] == qid]
        rec = {"query_id": qid}
        for probe in PROBES:
            rs = [int(r[f"wide_{probe}"]) for r in qrows if str(r[f"wide_{probe}"]).strip()]
            rec[f"wide_{probe}"] = min(rs) if rs else ""
        bq.append(rec)
    show(bq, f"Dante — passage hit rate, widened lemma (n={len(bq)} passages)", "wide")
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
