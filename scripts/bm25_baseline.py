"""BM25 baseline + reference points (random floor, candidate-generation ceiling).

Completes Phase 1's calibration (EXPERIMENT.md §8). Three things, same harness as the
n-gram / alignment baselines (query = commented_fragment else whole poem; rank the deduped
Poetree+extras pool; self excluded; hit = gold referent / dup / cycle part):

  * BM25            — the standard strong lexical baseline (Okapi, k1=1.5, b=0.75, word unigram)
  * random floor    — expected recall of a random ranking (k / N): the do-nothing baseline
  * candidate ceiling — fraction of golds the scorers *could* reach given how candidates are
                        generated: bigram (>=1 shared bigram) and alignment (>=2 shared rare
                        words). No scorer can beat its own candidate ceiling.

    python scripts/bm25_baseline.py        # from the release root

Stdlib only.
"""
from __future__ import annotations
import math, sys
from collections import defaultdict, Counter
import ngram_baseline as nb
import align_baseline as al

DF_CAP = al.DF_CAP   # rare-word bar used by alignment candidate generation
K1, B = 1.5, 0.75


def build_bm25(corpus):
    postings = defaultdict(list)            # word -> [(pid, tf)]
    doclen, df = {}, defaultdict(int)
    for pid, t in corpus.items():
        words = t.split()
        doclen[pid] = len(words) or 1
        for w, f in Counter(words).items():
            postings[w].append((pid, f))
            df[w] += 1
    N = len(corpus)
    avgdl = sum(doclen.values()) / N
    idf = {w: math.log((N - d + 0.5) / (d + 0.5) + 1) for w, d in df.items()}
    return postings, doclen, idf, avgdl, N, df


def bm25_rank(qwords, postings, doclen, idf, avgdl):
    score = defaultdict(float)
    for w in set(qwords):
        if w not in postings:
            continue
        iw = idf[w]
        for pid, f in postings[w]:
            dl = doclen[pid]
            score[pid] += iw * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / avgdl))
    return score


def metrics(ranks):
    n = len(ranks) or 1
    return {f"R@{k}": sum(1 for r in ranks if r and r <= k) / n for k in nb.KS} | \
           {"MRR": sum(1 / r for r in ranks if r) / n, "n": len(ranks)}


def show(lab, ranks):
    m = metrics(ranks)
    print(f"  {lab:13s} n={m['n']:3d}  " +
          "  ".join(f"R@{k}={m[f'R@{k}']:.2f}" for k in nb.KS) + f"  MRR={m['MRR']:.3f}")


def main():
    print("loading corpus…", file=sys.stderr)
    corpus = nb.load_corpus()
    dups = nb.dup_groups(corpus)
    gold = nb.load_gold(dups)
    postings, doclen, idf, avgdl, N, df = build_bm25(corpus)
    print(f"corpus {N} | grounded {len(gold)} | avgdl {avgdl:.0f}", file=sys.stderr)

    strat, band = defaultdict(list), defaultdict(list)
    allr = []
    # reachability (candidate ceilings) accumulators
    reach_big, reach_aln = defaultdict(list), defaultdict(list)
    rb_band, ra_band = defaultdict(list), defaultdict(list)

    for g in gold:
        qwords = g["q_frag"].split()
        scores = bm25_rank(qwords, postings, doclen, idf, avgdl)
        for pid in g["self"]:
            scores.pop(pid, None)
        r = nb.best_rank(scores, g["hit"])
        allr.append(r); strat[g["stratum"]].append(r)
        if g["band"]:
            band[g["band"]].append(r)

        # --- candidate ceilings: can the gold even be generated as a candidate? ---
        gtext = set()
        for h in g["hit"]:
            if h in corpus:
                gtext |= set(corpus[h].split())
        qrare = {w for w in set(qwords) if df.get(w, 0) <= DF_CAP}
        can_aln = len(qrare & gtext) >= al.MIN_SHARED
        qbg = nb.word_ngrams(g["q_frag"], 2)
        gbg = set()
        for h in g["hit"]:
            if h in corpus:
                gbg |= nb.word_ngrams(corpus[h], 2)
        can_big = len(qbg & gbg) >= 1
        reach_big[g["stratum"]].append(can_big); reach_aln[g["stratum"]].append(can_aln)
        if g["band"]:
            rb_band[g["band"]].append(can_big); ra_band[g["band"]].append(can_aln)

    print("\n### BM25 (Okapi, word unigram) — fragment → referent")
    show("ALL grounded", allr)
    for s in ("S1", "S2", "S3", "S4"):
        if strat[s]: show(s, strat[s])
    for b in ("verbatim", "close", "loose"):
        if band[b]: show("S1/" + b, band[b])

    print("\n### random floor — expected recall of a random ranking (k / N)")
    print("  " + "  ".join(f"R@{k}={k / N:.4f}" for k in nb.KS) + f"   (N={N})")

    print("\n### candidate-generation ceiling — fraction of golds reachable")
    print("  (no scorer can exceed this; gap to recall = pure ranking headroom)")
    def pct(xs): return sum(xs) / (len(xs) or 1)
    print(f"  {'stratum':13s}  bigram(≥1 shared bigram)   alignment(≥2 shared rare words)")
    for s in ("S1", "S2", "S3", "S4"):
        if reach_aln[s]:
            print(f"  {s:13s}  {pct(reach_big[s]):.2f}  ({sum(reach_big[s])}/{len(reach_big[s])})"
                  f"               {pct(reach_aln[s]):.2f}  ({sum(reach_aln[s])}/{len(reach_aln[s])})")
    for b in ("verbatim", "close", "loose"):
        if ra_band[b]:
            print(f"  S1/{b:9s}  {pct(rb_band[b]):.2f}  ({sum(rb_band[b])}/{len(rb_band[b])})"
                  f"               {pct(ra_band[b]):.2f}  ({sum(ra_band[b])}/{len(ra_band[b])})")


if __name__ == "__main__":
    main()
