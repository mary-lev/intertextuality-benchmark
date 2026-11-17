"""Phase 2 — alignment-style retrieval: shared rare words within a window.

Where the n-gram floor needs *contiguous exact* sequences, this rewards the query's rare
content words appearing **close together** in a candidate, regardless of order or gaps
(Tesserae/Passim idea). It should lift the `close` band (shared words, broken order/endings)
and turn the unigram recall reservoir into precision, *and* it returns the best-scoring
line window — passage localization (framing B) for free.

Score(query, poem) = max over line-windows W of  Σ idf(w) for w in query_words ∩ words(W),
counting only windows with ≥2 shared content words (the Tesserae minimum).

    python scripts/align_baseline.py        # from the release root

Stdlib only. Shares corpus/gold loading with ngram_baseline.
"""
from __future__ import annotations
import csv, math, sys
from collections import defaultdict
import ngram_baseline as nb   # norm, load_corpus, dup_groups, load_gold, KS, HERE

csv.field_size_limit(10**8)
WIN = 4            # window size in lines
DF_CAP = 4000      # words rarer than this seed candidate generation (skips stopwords)
MIN_SHARED = 2     # Tesserae minimum: ≥2 shared content words in the window


def poem_lines(text_by_id):
    """id -> list of line word-sets (for windowing)."""
    out = {}
    for pid, t in text_by_id.items():
        out[pid] = [set(ln.split()) for ln in t.split("\n") if ln.split()] or [set(t.split())]
    return out


def word_index(text_by_id):
    idx = defaultdict(set)
    for pid, t in text_by_id.items():
        for w in set(t.split()):
            idx[w].add(pid)
    N = len(text_by_id)
    idf = {w: math.log(N / len(p)) for w, p in idx.items()}
    return idx, idf


def best_window(qwords, lines, idf):
    """max Σ idf over a WIN-line window's shared query words; needs ≥MIN_SHARED."""
    best, bi = 0.0, -1
    for i in range(len(lines)):
        win = set().union(*lines[i:i + WIN])
        shared = qwords & win
        if len(shared) >= MIN_SHARED:
            s = sum(idf.get(w, 0.0) for w in shared)
            if s > best:
                best, bi = s, i
    return best, bi


def main():
    print("loading corpus…", file=sys.stderr)
    corpus_norm = {}                                   # need normalized text, with newlines kept
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        corpus_norm[p["poetree_poem_id"]] = "\n".join(nb.norm(ln) for ln in p["full_text"].split("\n"))
    for r in nb.load_added_referents():
        corpus_norm[r["extra_id"]] = "\n".join(nb.norm(ln) for ln in r["full_text"].split("\n"))

    flat = {pid: t.replace("\n", " ") for pid, t in corpus_norm.items()}
    dups = nb.dup_groups(flat)
    gold = nb.load_gold(dups)
    lines = poem_lines(corpus_norm)
    idx, idf = word_index(flat)
    print(f"corpus {len(flat)} | grounded {len(gold)}", file=sys.stderr)

    by = lambda: []
    allr, strat, band = [], defaultdict(by), defaultdict(by)
    for g in gold:
        qwords = set(g["q_frag"].split())
        # candidate generation: poems sharing ≥2 of the query's reasonably-rare words
        cand = defaultdict(int)
        for w in qwords:
            if w in idx and len(idx[w]) <= DF_CAP:
                for pid in idx[w]:
                    cand[pid] += 1
        cands = [pid for pid, c in cand.items() if c >= MIN_SHARED and pid not in g["self"]]
        scores = {}
        for pid in cands:
            s, _ = best_window(qwords, lines[pid], idf)
            if s > 0:
                scores[pid] = s
        r = nb.best_rank(scores, g["hit"])
        allr.append(r); strat[g["stratum"]].append(r)
        if g["band"]:
            band[g["band"]].append(r)

    def m(rs):
        n = len(rs) or 1
        return [sum(1 for r in rs if r and r <= k) / n for k in nb.KS] + [sum(1/r for r in rs if r)/n]
    def show(lab, rs):
        v = m(rs)
        print(f"  {lab:13s} n={len(rs):3d}  " + "  ".join(f"R@{k}={v[i]:.2f}" for i, k in enumerate(nb.KS)) + f"  MRR={v[-1]:.3f}")
    print(f"\n### alignment (shared rare words, {WIN}-line window) — fragment → referent")
    show("ALL grounded", allr)
    for s in ("S1", "S2", "S3", "S4"):
        if strat[s]: show(s, strat[s])
    for b in ("verbatim", "close", "loose"):
        if band[b]: show("S1/" + b, band[b])


if __name__ == "__main__":
    main()
