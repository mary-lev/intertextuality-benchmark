"""Phase 2 (proper) — canonical Tesserae on lemmas.

Our align_baseline approximated Tesserae with surface words. This is the real thing:

  * unit       : the verse line (Tesserae's "phrase")
  * matching   : two lines form a parallel iff they share >= 2 distinct LEMMAS (Mystem)
  * scoring    : Tesserae score = ln( (sum 1/freq over shared lemmas in the source line)
                 + (sum 1/freq over shared lemmas in the target line) ) — rarer shared lemmas
                 and denser matches score higher. (Distance term dropped: our query is a single
                 fragment, not a full source text, so per-line word-distance is degenerate.)
  * retrieval  : a poem scores = max Tesserae score over its lines vs the query lemmas.

Query lemmas = Mystem lemmas of the commented fragment. Candidate generation via a lemma
inverted index (lemmas with df <= cap). Same harness/strata as the other baselines.

    python scripts/tesserae_baseline.py        # needs cache/lemmas/poetree_lemmas.csv

Stdlib + pymystem3.
"""
from __future__ import annotations
import os, sys, csv, math, re
from collections import defaultdict
import ngram_baseline as nb

csv.field_size_limit(10**8)
LEMMAS = nb.HERE + "cache/lemmas/poetree_lemmas.csv"
DF_CAP = 4000
MIN_SHARED = 2
WORD = re.compile(r"\w+", re.U)


def content_lemmas(s):
    return [w for w in WORD.findall(s.lower().replace("ё", "е"))]


def load_lemma_corpus():
    """poem_id -> list of line lemma-lists; + global lemma doc-frequency (over lines)."""
    if not os.path.exists(LEMMAS):
        sys.exit("run scripts/lemmatize_corpus.py first")
    poems, df = {}, defaultdict(int)
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        lines = [content_lemmas(ln) for ln in r["lemma_text"].split("\n")]
        lines = [l for l in lines if l]
        poems[r["poem_id"]] = lines
        for l in lines:
            for w in set(l):
                df[w] += 1
    return poems, df


def main():
    from pymystem3 import Mystem
    print("loading lemma corpus…", file=sys.stderr)
    poems, df = load_lemma_corpus()
    N = sum(len(ls) for ls in poems.values())                 # total lines (Tesserae freq base)
    freq = {w: c for w, c in df.items()}                      # line doc-frequency as freq proxy
    flat = {pid: " ".join(w for l in ls for w in l) for pid, ls in poems.items()}
    dups = nb.dup_groups(flat)
    gold = nb.load_gold(dups)

    # lemma inverted index for candidate generation (rare lemmas only)
    inv = defaultdict(set)
    for pid, ls in poems.items():
        for w in {w for l in ls for w in l}:
            inv[w].add(pid)

    m = Mystem()
    print(f"corpus {len(poems)} poems / {N} lines | grounded {len(gold)}", file=sys.stderr)

    allr, strat, band = [], defaultdict(list), defaultdict(list)
    for g in gold:
        qlem = set(content_lemmas("".join(m.lemmatize(g["q_frag"]))))
        # candidates: share >= MIN_SHARED reasonably-rare query lemmas
        cand = defaultdict(int)
        for w in qlem:
            if w in inv and len(inv[w]) <= DF_CAP:
                for pid in inv[w]:
                    cand[pid] += 1
        cands = [pid for pid, c in cand.items() if c >= MIN_SHARED and pid not in g["self"]]
        scores = {}
        for pid in cands:
            best = 0.0
            for line in poems[pid]:
                shared = qlem & set(line)
                if len(shared) >= MIN_SHARED:
                    # Tesserae matching mass: rarer shared lemmas (low corpus freq) score higher.
                    s = sum(1.0 / freq[w] for w in shared)
                    if s > best:
                        best = s
            if best > 0:
                scores[pid] = best
        r = nb.best_rank(scores, g["hit"])
        allr.append(r); strat[g["stratum"]].append(r)
        if g["band"]:
            band[g["band"]].append(r)

    def mm(rs):
        n = len(rs) or 1
        return [sum(1 for r in rs if r and r <= k) / n for k in nb.KS] + [sum(1/r for r in rs if r)/n]
    def show(lab, rs):
        v = mm(rs)
        print(f"  {lab:13s} n={len(rs):3d}  " +
              "  ".join(f"R@{k}={v[i]:.2f}" for i, k in enumerate(nb.KS)) + f"  MRR={v[-1]:.3f}")

    print("\n### Tesserae (lemma, line unit, ≥2 shared lemmas, inverse-frequency) — fragment → referent")
    show("ALL grounded", allr)
    for s in ("S1", "S2", "S3", "S4"):
        if strat[s]: show(s, strat[s])
    for b in ("verbatim", "close", "loose"):
        if band[b]: show("S1/" + b, band[b])


if __name__ == "__main__":
    main()
