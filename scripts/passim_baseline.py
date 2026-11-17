"""Phase 2″ — Passim-style retrieval: rarity-weighted Smith–Waterman local alignment.

Tesserae scores a single line by its shared lemmas. Passim instead finds the best *contiguous
run* of aligned lemmas, tolerating gaps and mismatches — a true local alignment (Smith–Waterman)
over the lemma token sequences, with each lemma match rewarded by its rarity (idf). This catches
reordered/expanded quotations that don't sit inside one line.

  * sequences : query lemmas (Mystem) vs each candidate poem's lemma sequence
  * cand gen  : poems sharing >= 2 reasonably-rare query lemmas (lemma inverted index)
  * align     : SW with match = idf(lemma), mismatch and gap = small negative; poem score = max H
  * retrieval : rank poems by best local-alignment score; same harness/strata as the others.

    python scripts/passim_baseline.py        # needs cache/lemmas/poetree_lemmas.csv

Stdlib + pymystem3.
"""
from __future__ import annotations
import os, sys, csv, math
from collections import defaultdict
import ngram_baseline as nb
from tesserae_baseline import content_lemmas, LEMMAS, DF_CAP, MIN_SHARED

csv.field_size_limit(10**8)
MISMATCH = -0.5
GAP = -0.5


def load_lemma_seqs():
    """poem_id -> flat lemma sequence; + poem-level doc-frequency per lemma."""
    if not os.path.exists(LEMMAS):
        sys.exit("run scripts/lemmatize_corpus.py first")
    seqs, df = {}, defaultdict(int)
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        seq = content_lemmas(r["lemma_text"].replace("\n", " "))
        seqs[r["poem_id"]] = seq
        for w in set(seq):
            df[w] += 1
    return seqs, df


def sw(Q, P, idf):
    """Smith–Waterman max score: match=idf(lemma), mismatch/gap negative. Two-row DP."""
    nP = len(P)
    prev = [0.0] * (nP + 1)
    best = 0.0
    for a in Q:
        wa = idf.get(a, 0.0)
        cur = [0.0] * (nP + 1)
        up = prev
        for j in range(1, nP + 1):
            d = up[j - 1] + (wa if a == P[j - 1] else MISMATCH)
            v = d
            u = up[j] + GAP
            if u > v:
                v = u
            l = cur[j - 1] + GAP
            if l > v:
                v = l
            if v < 0.0:
                v = 0.0
            cur[j] = v
            if v > best:
                best = v
        prev = cur
    return best


def main():
    from pymystem3 import Mystem
    print("loading lemma corpus…", file=sys.stderr)
    seqs, df = load_lemma_seqs()
    Npoems = len(seqs)
    idf = {w: math.log(Npoems / d) for w, d in df.items()}
    flat = {pid: " ".join(s) for pid, s in seqs.items()}
    dups = nb.dup_groups(flat)
    gold = nb.load_gold(dups)
    inv = defaultdict(set)
    for pid, s in seqs.items():
        for w in set(s):
            inv[w].add(pid)
    m = Mystem()
    print(f"corpus {Npoems} poems | grounded {len(gold)}", file=sys.stderr)

    # Passim aligns a short borrowed fragment, not a whole poem: scope to fragment-bearing
    # connections (S1/S2). S3/S4 have only the whole commented poem as query (up to 1000+
    # lemmas), for which local line-alignment is meaningless and SW is O(|Q||P|)-explosive.
    gold = [g for g in gold if g["stratum"] in ("S1", "S2")]
    allr, strat, band = [], defaultdict(list), defaultdict(list)
    for gi, g in enumerate(gold):
        Q = content_lemmas("".join(m.lemmatize(g["q_frag"])))
        qset = set(Q)
        cand = defaultdict(int)
        for w in qset:
            if w in inv and len(inv[w]) <= DF_CAP:
                for pid in inv[w]:
                    cand[pid] += 1
        cands = [pid for pid, c in cand.items() if c >= MIN_SHARED and pid not in g["self"]]
        scores = {pid: sw(Q, seqs[pid], idf) for pid in cands}
        r = nb.best_rank(scores, g["hit"])
        allr.append(r); strat[g["stratum"]].append(r)
        if g["band"]:
            band[g["band"]].append(r)
        if (gi + 1) % 25 == 0:
            print(f"  {gi+1}/{len(gold)}", file=sys.stderr)

    def mm(rs):
        n = len(rs) or 1
        return [sum(1 for r in rs if r and r <= k) / n for k in nb.KS] + [sum(1/r for r in rs if r)/n]
    def show(lab, rs):
        v = mm(rs)
        print(f"  {lab:13s} n={len(rs):3d}  " +
              "  ".join(f"R@{k}={v[i]:.2f}" for i, k in enumerate(nb.KS)) + f"  MRR={v[-1]:.3f}")

    print("\n### Passim (lemma, Smith–Waterman local alignment, idf-weighted) — fragment → referent")
    show("ALL grounded", allr)
    for s in ("S1", "S2", "S3", "S4"):
        if strat[s]: show(s, strat[s])
    for b in ("verbatim", "close", "loose"):
        if band[b]: show("S1/" + b, band[b])


if __name__ == "__main__":
    main()
