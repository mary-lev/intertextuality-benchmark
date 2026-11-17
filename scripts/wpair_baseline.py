"""Distinctive word-PAIR retrieval (port of the CnP→NT `wpair` method) — for the `close` band.

Tesserae/Passim score a candidate by the summed *individual* rarity of shared lemmas. `wpair`
instead scores by the rarity of shared **unordered lemma PAIRS**: a pair like {новый, иерусалим}
is distinctive even when neither word alone is, and the pair survives reordering/inflection (the
`close` band = "shared words, reordered/inflected"). Lemma-based so it also absorbs Russian
inflection (the win that makes Tesserae/Passim lead).

  * unit        : flattened content-lemma sequence per poem (Mystem lemmas, ё→е)
  * content     : drop lemmas with poem-df > LEMMA_DF_CAP (function words / ubiquitous lemmas)
  * pairs       : unordered lemma pairs co-occurring within a W-token window of that sequence
  * distinctive : keep only pairs with poem-df <= PAIR_DF_CAP (a pair in more poems is an
                  ordinary collocation, not a borrowing fingerprint)
  * score       : poem score = Σ pair-idf over the distinctive pairs it shares with the query
  * retrieval   : candidates = poems sharing ≥1 distinctive pair; scoped to fragment queries (S1/S2)

    python scripts/wpair_baseline.py [PAIR_DF_CAP] [LEMMA_DF_CAP] [W]   # needs the lemma cache

Stdlib + pymystem3. Mirrors the tesserae_baseline harness (same gold/strata/bands).
"""
from __future__ import annotations
import os, sys, csv, math, re
from collections import defaultdict
import ngram_baseline as nb

csv.field_size_limit(10**8)
LEMMAS = nb.HERE + "cache/lemmas/poetree_lemmas.csv"
PAIR_DF_CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 100   # pair in > this many poems = common
LEMMA_DF_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 8000  # lemma in > this many poems = function-ish
W = int(sys.argv[3]) if len(sys.argv) > 3 else 4               # co-occurrence window (content lemmas)
WORD = re.compile(r"\w+", re.U)


def content_lemmas(s):
    return [w for w in WORD.findall(s.lower().replace("ё", "е"))]


def load_lemma_corpus():
    """poem_id -> flat lemma list; + lemma poem-frequency (df over poems)."""
    if not os.path.exists(LEMMAS):
        sys.exit("run scripts/lemmatize_corpus.py first")
    poems, df = {}, defaultdict(int)
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        flat = content_lemmas(r["lemma_text"].replace("\n", " "))
        poems[r["poem_id"]] = flat
        for w in set(flat):
            df[w] += 1
    return poems, df


def pairs(seq, keep, w=W):
    """unordered lemma pairs within window w over the content-lemma subsequence (keep = content set)."""
    t = [x for x in seq if x in keep]
    out = set()
    for i in range(len(t)):
        for j in range(i + 1, min(i + w, len(t))):
            if t[i] != t[j]:
                out.add((t[i], t[j]) if t[i] < t[j] else (t[j], t[i]))
    return out


def build_index(seqs, pair_cap=PAIR_DF_CAP, lemma_cap=LEMMA_DF_CAP, w=W):
    """seqs: pid -> flat lemma list. Returns (keep, pidf, inv) for distinctive-pair scoring.
    Reusable by the fusion harness so it builds the index from the lemma corpus it already loaded."""
    Np = len(seqs)
    ldf = defaultdict(int)
    for seq in seqs.values():
        for x in set(seq):
            ldf[x] += 1
    keep = {x for x, d in ldf.items() if d <= lemma_cap}
    pdf = defaultdict(int)
    ppairs = {}
    for pid, seq in seqs.items():
        ps = pairs(seq, keep, w)
        ppairs[pid] = ps
        for p in ps:
            pdf[p] += 1
    pidf = {p: math.log(Np / d) for p, d in pdf.items() if d <= pair_cap}
    inv = defaultdict(list)
    for pid, ps in ppairs.items():
        for p in ps:
            if p in pidf:
                inv[p].append(pid)
    return keep, pidf, inv


def score(qseq, keep, pidf, inv, self_ids=(), w=W):
    """qseq: query content-lemma list -> {poem_id: summed pair-idf} over shared distinctive pairs."""
    scores = defaultdict(float)
    for p in pairs(qseq, keep, w):
        if p in pidf:
            wt = pidf[p]
            for pid in inv[p]:
                if pid not in self_ids:
                    scores[pid] += wt
    return scores


def main():
    from pymystem3 import Mystem
    print("loading lemma corpus…", file=sys.stderr)
    poems, _ = load_lemma_corpus()
    flat = {pid: " ".join(seq) for pid, seq in poems.items()}
    dups = nb.dup_groups(flat)
    gold = nb.load_gold(dups)

    print("building pair index…", file=sys.stderr)
    keep, pidf, inv = build_index(poems)
    print(f"corpus {len(poems)} poems | content lemmas {len(keep)} | distinctive pairs {len(pidf)} "
          f"(cap df<={PAIR_DF_CAP}, lemma df<={LEMMA_DF_CAP}, W={W}) | grounded {len(gold)}",
          file=sys.stderr)

    m = Mystem()
    allr, strat, band = [], defaultdict(list), defaultdict(list)
    for g in gold:
        qseq = content_lemmas("".join(m.lemmatize(g["q_frag"])))
        scores = score(qseq, keep, pidf, inv, g["self"])
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

    print(f"\n### wpair (distinctive lemma-pairs, df<={PAIR_DF_CAP}, W={W}) — fragment → referent")
    show("ALL grounded", allr)
    for s in ("S1", "S2", "S3", "S4"):
        if strat[s]: show(s, strat[s])
    for b in ("verbatim", "close", "loose"):
        if band[b]: show("S1/" + b, band[b])


if __name__ == "__main__":
    main()
