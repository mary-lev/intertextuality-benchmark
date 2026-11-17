"""Line-containment retrieval (`lc`) — ungated, idf-weighted line/window match.

Motivation: the gated lexical methods (Tesserae/Passim require >=2 shared rare lemmas, df<=4000)
*exclude* connections whose distinctive word is mid-frequency or whose fragment is
short — making ~1/3 of the hard cases un-rankable at any depth. `lc` removes the gate:

  * candidate = any poem sharing >=1 query content-lemma (no df cap, no >=2 rule)
  * poem score = max over UNITS of  sum idf(distinct query lemmas present in the unit)
      - unit = a single verse line            (WIN=1, the default; best in fusion)
      - unit = a window of <=WIN consecutive lines (WIN>=2; better standalone recall,
               recovers borrowings whose words spread across source lines, but it is
               redundant with WIN=1 inside RRF — keep WIN=1 for the fusion)

idf = log(N / poem-df(lemma)); function words get idf~=0 so no stoplist is needed.
Deterministic poem_id tie-break. Scoped to fragment connections (S1/S2).

    python scripts/lc_baseline.py            # WIN=1 (line)   — fusion default
    WIN=2 python scripts/lc_baseline.py      # <=2-line window — best standalone

Needs cache/lemmas/poetree_lemmas.csv. Stdlib + pymystem3.
"""
from __future__ import annotations
import os, sys, csv, math
from collections import defaultdict
import ngram_baseline as nb
from tesserae_baseline import content_lemmas, LEMMAS

WIN = int(os.environ.get("WIN", "1"))      # 1 = single line (fusion default); >=2 = multi-line window


def build_lemma_lines(corpus):
    """poem_id -> list[list[lemma]] per line; + inverted index + idf over poem-df."""
    lem_lines = {}
    inv = defaultdict(set)
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        pid = r["poem_id"]
        if pid not in corpus:
            continue
        lines = [content_lemmas(ln) for ln in r["lemma_text"].split("\n")]
        lines = [l for l in lines if l]
        lem_lines[pid] = lines
        for w in set(w for l in lines for w in l):
            inv[w].add(pid)
    N = len(lem_lines)
    idf = {w: math.log(N / len(ps)) for w, ps in inv.items()}
    return lem_lines, inv, idf


def score_poem(q_keys, qidf, lines, win=WIN):
    """max over <=win-line windows of sum idf(distinct query lemmas in the window)."""
    n = len(lines)
    ql = [q_keys & set(line) for line in lines]
    best = 0.0
    for i in range(n):
        for w in range(1, win + 1):
            if i + w > n:
                break
            u = set()
            for j in range(i, i + w):
                u |= ql[j]
            s = sum(qidf[x] for x in u)
            if s > best:
                best = s
    return best


def lc_rank(Q, lem_lines, inv, idf, self_ids):
    """ranked {poem_id: score} for the ungated line-containment of query lemmas Q."""
    qidf = {w: idf[w] for w in set(Q) if w in idf and idf[w] > 0}
    qk = set(qidf)
    cand = set()
    for w in qk:
        cand |= inv[w]
    cand -= set(self_ids)
    return {pid: score_poem(qk, qidf, lem_lines[pid]) for pid in cand}


def main():
    print("loading corpus + lemma lines…", file=sys.stderr)
    corpus = nb.load_corpus()
    dups = nb.dup_groups(corpus)
    gold = [g for g in nb.load_gold(dups) if g["stratum"] in ("S1", "S2")]
    lem_lines, inv, idf = build_lemma_lines(corpus)
    from pymystem3 import Mystem
    mys = Mystem()
    print(f"corpus {len(lem_lines)} | fragment connections {len(gold)} | WIN={WIN}", file=sys.stderr)

    by_all, by_strat, by_band = [], defaultdict(list), defaultdict(list)
    for g in gold:
        Q = content_lemmas("".join(mys.lemmatize(g["q_frag"])))
        scores = lc_rank(Q, lem_lines, inv, idf, g["self"])
        r = nb.best_rank(scores, g["hit"])
        by_all.append(r); by_strat[g["stratum"]].append(r)
        if g["band"]:
            by_band[g["band"]].append(r)

    def show(label, rs):
        m = nb.metrics(rs)
        print(f"  {label:14s} n={m['n']:3d}  " + "  ".join(f"R@{k}={m[f'R@{k}']:.2f}" for k in nb.KS) + f"  MRR={m['MRR']:.3f}")
    print(f"\n### lc (line-containment, ungated, WIN={WIN}) — fragment connections")
    show("ALL", by_all)
    for s in ("S1", "S2"):
        if by_strat[s]:
            show(s, by_strat[s])
    for b in ("verbatim", "close", "loose"):
        if by_band[b]:
            show("S1/" + b, by_band[b])


if __name__ == "__main__":
    main()
