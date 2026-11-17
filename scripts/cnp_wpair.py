"""Distinctive word-pair (collocation) enricher · toggle `wpair` — for REORDERED allusions.

Single-word rarity can't make «Новый Иерусалим» distinctive (иерусалим is mid-frequency), but the
unordered word-PAIR {новый, иерусалим} is. This scores each (sentence, NT verse) by their shared
**distinctive word-pairs** — unordered content-word pairs co-occurring within a window, weighted by
CnP-pair-idf (pair rarity in the NOVEL only; the NT-pair-idf factor was removed — it boosted
NT-rare-but-novel-common idiom pairs). Pairs of function words are excluded up front by the shared
function-word stoplist in `cnp_surface.toks`. It is order-independent and runs on SURFACE words
(the lemmatizer drops «Иерусалим»), so it uniquely recovers reordered allusions: c102 «Новый
Иерусалим»→`Откр 21:2` (pair «иерусалим+новый») and c131 «царствие божие»→`Мк 10:14` both rank 1.

**Per-segment top-1 ENRICHER, by design.** The score sums over shared pairs, so it scales with overlap
length and is NOT globally thresholdable (a coincidental long rare-phrase overlap outscores a short
reordered pair). It is a per-*segment* rank signal: we emit only each sentence's top pair-match (ties
kept) and wire it as an ENRICHER, so rule A drops the coincidental tail that no source seeded, leaving
the reordered cases tagged with their distinctive pair (stored as evidence in `top[].pairs`).

    python scripts/cnp_wpair.py   # -> results/crime_and_punishment/cnp_wpair_top5.jsonl
"""
import json
import math
import sys
from collections import defaultdict

import ngram_baseline as nb
from cnp_surface import toks

H = nb.HERE
W = 4              # window: unordered content-word pairs co-occurring within W surface tokens
PAIR_DF_CAP = 40   # only DISTINCTIVE pairs seed (a pair in > 40 NT verses is an ordinary collocation)


def pairs(text):
    t = toks(text)
    out = set()
    for i in range(len(t)):
        for j in range(i + 1, min(i + W, len(t))):
            if t[i] != t[j]:
                out.add(frozenset((t[i], t[j])))
    return out


def main():
    verses = [json.loads(l) for l in open(H + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8")]
    vp = [pairs(v["text"]) for v in verses]
    vref = [v["ref"] for v in verses]
    pdf = defaultdict(int)              # NT pair df — used ONLY for the distinctiveness cap below
    for s in vp:
        for p in s:
            pdf[p] += 1

    segs = [json.loads(l) for l in open(H + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8")]
    cpdf = defaultdict(int)                                       # pair rarity over the novel's SENTENCES
    sents = [s for s in segs if s["level"] == "sentence"]
    Ns = len(sents)
    for s in sents:
        for p in pairs(s["text"]):
            cpdf[p] += 1
    cnp_idf = {p: math.log(Ns / d) for p, d in cpdf.items()}

    inv = defaultdict(list)
    for vi, s in enumerate(vp):
        for p in s:
            if pdf[p] <= PAIR_DF_CAP:
                inv[p].append(vi)

    def weight(p):
        return cnp_idf.get(p, 0.0)      # CnP-pair rarity only — NT-pair-idf removed (it boosted the
        #                                 NT-rare-but-novel-common idiom pairs like «тех+пор»)

    rows = []
    # process ALL segment levels (sentence + clause), like cnp_surface — short embedded quotes
    # (c131's «Сих есть царствие божие» clause) surface here, diluted at the full-sentence level.
    for s in segs:
        cand = defaultdict(float)
        shared = defaultdict(list)
        for p in pairs(s["text"]):
            for vi in inv.get(p, ()):
                cand[vi] += weight(p)
                shared[vi].append("+".join(sorted(p)))
        if not cand:
            continue
        best = max(cand.values())                                # top-1 per segment (ties kept)
        top = [{"nt_ref": vref[vi], "score": round(sc, 2), "pairs": shared[vi]}
               for vi, sc in cand.items() if sc >= best]
        rows.append({"seg_id": s["id"], "paragraph_ref": s["id"].split("/")[0], "top": top})

    out = H + "results/crime_and_punishment/cnp_wpair_top5.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} sentences with a distinctive pair-match -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
