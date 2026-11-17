"""1823 Russian NT surface matcher (thin caller over cnp_surface.surface_match).

Scores the shared words by **CnP-idf** (rarity in the novel) instead of NT-idf: a word common in
Dostoevsky's prose (e.g. «по крайней мере») is down-weighted to near-zero, so novel-idiom attractors
collapse — `1Кор 9:2` («крайней мере») went 121×→4×, the worst attractor profile flattened (max ~23×
vs 121×), and real finds rise (`Лк 7:47` #6→#1, with the synoptic «прощаются грехи» parallels). Gold
preserved; candidate generation, floor (15), and top-5 unchanged (compared head to head, 2026-06-25).
"""
import json
import math
import sys
from collections import defaultdict

from cnp_surface import surface_match, toks, H


def cnp_idf():
    """idf of each surface word over the novel's SENTENCES (the prose control corpus)."""
    cdf = defaultdict(int)
    n = 0
    for l in open(H + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8"):
        s = json.loads(l)
        if s["level"] != "sentence":
            continue
        n += 1
        for w in set(toks(s["text"])):
            cdf[w] += 1
    return {w: math.log(n / d) for w, d in cdf.items()}


if __name__ == "__main__":
    # FLOOR=22 (raised from 15 after the floor-vs-anchor sweep): the 15–22 band was ~pure
    # common-word coincidence; 22 keeps every gold case (Лк 7:47, the «прощаются» family, Lazarus)
    # and cuts the candidate volume ~95% (3,502 → ~186), ~20× cleaner. The residual margin is the
    # scattered-word class that needs a proximity bonus, not a higher floor.
    ci = cnp_idf()
    # score = summed CnP-idf of shared words (rarity, gated at floor 22)  +  contiguity bonus
    # (the longest adjacent shared-word run in the verse, summed CnP-idf). The bonus is re-rank
    # only — it surfaces contiguous quotations (the Lord's Prayer «приидет царствие твое» rises
    # to the top) and sinks scattered-word coincidences of the same rarity, without changing the
    # candidate set.
    n = surface_match("data/crime_and_punishment/nt1823.jsonl", "nt_ref", "results/crime_and_punishment/cnp_nt_surface.jsonl",
                      floor=22.0, topk=5, score_idf=ci, proximity_idf=ci)
    print(f"{n} NT surface connections (CnP-idf + contiguity bonus, floor 22, top-5) "
          f"-> results/crime_and_punishment/cnp_nt_surface.jsonl", file=sys.stderr)
