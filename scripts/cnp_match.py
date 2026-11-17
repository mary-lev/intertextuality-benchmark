"""Match CnP segments to NT verses by rarity-weighted SW local alignment.

    python scripts/cnp_match.py    # -> results/crime_and_punishment/cnp_match_raw.jsonl
"""
import json, math, sys
from collections import defaultdict
import ngram_baseline as nb
from cnp_align import sw_span

CACHE = nb.HERE + "cache/lemmas/"
STOP = set("и в во не на он что с а как его это к но они мы за бы по ее так все".split())
DF_CAP = 400        # a query lemma in > DF_CAP verses is too common to seed
SCORE_FLOOR = 3.0   # raw SW floor to record a candidate at all


def _content(lemmas):
    return [w for w in lemmas.split() if w not in STOP and len(w) > 1 and w.isalpha()]


def load_lemmas():
    seg = [json.loads(x) for x in open(CACHE + "cnp_segments_lemmas.jsonl", encoding="utf-8")]
    verses = [json.loads(x) for x in open(CACHE + "nt1823_lemmas.jsonl", encoding="utf-8")]
    for v in verses:
        v["seq"] = _content(v["lemmas"])
    df = defaultdict(int)
    for v in verses:
        for w in set(v["seq"]):
            df[w] += 1
    N = len(verses)
    idf = {w: math.log(N / d) for w, d in df.items()}
    return seg, verses, idf


def match_all(seg, verses, idf, topk=3):
    inv = defaultdict(list)
    for vi, v in enumerate(verses):
        for w in set(v["seq"]):
            inv[w].append(vi)
    out = []
    for s in seg:
        Q = _content(s["lemmas"])
        if len(Q) < 2:
            continue
        cand = set()
        for w in set(Q):
            if idf.get(w, 0) > 0 and len(inv.get(w, ())) <= DF_CAP:
                cand.update(inv[w])
        scored = []
        for vi in cand:
            v = verses[vi]
            score, qi, qj, pi, pj = sw_span(Q, v["seq"], idf)
            if score >= SCORE_FLOOR:
                scored.append((score, vi, qi, qj, pi, pj))
        scored.sort(reverse=True)
        for score, vi, qi, qj, pi, pj in scored[:topk]:
            out.append({"seg_id": s["id"], "nt_ref": verses[vi]["id"],
                        "score": round(score, 3), "norm": round(score / max(len(Q), 1), 3),
                        "q_lemmas": Q[qi:qj], "v_lemmas": verses[vi]["seq"][pi:pj]})
    return out


def main():
    seg, verses, idf = load_lemmas()
    hits = match_all(seg, verses, idf, topk=3)
    out = nb.HERE + "results/crime_and_punishment/cnp_match_raw.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"{len(hits)} candidate links -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
