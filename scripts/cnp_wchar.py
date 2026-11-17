"""CnP-frequency-weighted char-4gram scorer  (toggle: `wchar`).

The validated fix for `char`'s degenerate `overlap_coef` (short-verse attractors, perfect-1.0
short-interjection noise). Each (sentence, NT verse) is scored by the summed weight of their shared character
4-grams, where every gram is weighted by its rarity in the NOVEL:

    score = sum over shared grams of  CnP_idf(g)
    CnP_idf(g) = log(N_sentences / df over the novel sentences)
                 -> ordinary-prose grams (the «понимаете-с» noise) collapse to ~0; rare-in-the-novel
                    grams («ерусал», «царств») dominate. The novel is the prose control corpus.

The NT-side idf was REMOVED after measurement: it doesn't help (the candidates already share grams),
it compresses the gold-vs-noise separation (ratio 1.17 -> 1.57 without it), and it pushed c102 from
rank 2 to 3 — because rare-in-NT grams can sit in coincidental noise too. The borrowing signal is
purely "is this gram unusual in Dostoevsky's prose", i.e. CnP rarity. (DF over the novel's sentences,
not words — wchar is character-level throughout.)

Unlike `char`, this KEEPS the per-candidate score (in `top[].score`). In the output, `top` is a list
of {nt_ref, score}.

    python scripts/cnp_wchar.py   # -> results/crime_and_punishment/cnp_wchar_top5.jsonl  (+ gold-reach self-check)
"""
import json
import math
import re
import sys
from collections import defaultdict

import ngram_baseline as nb
from cnp_char import char_ngrams   # reuse the exact normalize + 4-gram tokenizer

H = nb.HERE
TOPK = 5
MIN_SHARE = 3      # >= shared 4-grams to be a candidate (same gate as `char`)
FLOOR = 0.0        # 0 = emit top-k for every sentence (enrich-only). Raise to use as a precision source.


def gram_idf(gram_sets, N):
    df = defaultdict(int)
    for s in gram_sets:
        for g in s:
            df[g] += 1
    return {g: math.log(N / d) for g, d in df.items()}


def main():
    verses = [json.loads(l) for l in open(H + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8")]
    vcg = [char_ngrams(v["text"]) for v in verses]
    vref = [v["ref"] for v in verses]

    sents = [s for s in (json.loads(l) for l in open(H + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8"))
             if s["level"] == "sentence"]
    scg = {s["id"]: char_ngrams(s["text"]) for s in sents}
    cnp_idf = gram_idf(list(scg.values()), len(scg))   # the prose control corpus = the novel itself

    inv = defaultdict(list)
    for vi, s in enumerate(vcg):
        for g in s:
            inv[g].append(vi)

    def weight(g):
        return cnp_idf.get(g, 0.0)        # CnP rarity only — NT-idf removed (measured: it hurt)

    rows = []
    for s in sents:
        S = scg[s["id"]]
        if len(S) < MIN_SHARE:
            continue
        cand = defaultdict(int)
        for g in S:
            for vi in inv.get(g, ()):
                cand[vi] += 1
        scored = []
        for vi, sh in cand.items():
            if sh < MIN_SHARE:
                continue
            sc = sum(weight(g) for g in (S & vcg[vi]))
            if sc >= FLOOR:
                scored.append((sc, vref[vi]))
        if not scored:
            continue
        scored.sort(reverse=True)
        rows.append({"seg_id": s["id"], "paragraph_ref": s["id"].split("/")[0],
                     "top": [{"nt_ref": r, "score": round(sc, 2)} for sc, r in scored[:TOPK]]})

    out = H + "results/crime_and_punishment/cnp_wchar_top5.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} sentences scored -> {out}", file=sys.stderr)
    _gold_reach(rows)


# ---- gold-reach self-check (verse-gated, against the frozen verbal gold) ----
def _parse(ref):
    m = re.match(r"(.+?) (\d+):(\d+)(?:-(\d+))?", ref)
    return (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4) or m.group(3))) if m else None


def _in_range(cand, gold):
    pc, pg = _parse(cand if ":" in cand else cand + ":1"), _parse(gold if ":" in gold else gold + ":1")
    return bool(pc and pg and pc[0] == pg[0] and pc[1] == pg[1] and pg[2] <= pc[2] <= pg[3])


def _gold_reach(rows):
    gp = H + "data/crime_and_punishment/gold.jsonl"
    try:
        gold = [g for g in (json.loads(l) for l in open(gp, encoding="utf-8"))
                if g["type"] in ("quotation", "allusion")]
    except FileNotFoundError:
        return
    by_para = defaultdict(list)
    for r in rows:
        by_para[r["paragraph_ref"]].append(r)
    print(f"\n  gold-reach self-check ({len(gold)} verbal notes): best rank of a gold-range verse", file=sys.stderr)
    for g in gold:
        best = None
        for r in by_para.get(g["paragraph_ref"], []):
            for i, h in enumerate(r["top"]):
                if _in_range(h["nt_ref"], g["nt_ref"]):
                    best = i + 1 if best is None else min(best, i + 1)
        print(f"    {g['id']:6} {g['nt_ref']:13} rank={best}", file=sys.stderr)


if __name__ == "__main__":
    main()
