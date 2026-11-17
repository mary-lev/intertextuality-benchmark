"""Experiment: match CnP segments against the Church-Slavonic NT (cs_nt) with the SAME
rarity-weighted Smith-Waterman matcher as cnp_match.py, and compare to the 1823-Russian
(nt1823) result. Tests whether the CS NT recovers the «се»-register quotations the 1823
Russian NT misses.

    python scripts/cnp_match_cs.py    # -> results/crime_and_punishment/cnp_match_cs_raw.jsonl + a comparison report
"""
import json, math, os, sys
from collections import defaultdict
import ngram_baseline as nb
import cnp_match as M
from cnp_lemmatize import _lemmatize_file

CACHE = nb.HERE + "cache/lemmas/"
CS = nb.HERE + "data/crime_and_punishment/cs_nt.jsonl"
CS_LEM = CACHE + "cs_nt_lemmas.jsonl"


def ensure_cs_lemmas():
    if os.path.exists(CS_LEM):
        return
    from pymystem3 import Mystem
    m = Mystem()
    cs = [json.loads(x) for x in open(CS, encoding="utf-8")]
    _lemmatize_file(m, [v["ref"] for v in cs], [v["text"] for v in cs], CS_LEM)


def load(verse_lem_path):
    seg = [json.loads(x) for x in open(CACHE + "cnp_segments_lemmas.jsonl", encoding="utf-8")]
    verses = [json.loads(x) for x in open(verse_lem_path, encoding="utf-8")]
    for v in verses:
        v["seq"] = M._content(v["lemmas"])
    df = defaultdict(int)
    for v in verses:
        for w in set(v["seq"]):
            df[w] += 1
    N = len(verses)
    idf = {w: math.log(N / d) for w, d in df.items()}
    return seg, verses, idf


def best_per_seg(hits):
    b = {}
    for h in hits:
        if h["seg_id"] not in b or h["score"] > b[h["seg_id"]]["score"]:
            b[h["seg_id"]] = h
    return b


def main():
    ensure_cs_lemmas()
    seg, cs_v, cs_idf = load(CS_LEM)
    _, ru_v, ru_idf = load(CACHE + "nt1823_lemmas.jsonl")
    cs_hits = M.match_all(seg, cs_v, cs_idf, topk=3)
    ru_hits = M.match_all(seg, ru_v, ru_idf, topk=3)
    with open(nb.HERE + "results/crime_and_punishment/cnp_match_cs_raw.jsonl", "w", encoding="utf-8") as f:
        for h in cs_hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"CS candidate links: {len(cs_hits)}  |  RU(nt1823): {len(ru_hits)}", file=sys.stderr)

    csb, rub = best_per_seg(cs_hits), best_per_seg(ru_hits)
    seg_ids = {s["id"] for s in seg}

    print("\n=== flagged CS-register paragraphs: best CS vs best RU match ===")
    for prefix, label in [("ч1.гл2:p63", "«Се человек»"),
                          ("ч1.гл2:p73", "«царствие божие»"),
                          ("ч4.гл4:p2466", "«Сих есть…»")]:
        for sid in sorted(s for s in seg_ids if s.startswith(prefix + "/")):
            cs, ru = csb.get(sid), rub.get(sid)
            if not (cs or ru):
                continue
            print(f"  {sid}  {label}")
            print(f"     CS: {cs['nt_ref']+'  score '+str(cs['score']) if cs else '— (no match)'}"
                  + (f"  v={cs['v_lemmas']}" if cs else ""))
            print(f"     RU: {ru['nt_ref']+'  score '+str(ru['score']) if ru else '— (no match)'}"
                  + (f"  v={ru['v_lemmas']}" if ru else ""))

    cs_only = [s for s in csb if s not in rub]
    print(f"\n=== segments matched by CS but NOT by the 1823 Russian: {len(cs_only)} ===")
    for sid in sorted(cs_only, key=lambda s: -csb[s]["score"])[:20]:
        h = csb[sid]
        print(f"  {sid:24} -> {h['nt_ref']:11} score {h['score']:>5}  v={h['v_lemmas']}")


if __name__ == "__main__":
    main()
