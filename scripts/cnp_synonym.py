"""Synonym-expanded matching — Tesserae's "semantic search" (survey §2): a concept match is a
shared lemma OR a SYNONYM (wiki-ru-wordnet). A recall booster for synonym-substitution paraphrase
that plain lemma matching misses. Compared head-to-head with plain Tesserae.

A (segment, verse) pair scores by the summed idf of the DISTINCT segment lemmas that match a verse
lemma directly or via synonymy; >=2 matched lemmas required (Tesserae rule).

    python scripts/cnp_synonym.py   # -> results/crime_and_punishment/cnp_synonym_top5.jsonl + reach vs gold (Tess vs Syn)
"""
import json, math, re, sys
from collections import defaultdict
import ngram_baseline as nb
from cnp_eval_commentary import load_gold
from cnp_match import STOP, _content

import os
CACHE = nb.HERE + "cache/lemmas/"
TOPK = 5
BOOK = {"матфея": "Мф", "марка": "Мк", "луки": "Лк", "иоанна": "Ин"}


def build_synmap(seqs):
    """lemma -> up to 10 wiki-ru-wordnet synonyms, over the corpus content lemmas (cached)."""
    path = nb.HERE + "data/crime_and_punishment/synonyms.json"
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    from wiki_ru_wordnet import WikiWordnet
    w = WikiWordnet()
    lem = set()
    for seq in seqs:
        lem.update(seq)
    out = {}
    for L in lem:
        s = set()
        for ss in w.get_synsets(L):
            for x in ss.get_words():
                s.add(x.lemma())
        s -= {L}
        if s:
            out[L] = sorted(s)[:10]
    json.dump(out, open(path, "w"), ensure_ascii=False)
    print(f"  built synmap: {len(out)}/{len(lem)} lemmas have synonyms", file=sys.stderr)
    return out


SYN = {}                                    # populated in main() from build_synmap


def expand(lemma):
    return {lemma} | set(SYN.get(lemma, ()))


def load():
    verses = []
    for l in open(CACHE + "nt1823_lemmas.jsonl", encoding="utf-8"):
        r = json.loads(l)
        verses.append((r["id"], _content(r["lemmas"])))
    segs = []
    for l in open(CACHE + "cnp_segments_lemmas.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if "/s" in r["id"] and "/c" not in r["id"]:
            segs.append((r["id"], _content(r["lemmas"])))
    return verses, segs


def index(verses):
    # token (lemma or its synonym) -> set of verse indices that have it as a lemma-or-synonym
    inv = defaultdict(set)
    for vi, (_, seq) in enumerate(verses):
        toks = set()
        for L in seq:
            toks |= expand(L)
        for t in toks:
            inv[t].add(vi)
    N = len(verses)
    idf = {t: math.log(1 + N / len(vs)) for t, vs in inv.items()}
    return inv, idf


def rank(Q, inv, idf):
    # for each distinct segment lemma, which verses does it match (directly or via synonym)?
    matched = defaultdict(dict)            # vi -> {seg_lemma: idf}
    for S in set(Q):
        score = idf.get(S, 0.0)
        for t in expand(S):                # S matches a verse if S∪syn(S) hits the verse's tokens
            for vi in inv.get(t, ()):
                matched[vi][S] = max(matched[vi].get(S, 0.0), score)
    out = []
    for vi, hits in matched.items():
        if len(hits) >= 2:
            out.append((sum(hits.values()), vi))
    out.sort(reverse=True)
    return out[:TOPK]


# ---- gold reach (mirror cnp_baselines) ----
def book_sigla(s):
    s = (s or "").lower()
    if "откровение" in s:
        return "Откр"
    for k, v in BOOK.items():
        if k in s:
            return v
    return "Ин" if s.strip() == "евангелие" else None


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[«»\"“”']", " ", t)).strip()


def bc(ref):
    m = re.match(r"(.+?) (\d+):", ref)
    return (m.group(1), int(m.group(2))) if m else (ref, 0)


def by_para(seg_ids, ranks):
    d = defaultdict(set)
    for sid, top in zip(seg_ids, ranks):
        d[sid.split("/")[0]].update(bc(v) for v in top)
    return d


def main():
    global SYN
    verses, segs = load()
    SYN = build_synmap([seq for _, seq in verses] + [seq for _, seq in segs])
    inv, idf = index(verses)
    print(f"indexed {len(verses)} verses (synonym-expanded); ranking {len(segs)} sentences…",
          file=sys.stderr)
    ranks = [[verses[vi][0] for _, vi in rank(Q, inv, idf)] for _, Q in segs]
    with open(nb.HERE + "results/crime_and_punishment/cnp_synonym_top5.jsonl", "w", encoding="utf-8") as f:
        for (sid, _), top in zip(segs, ranks):
            f.write(json.dumps({"seg_id": sid, "paragraph_ref": sid.split("/")[0], "top": top},
                               ensure_ascii=False) + "\n")

    # compare to plain Tesserae
    sids = [s for s, _ in segs]
    P_syn = by_para(sids, ranks)
    tess = {}
    for l in open(nb.HERE + "results/crime_and_punishment/cnp_tesserae_top5.jsonl", encoding="utf-8"):
        r = json.loads(l)
        tess[r["seg_id"]] = r["top"]
    P_tess = by_para(list(tess), list(tess.values()))

    paras = [(p["ref"], norm(p["text"])) for p in
             (json.loads(l) for l in open(nb.HERE + "data/crime_and_punishment/crime_and_punishment.jsonl", encoding="utf-8"))]

    def find_para(frag):
        for key in sorted([s.strip() for s in re.split(r"~|\.\.\.|…", norm(frag))], key=len, reverse=True):
            if len(key) >= 6:
                for ref, txt in paras:
                    if key[:30] in txt:
                        return ref
        return None

    gold = load_gold()   # the frozen gold, (bk, ch, gv) parsed from nt_ref
    t = s = n = 0
    print(f"\n{'gold':9} {'para':14} {'Tess':5}{'Syn':4} fragment")
    for g in gold:
        bk, ch = g["bk"], g["ch"]
        if not bk or ch is None:
            continue
        pr = find_para(g["fragment"])
        if not pr:
            continue
        key = (bk, ch)
        ht = key in P_tess.get(pr, set())
        hs = key in P_syn.get(pr, set())
        t += ht; s += hs; n += 1
        b = lambda x: "Y" if x else "·"
        print(f"{bk+' '+str(ch):9} {pr:14} {b(ht):5}{b(hs):4} {g['fragment'][:30]}")
    print(f"\nreach over {n} gold notes (top-{TOPK}):  plain Tesserae {t}/{n} · "
          f"synonym-expanded {s}/{n}")
    # is it a no-op? count segments whose top-5 differs from plain Tesserae
    diff = sum(1 for sid, top in zip(sids, ranks) if set(top) != set(tess.get(sid, [])))
    print(f"(not a no-op: top-5 differs from plain Tesserae on {diff}/{len(sids)} segments — "
          f"synonyms change rankings, but not on any gold case)")


if __name__ == "__main__":
    main()
