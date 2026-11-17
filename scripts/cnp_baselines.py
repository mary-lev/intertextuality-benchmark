"""BM25 and Tesserae baselines for CnP->NT, the lexical floor methods from EXPERIMENT.md §5.1-5.2.
Both index the 7,952 NT verses and rank them per CnP sentence segment. Expected to be redundant
with our Smith-Waterman alignment (all lexical, correlated) — run for completeness of the sweep.

  * BM25     : Okapi BM25 over verse content-lemma bags (k1=1.5, b=0.75).
  * Tesserae : verses sharing >=2 content lemmas, scored by summed idf of the shared lemmas.

    python scripts/cnp_baselines.py   # -> results/crime_and_punishment/cnp_bm25_top5.jsonl, results/crime_and_punishment/cnp_tesserae_top5.jsonl
                                        #    + reach comparison vs gold (lex/dense/bm25/tess/union)
"""
import json, math, re, sys
from collections import defaultdict
import ngram_baseline as nb
from cnp_eval_commentary import load_gold
from cnp_match import STOP, _content

CACHE = nb.HERE + "cache/lemmas/"
TOPK = 5
K1, B = 1.5, 0.75
BM25_FLOOR = 18.0   # BM25 is CnP-idf-weighted + floored to a shortlist (see cnp_idf / bm25_rank)
BOOK = {"матфея": "Мф", "марка": "Мк", "луки": "Лк", "иоанна": "Ин"}


def load():
    verses = []
    for l in open(CACHE + "nt1823_lemmas.jsonl", encoding="utf-8"):
        r = json.loads(l)
        verses.append((r["id"], _content(r["lemmas"])))
    segs = []
    for l in open(CACHE + "cnp_segments_lemmas.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if "/s" in r["id"] and "/c" not in r["id"]:        # sentence-level only (match dense)
            segs.append((r["id"], _content(r["lemmas"])))
    return verses, segs


def index(verses):
    N = len(verses)
    df = defaultdict(int)
    inv = defaultdict(list)
    dl = []
    for vi, (_, seq) in enumerate(verses):
        dl.append(len(seq))
        for w in set(seq):
            df[w] += 1
            inv[w].append(vi)
    idf = {w: math.log(1 + (N - d + 0.5) / (d + 0.5)) for w, d in df.items()}
    avgdl = sum(dl) / max(N, 1)
    return idf, inv, dl, avgdl


def cnp_idf(segs):
    """idf of each content lemma over the novel's sentence segments (the prose control corpus)."""
    cdf = defaultdict(int)
    n = len(segs)
    for _, Q in segs:
        for w in set(Q):
            cdf[w] += 1
    return {w: math.log(n / d) for w, d in cdf.items()}


def bm25_rank(Q, verses, idf, inv, dl, avgdl, floor=0.0):
    """BM25 with `idf` = CnP-idf (rarity in the novel, not the NT) and a score `floor`, so the
    output is a clean shortlist of strong matches rather than top-5-for-every-segment.
    Returns [(score, ref)] (scores kept, unlike the old refs-only version)."""
    cand = set()
    for w in set(Q):
        cand.update(inv.get(w, ()))
    qset = set(Q)
    out = []
    for vi in cand:
        seq = verses[vi][1]
        tf = defaultdict(int)
        for w in seq:
            if w in qset:
                tf[w] += 1
        s = 0.0
        for w, f in tf.items():
            s += idf.get(w, 0) * f * (K1 + 1) / (f + K1 * (1 - B + B * dl[vi] / avgdl))
        if s >= floor:
            out.append((round(s, 2), verses[vi][0]))
    out.sort(reverse=True)
    return out[:TOPK]


def tesserae_rank(Q, verses, idf, inv):
    cand = defaultdict(int)
    for w in set(Q):
        for vi in inv.get(w, ()):
            cand[vi] += 1
    qset = set(Q)
    out = []
    for vi, shared in cand.items():
        if shared < 2:
            continue
        s = sum(idf.get(w, 0) for w in set(verses[vi][1]) & qset)
        out.append((s, verses[vi][0]))
    out.sort(reverse=True)
    return [r for _, r in out[:TOPK]]


def write(path, segs, ranks):
    with open(path, "w", encoding="utf-8") as f:
        for (sid, _), top in zip(segs, ranks):
            # bm25 returns [(score, ref)] (keep scores); tesserae returns [ref]
            out_top = [{"nt_ref": t[1], "score": t[0]} if isinstance(t, tuple) else t for t in top]
            f.write(json.dumps({"seg_id": sid, "paragraph_ref": sid.split("/")[0],
                                "top": out_top}, ensure_ascii=False) + "\n")


# ---------- gold reach comparison ----------
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
        d[sid.split("/")[0]].update(bc(t[1] if isinstance(t, tuple) else t) for t in top[:TOPK])
    return d


def load_top(path):
    ids, tops = [], []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        ids.append(r["seg_id"]); tops.append([h["nt_ref"] if isinstance(h, dict) else h for h in r["top"]])
    return ids, tops


def main():
    verses, segs = load()
    idf, inv, dl, avgdl = index(verses)
    cidf = cnp_idf(segs)        # CnP-rarity weights for BM25 (kills novel-common-word noise)
    print(f"indexed {len(verses)} verses; ranking {len(segs)} sentence segments…", file=sys.stderr)
    bm = [bm25_rank(Q, verses, cidf, inv, dl, avgdl, floor=BM25_FLOOR) for _, Q in segs]
    te = [tesserae_rank(Q, verses, idf, inv) for _, Q in segs]
    write(nb.HERE + "results/crime_and_punishment/cnp_bm25_top5.jsonl", segs, bm)
    write(nb.HERE + "results/crime_and_punishment/cnp_tesserae_top5.jsonl", segs, te)
    nbm = sum(len(t) for t in bm)
    print(f"BM25 (CnP-idf, floor {BM25_FLOOR}): {nbm} shortlist candidates "
          f"(was top-5×all = {5*len(segs)})", file=sys.stderr)

    # reach vs gold, alongside lexical + dense
    sids = [s for s, _ in segs]
    P_bm = by_para(sids, bm)
    P_te = by_para(sids, te)
    di, dt = load_top(nb.HERE + "results/crime_and_punishment/cnp_dense_top5.jsonl")
    P_dense = by_para(di, dt)
    lex = defaultdict(list)
    for l in open(nb.HERE + "results/crime_and_punishment/cnp_match_features.jsonl", encoding="utf-8"):
        r = json.loads(l)
        lex[r["seg_id"]].append((r["nt_ref"], r["score"]))
    P_lex = defaultdict(set)
    for s, vs in lex.items():
        for ref, _ in sorted(vs, key=lambda x: -x[1])[:TOPK]:
            P_lex[s.split("/")[0]].add(bc(ref))

    paras = [(p["ref"], norm(p["text"])) for p in
             (json.loads(l) for l in open(nb.HERE + "data/crime_and_punishment/crime_and_punishment.jsonl", encoding="utf-8"))]

    def find_para(frag):
        for key in sorted([s.strip() for s in re.split(r"~|\.\.\.|…", norm(frag))], key=len, reverse=True):
            if len(key) >= 6:
                for ref, txt in paras:
                    if key[:30] in txt:
                        return ref
        fk = norm(frag)[:18]
        return next((ref for ref, txt in paras if fk and fk in txt), None)

    gold = load_gold()   # the frozen gold, (bk, ch, gv) parsed from nt_ref
    tot = defaultdict(int)
    print(f"\n{'gold':9} {'para':14} {'lex':4}{'dense':6}{'bm25':5}{'tess':5}{'union':6} fragment")
    for g in gold:
        bk, ch = g["bk"], g["ch"]
        if not bk or ch is None:
            continue
        pr = find_para(g["fragment"])
        if not pr:
            continue
        key = (bk, ch)
        h = {m: key in P.get(pr, set()) for m, P in
             (("lex", P_lex), ("dense", P_dense), ("bm25", P_bm), ("tess", P_te))}
        h["union"] = any(h.values())
        for m, v in h.items():
            tot[m] += v
        tot["n"] += 1
        b = lambda x: "Y" if x else "·"
        print(f"{bk+' '+str(ch):9} {pr:14} {b(h['lex']):4}{b(h['dense']):6}{b(h['bm25']):5}"
              f"{b(h['tess']):5}{b(h['union']):6} {g['fragment'][:30]}")
    n = tot["n"]
    print(f"\nreach over {n} gold notes (top-{TOPK}, chapter-level, incl. some same-chapter noise):"
          f"\n  lex {tot['lex']}/{n} · dense {tot['dense']}/{n} · bm25 {tot['bm25']}/{n} · "
          f"tesserae {tot['tess']}/{n} · UNION(all) {tot['union']}/{n}")


if __name__ == "__main__":
    main()
