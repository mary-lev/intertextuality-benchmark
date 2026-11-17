"""Two more lexical methods from the survey §5, both cheap:

  * OVERLAP COEFFICIENT  |A∩B| / min(|A|,|B|) over content-lemma sets — the survey notes it is
    "better when lengths differ greatly, e.g. a short quote in a long work", i.e. the embedded-quote
    case. (Jaccard would penalize the short side; the min-normalization does not.)
  * CHAR n-gram  overlap coefficient over character 4-grams of the normalized surface — the
    character-level family the survey recommends for OCR/orthographic noise (here a robustness check;
    our corpus is clean so it should track the lemma methods). This is the scalable stand-in for
    edit/Levenshtein distance (§5), which is the same idea at pairwise granularity.

    python scripts/cnp_lexical_extra.py   # -> results/crime_and_punishment/cnp_overlap_top5.jsonl, results/crime_and_punishment/cnp_char_top5.jsonl
                                            #    + reach vs gold (overlap / char / BM25 / Tesserae)
"""
import json, re, sys
from collections import defaultdict
import ngram_baseline as nb
from cnp_eval_commentary import load_gold
from cnp_match import _content

CACHE = nb.HERE + "cache/lemmas/"
TOPK = 5
BOOK = {"матфея": "Мф", "марка": "Мк", "луки": "Лк", "иоанна": "Ин"}


def norm_surface(t):
    return re.sub(r"[^а-я0-9]", "", t.lower().replace("ё", "е"))


def char_ngrams(t, n=4):
    s = norm_surface(t)
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()


def overlap_coef(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def load_lemma_sets():
    verses = {}
    for l in open(CACHE + "nt1823_lemmas.jsonl", encoding="utf-8"):
        r = json.loads(l)
        verses[r["id"]] = set(_content(r["lemmas"]))
    segs = {}
    for l in open(CACHE + "cnp_segments_lemmas.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if "/s" in r["id"] and "/c" not in r["id"]:
            segs[r["id"]] = set(_content(r["lemmas"]))
    return verses, segs


def load_surface():
    verses = {v["ref"]: v["text"] for v in
              (json.loads(l) for l in open(nb.HERE + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8"))}
    segs = {s["id"]: s["text"] for s in
            (json.loads(l) for l in open(nb.HERE + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8"))
            if s["level"] == "sentence"}
    return verses, segs


def retrieve(seg_sets, verse_sets, min_share=2):
    """generic set-overlap retrieval with an inverted index for candidate gen."""
    inv = defaultdict(list)
    vids = list(verse_sets)
    for vi, vid in enumerate(vids):
        for tok in verse_sets[vid]:
            inv[tok].append(vi)
    out = {}
    for sid, S in seg_sets.items():
        cand = defaultdict(int)
        for tok in S:
            for vi in inv.get(tok, ()):
                cand[vi] += 1
        scored = []
        for vi, shared in cand.items():
            if shared >= min_share:
                scored.append((overlap_coef(S, verse_sets[vids[vi]]), vids[vi]))
        scored.sort(reverse=True)
        out[sid] = [r for _, r in scored[:TOPK]]
    return out


# ---- gold reach (mirror cnp_baselines) ----
def book_sigla(s):
    s = (s or "").lower()
    if "откровение" in s:
        return "Откр"
    for k, v in BOOK.items():
        if k in s:
            return v
    return "Ин" if s.strip() == "евангелие" else None


def normg(t):
    return re.sub(r"\s+", " ", re.sub(r"[«»\"“”']", " ", t)).strip()


def bc(ref):
    m = re.match(r"(.+?) (\d+):", ref)
    return (m.group(1), int(m.group(2))) if m else (ref, 0)


def by_para(d):
    out = defaultdict(set)
    for sid, top in d.items():
        out[sid.split("/")[0]].update(bc(v) for v in top)
    return out


def load_top(path):
    return {r["seg_id"]: [h["nt_ref"] if isinstance(h, dict) else h for h in r["top"]]
            for r in (json.loads(l) for l in open(path, encoding="utf-8"))}


def write(path, d):
    with open(path, "w", encoding="utf-8") as f:
        for sid, top in d.items():
            f.write(json.dumps({"seg_id": sid, "paragraph_ref": sid.split("/")[0], "top": top},
                               ensure_ascii=False) + "\n")


def main():
    vlem, slem = load_lemma_sets()
    print(f"overlap-coefficient over content lemmas ({len(slem)} sentences)…", file=sys.stderr)
    ov = retrieve(slem, vlem, min_share=2)
    write(nb.HERE + "results/crime_and_punishment/cnp_overlap_top5.jsonl", ov)

    vsurf, ssurf = load_surface()
    print("char-4gram overlap…", file=sys.stderr)
    vchar = {r: char_ngrams(t) for r, t in vsurf.items()}
    schar = {s: char_ngrams(t) for s, t in ssurf.items()}
    ch = retrieve(schar, vchar, min_share=3)
    write(nb.HERE + "results/crime_and_punishment/cnp_char_top5.jsonl", ch)

    # reach vs gold, alongside BM25/Tesserae
    P_ov, P_ch = by_para(ov), by_para(ch)
    P_bm = by_para(load_top(nb.HERE + "results/crime_and_punishment/cnp_bm25_top5.jsonl"))
    P_te = by_para(load_top(nb.HERE + "results/crime_and_punishment/cnp_tesserae_top5.jsonl"))
    paras = [(p["ref"], normg(p["text"])) for p in
             (json.loads(l) for l in open(nb.HERE + "data/crime_and_punishment/crime_and_punishment.jsonl", encoding="utf-8"))]

    def find_para(frag):
        for key in sorted([s.strip() for s in re.split(r"~|\.\.\.|…", normg(frag))], key=len, reverse=True):
            if len(key) >= 6:
                for ref, txt in paras:
                    if key[:30] in txt:
                        return ref
        return None

    gold = load_gold()   # the frozen gold, (bk, ch, gv) parsed from nt_ref
    tot = defaultdict(int)
    print(f"\n{'gold':9} {'para':14} {'over':5}{'char':5}{'bm25':5}{'tess':5} fragment")
    for g in gold:
        bk, ch_ = g["bk"], g["ch"]
        if not bk or ch_ is None:
            continue
        pr = find_para(g["fragment"])
        if not pr:
            continue
        key = (bk, ch_)
        h = {"over": key in P_ov.get(pr, set()), "char": key in P_ch.get(pr, set()),
             "bm25": key in P_bm.get(pr, set()), "tess": key in P_te.get(pr, set())}
        for k, v in h.items():
            tot[k] += v
        tot["n"] += 1
        b = lambda x: "Y" if x else "·"
        print(f"{bk+' '+str(ch_):9} {pr:14} {b(h['over']):5}{b(h['char']):5}{b(h['bm25']):5}"
              f"{b(h['tess']):5} {g['fragment'][:28]}")
    n = tot["n"]
    print(f"\nreach over {n} gold notes (top-{TOPK}): overlap-coef {tot['over']}/{n} · "
          f"char-4gram {tot['char']}/{n} · bm25 {tot['bm25']}/{n} · tesserae {tot['tess']}/{n}")


if __name__ == "__main__":
    main()
