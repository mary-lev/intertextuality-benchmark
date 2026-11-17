"""Character 4-gram retrieval for the Crime and Punishment set: overlap coefficient over the
character 4-grams of the normalized surface, top five verses per sentence.

    python scripts/cnp_char.py   # -> results/crime_and_punishment/cnp_char_top5.jsonl
"""
import json, re, sys
from collections import defaultdict
import ngram_baseline as nb
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
    vsurf, ssurf = load_surface()
    print("char-4gram overlap…", file=sys.stderr)
    vchar = {r: char_ngrams(t) for r, t in vsurf.items()}
    schar = {s: char_ngrams(t) for s, t in ssurf.items()}
    write(nb.HERE + "results/crime_and_punishment/cnp_char_top5.jsonl", retrieve(schar, vchar, min_share=3))


if __name__ == "__main__":
    main()
