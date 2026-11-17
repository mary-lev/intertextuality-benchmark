"""First baseline — idf-weighted n-gram overlap retrieval.

The surface floor for the verbal benchmark (EXPERIMENT.md §5/§8). For each grounded
poetry connection we take the source-side anchor (commented_fragment if present, else the
whole commented poem), score every corpus poem by the rarity-weighted fraction of the
query's n-grams it shares, rank, and check whether the gold referent (or a text-duplicate /
cycle part) lands in the top-k.

Two representations, run side by side:
  * word bigrams   — the classic text-reuse unit; brittle to Russian inflection
  * char 4-grams   — shares root substrings, survives inflection

Plus a Step-0 sanity run (query = the referent's own citation) that should score ~100% —
if it doesn't, the corpus/linkage/dedup is wrong, not the method.

    python scripts/ngram_baseline.py        # run from the release root

Stdlib only.
"""
from __future__ import annotations
import csv, os, re, math, sys
from collections import defaultdict, Counter
from difflib import SequenceMatcher

from rank_metrics import metrics as _rm_metrics

csv.field_size_limit(10**8)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"   # release root
KS = (1, 5, 10, 50)


# ---------------------------------------------------------------- text / n-grams
def norm(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"[̀-ͯ]", "", s)        # strip editorial stress marks
    s = re.sub(r"[^\w\s]", " ", s, flags=re.U)
    return re.sub(r"\s+", " ", s).strip()


def word_ngrams(t: str, n: int = 2) -> set[str]:
    w = t.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= n else set(w)


def char_ngrams(t: str, n: int = 4) -> set[str]:
    return {t[i:i + n] for i in range(len(t) - n + 1)} if len(t) >= n else {t}


# ---------------------------------------------------------------- corpus + index
def load_added_referents() -> list[dict]:
    """The referent poems PoeTree lacks, read from the gold (rows whose referent_source is
    'extra'): id, text, year, author and title, in id order."""
    rows = {}
    for r in csv.DictReader(open(HERE + "data/russian_poetry/gold.csv", encoding="utf-8")):
        if r["referent_source"] == "extra":
            rows.setdefault(r["referent_text_id"], {
                "extra_id": r["referent_text_id"], "full_text": r["referent_text"],
                "year": r["referent_year"], "ref_author": r["ref_author"], "ref_title": r["ref_title"]})
    return sorted(rows.values(), key=lambda x: int(x["extra_id"].split("-")[1]))


def load_corpus() -> dict[str, str]:
    """poetree_poem_id / extra_id -> normalized full text (minus excluded near-duplicates)."""
    corpus = {}
    with open(HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8") as f:
        for p in csv.DictReader(f):
            corpus[p["poetree_poem_id"]] = norm(p["full_text"])
    for r in load_added_referents():
        corpus[r["extra_id"]] = norm(r["full_text"])          # off-corpus referents = first-class
    return corpus


def load_years() -> dict[str, int]:
    """poem id -> year of composition (poems without a usable year are absent)."""
    years = {}
    with open(HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8") as f:
        for p in csv.DictReader(f):
            try:
                years[p["poetree_poem_id"]] = int(float(p["year_created_from"]))
            except (ValueError, TypeError):
                pass
    for r in load_added_referents():
        try:
            years[r["extra_id"]] = int(float(r["year"]))
        except (ValueError, TypeError):
            pass
    return years


def not_later(years, self_ids):
    """Chronology filter: -> predicate keeping the candidates not dated after the commented
    poem (its latest dated copy). Undated candidates are kept; with an undated commented
    poem nothing is dropped. Uses nothing about the referent."""
    qy = [years[s] for s in self_ids if s in years]
    if not qy:
        return lambda pid: True
    qy = max(qy)
    return lambda pid: years.get(pid, qy) <= qy


def dup_groups(corpus: dict[str, str]) -> dict[str, set[str]]:
    """id -> set of ids with text-identical content (so a duplicate counts as a hit)."""
    by_text = defaultdict(set)
    for pid, t in corpus.items():
        by_text[t].add(pid)
    return {pid: by_text[t] for pid, t in corpus.items()}


def build_index(corpus: dict[str, str], gram):
    index = defaultdict(set)
    for pid, t in corpus.items():
        for ng in gram(t):
            index[ng].add(pid)
    N = len(corpus)
    idf = {ng: math.log(N / len(pids)) for ng, pids in index.items()}
    return index, idf


def rank(query_grams: set[str], index, idf) -> dict[str, float]:
    qtot = sum(idf.get(ng, 0.0) for ng in query_grams) or 1.0
    score = defaultdict(float)
    for ng in query_grams:
        w = idf.get(ng)
        if w is None:
            continue
        for pid in index[ng]:
            score[pid] += w
    return {pid: s / qtot for pid, s in score.items()}


def best_rank(scores: dict[str, float], hit_ids: set[str]) -> int | None:
    """1-based rank of the highest-scoring hit id (None if none scored).

    Tie-break by poem_id so equal scores resolve deterministically across runs
    (otherwise set/dict iteration order, which depends on PYTHONHASHSEED, made the
    reported R@k wobble ~±0.03 between runs)."""
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    for i, (pid, _) in enumerate(order, 1):
        if pid in hit_ids:
            return i
    return None


# ---------------------------------------------------------------- gold
def load_gold(dups):
    """Grounded poetry connections with query anchor, hit set, stratum, exactness."""
    rows = [r for r in csv.DictReader(open(HERE + "data/russian_poetry/gold.csv", encoding="utf-8"))
            if r["category"] == "poetry" and r["referent_text"].strip() and r["commented_text"].strip()]
    out = []
    for r in rows:
        # prefer the analyst-restored span over the scholar's (possibly elided) fragment/citation,
        # consistent with the line-level eval — the analyst span is the corrected, full text
        frag = (r["analyst_commented_span"] or r["commented_fragment"]).strip()
        cit = (r["analyst_referent_span"] or r["ref_citation"]).strip()
        # span-availability stratum (EXPERIMENT §9)
        has_cm = bool(frag); has_rf = bool(cit)
        stratum = "S1" if (has_cm and has_rf) else "S2" if has_cm else "S3" if has_rf else "S4"
        # hit set = gold referent id(s) + their text-duplicates
        hit = set()
        for pid in r["referent_text_id"].split(";"):
            hit |= dups.get(pid, {pid})
        # the query's OWN commented poem must be excluded from candidates (you don't
        # retrieve the poem you already hold; otherwise a fragment self-retrieves at rank 1)
        selfids = set()
        for pid in (r["commented_text_id"] or "").split(";"):
            if pid:
                selfids |= dups.get(pid, {pid})
        # exactness band (S1 only)
        band = ""
        if stratum == "S1":
            sr = SequenceMatcher(None, norm(frag), norm(cit)).ratio()
            band = "verbatim" if sr >= 0.85 else "close" if sr >= 0.5 else "loose"
        out.append({
            "conn_id": r["conn_id"], "stratum": stratum, "band": band,
            "q_frag": norm(frag) if frag else norm(r["commented_text"]),   # fragment else whole poem
            "q_cit": norm(cit),                                            # for the sanity run
            "hit": hit, "self": selfids,                                   # exclude commented poem
        })
    return out


# ---------------------------------------------------------------- metrics / report
def metrics(ranks: list[int | None]) -> dict:
    return _rm_metrics(ranks, KS)


def run(gold, qkey, index, idf, label):
    by_all, by_strat, by_band = [], defaultdict(list), defaultdict(list)
    sub = [g for g in gold if g[qkey]]
    for g in sub:
        gram_fn = char_ngrams if "char" in label else word_ngrams
        scores = rank(gram_fn(g[qkey]), index, idf)
        for pid in g["self"]:                       # drop the query's own commented poem
            scores.pop(pid, None)
        r = best_rank(scores, g["hit"])
        by_all.append(r); by_strat[g["stratum"]].append(r)
        if g["band"]:
            by_band[g["band"]].append(r)
    print(f"\n### {label} — query = {qkey}  (n={len(sub)})")
    m = metrics(by_all)
    print(f"  ALL      n={m['n']:3d}  " + "  ".join(f"R@{k}={m[f'R@{k}']:.2f}" for k in KS) + f"  MRR={m['MRR']:.3f}")
    for s in ("S1", "S2", "S3", "S4"):
        if by_strat[s]:
            m = metrics(by_strat[s])
            print(f"  {s}       n={m['n']:3d}  " + "  ".join(f"R@{k}={m[f'R@{k}']:.2f}" for k in KS) + f"  MRR={m['MRR']:.3f}")
    for b in ("verbatim", "close", "loose"):
        if by_band[b]:
            m = metrics(by_band[b])
            print(f"  S1/{b:8s} n={m['n']:3d}  " + "  ".join(f"R@{k}={m[f'R@{k}']:.2f}" for k in KS) + f"  MRR={m['MRR']:.3f}")


def main():
    print("loading corpus…", file=sys.stderr)
    corpus = load_corpus()
    dups = dup_groups(corpus)
    gold = load_gold(dups)
    print(f"corpus: {len(corpus)} texts | grounded poetry connections: {len(gold)}", file=sys.stderr)

    for label, gram in (("word-2gram", lambda t: word_ngrams(t, 2)),
                        ("char-4gram", lambda t: char_ngrams(t, 4))):
        print(f"building {label} index…", file=sys.stderr)
        index, idf = build_index(corpus, gram)
        run(gold, "q_cit", index, idf, label + " [SANITY citation→referent]")
        run(gold, "q_frag", index, idf, label + " [fragment→referent]")
        del index, idf


if __name__ == "__main__":
    main()
