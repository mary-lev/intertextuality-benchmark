"""Background phrase-frequency tables for phrase-distinctiveness rescoring.

NOTE (superseded): the Poetree table built here is GENRE-MISMATCHED — Poetree is *poetry*, the
wrong background for *prose* phraseology (it scored ordinary prose phrases like «изъявить желание»
as distinctive). With only NT + CnP available, the prose background is CnP self-frequency and final
tiering lives in `cnp_tier.py`; the Poetree `phrase_bg_idf` column is retained but UNUSED. The CnP
self-frequency table is still used.

Builds two cached tables under .lemma_cache/ (git-ignored):

 1. Poetree document-frequency of content-lemma bi/trigrams (background Russian) ->
    poetree_phrase_df.json  {"N": N_poems, "bi": {ngram: df}, "tri": {ngram: df}}
 2. CnP self raw-frequency of content-lemma bigrams (authorial recurring collocations) ->
    cnp_phrase_freq.json    {"bi": {ngram: count}}

    python scripts/cnp_background.py
"""
import csv
import json
import sys
from collections import defaultdict

import ngram_baseline as nb
from cnp_match import STOP, _content

CACHE = nb.HERE + "cache/lemmas/"
POETREE_LEMMAS = CACHE + "poetree_lemmas.csv"
CNP_SEGMENTS = nb.HERE + "results/crime_and_punishment/cnp_segments.jsonl"
CNP_SEG_TOKENS = CACHE + "cnp_segments_tokens.jsonl"

POETREE_OUT = CACHE + "poetree_phrase_df.json"
CNP_OUT = CACHE + "cnp_phrase_freq.json"


def _is_content(lem):
    """The cnp_match._content rule applied to a single lemma."""
    return lem not in STOP and len(lem) > 1 and lem.isalpha()


def content_lemmas(lemmas):
    """Keep content lemmas from a list of lemma tokens."""
    return [w for w in lemmas if _is_content(w)]


def bigrams(seq):
    return [seq[i] + " " + seq[i + 1] for i in range(len(seq) - 1)]


def trigrams(seq):
    return [seq[i] + " " + seq[i + 1] + " " + seq[i + 2] for i in range(len(seq) - 2)]


def build_poetree_df():
    """Document-frequency of content-lemma bi/trigrams over the Poetree corpus.

    SUPERSEDED (see module docstring) -- returns an empty table if the gitignored
    78MB lemma cache isn't present locally, instead of crashing a fresh clone."""
    csv.field_size_limit(10 ** 8)
    bi_df = defaultdict(int)
    tri_df = defaultdict(int)
    n_poems = 0
    try:
        f = open(POETREE_LEMMAS, encoding="utf-8")
    except FileNotFoundError:
        print(f"  [skip] {POETREE_LEMMAS} not found (gitignored cache; superseded signal)",
              file=sys.stderr)
        return {"N": 0, "bi": {}, "tri": {}}
    with f:
        r = csv.reader(f)
        header = next(r)  # poem_id,lemma_text
        for row in r:
            if len(row) < 2:
                continue
            lemma_text = row[1].replace("\n", " ")
            seq = content_lemmas(lemma_text.split())
            n_poems += 1
            # document-frequency: set per poem
            for ng in set(bigrams(seq)):
                bi_df[ng] += 1
            for ng in set(trigrams(seq)):
                tri_df[ng] += 1
    # drop df==1 to keep the file reasonable
    bi = {k: v for k, v in bi_df.items() if v > 1}
    tri = {k: v for k, v in tri_df.items() if v > 1}
    return {"N": n_poems, "bi": bi, "tri": tri}


def build_cnp_freq():
    """Raw occurrence count of content-lemma bigrams over CnP sentence segments."""
    sent_ids = set()
    for line in open(CNP_SEGMENTS, encoding="utf-8"):
        o = json.loads(line)
        if o.get("level") == "sentence":
            sent_ids.add(o["id"])
    bi_freq = defaultdict(int)
    for line in open(CNP_SEG_TOKENS, encoding="utf-8"):
        o = json.loads(line)
        if o["id"] not in sent_ids:
            continue
        seq = content_lemmas(o["lem"])
        for ng in bigrams(seq):  # raw occurrences, not set
            bi_freq[ng] += 1
    return {"bi": dict(bi_freq)}


def main():
    poetree = build_poetree_df()
    with open(POETREE_OUT, "w", encoding="utf-8") as f:
        json.dump(poetree, f, ensure_ascii=False)
    cnp = build_cnp_freq()
    with open(CNP_OUT, "w", encoding="utf-8") as f:
        json.dump(cnp, f, ensure_ascii=False)

    print(f"Poetree: N={poetree['N']} poems, "
          f"{len(poetree['bi'])} bigrams (df>1), {len(poetree['tri'])} trigrams (df>1)",
          file=sys.stderr)
    top_bi = sorted(poetree["bi"].items(), key=lambda kv: -kv[1])[:8]
    print("  top Poetree bigrams (df):", top_bi, file=sys.stderr)
    print(f"CnP: {len(cnp['bi'])} distinct content-lemma bigrams", file=sys.stderr)
    top_cnp = sorted(cnp["bi"].items(), key=lambda kv: -kv[1])[:8]
    print("  top CnP bigrams (count):", top_cnp, file=sys.stderr)


if __name__ == "__main__":
    main()
