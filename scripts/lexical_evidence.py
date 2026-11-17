"""Computed lexical evidence between the two spans of each banded Russian relation.

Post hoc and descriptive: added after the retrieval results were seen; no method reads it
and the reported bands (ngram_baseline.load_gold: character ratio over the whole spans) are
not changed. The band mixes how far the wording drifted with how long the spans are; these
columns describe the surviving wording directly, from the two spans alone:

  shorter_span_words     word count of the shorter span
  longest_shared_run     longest contiguous run of identical word forms
  shared_content_words   content words (noun, verb, adjective, adverb, numeral; Mystem;
                         «быть» excluded) present in both spans, matched by lemma or by
                         5-letter prefix
  shared_words           the matched pairs
  rarest_shared_df       poem frequency in the Poetree corpus of the rarest shared lemma
  wording_class          the stage of drift from the cited wording:
        verbatim     the shorter span reproduced whole: a run of >= 85% of its words, and
                     every one of its content words shared
        varied       the same words in another form or order: not verbatim, but every
                     content word of the shorter span (at least two) is shared
        reworked     part of the wording kept, part replaced: a copied run of >= 3 word
                     forms, or >= 2 shared content words, or a run of 2 with a shared
                     content word
        single_word  the shorter span is one word and the other span has it
        loose        none of the above: at most one shared content word
  lexical_anchor         yes for every class but loose, and for a loose relation whose one
                         shared content word is rare (in <= RARE_DF poems)

Judgment calls, tried on these same 100 relations and revised once after a read-through of
all pairs (wording_class_review.csv, made on the first version of the rule): the 85%, 3-word
and 2-word cut-offs, the 5-letter prefix, RARE_DF. What the rule cannot see: derivational
relatives (хмель / хмелен), synonyms, and any relation that keeps no word.

    python scripts/lexical_evidence.py     # from the release root; needs pymystem3

Writes results/russian_poetry/lexical_evidence.csv.
"""
from __future__ import annotations
import csv, re, sys
from difflib import SequenceMatcher

sys.argv = sys.argv[:1]            # wpair_baseline reads sys.argv at import
import ngram_baseline as nb
import wpair_baseline as wp

csv.field_size_limit(10 ** 9)
CONTENT = ("S", "V", "A", "ADV", "NUM", "ANUM")
RARE_DF = 200
CLASSES = ("verbatim", "varied", "reworked", "single_word", "loose")


def analyse(mys, t):
    """-> (word forms, [(lemma, form)] for the content words of at least three letters)."""
    words, content = [], []
    for a in mys.analyze(nb.norm(t)):
        w = a["text"].strip()
        if not re.fullmatch(r"\w+", w or ""):
            continue
        an = a.get("analysis") or []
        words.append(w)
        lem = an[0]["lex"].replace("ё", "е") if an else w
        if len(w) >= 3 and lem != "быть" and (re.split(r"[,=]", an[0]["gr"])[0] if an else "S") in CONTENT:
            key = (lem, w)
            if key not in content:
                content.append(key)
    return words, content


def shared(qc, dc):
    """Greedy one-to-one matching of content words by lemma or 5-letter prefix."""
    used, out = set(), []
    for la, wa in qc:
        for j, (lb, wb) in enumerate(dc):
            if j not in used and (la == lb or (len(wa) >= 5 and len(wb) >= 5 and wa[:5] == wb[:5])):
                used.add(j); out.append((la, wa, wb)); break
    return out


def main():
    from pymystem3 import Mystem
    mys = Mystem()
    _, df = wp.load_lemma_corpus()
    gold = {r["conn_id"]: r for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/gold.csv", encoding="utf-8"))}
    banded = [r for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/factorial_ranks.csv", encoding="utf-8"))
              if r["band"]]
    rows = []
    for r in banded:
        g = gold[r["conn_id"]]
        q = (g["analyst_commented_span"] or g["commented_fragment"]).strip()
        d = (g["analyst_referent_span"] or g["ref_citation"]).strip()
        (qw, qc), (dw, dc) = analyse(mys, q), analyse(mys, d)
        short = min(len(qw), len(dw))
        run = SequenceMatcher(None, qw, dw, autojunk=False).find_longest_match(0, len(qw), 0, len(dw)).size
        sh = shared(qc, dc)
        rare = min((df.get(l, 0) for l, _, _ in sh), default="")
        # content words of the shorter span (of either span when they are equally long)
        n_short = [len(c) for w_, c in ((qw, qc), (dw, dc)) if len(w_) == short]
        covered = any(n and len(sh) >= n for n in n_short)          # all of them are shared
        cls = "single_word" if short == 1 and (run == 1 or sh) \
            else "verbatim" if short >= 2 and run >= max(2, round(.85 * short)) and covered \
            else "varied" if covered and len(sh) >= 2 \
            else "reworked" if run >= 3 or len(sh) >= 2 or (run >= 2 and sh) \
            else "loose"
        anchor = cls != "loose" or (len(sh) == 1 and rare <= RARE_DF)
        rows.append({"conn_id": r["conn_id"], "band": r["band"], "shorter_span_words": short,
                     "longest_shared_run": run, "shared_content_words": len(sh),
                     "shared_words": "; ".join(a if a == b else f"{a} / {b}" for _, a, b in sh),
                     "rarest_shared_df": rare, "wording_class": cls,
                     "lexical_anchor": "yes" if anchor else "no"})
    with open(nb.HERE + "results/russian_poetry/lexical_evidence.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote results/russian_poetry/lexical_evidence.csv ({len(rows)} rows)")
    for b in ("verbatim", "close", "loose"):
        sel = [x for x in rows if x["band"] == b]
        print(f"band {b:8s} n={len(sel):3d}  class:",
              {c: sum(x["wording_class"] == c for x in sel) for c in CLASSES},
              " anchor yes:", sum(x["lexical_anchor"] == "yes" for x in sel))


if __name__ == "__main__":
    main()
