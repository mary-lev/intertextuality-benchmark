"""Extract from the DanteSources graph the records build_dante_citations.py needs.

Input : data/dante/source/note.ttl — the full dump of the DanteSources notes graph
        (36 MB; Bartalesi et al. 2015). NOT shipped in this release: the extract
        below is; fetch the dump from DanteSources to rerun this step.
Output: data/dante/source/note_lemmata.ttl — the same lines, verbatim, restricted to
        the passages cited in citation_pairs_with_snippets.json and to four properties:
          Dante-side fragment (…/exprFrag/N):   efChars (the lemma), numVerse, hasNote
          citing fragment     (…/exprFrag1/N):  efChars (the note's wording), hasSource

    python scripts/extract_dante_lemmata.py

Stdlib only, deterministic.
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"   # release root
DANTE = HERE + "data/dante/source/"
P = "http://perunaenciclopediadantescadigitale.eu/resource/work/"


def main():
    raw = json.load(open(DANTE + "citation_pairs_with_snippets.json", encoding="utf-8"))
    paragraphs = {p["dante_paragraph"] for p in raw}
    keep, n = [], 0
    for line in open(DANTE + "note.ttl", encoding="utf-8"):
        n += 1
        if not line.startswith("<" + P):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        subj, pred = parts[0][1:-1], parts[1]
        if "/exprFrag/" in subj:
            ok = pred.endswith(("numVerse", ":efChars", "hasNote"))
        elif "/exprFrag1/" in subj:
            ok = pred.endswith((":efChars", "hasSource"))
        else:
            ok = False
        if ok and subj.split("/exprFrag")[0] in paragraphs:
            keep.append(line)
    with open(DANTE + "note_lemmata.ttl", "w", encoding="utf-8") as f:
        f.writelines(keep)
    print(f"{n} lines -> {len(keep)} kept -> {DANTE}note_lemmata.ttl")


if __name__ == "__main__":
    main()
