"""Lemmatize the whole corpus with Mystem → poetree_lemmas.csv (for Tesserae/Passim, lemma track).

Mystem (pymystem3) is the lemmatizer of record: it is what Poetree used, it is context-aware,
and it is fast (~5000 lines/s, full corpus in minutes). We lemmatize each poem WITH its line
breaks (Mystem keeps \n tokens), so the output preserves the line structure 1:1.

Output: cache/lemmas/poetree_lemmas.csv — poem_id, lemma_text (lines joined by \n,
each line = space-joined lemmas). Regenerable; gitignored.

    python scripts/lemmatize_corpus.py        # from the release root  (~5 min)
"""
from __future__ import annotations
import os, sys, csv
import ngram_baseline as nb
from pymystem3 import Mystem

csv.field_size_limit(10**8)
OUT = nb.HERE + "cache/lemmas/"
CH = 800            # poems per Mystem call, joined by a sentinel line


def lemmatize_texts(m, texts):
    """Lemmatize a list of full-poem texts; return list of lemmatized texts (\n preserved)."""
    SENT = "\nщщщ\n"                                # sentinel unlikely in verse; Mystem passes it
    out = []
    for i in range(0, len(texts), CH):
        chunk = texts[i:i + CH]
        joined = SENT.join(chunk)
        s = "".join(m.lemmatize(joined))
        parts = s.split("щщщ")
        if len(parts) != len(chunk):               # sentinel got mangled — fall back per poem
            parts = ["".join(m.lemmatize(t)) for t in chunk]
        out.extend(p.strip("\n") for p in parts)
        print(f"  {min(i+CH,len(texts))}/{len(texts)}", file=sys.stderr)
    return out


def main():
    print("reading corpus…", file=sys.stderr)
    ids, texts = [], []
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        ids.append(p["poetree_poem_id"]); texts.append(p["full_text"])
    for r in nb.load_added_referents():
        ids.append(r["extra_id"]); texts.append(r["full_text"])
    print(f"  {len(ids)} poems", file=sys.stderr)

    m = Mystem()
    lem = lemmatize_texts(m, texts)

    os.makedirs(OUT, exist_ok=True)
    with open(OUT + "poetree_lemmas.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["poem_id", "lemma_text"])
        for pid, lt in zip(ids, lem):
            w.writerow([pid, lt])
    print(f"wrote {OUT}poetree_lemmas.csv ({len(ids)} poems)", file=sys.stderr)


if __name__ == "__main__":
    main()
