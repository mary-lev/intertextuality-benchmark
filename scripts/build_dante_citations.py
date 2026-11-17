"""Rebuild the Dante↔Duecento gold from the DanteSources citations, one row per citation.

The source benchmark keeps one row
per (passage, chunk) and no Dante-side span. The scholarly source records more:

  * note_lemmata.ttl — the lemma records of the DanteSources notes graph (extracted from
    its 36 MB dump note.ttl by extract_dante_lemmata.py): every commentary note hangs on a Dante-side fragment with a LEMMA (the words
    commented, often abbreviated "FIRST ... LAST") and holds the citing fragments;
  * citation_pairs_with_snippets.json — each citing fragment with the cited poem and the
    note's own wording ("Amor m'assale - verso 3": quoted words and/or a verse number).

One output row = one relation: (Dante passage, lemma locus) -> (cited poem, cited verses).

  query side     commented_fragment   the lemma located in the passage (original wording);
                                      empty when it cannot be located (lemma kept as recorded)
  referent side  cited_verses         verse numbers of the cited poem; from the note, checked
                                      against the note's quoted words when it quotes any
                 referent_text_id     the corpus chunk(s) covering those verses (";"-joined)
                 referent_span        the corpus text of the cited verses
                 scholar_quote        the words the note quotes (commentary's edition spelling)

Placement rules (quote_check column):
  exact / fuzzy   the quoted words are found in the cited poem (ratio >= QUOTE_MIN); the
                  matched verses are used (this corrects a few off-by-one verse numbers)
  number_only     the note gives a verse number and quotes nothing; the number is trusted
                  (numbering agrees with the quoted words in 291 of 298 checked citations)
Excluded, with the reason, in the report: neither verse number nor locatable quotation;
quoted words not found in the cited poem (wrong-poem links).

Cited poems (referent_corpus column):
  verified             the poem is in the 431-chunk corpus of the source benchmark
                       (retrieval_corpus.json; 60 poems), whose positives were verified there
  biblioteca_italiana  the poem is not: the source benchmark dropped its citations because
                       the chunk could not be verified by line number within an author's
                       collection. The citation file links each of them to a poem by incipit
                       (match score >= 0.80), and that poem is in BibliotecaItaliana/
                       poems_index.json; it is cut into the same four-line chunks and the
                       verses are placed by the same rules. Chunk ids are those of the open
                       search space of scripts/dante_factorial.py (open{i}_chunk_{n}@slug,
                       i = position of the poem in poems_index.json).

The two annotated Vita nuova editions overlap: a `vitanova` passage that is a near-duplicate
(token Jaccard >= JACCARD_MERGE, or word containment >= CONTAIN_MERGE) of a
`vitanuovaderobertis` passage is merged into it, as in the source benchmark's conversion; a relation attested in both keeps both in `attested_by`.

Output: data/dante/gold.csv (+ report on stderr). Stdlib only, deterministic.

    python scripts/build_dante_citations.py
"""
from __future__ import annotations
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"   # release root
DANTE = HERE + "data/dante/source/"
GT = DANTE
OUT = HERE + "data/dante/gold.csv"
P = "http://perunaenciclopediadantescadigitale.eu/resource/work/"

JACCARD_MERGE = 0.6   # cross-edition twin threshold, as in the source benchmark's conversion
CONTAIN_MERGE = 0.7   # ... or one passage's words largely contained in the other's
QUOTE_MIN = 0.75      # quoted words vs corpus verses (edition spelling differs)


def fix_text(t: str) -> str:
    return (t or "").replace("\\n", "\n")


def toks(t: str) -> set:
    return set(re.findall(r"[a-zàèéìòóùäëïöü’']+", (t or "").lower()))


def norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace("’", "'").lower()).strip()


def norm(t: str) -> str:
    """Lowercase, punctuation to spaces — the comparison form."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (t or "").lower())).strip()


def slug(t: str, n: int = 4) -> str:
    return "-".join(re.findall(r"[a-zàèéìòóù]+", (t or "").lower())[:n])


def lit(o: str) -> str:
    o = o.strip()
    if o.startswith('"'):
        try:
            return json.loads(o[:o.rindex('"') + 1])
        except Exception:
            return o.strip('"')
    return o


def load_notes():
    """note_lemmata.ttl -> (lemma, verse) per Dante fragment; citing fragment -> Dante fragment."""
    nv, lem, note, src, fchars = {}, {}, {}, {}, {}
    pat = re.compile(r'^<([^>]+)>\t(\S+)\t(.*?)\s[.,;]\s*$')
    for line in open(DANTE + "note_lemmata.ttl", encoding="utf-8"):
        m = pat.match(line)
        if not m:
            continue
        s, p, o = m.groups()
        s = s.replace(P, "")
        if "/exprFrag/" in s:
            if p.endswith("numVerse"):
                nv[s] = lit(o)
            elif p.endswith(":efChars"):
                lem[s] = lit(o)
            elif p.endswith("hasNote"):
                note[o.strip("<>").replace(P, "")] = s
        elif "/exprFrag1/" in s:
            if p.endswith(":efChars"):
                fchars[s] = lit(o)
            elif p.endswith("hasSource"):
                src[s] = o.strip("<>").replace(P, "")
    by = defaultdict(list)                      # (paragraph, note wording) -> Dante fragments
    for f, c in fchars.items():
        d = note.get(src.get(f, ""))
        if d:
            by[(f.split("/exprFrag1/")[0], norm_ws(c))].append(d)
    return nv, lem, by


def locate_lemma(lemma: str, text: str):
    """Find 'FIRST ... LAST' (or a plain phrase) in the passage. Returns (original-wording
    span, number of places the first part occurs); ('', 0) when not found."""
    lemma = re.sub(r"^\s*vv?\.\s*[\d\-–, ]+\.?", "", lemma or "").replace("(", "").replace(")", "")
    parts = [norm(x).split() for x in re.split(r"\.\.\.|…", lemma) if norm(x)]
    words = [(m.start(), m.end(), norm(m.group())) for m in re.finditer(r"\S+", text)]
    words = [(a, b, w2) for a, b, w in words for w2 in (w.split() or [""]) if w2]
    seq = [w for _, _, w in words]

    def find(sub, start=0):
        for i in range(start, len(seq) - len(sub) + 1):
            if seq[i:i + len(sub)] == sub:
                return i
        return -1

    if not parts:
        return "", 0
    a = find(parts[0])
    if a < 0:
        return "", 0
    e = a + len(parts[0])
    if len(parts) > 1:
        b = find(parts[-1], e)
        if b < 0:
            return "", 0
        e = b + len(parts[-1])
    occ, i = 0, find(parts[0])
    while i >= 0:
        occ += 1
        i = find(parts[0], i + 1)
    return text[words[a][0]:words[e - 1][1]], occ


def parse_verses(cited: str, method: str):
    """Verse numbers named by the note ('verso 3', 'versi 16-7', 'versi 5-6 e 8-9'); falls
    back to the numbers the source file extracted."""
    m = re.search(r"\bvers[oi]\s+([\d\s,e\-–]+)", cited or "")
    out = []
    if m:
        for part in re.split(r",|\be\b", m.group(1)):
            r = re.findall(r"\d+", part)
            if len(r) == 1:
                out.append(int(r[0]))
            elif len(r) >= 2:
                a, b = r[0], r[1]
                if int(b) < int(a):                       # abbreviated end: 16-7 -> 16-17
                    b = a[:len(a) - len(b)] + b
                if int(a) <= int(b) <= int(a) + 30:
                    out.extend(range(int(a), int(b) + 1))
    if not out:
        m = re.match(r"lines \[([\d, ]+)\] from poem", method or "")
        if m:
            out = [int(x) for x in m.group(1).split(",")]
    return sorted(set(out))


def quoted_words(cited: str, stype: str, snippet: str) -> str:
    """The words of the source the note itself quotes, if any."""
    if " - vers" in (cited or ""):
        return re.split(r"\s+-\s+vers", cited)[0].strip()
    if stype == "quote":
        return (snippet or "").strip()
    return ""


def match_quote(quote: str, lines: dict):
    """Best place of the quoted words in the poem -> (ratio, [verse numbers])."""
    q = norm(quote.replace("/", " "))
    n0 = max(1, quote.count("/") + 1)
    nums = sorted(lines)
    best = (0.0, [])
    for s in nums:
        for n in {n0, max(1, n0 - 1), n0 + 1}:
            vs = [s + i for i in range(n) if s + i in lines]
            if len(vs) != n:
                continue
            txt = norm(" ".join(lines[v] for v in vs))
            if not txt:
                continue
            r = 1.0 if (q in txt or (len(txt.split()) >= 3 and txt in q and n >= n0)) \
                else SequenceMatcher(None, q, txt).ratio()
            if n == 1 and r < 1.0 and len(txt) > len(q):          # words within a longer line
                w = len(q)
                r = max(r, max(SequenceMatcher(None, q, txt[i:i + w]).ratio()
                               for i in range(0, len(txt) - w + 1, 2)))
            if r > best[0] + 1e-9 or (abs(r - best[0]) < 1e-9 and n < len(best[1])):
                best = (r, vs)
    return best


def main():
    corpus_list = json.load(open(GT + "retrieval_corpus.json", encoding="utf-8"))
    raw = json.load(open(DANTE + "citation_pairs_with_snippets.json", encoding="utf-8"))
    nv, lem, by = load_notes()

    # -- corpus: poems by incipit; verse number -> (text, chunk id) ----------------------
    poems = defaultdict(dict)
    meta = {}
    for c in corpus_list:
        k = norm_ws(c["poem_incipit"])[:40]
        cid = f'{c["chunk_id"]}@{slug(c["poem_incipit"])}'
        meta[k] = (c["poem_author"], c["poem_incipit"])
        ls = [l for l in c["chunk_text"].split("\n") if l.strip()]
        assert len(ls) == c["end_line"] - c["start_line"] + 1, cid
        for i, l in enumerate(ls):
            poems[k][c["start_line"] + i] = (l, cid)
    verified_keys = set(poems)
    # cited poems outside that corpus: taken from the Biblioteca Italiana poem index
    for i, bp in enumerate(json.load(open(DANTE + "poems_index.json", encoding="utf-8"))):
        k = norm_ws(bp["incipit"])[:40]
        if k in poems or not k:
            continue                                  # verified poem, or an earlier printing
        meta[k] = (bp["work_author"], bp["incipit"])
        ls = [l for l in bp["full_text"].split("\n") if l.strip()]
        for a, l in enumerate(ls):
            poems[k][a + 1] = (l, f'open{i}_chunk_{a // 4 + 1}@{slug(bp["incipit"])}')

    # -- passages; merge vitanova twins into De Robertis ---------------------------------
    ptext, pwork = {}, {}
    for p in raw:
        ptext[p["dante_paragraph"]] = fix_text(p["dante_text"])
        pwork[p["dante_paragraph"]] = p["dante_work"]
    vdr = [q for q in ptext if pwork[q] == "vitanuovaderobertis"]
    target, merged = {}, []
    for q in ptext:
        target[q] = q
        if pwork[q] != "vitanova" or not vdr:
            continue
        A = toks(ptext[q])
        jac = lambda b: len(A & toks(ptext[b])) / max(1, len(A | toks(ptext[b])))
        cont = lambda b: len(A & toks(ptext[b])) / max(1, min(len(A), len(toks(ptext[b]))))
        best = max(vdr, key=jac)
        if jac(best) < JACCARD_MERGE:                 # same text, different spelling/extent
            best = max(vdr, key=cont)
        if jac(best) >= JACCARD_MERGE or cont(best) >= CONTAIN_MERGE:
            target[q] = best
            merged.append((q, best, round(jac(best), 2)))

    # -- one relation per (passage, lemma locus, poem, verses) ---------------------------
    rel, excl = {}, Counter()
    for p in raw:
        k = norm_ws(p.get("source_incipit", ""))[:40]
        if k not in poems:
            excl["cited poem outside the corpus"] += 1
            continue
        lines = {v: t for v, (t, _) in poems[k].items()}
        cited = p["original_cited_text"]
        verses = [v for v in parse_verses(cited, p["source_snippet_method"]) if v in lines]
        quote = quoted_words(cited, p["source_snippet_type"], fix_text(p["source_snippet"]))
        check = "number_only"
        if quote and len(norm(quote).split()) >= 2:
            r, vs = match_quote(quote, lines)
            if r >= QUOTE_MIN:
                check = "exact" if r == 1.0 else "fuzzy"
                if verses and not (set(vs) & set(verses)) and min(abs(a - b) for a in vs for b in verses) > 1:
                    excl["[kept] quoted words found away from the cited verse number"] += 1
                verses = vs
            else:
                excl["quoted words not found in the cited poem"] += 1
                continue
        if not verses:
            excl["no verse number and no locatable quotation"] += 1
            continue
        q = target[p["dante_paragraph"]]
        frags = by.get((p["dante_paragraph"].replace(P, ""), norm_ws(cited)), [])
        if not frags:
            excl["[kept] note not found in the notes graph (no lemma)"] += 1
        for d in (sorted(set(frags)) or [""]):
            lemma = lem.get(d, "")
            span, occ = locate_lemma(lemma, ptext[q])
            key = (q, norm(span) or norm(lemma), k, tuple(verses))
            row = rel.get(key)
            if row is None:
                chunk_ids = sorted({poems[k][v][1] for v in verses})
                row = rel[key] = {
                    "edition": pwork[q], "commented_author": "Dante Alighieri",
                    "commented_work": pwork[q], "commented_text_id": q,
                    "commented_text": ptext[q], "lemma": lemma, "lemma_verse": nv.get(d, ""),
                    "commented_fragment": span, "lemma_occurrences": occ,
                    "ref_author": meta[k][0], "ref_title": meta[k][1],
                    "referent_poem_id": slug(meta[k][1]),
                    "cited_verses": ",".join(map(str, verses)),
                    "referent_text_id": ";".join(chunk_ids),
                    "referent_span": "\n".join(lines[v] for v in verses),
                    "scholar_quote": quote, "quote_check": check, "note_text": cited,
                    "citation_type": p.get("citation_type", ""),
                    "attested_by": set(), "merged_from": set(),
                    "referent_corpus": "verified" if k in verified_keys else "biblioteca_italiana",
                    "category": "poetry", "verified": "source-benchmark",
                    "provenance": "DanteSources notes graph and citations",
                }
            row["attested_by"].add(p["dante_work"])
            if p["dante_paragraph"] != q:
                row["merged_from"].add(p["dante_paragraph"])
            if check != "number_only" and row["quote_check"] == "number_only":
                row["scholar_quote"], row["quote_check"] = quote, check

    # The two commentaries often hang the same citation on lemmata of different extent
    # ("ANDAR SICURAMENTE" / "sicuramente"). Within one passage, rows that cite the same
    # verses of the same poem from DIFFERENT editions are one relation with two
    # attestations; the lemma located in the passage text (De Robertis first) is kept and
    # the other is recorded in lemma_other. Rows from the same edition stay separate.
    groups = defaultdict(list)
    for key in sorted(rel, key=lambda x: (x[0], x[2], x[3], x[1])):
        groups[(key[0], key[2], key[3])].append(rel[key])
    rows, n_cross = [], 0
    for g in groups.values():
        if len({e for r in g for e in r["attested_by"]}) > 1 and len(g) > 1:
            n_cross += 1
            g.sort(key=lambda r: (not r["commented_fragment"], "vitanuovaderobertis" not in r["attested_by"]))
            keep = g[0]
            keep["lemma_other"] = " | ".join(r["lemma"] for r in g[1:] if r["lemma"])
            for r in g[1:]:
                keep["attested_by"] |= r["attested_by"]
                keep["merged_from"] |= r["merged_from"]
                if keep["quote_check"] == "number_only" and r["quote_check"] != "number_only":
                    keep["scholar_quote"], keep["quote_check"] = r["scholar_quote"], r["quote_check"]
            g = [keep]
        for r in g:
            r.setdefault("lemma_other", "")
            r["attested_by"] = ";".join(sorted(r["attested_by"]))
            r["merged_from"] = ";".join(sorted(r["merged_from"]))
            rows.append(r)
    seen = Counter()
    for r in rows:
        base = f'{r["commented_text_id"].split("resource/work/")[1]}#{r["referent_poem_id"]}:{r["cited_verses"].replace(",", "_")}'
        seen[base] += 1
        r["conn_id"] = base if seen[base] == 1 else f"{base}~{seen[base]}"
    cols = ["conn_id"] + [c for c in rows[0] if c != "conn_id"]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # -- report ---------------------------------------------------------------------------
    e = sys.stderr
    print(f"source citations {len(raw)} -> relations {len(rows)} -> {OUT}", file=e)
    print(f"passages {len({r['commented_text_id'] for r in rows})} "
          f"({dict(Counter(r['commented_work'] for r in {r['commented_text_id']: r for r in rows}.values()))}); "
          f"vitanova passages merged into De Robertis: {len(merged)}", file=e)
    print(f"relations attested by both commentaries and merged: {n_cross}", file=e)
    print(f"cited poems {len({r['referent_poem_id'] for r in rows})}; "
          f"distinct (passage, poem) {len({(r['commented_text_id'], r['referent_poem_id']) for r in rows})}", file=e)
    for k_, v in sorted(excl.items()):
        print(f"  {v:4d} {k_}", file=e)
    print("quote_check:", dict(Counter(r["quote_check"] for r in rows)), file=e)
    print("lemma located:", sum(1 for r in rows if r["commented_fragment"]),
          "| recorded but not located:", sum(1 for r in rows if r["lemma"] and not r["commented_fragment"]),
          "| no lemma:", sum(1 for r in rows if not r["lemma"]),
          "| located in more than one place:", sum(1 for r in rows if r["lemma_occurrences"] > 1), file=e)
    print("both sides have a span (lemma located + quoted words verified):",
          sum(1 for r in rows if r["commented_fragment"] and r["quote_check"] != "number_only"), file=e)
    print("chunks per relation:", dict(Counter(len(r["referent_text_id"].split(";")) for r in rows)), file=e)
    print("attested by:", dict(Counter(r["attested_by"] for r in rows)), file=e)
    print("by cited author:", dict(Counter(r["ref_author"] for r in rows)), file=e)
    print("referent corpus:", dict(Counter(r["referent_corpus"] for r in rows)), "| poems:",
          {c: len({r["referent_poem_id"] for r in rows if r["referent_corpus"] == c})
           for c in ("verified", "biblioteca_italiana")}, file=e)


if __name__ == "__main__":
    main()
