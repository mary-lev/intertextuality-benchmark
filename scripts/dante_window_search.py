"""Exploration: slide word windows over a Dante passage and list what each probe retrieves
from the Duecento poems of Biblioteca Italiana.

Not an evaluation. The Russian counterpart is window_search.py. Differences, all forced by
the material:
  queries     the gold stores Dante's passages as running text, so the windows are runs of
              6 and 12 words (step 3), plus the whole passage
  candidates  the poems of data/dante/source/poems_index.json (929,
              about 60 authors; a poem printed in two collections is kept once, by its
              incipit), not only the 60 cited poems that dante_factorial.py searches. Each
              poem is cut into chunks of four lines, as there, and scored by its best chunk,
              so that long poems are not favoured. No dates, so no chronology filter: some
              authors (Cino da Pistoia, Cecco Angiolieri) are Dante's contemporaries.
  probes      surface forms throughout (no Duecento lemmatizer): word 2-grams, character
              4-grams, local alignment, BM25, dense E5 (best window x poem line), and an RRF
              of the five. No collocation probe.
A candidate poem the commentary cites for the passage is marked as attested (matched by the
incipit); every other candidate is unjudged.

    PASSAGES="Tanto gentile|Donne ch'avete|Ne li occhi porta" python scripts/dante_window_search.py

CPU, no API calls; encodes about 38,000 lines with E5 (minutes). Writes
results/dante/dante_window_search.csv (top 10) and results/dante/dante_window_search.html (top 5).
"""
from __future__ import annotations
import csv, json, math, os, re, sys
from collections import defaultdict

import numpy as np
import ngram_baseline as nb
import bm25_baseline as bm
from passim_baseline import sw
from window_search import write_html

csv.field_size_limit(10 ** 9)
PASSAGES = os.environ.get("PASSAGES", "Tanto gentile|Donne ch'avete|Ne li occhi porta").split("|")
SIZES, STEP = (6, 12, 0), 3              # 0 = the whole passage
PROBES = [("ngram", "surface (word 2-grams)"), ("char4", "surface (character 4-grams)"),
          ("sw", "alignment (surface forms)"), ("bm25", "bag (BM25)"), ("dense", "dense (E5)"),
          ("fusion", "fusion (RRF of the five)")]
KEEP, SHOW, TOPN, C = 10, 5, 100, 60
BIBIT = nb.HERE + "data/dante/source/poems_index.json"
slug = lambda t, n=4: "-".join(re.findall(r"[a-zàèéìòóù]+", (t or "").lower())[:n])
INTRO = ("Every run of 6 and of 12 words of the passage, and the whole passage, is a query. Candidates are "
         "the Duecento poems of Biblioteca Italiana, each scored by its best four-line chunk (no dates, so "
         "contemporaries of Dante are included). For each probe and window size the five best candidates are shown, with the window "
         "that scored best and the candidate's closest line. A green row is a poem the commentary cites for "
         "this passage. Every other row is unjudged. Scores are comparable only within one table. "
         "Window size 0 means the whole passage.")


def main():
    from sentence_transformers import SentenceTransformer
    seen, poems = set(), []
    for p in json.load(open(BIBIT, encoding="utf-8")):
        if slug(p["incipit"], 6) not in seen and p["full_text"].strip():
            seen.add(slug(p["incipit"], 6)); poems.append(p)
    ids = [f'{p["bibit_id"]}#{i}' for i, p in enumerate(poems)]
    meta = {i: (p["work_author"] or "anonymous", p["incipit"]) for i, p in zip(ids, poems)}
    corpus, owner = {}, {}                       # chunk id -> text, chunk id -> poem id
    for i, p in zip(ids, poems):
        lns = [l for l in p["full_text"].split("\n") if l.strip()]
        for a in range(0, len(lns), 4):
            corpus[f"{i}/{a}"] = nb.norm(" ".join(lns[a:a + 4])); owner[f"{i}/{a}"] = i
    by_slug = defaultdict(set)
    for i, p in zip(ids, poems):
        by_slug[slug(p["incipit"])].add(i)
    gold = list(csv.DictReader(open(nb.HERE + "data/dante/gold.csv", encoding="utf-8")))

    ng_index, ng_idf = nb.build_index(corpus, lambda t: nb.word_ngrams(t, 2))
    c4_index, c4_idf = nb.build_index(corpus, lambda t: nb.char_ngrams(t, 4))
    postings, doclen, idf_bm, avgdl, _, _ = bm.build_bm25(corpus)
    seqs = {i: t.split() for i, t in corpus.items()}
    df = defaultdict(int)
    for s in seqs.values():
        for w in set(s):
            df[w] += 1
    idf = {w: math.log(len(seqs) / d) for w, d in df.items()}

    model = SentenceTransformer("intfloat/multilingual-e5-base")
    lines, starts = [], []
    for p in poems:
        starts.append(len(lines))
        lines.extend([l.strip() for l in p["full_text"].split("\n") if l.strip()] or [p["incipit"]])
    starts = np.array(starts, dtype=np.int64)
    print(f"encoding {len(lines)} lines…", file=sys.stderr)
    lvecs = model.encode(["passage: " + l for l in lines], normalize_embeddings=True,
                         convert_to_numpy=True, batch_size=128).astype(np.float32)
    top = lambda sc, n: sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    out, pages = [], []
    for key in PASSAGES:
        rels = [g for g in gold if g["commented_text"].strip().startswith(key)]
        if not rels:
            sys.exit(f"no gold passage starts with: {key}")
        text = rels[0]["commented_text"].strip()
        rels = [g for g in rels if g["commented_text"].strip() == text]
        words = text.split()
        attested = {}
        for g in rels:
            for rid in g["referent_text_id"].split(";"):
                for i in by_slug.get(rid.split("@")[-1], ()):
                    attested.setdefault(i, g["conn_id"])
        print(f"{key}: {len(words)} words, {len(rels)} citations, {len(attested)} cited poems located",
              file=sys.stderr)
        best, wins = defaultdict(dict), {}
        for size in SIZES:
            spans = [(0, len(words))] if size == 0 else \
                [(a, a + size) for a in range(0, max(1, len(words) - size + 1), STEP)]
            texts = [" ".join(words[a:b]) for a, b in spans]
            qv = model.encode(["query: " + t for t in texts], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
            sims = qv @ lvecs.T
            for k, t in enumerate(texts):
                wins[(size, k)] = (t, sims[k])
                q = nb.norm(t)
                dense = np.maximum.reduceat(sims[k], starts)
                def per_poem(chunk_scores):          # a poem scores as its best chunk
                    out_ = {}
                    for c, v in chunk_scores.items():
                        if v > out_.get(owner[c], 0):
                            out_[owner[c]] = v
                    return out_
                S = {"ngram": per_poem(nb.rank(nb.word_ngrams(q, 2), ng_index, ng_idf)),
                     "char4": per_poem(nb.rank(nb.char_ngrams(q, 4), c4_index, c4_idf)),
                     "sw": per_poem({c: sw(q.split(), seqs[c], idf) for c in seqs}),
                     "bm25": per_poem(bm.bm25_rank(q.split(), postings, doclen, idf_bm, avgdl)),
                     "dense": {i: float(v) for i, v in zip(ids, dense)}}
                rrf = defaultdict(float)
                for probe, sc in S.items():
                    S[probe] = sc = {i: v for i, v in sc.items() if v > 0}
                    for rank, (i, _) in enumerate(top(sc, TOPN), 1):
                        rrf[i] += 1.0 / (C + rank)
                S["fusion"] = dict(rrf)
                for probe, sc in S.items():
                    b = best[(probe, size)]
                    for i, v in top(sc, KEEP):
                        if i not in b or v > b[i][0]:
                            b[i] = (v, k)

        page = {"poem": "", "work": f'{rels[0]["commented_work"]}: {" ".join(words[:6])}…', "year": "",
                "lines": [text], "attested": sorted({g["conn_id"] for g in rels}), "tables": []}
        for probe, label in PROBES:
            for size in SIZES:
                rows = []
                for rank, (i, _) in enumerate(top({p: sv[0] for p, sv in best[(probe, size)].items()}, KEEP), 1):
                    v, k = best[(probe, size)][i]
                    t, s = wins[(size, k)]
                    n = ids.index(i)
                    lo = int(starts[n]); hi = int(starts[n + 1]) if n + 1 < len(starts) else len(lines)
                    row = {"poem": key, "probe": probe, "window_lines": size, "rank": rank,
                           "window_start_line": k * STEP + 1, "window": t, "candidate_id": i,
                           "candidate_author": meta[i][0], "candidate_title": meta[i][1],
                           "candidate_year": "", "candidate_line": lines[lo + int(s[lo:hi].argmax())],
                           "score": round(v, 4), "attested": "yes" if i in attested else ""}
                    rows.append(row); out.append(row)
                page["tables"].append((label, size, rows[:SHOW]))
        pages.append(page)

    with open(nb.HERE + "results/dante/dante_window_search.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    write_html(pages, nb.HERE + "results/dante/dante_window_search.html", INTRO, "word")
    print("wrote results/dante/dante_window_search.csv / .html", file=sys.stderr)


if __name__ == "__main__":
    main()
