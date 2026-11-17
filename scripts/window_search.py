"""Exploration: slide windows over a commented poem and list what each probe retrieves from
the poems written before it.

Not an evaluation. For each chosen poem, every window of 1, 2 and 4 consecutive verse lines
is a query; the candidates are the corpus poems not dated after the poem (ngram_baseline.
not_later; undated poems are kept, the poem and its text-duplicates are excluded). For every
probe and window size the candidates are ranked by their best score over the windows of
that size; the window that gave the score and the closest candidate line are kept as
evidence. A candidate the gold records for this poem is marked as attested; every other
candidate is unjudged — it may be noise, a commonplace, or a relation no commentator
recorded.

Probes (the factorial's scoring rules): surface word 2-grams, lemma alignment, BM25,
collocation, dense E5 (line-to-line maximum), and an RRF fusion of these five (top 100
each, C = 60). The fusion is NOT the paper's fusion, which has other components.

    POEMS=mandl-009,mandl-146,mandl-256 python scripts/window_search.py   # from the release root

Needs the lemma cache and the E5 line cache. CPU, no API calls. Writes
results/russian_poetry/window_search.csv (top 10 per poem x probe x window size) and
results/russian_poetry/window_search.html (top 5, readable).
"""
from __future__ import annotations
import csv, hashlib, html, math, os, sys
from collections import defaultdict

os.environ["EMB_MODEL"] = "intfloat/multilingual-e5-base"   # read by line_dense at import
sys.argv = sys.argv[:1]                                     # wpair_baseline reads sys.argv

import numpy as np
import ngram_baseline as nb
import bm25_baseline as bm
import wpair_baseline as wp
import line_dense as ld
import passage_baseline as pb
from passim_baseline import sw
from tesserae_baseline import content_lemmas, DF_CAP, MIN_SHARED

csv.field_size_limit(10 ** 9)
POEMS = os.environ.get("POEMS", "mandl-009,mandl-146,mandl-256").split(",")
SIZES = (1, 2, 4)
PROBES = [("ngram", "surface (word 2-grams)"), ("passim", "alignment (lemmas)"),
          ("bm25", "bag (BM25)"), ("wpair", "collocation"), ("dense", "dense (E5)"),
          ("fusion", "fusion (RRF of the five)")]
KEEP, SHOW, TOPN, C = 10, 5, 100, 60
LEMMAS = nb.HERE + "cache/lemmas/poetree_lemmas.csv"


def main():
    from sentence_transformers import SentenceTransformer
    from pymystem3 import Mystem

    surf = nb.load_corpus()
    dups = nb.dup_groups(surf)
    years = nb.load_years()
    meta = {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        meta[p["poetree_poem_id"]] = (p["author_name"], p["title"] or p["first_line"])
    for r in nb.load_added_referents():
        meta[r["extra_id"]] = (r["ref_author"], r["ref_title"])
    gold = list(csv.DictReader(open(nb.HERE + "data/russian_poetry/gold.csv", encoding="utf-8")))

    ng_index, ng_idf = nb.build_index(surf, lambda t: nb.word_ngrams(t, 2))
    postings, doclen, idf_bm, avgdl, _, _ = bm.build_bm25(surf)
    seqs, df_lem = {}, defaultdict(int)
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        lines = [l for l in (content_lemmas(ln) for ln in r["lemma_text"].split("\n")) if l]
        seqs[r["poem_id"]] = [w for l in lines for w in l]
        for l in lines:
            for w in set(l):
                df_lem[w] += 1
    idf_lem = {w: math.log(len(seqs) / d) for w, d in df_lem.items()}
    inv = defaultdict(set)
    for pid, s in seqs.items():
        for w in set(s):
            inv[w].add(pid)
    wp_keep, wp_pidf, wp_inv = wp.build_index(seqs)

    lines_by_id = pb.corpus_lines()
    pids_d, starts_d, dtexts = [], [], []
    for pid, lns in lines_by_id.items():
        starts_d.append(len(dtexts)); pids_d.append(pid)
        dtexts.extend(lns)
    starts_d = np.array(starts_d, dtype=np.int64)
    sig = hashlib.md5((ld.MODEL + "|line1|" + str(len(dtexts)) + "|" +
                       hashlib.md5("".join(dtexts[::997]).encode()).hexdigest()).encode()).hexdigest()[:16]
    vpath = ld.CACHE + f"line1_{sig}.npy"
    if not os.path.exists(vpath):
        sys.exit("E5 line cache missing — run line_dense.py with EMB_MODEL set first")
    dvecs = np.asarray(np.load(vpath, mmap_mode="r"), dtype=np.float32)
    pidx_d = {p: i for i, p in enumerate(pids_d)}
    model = SentenceTransformer(ld.MODEL)
    mys = Mystem()
    lem = lambda t: content_lemmas("".join(mys.lemmatize(t)))
    top = lambda sc, n: sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    out, pages = [], []
    for poem in POEMS:
        rels = [g for g in gold if poem in g["commented_text_id"].split(";")]
        if not rels:
            sys.exit(f"{poem}: not a commented poem of the gold")
        raw = [l.strip() for l in rels[0]["commented_text"].split("\n") if l.strip()]
        self_ids = set().union(*(dups.get(p, {p}) for p in rels[0]["commented_text_id"].split(";")))
        ok = nb.not_later(years, self_ids)
        attested = {}
        for g in rels:
            for pid in g["referent_text_id"].split(";"):
                for d in dups.get(pid, {pid}):
                    if d:
                        attested.setdefault(d, g["conn_id"])
        print(f"{poem}: {len(raw)} lines, {len(rels)} gold relations", file=sys.stderr)
        qv_all = model.encode([ld.QPFX + nb.norm(l) for l in raw], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
        line_sims = qv_all @ dvecs.T          # query line x corpus line
        poem_line = np.maximum.reduceat(line_sims, starts_d, axis=1)       # query line x poem

        # best[(probe, size)][candidate] = (score, window start)
        best = defaultdict(dict)
        for size in SIZES:
            for a in range(0, len(raw) - size + 1):
                q = nb.norm(" ".join(raw[a:a + size]))
                Q = lem(q)
                cand = defaultdict(int)
                for w in set(Q):
                    if w in inv and len(inv[w]) <= DF_CAP:
                        for pid in inv[w]:
                            cand[pid] += 1
                dense = poem_line[a:a + size].max(axis=0)
                S = {"ngram": nb.rank(nb.word_ngrams(q, 2), ng_index, ng_idf),
                     "bm25": bm.bm25_rank(q.split(), postings, doclen, idf_bm, avgdl),
                     "passim": {pid: sw(Q, seqs[pid], idf_lem) for pid, c in cand.items()
                                if c >= MIN_SHARED and pid not in self_ids and ok(pid)},
                     "wpair": wp.score(Q, wp_keep, wp_pidf, wp_inv, self_ids),
                     "dense": {pids_d[i]: float(dense[i]) for i in np.argsort(-dense)[:TOPN * 20]}}
                rrf = defaultdict(float)
                for probe, sc in S.items():
                    sc = {p: v for p, v in sc.items() if v > 0 and p not in self_ids and ok(p)}
                    for rank, (pid, _) in enumerate(top(sc, TOPN), 1):
                        rrf[pid] += 1.0 / (C + rank)
                    S[probe] = sc
                S["fusion"] = dict(rrf)
                for probe, sc in S.items():
                    b = best[(probe, size)]
                    for pid, v in top(sc, KEEP):
                        if pid not in b or v > b[pid][0]:
                            b[pid] = (v, a)

        def evidence(pid, a, size):
            """the candidate's line closest to the window (E5 line-to-line cosine)"""
            i = pidx_d.get(pid)
            if i is None:
                return ""
            lo = int(starts_d[i]); hi = int(starts_d[i + 1]) if i + 1 < len(starts_d) else len(dtexts)
            return dtexts[lo + int(line_sims[a:a + size, lo:hi].max(axis=0).argmax())]

        page = {"poem": poem, "work": rels[0]["commented_work"], "lines": raw,
                "year": max((years[s] for s in self_ids if s in years), default=""),
                "attested": sorted({g["conn_id"] for g in rels if g["referent_text_id"].strip()}),
                "tables": []}
        for probe, label in PROBES:
            for size in SIZES:
                rows = []
                for rank, (pid, _) in enumerate(top({p: sv[0] for p, sv in best[(probe, size)].items()}, KEEP), 1):
                    v, a = best[(probe, size)][pid]
                    au, ti = meta.get(pid, ("", ""))
                    row = {"poem": poem, "probe": probe, "window_lines": size, "rank": rank,
                           "window_start_line": a + 1, "window": " / ".join(raw[a:a + size]),
                           "candidate_id": pid, "candidate_author": au, "candidate_title": ti,
                           "candidate_year": years.get(pid, ""),
                           "candidate_line": evidence(pid, a, size), "score": round(v, 4),
                           "attested": attested.get(pid, "")}
                    rows.append(row); out.append(row)
                page["tables"].append((label, size, rows[:SHOW]))
        pages.append(page)

    with open(nb.HERE + "results/russian_poetry/window_search.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    write_html(pages)
    print("wrote results/russian_poetry/window_search.csv, results/russian_poetry/window_search.html", file=sys.stderr)


INTRO = ("Every window of 1, 2 and 4 consecutive lines of the poem is a query. Candidates are "
         "corpus poems not dated after the poem. For each probe and window size the five best candidates are "
         "shown, with the window that scored best and the candidate's closest line. A green row is a referent "
         "the commentary records for this poem. Every other row is unjudged: it may be noise, a commonplace, "
         "or a relation no commentator recorded. Scores are comparable only within one table.")


def write_html(pages, path=None, intro=INTRO, unit="line"):
    e = html.escape
    h = ["<!doctype html><html lang='ru'><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width, initial-scale=1'>",
         "<title>Window search</title><style>",
         "body{font:15px/1.45 Georgia,serif;max-width:1100px;margin:2em auto;padding:0 16px;color:#222;background:#fff}",
         "h1{font-size:1.5em}h2{margin-top:2.2em;border-top:2px solid #999;padding-top:.6em}h3{margin:1.6em 0 .2em}",
         "h4{margin:1em 0 .2em;font-weight:normal;color:#555}",
         "pre{font:inherit;background:#f6f4ee;padding:.8em 1em;white-space:pre-wrap}",
         "table{border-collapse:collapse;width:100%;font-size:.92em}td,th{border-bottom:1px solid #ddd;padding:.3em .5em;",
         "text-align:left;vertical-align:top}th{color:#555;font-weight:normal}",
         ".att{background:#e3f1df}.n{color:#777;white-space:nowrap}.note{color:#555;font-size:.95em}",
         "</style>",
         "<h1>Window search: what each probe retrieves from earlier poems</h1>",
         "<p class='note'>" + e(intro) + "</p>"]
    for p in pages:
        h.append(f"<h2>{e(p['work'])}{' (' + e(str(p['year'])) + ')' if p['year'] else ''} <span class='n'>{e(p['poem'])}</span></h2>")
        h.append("<pre>" + "\n".join(f"{i:>2}  {e(l)}" for i, l in enumerate(p["lines"], 1)) + "</pre>")
        h.append(f"<p class='note'>Gold relations of this poem with a referent in the corpus: "
                 f"{', '.join('conn ' + c for c in p['attested']) or 'none'}.</p>")
        last = None
        for label, size, rows in p["tables"]:
            if label != last:
                h.append(f"<h3>{e(label)}</h3>"); last = label
            h.append(f"<h4>windows of {size} {unit}{'s' if size > 1 else ''}</h4>")
            if not rows:
                h.append("<p class='note'>no candidate scored</p>"); continue
            h.append("<table><tr><th>#</th><th>window (line)</th><th>candidate</th><th>closest line</th><th>score</th></tr>")
            for r in rows:
                cand = f"{e(r['candidate_author'])}, {e(r['candidate_title'])}" + \
                    (f" ({e(str(r['candidate_year']))})" if r["candidate_year"] else "")
                if r["attested"]:
                    cand += " — attested" + ("" if r["attested"] == "yes" else f", conn {e(r['attested'])}")
                h.append(f"<tr class='{'att' if r['attested'] else ''}'><td class='n'>{r['rank']}</td>"
                         f"<td>{e(r['window'])} <span class='n'>({r['window_start_line']})</span></td>"
                         f"<td>{cand}</td><td>{e(r['candidate_line'])}</td><td class='n'>{r['score']}</td></tr>")
            h.append("</table>")
    with open(path or nb.HERE + "results/russian_poetry/window_search.html", "w", encoding="utf-8") as f:
        f.write("\n".join(h))


if __name__ == "__main__":
    main()
