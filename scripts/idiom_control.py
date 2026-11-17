"""Idiom control: do the first-stage probes retrieve poems that merely share a language idiom?

Post hoc and descriptive. The idioms are Wiktionary's «Фразеологизмы/ru» entries of three or
more words found in 2-99 corpus poems (data/russian_poetry/idiom_candidates.csv; not reviewed by a reader).
An idiom "occurs" in a poem when its Mystem lemmas appear contiguously and in order in the
poem's lemma text. For every poem that has an idiom, the verse line(s) holding it are the
query; the other poems that have the same idiom are the targets. Each probe ranks the corpus
as in factorial.py (query poem and its text-duplicates excluded) and the rank of every target
is recorded. No relation between the poems is assumed.

    python scripts/idiom_control.py lexical                      # from the release root; CPU
    python scripts/idiom_control.py dense                        # MiniLM line cache
    EMB_MODEL=intfloat/multilingual-e5-base python scripts/idiom_control.py dense
    python scripts/idiom_control.py summary

Writes results/russian_poetry/idiom_control_{lexical,dense_minilm,dense_e5}.csv (one row per query-target
pair) and prints the summary. No API calls.
"""
from __future__ import annotations
import csv, hashlib, math, os, sys
from collections import defaultdict

STAGE = sys.argv[1] if len(sys.argv) > 1 else "summary"
sys.argv = sys.argv[:1]                    # wpair_baseline reads sys.argv at import
import numpy as np
import ngram_baseline as nb
import wpair_baseline as wp

csv.field_size_limit(10 ** 9)
R = nb.HERE + "results/russian_poetry/"
LEMMAS = nb.HERE + "cache/lemmas/poetree_lemmas.csv"


def load_queries():
    """-> [{idiom, q_poem, q_lines (normalized verse lines), self, targets}]"""
    idioms = {}
    for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/idiom_candidates.csv", encoding="utf-8")):
        idioms.setdefault(tuple(r["lemmas"].split()), r["idiom"])     # aspect pairs share lemmas
    surf_lines, author = {}, {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        surf_lines[p["poetree_poem_id"]] = p["full_text"].split("\n"); author[p["poetree_poem_id"]] = p["author_name"]
    for r in nb.load_added_referents():
        surf_lines[r["extra_id"]] = r["full_text"].split("\n")
    first = defaultdict(lambda: defaultdict(list))                    # first lemma -> rest -> idiom keys
    for k in idioms:
        first[k[0]][k[1]].append(k)
    occ = defaultdict(dict)                                           # idiom key -> poem -> (line_a, line_b)
    misaligned = 0
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        pid = r["poem_id"]; ll = r["lemma_text"].split("\n")
        flat, where = [], []
        for i, ln in enumerate(ll):
            for w in wp.content_lemmas(ln):
                flat.append(w); where.append(i)
        for j in range(len(flat) - 2):
            for k in first.get(flat[j], {}).get(flat[j + 1], ()):
                if pid not in occ[k] and tuple(flat[j:j + len(k)]) == k:
                    if len(ll) != len(surf_lines.get(pid, [])):
                        misaligned += 1; continue
                    occ[k][pid] = (where[j], where[j + len(k) - 1])
    dups = nb.dup_groups(nb.load_corpus())
    out = []
    for k, poems in occ.items():
        for pid, (a, b) in poems.items():
            self_ids = dups.get(pid, {pid})
            targets = sorted(set(poems) - self_ids)
            q_lines = [x for x in (nb.norm(l) for l in surf_lines[pid][a:b + 1]) if x]
            if targets and q_lines:
                out.append({"idiom": idioms[k], "q_poem": pid, "q_lines": q_lines, "self": self_ids,
                            "targets": targets, "author": author})
    print(f"{len(occ)} idioms with a located occurrence, {len(out)} queries, "
          f"{sum(len(q['targets']) for q in out)} pairs; {misaligned} occurrences skipped (line count mismatch)",
          file=sys.stderr)
    return out


def ranks_of(scores, targets):
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    pos = {pid: i for i, (pid, _) in enumerate(order, 1)}
    return [pos.get(t, "") for t in targets]


def write(name, queries, probes, per_query):
    with open(R + name, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idiom", "q_poem", "query", "t_poem", "same_author", *probes])
        for q, pr in zip(queries, per_query):
            au = q["author"]
            for i, t in enumerate(q["targets"]):
                w.writerow([q["idiom"], q["q_poem"], " / ".join(q["q_lines"]), t,
                            int(bool(au.get(t)) and au.get(t) == au.get(q["q_poem"])), *[pr[p][i] for p in probes]])
    print("wrote", name, file=sys.stderr)


def lexical():
    from pymystem3 import Mystem
    from passim_baseline import sw
    from tesserae_baseline import content_lemmas, DF_CAP, MIN_SHARED
    import bm25_baseline as bm
    queries = load_queries()
    surf = nb.load_corpus()
    ng_index, ng_idf = nb.build_index(surf, lambda t: nb.word_ngrams(t, 2))
    c4_index, c4_idf = nb.build_index(surf, lambda t: nb.char_ngrams(t, 4))
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
    mys = Mystem()
    probes = ("ngram", "char4", "bm25", "passim", "wpair")
    per_query = []
    for qi, q in enumerate(queries):
        text = " ".join(q["q_lines"]); pr = {}
        def drop(sc):
            for s in q["self"]:
                sc.pop(s, None)
            return sc
        pr["ngram"] = ranks_of(drop(nb.rank(nb.word_ngrams(text, 2), ng_index, ng_idf)), q["targets"])
        pr["char4"] = ranks_of(drop(nb.rank(nb.char_ngrams(text, 4), c4_index, c4_idf)), q["targets"])
        pr["bm25"] = ranks_of(drop(bm.bm25_rank(text.split(), postings, doclen, idf_bm, avgdl)), q["targets"])
        Q = content_lemmas("".join(mys.lemmatize(text)))
        cand = defaultdict(int)
        for w in set(Q):
            if w in inv and len(inv[w]) <= DF_CAP:
                for pid in inv[w]:
                    cand[pid] += 1
        cands = [pid for pid, c in cand.items() if c >= MIN_SHARED and pid not in q["self"]]
        pr["passim"] = ranks_of({pid: sw(Q, seqs[pid], idf_lem) for pid in cands}, q["targets"])
        pr["wpair"] = ranks_of(dict(wp.score(Q, wp_keep, wp_pidf, wp_inv, q["self"])), q["targets"])
        per_query.append(pr)
        if (qi + 1) % 100 == 0:
            print(f"  {qi+1}/{len(queries)}", file=sys.stderr, flush=True)
    write("idiom_control_lexical.csv", queries, probes, per_query)


def dense():
    from sentence_transformers import SentenceTransformer
    import passage_baseline as pb
    import line_dense as ld
    name = "dense_e5" if "e5" in ld.MODEL else "dense_minilm"
    queries = load_queries()
    pids, starts, texts = [], [], []
    for pid, lns in pb.corpus_lines().items():
        starts.append(len(texts)); pids.append(pid); texts.extend(lns)
    starts = np.array(starts, dtype=np.int64)
    sig = hashlib.md5((ld.MODEL + "|line1|" + str(len(texts)) + "|" +
                       hashlib.md5("".join(texts[::997]).encode()).hexdigest()).encode()).hexdigest()[:16]
    vpath = ld.CACHE + f"line1_{sig}.npy"
    if not os.path.exists(vpath):
        sys.exit("line cache missing for " + ld.MODEL)
    vecs = np.asarray(np.load(vpath, mmap_mode="r"), dtype=np.float32)
    pidx = {p: i for i, p in enumerate(pids)}
    model = SentenceTransformer(ld.MODEL)
    per_query = []
    for qi, q in enumerate(queries):
        Qv = model.encode([ld.QPFX + l for l in q["q_lines"]], normalize_embeddings=True,
                          convert_to_numpy=True).astype(np.float32)
        pm = np.maximum.reduceat(Qv @ vecs.T, starts, axis=1).max(axis=0)
        for s in q["self"]:
            if s in pidx:
                pm[pidx[s]] = -np.inf
        srt = np.sort(pm)
        rk = [int(len(pm) - np.searchsorted(srt, pm[pidx[t]], side="right") + 1) if t in pidx else ""
              for t in q["targets"]]
        per_query.append({name: rk})
        if (qi + 1) % 100 == 0:
            print(f"  {qi+1}/{len(queries)}", file=sys.stderr, flush=True)
    write(f"idiom_control_{name}.csv", queries, (name,), per_query)


def summary():
    rows = {}
    for f in ("lexical", "dense_minilm", "dense_e5"):
        if os.path.exists(R + f"idiom_control_{f}.csv"):
            for r in csv.DictReader(open(R + f"idiom_control_{f}.csv", encoding="utf-8")):
                rows.setdefault((r["idiom"], r["q_poem"], r["t_poem"]), {}).update(r)
    rows = list(rows.values())
    probes = [p for p in ("ngram", "char4", "passim", "bm25", "wpair", "dense_minilm", "dense_e5") if p in rows[0]]
    hit = lambda r, p, k: bool(r[p]) and int(r[p]) <= k
    def report(label, sel):
        byq, byi = defaultdict(list), defaultdict(list)
        for r in sel:
            byq[(r["idiom"], r["q_poem"])].append(r); byi[r["idiom"]].append(r)
        print(f"\n{label}: {len(byi)} idioms, {len(byq)} queries, {len(sel)} query-target pairs")
        print("  probe          any target in top 10 (queries)   pair R@1  R@10  R@50   pair R@10, mean over idioms")
        for p in probes:
            anyq = sum(any(hit(r, p, 10) for r in v) for v in byq.values()) / len(byq)
            pr = [sum(hit(r, p, k) for r in sel) / len(sel) for k in (1, 10, 50)]
            macro = sum(sum(hit(r, p, 10) for r in v) / len(v) for v in byi.values()) / len(byi)
            print(f"  {p:13s}  {anyq:.2f}                             {pr[0]:.2f}     {pr[1]:.2f}  {pr[2]:.2f}   {macro:.2f}")
    report("ALL", rows)
    report("different authors only", [r for r in rows if r["same_author"] == "0"])
    n = defaultdict(set)
    for r in rows:
        n[r["idiom"]].add(r["q_poem"])
    for lo, hi in ((2, 3), (4, 9), (10, 99)):
        report(f"idioms in {lo}-{hi} poems", [r for r in rows if lo <= len(n[r["idiom"]]) <= hi])


{"lexical": lexical, "dense": dense, "summary": summary}[STAGE]()
