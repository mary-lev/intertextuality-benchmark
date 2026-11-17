"""Phase 3c — line-to-line dense retrieval. No whole-poem vectors anywhere.

The fix to 3b's asymmetry: split the QUERY into its real verse lines and match each line
against single corpus lines. The commented_fragment is stored flattened (line breaks lost), so
we recover the line borders by locating the fragment inside its commented poem (which keeps
them) — 97/105 fragments are exact substrings.

  * query   : fragment → recovered verse lines → one vector PER line
  * corpus  : every poem → single lines → one vector per line (no windows, no whole poem)
  * score   : poem = max over (query line × poem line) cosine — the single best line match;
              we keep the matched (query-line ↔ corpus-line) pair for inspection
  * scope   : the 105 connections that HAVE a fragment (S1/S2). S3/S4 have only a whole poem,
              no localized borrowing to query line-wise — excluded by design.

    python scripts/line_dense.py        # from the release root  (encodes single-line corpus once)

Writes line_dense_top5.md. Needs sentence-transformers, torch, numpy.
"""
from __future__ import annotations
import os, sys, csv, hashlib
from collections import defaultdict
import numpy as np
import ngram_baseline as nb
import passage_baseline as pb

csv.field_size_limit(10**8)
MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CACHE = nb.HERE + "cache/embeddings/"
BATCH = 64
TOPK = 5
# E5 models require asymmetric prefixes; other models use none.
_E5 = "e5" in MODEL.lower()
QPFX = "query: " if _E5 else ""
PPFX = "passage: " if _E5 else ""


def commented_poem_lines():
    out = {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        lns = [nb.norm(l) for l in p["full_text"].split("\n")]
        out[p["poetree_poem_id"]] = [l for l in lns if l]
    return out


def recover_query_lines(frag_norm, poem_lines):
    """Split the flat fragment into the verse lines it spans, via its position in the poem."""
    fw = frag_norm.split()
    seq = [(w, i) for i, ln in enumerate(poem_lines) for w in ln.split()]
    pw = [w for w, _ in seq]
    L = len(fw)
    if L == 0 or len(pw) < L:
        return None
    sset = set(fw)
    best, bi = -1, 0
    for i in range(len(pw) - L + 1):
        ov = sum(1 for w in pw[i:i + L] if w in sset)
        if ov > best:
            best, bi = ov, i
    if best < max(1, L * 0.6):
        return None
    li = sorted({seq[j][1] for j in range(bi, bi + L)})
    return [poem_lines[i] for i in li]


def encode_corpus_lines(model):
    """Flat single-line corpus. Returns poem_ids, starts, line_texts, vecs (mmap->RAM)."""
    lines_by_id = pb.corpus_lines()                       # id -> [normalized non-empty lines]
    pids, starts, texts = [], [], []
    for pid, lns in lines_by_id.items():
        starts.append(len(texts)); pids.append(pid)
        texts.extend(lns)
    starts = np.array(starts, dtype=np.int64)
    sig = hashlib.md5((MODEL + "|line1|" + str(len(texts)) + "|" +
                       hashlib.md5("".join(texts[::997]).encode()).hexdigest()).encode()).hexdigest()[:16]
    vpath = CACHE + f"line1_{sig}.npy"
    if os.path.exists(vpath):
        print("  (cached single-line embeddings)", file=sys.stderr)
        vecs = np.asarray(np.load(vpath, mmap_mode="r"), dtype=np.float32)
    else:
        print(f"  encoding {len(texts)} single lines on CPU (one-time)…", file=sys.stderr)
        v = model.encode([PPFX + t for t in texts], batch_size=BATCH, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float16)
        os.makedirs(CACHE, exist_ok=True)
        np.save(vpath, v)
        vecs = np.asarray(v, dtype=np.float32)
    return pids, starts, texts, vecs


def load_meta():
    meta = {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        meta[p["poetree_poem_id"]] = f'{p["author_name"]} — {p["title"]}'.strip(" —")
    for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/added_referents.csv", encoding="utf-8")):
        meta[r["extra_id"]] = f'{r["ref_author"]} — {r["ref_title"]}'.strip(" —")
    return meta


def main():
    from sentence_transformers import SentenceTransformer
    print("loading corpus…", file=sys.stderr)
    lines_by_id = pb.corpus_lines()
    flat = {pid: " ".join(lns) for pid, lns in lines_by_id.items()}
    dups = nb.dup_groups(flat)
    gold = {g["conn_id"]: g for g in nb.load_gold(dups)}
    cposlines = commented_poem_lines()
    meta = load_meta()

    print(f"loading {MODEL}…", file=sys.stderr)
    model = SentenceTransformer(MODEL)
    pids, starts, ltext, wvecs = encode_corpus_lines(model)
    pidx = {p: i for i, p in enumerate(pids)}
    ends = np.append(starts[1:], len(ltext))

    # build per-connection query lines (only fragment-bearing connections)
    rows = [r for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/gold.csv", encoding="utf-8"))
            if r["category"] == "poetry" and r["referent_text"].strip()
            and (r["commented_fragment"] or r["analyst_commented_span"]).strip()
            and r["conn_id"] in gold]
    qlines, recovered = {}, 0
    for r in rows:
        frag = nb.norm(r["commented_fragment"] or r["analyst_commented_span"])
        cids = [c for c in (r["commented_text_id"] or "").split(";") if c in cposlines]
        rec = recover_query_lines(frag, cposlines[cids[0]]) if cids else None
        if rec and len(rec) > 1:
            recovered += 1
        qlines[r["conn_id"]] = rec or [frag]               # fallback: the flat fragment as 1 line
    print(f"connections {len(rows)} | multi-line recovered {recovered}", file=sys.stderr)

    # encode every distinct query line once
    alllines = sorted({l for ls in qlines.values() for l in ls})
    lvec = dict(zip(alllines, model.encode([QPFX + l for l in alllines], batch_size=BATCH,
                                           normalize_embeddings=True,
                                           convert_to_numpy=True).astype(np.float32)))

    strat, band, md, rows_csv = defaultdict(list), defaultdict(list), [], []
    md.append(f"# Top-5 line-to-line dense matches per query — {MODEL.split('/')[-1]}, single lines\n")
    md.append("Each query verse line is matched against every corpus line; a poem ranks by its "
              "single best line↔line cosine. Candidates show that best pair (which query line "
              "matched which corpus line).\n")

    for r in rows:
        g = gold[r["conn_id"]]
        qls = qlines[r["conn_id"]]
        Q = np.stack([lvec[l] for l in qls])               # (k, 384)
        sims = Q @ wvecs.T                                 # (k, Nlines)
        per_poem = np.maximum.reduceat(sims, starts, axis=1)   # (k, npoems)
        poem_score = per_poem.max(axis=0)                  # best query line per poem
        best_q = per_poem.argmax(axis=0)                   # which query line won, per poem
        for s in g["self"]:
            if s in pidx:
                poem_score[pidx[s]] = -1e9
        order = np.argsort(-poem_score)
        hitset = {pidx[h] for h in g["hit"] if h in pidx}
        rank = next((i for i, idx in enumerate(order, 1) if idx in hitset), None)
        strat[g["stratum"]].append(rank)
        if g["band"]:
            band[g["band"]].append(rank)

        md.append(f"### conn {g['conn_id']}  ·  gold rank {rank if rank else '—'}"
                  f"{' · ' + g['band'] if g['band'] else ''}")
        for ql in qls:
            md.append(f"- **q-line:** `{ql}`")
        gold_names = " / ".join(meta.get(h, h) for h in sorted(g["hit"]) if h in meta) or "—"
        md.append(f"- **gold referent:** {gold_names}")
        for ri, i in enumerate(order[:TOPK], 1):
            qi = int(best_q[i])
            cli = starts[i] + int(np.argmax(sims[qi, starts[i]:ends[i]]))
            flag = "✓" if i in hitset else " "
            md.append(f"  {ri}. [{flag}] **{poem_score[i]:.2f}**  {meta.get(pids[i], pids[i])}  `({pids[i]})`")
            md.append(f"       ↳ `{qls[qi]}`  ⇄  `{ltext[cli]}`")
            rows_csv.append({
                "conn_id": g["conn_id"], "stratum": g["stratum"], "band": g["band"],
                "gold_rank": rank or "", "rank": ri, "score": f"{poem_score[i]:.4f}",
                "is_gold": int(i in hitset), "cand_id": pids[i],
                "cand_label": meta.get(pids[i], pids[i]),
                "q_line": qls[qi], "matched_corpus_line": ltext[cli],
            })
        md.append("")

    def m(rs):
        n = len(rs) or 1
        return [sum(1 for r in rs if r and r <= k) / n for k in nb.KS] + [sum(1/r for r in rs if r)/n]
    def show(lab, rs):
        v = m(rs)
        print(f"  {lab:13s} n={len(rs):3d}  " +
              "  ".join(f"R@{k}={v[i]:.2f}" for i, k in enumerate(nb.KS)) + f"  MRR={v[-1]:.3f}")

    print(f"\n### line-to-line dense ({MODEL.split('/')[-1]} single lines, query split into verse lines)")
    show("S1", strat["S1"])
    if strat["S2"]: show("S2", strat["S2"])
    for b in ("verbatim", "close", "loose"):
        if band[b]: show("S1/" + b, band[b])

    tag = "" if "MiniLM" in MODEL else "_" + MODEL.split("/")[-1]
    open(nb.HERE + f"results/russian_poetry/line_dense_top5{tag}.md", "w", encoding="utf-8").write("\n".join(md))
    with open(nb.HERE + f"results/russian_poetry/line_dense_top5{tag}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conn_id", "stratum", "band", "gold_rank", "rank",
                                          "score", "is_gold", "cand_id", "cand_label",
                                          "q_line", "matched_corpus_line"])
        w.writeheader(); w.writerows(rows_csv)
    print(f"wrote line_dense_top5.md + .csv ({len(rows_csv)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
