"""Phase 3b — passage-level dense retrieval (LaBSE, max-sim late interaction).

Phase 3a failed because a 1-line query was matched against a mean-pooled whole-poem vector
(RESULTS.md). The fix: make the *passage* the unit. Chunk every poem into overlapping
line-windows, embed each window, and score a poem by the **max** cosine of any of its windows
to the query (DPR/ColBERT-style late interaction). A borrowed line now matches the window that
contains it, instead of being averaged away.

Same harness as the other baselines (query = commented_fragment else whole commented poem;
deduped Poetree+extras pool; self excluded; hit = gold referent / dup / cycle part).

Window vectors are cached (float16) under cache/embeddings/ — first run pays the CPU encode.
Owners are contiguous per poem, so per-poem max is a single np.maximum.reduceat.

    python scripts/passage_baseline.py [WIN]      # default WIN=2 lines, step 1

Needs: sentence-transformers, torch (CPU ok), numpy.
"""
from __future__ import annotations
import sys, os, csv, hashlib
from collections import defaultdict
import numpy as np
import ngram_baseline as nb

csv.field_size_limit(10**8)
MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/LaBSE")
CACHE = nb.HERE + "cache/embeddings/"
BATCH = 64
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 2     # window size in lines, step 1


def corpus_lines():
    """poem_id -> [normalized non-empty lines]. Includes extras."""
    out = {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        lns = [nb.norm(ln) for ln in p["full_text"].split("\n")]
        lns = [l for l in lns if l]
        if lns:
            out[p["poetree_poem_id"]] = lns
    for r in nb.load_added_referents():
        lns = [nb.norm(ln) for ln in r["full_text"].split("\n")]
        lns = [l for l in lns if l]
        if lns:
            out[r["extra_id"]] = lns
    return out


def build_windows(lines_by_id):
    """Returns poem_ids, poem_start offsets, and the flat list of window texts (contiguous)."""
    pids, starts, texts = [], [], []
    for pid, lns in lines_by_id.items():
        starts.append(len(texts))
        pids.append(pid)
        for i in range(len(lns)):
            texts.append(" ".join(lns[i:i + WIN]))
    return pids, np.array(starts, dtype=np.int64), texts


def embed_windows(model, texts, sig):
    vpath = CACHE + f"win{WIN}_{sig}.npy"
    if os.path.exists(vpath):
        print("  (cached window embeddings)", file=sys.stderr)
        return np.load(vpath, mmap_mode="r")
    print(f"  encoding {len(texts)} windows on CPU (one-time)…", file=sys.stderr)
    vecs = model.encode(texts, batch_size=BATCH, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float16)
    os.makedirs(CACHE, exist_ok=True)
    np.save(vpath, vecs)
    return vecs


def main():
    from sentence_transformers import SentenceTransformer
    print("loading corpus…", file=sys.stderr)
    lines_by_id = corpus_lines()
    flat = {pid: " ".join(lns) for pid, lns in lines_by_id.items()}
    dups = nb.dup_groups(flat)
    gold = nb.load_gold(dups)

    pids, starts, texts = build_windows(lines_by_id)
    pidx = {p: i for i, p in enumerate(pids)}
    sig = hashlib.md5((MODEL + f"|win{WIN}|" + str(len(texts)) + "|" +
                       hashlib.md5("".join(texts[::997]).encode()).hexdigest()).encode()).hexdigest()[:16]
    print(f"corpus poems {len(pids)} | windows {len(texts)} | grounded {len(gold)}", file=sys.stderr)

    print(f"loading {MODEL}…", file=sys.stderr)
    model = SentenceTransformer(MODEL)
    wvecs = embed_windows(model, texts, sig)
    wvecs = np.asarray(wvecs, dtype=np.float32)           # cast once into RAM for fast matmul

    qvecs = model.encode([g["q_frag"] for g in gold], batch_size=BATCH,
                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    allr, strat, band = [], defaultdict(list), defaultdict(list)
    for g, qv in zip(gold, qvecs):
        sims = wvecs @ qv                                  # (n_windows,)
        poem_max = np.maximum.reduceat(sims, starts)       # per-poem max (owners contiguous)
        for s in g["self"]:
            if s in pidx:
                poem_max[pidx[s]] = -1e9
        order = np.argsort(-poem_max)
        hitset = {pidx[h] for h in g["hit"] if h in pidx}
        rank = next((i for i, idx in enumerate(order, 1) if idx in hitset), None)
        allr.append(rank); strat[g["stratum"]].append(rank)
        if g["band"]:
            band[g["band"]].append(rank)

    def m(rs):
        n = len(rs) or 1
        return [sum(1 for r in rs if r and r <= k) / n for k in nb.KS] + [sum(1/r for r in rs if r)/n]
    def show(lab, rs):
        v = m(rs)
        print(f"  {lab:13s} n={len(rs):3d}  " +
              "  ".join(f"R@{k}={v[i]:.2f}" for i, k in enumerate(nb.KS)) + f"  MRR={v[-1]:.3f}")

    print(f"\n### dense passage max-sim ({MODEL.split('/')[-1]}, {WIN}-line windows) — fragment → referent")
    show("ALL grounded", allr)
    for s in ("S1", "S2", "S3", "S4"):
        if strat[s]: show(s, strat[s])
    for b in ("verbatim", "close", "loose"):
        if band[b]: show("S1/" + b, band[b])


if __name__ == "__main__":
    main()
