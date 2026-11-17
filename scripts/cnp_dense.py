"""Dense (semantic) line-to-line retrieval for the CnP→NT Bible-reference pipeline.

The lexical pipeline (Smith–Waterman on lemmas) recovers verbatim/close quotation but
MISSES thematic / paraphrase allusions that share meaning, not wording. This is an
additive, parallel method: embed every NT verse and every CnP *sentence*, then for each
sentence keep the top-5 cosine-nearest NT verses. The point is to test whether dense
embeddings reach the thematic gold notes the lexical method missed.

  * corpus  : NT verses, data/crime_and_punishment/nt1823.jsonl (fields ref, text; 7952 verses)
  * query   : CnP segments, results/crime_and_punishment/cnp_segments.jsonl, sentence-level only (~15,555)
  * model   : EMB_MODEL env, default paraphrase-multilingual-MiniLM-L12-v2;
              e5 models get query:/passage: prefixes. normalize_embeddings=True.
  * score   : cosine top-k (k=5) NT verses per CnP sentence (matrix product, chunked).
  * cache   : embeddings cached to .emb_cache/ keyed by hash(model, texts).

    python scripts/cnp_dense.py

Writes results/crime_and_punishment/cnp_dense_top5.jsonl and prints an evaluation report to stderr.
Needs sentence-transformers, torch, numpy.
"""
from __future__ import annotations
import os, sys, json, time, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)            # release root
EXP = ROOT
CACHE = os.path.join(ROOT, "cache", "embeddings")

MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
BATCH = 64
TOPK = 5
CHUNK = 1000                                             # segments per matmul chunk
_E5 = "e5" in MODEL.lower()
QPFX = "query: " if _E5 else ""
PPFX = "passage: " if _E5 else ""


def load_nt():
    refs, texts = [], []
    with open(os.path.join(EXP, "data/crime_and_punishment/nt1823.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            refs.append((o["ref"], o["code"], o["chapter"]))
            texts.append(o["text"])
    return refs, texts


def load_segments():
    segs = []
    with open(os.path.join(EXP, "results/crime_and_punishment/cnp_segments.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("level") == "sentence":
                segs.append((o["id"], o["paragraph_ref"], o["text"]))
    return segs


def encode(model, texts, prefix, tag):
    """Encode texts (normalized), caching to .emb_cache keyed by hash(model, texts)."""
    sig = hashlib.md5(
        (MODEL + "|" + tag + "|" + str(len(texts)) + "|" +
         hashlib.md5("".join(texts[::997]).encode("utf-8")).hexdigest()).encode("utf-8")
    ).hexdigest()[:16]
    vpath = os.path.join(CACHE, f"cnp_{tag}_{sig}.npy")
    if os.path.exists(vpath):
        print(f"  (cached {tag} embeddings: {len(texts)})", file=sys.stderr)
        return np.asarray(np.load(vpath, mmap_mode="r"), dtype=np.float32)
    print(f"  encoding {len(texts)} {tag} texts on CPU…", file=sys.stderr)
    v = model.encode([prefix + t for t in texts], batch_size=BATCH,
                     normalize_embeddings=True, show_progress_bar=True,
                     convert_to_numpy=True).astype(np.float16)
    os.makedirs(CACHE, exist_ok=True)
    np.save(vpath, v)
    return np.asarray(v, dtype=np.float32)


# ---- evaluation probes -------------------------------------------------------
# Each: label, paragraph_ref, gold book code, gold chapter, note.
PROBES = [
    # THEMATIC gold the LEXICAL method MISSED — the real test
    ("THEMATIC «Новый Иерусалим»",       "ч3.гл5:p1855", "Откр", 21, ""),
    ("THEMATIC «Звезда Полынь»",         "эп.гл2:p3743", "Откр",  8, "v11"),
    ("THEMATIC «Лазаря петь»",           "ч3.гл4:p1738", "Лк",   16, ""),
    ("THEMATIC «всё тайное явным»",      "ч1.гл2:p63",   "Мф",   10, "26-27"),
    ("THEMATIC «Се человек»",            "ч1.гл2:p63",   "Ин",   19, "5"),
    # POSITIVE CONTROLS the lexical method FOUND — dense should also get these
    ("CONTROL Lazarus «воскресение»",    "ч4.гл4:p2440", "Ин",   11, ""),
    ("CONTROL «грехи твои мнози»",       "ч1.гл2:p83",   "Лк",    7, ""),
    ("CONTROL «царствие твое» (Мф)",     "ч1.гл2:p83",   "Мф",    6, ""),
    ("CONTROL «царствие твое» (Лк)",     "ч1.гл2:p83",   "Лк",   11, ""),
]


def main():
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    print(f"loading corpora…", file=sys.stderr)
    nt_meta, nt_texts = load_nt()
    segs = load_segments()
    print(f"  NT verses={len(nt_texts)}  CnP sentences={len(segs)}", file=sys.stderr)

    print(f"loading {MODEL}…", file=sys.stderr)
    try:
        model = SentenceTransformer(MODEL)
    except Exception as e:
        print(f"BLOCKED: could not load model {MODEL}: {e}", file=sys.stderr)
        sys.exit(2)

    nt_vecs = encode(model, nt_texts, PPFX, "nt")               # (V, d)
    seg_vecs = encode(model, [s[2] for s in segs], QPFX, "seg")  # (N, d)

    ntT = nt_vecs.T                                             # (d, V)
    out_path = os.path.join(EXP, "results/crime_and_punishment/cnp_dense_top5.jsonl")

    # ---- pass 1: per-VERSE hubness (mean cosine of each verse across ALL segments) ----
    # Raw cosines are anisotropic: some NT verses are "hubs" close to everything (mean cos up to
    # ~0.45), flooding every segment's top-k. Centering each verse by its hubness — cos_c =
    # cos - vmean[verse] — surfaces the SPECIFIC matches and, unlike a per-segment z-score, does NOT
    # penalise short biblical-register phrases. It also makes the score COMPARABLE ACROSS segments
    # (raw thematic golds sit at/below the 0.666 corpus median; centered they rise above p90), so the
    # top-1 cos_c becomes thresholdable. Measured 2026-06-25.
    print(f"pass 1/2: per-verse hubness…", file=sys.stderr)
    vsum = np.zeros(len(nt_texts), np.float64)
    for c0 in range(0, len(segs), CHUNK):
        vsum += (seg_vecs[c0:c0 + CHUNK] @ ntT).sum(axis=0)
    vmean = (vsum / len(segs)).astype(np.float32)               # (V,) verse hubness

    # ---- pass 2: top-k by raw cosine; emit cos_c (centered) as a comparable/diagnostic score ----
    # Measured 2026-06-25: re-ranking the top-k BY cos_c is recall-neutral on the thematic gold
    # (it trades «Лазаря петь»→Лк 16 for «Звезда Полынь»→Откр 8) and centering does NOT make dense
    # thresholdable as a source (the thematic matches still sit below the background p90). So retrieval
    # stays cos-ranked (preserving the documented thematic adds); cos_c rides along as the
    # cross-segment-comparable score. dense remains a recall-only enricher.
    print(f"pass 2/2: retrieving top-{TOPK} by cosine (cos_c emitted alongside)…", file=sys.stderr)
    top1c = []                                                 # top-1 centered score per segment
    top1 = []                                                  # top-1 raw cosine per segment
    # per-paragraph store of (seg_id, seg_text, list[(rank, code, chap, ref, text, cos, cos_c)])
    para_hits = {}
    probe_paras = {p[1] for p in PROBES}

    with open(out_path, "w", encoding="utf-8") as out:
        for c0 in range(0, len(segs), CHUNK):
            chunk = segs[c0:c0 + CHUNK]
            sims = seg_vecs[c0:c0 + CHUNK] @ ntT                # (chunk, V) raw cosine
            simc = sims - vmean[None, :]                        # (chunk, V) centered (comparable)
            idx = np.argpartition(-sims, TOPK, axis=1)[:, :TOPK]   # top-k by raw COSINE
            for r in range(len(chunk)):
                seg_id, para, text = chunk[r]
                ti = idx[r]
                ti = ti[np.argsort(-sims[r, ti])]              # sort the topk by cos
                tops = []
                for j in ti:
                    ref, code, chap = nt_meta[j]
                    tops.append({"nt_ref": ref, "nt_text": nt_texts[j],
                                 "cos": round(float(sims[r, j]), 4),
                                 "cos_c": round(float(simc[r, j]), 4)})
                out.write(json.dumps({"seg_id": seg_id, "paragraph_ref": para,
                                      "seg_text": text, "top": tops},
                                     ensure_ascii=False) + "\n")
                top1c.append(float(simc[r].max()))            # best centered score (thresholdability)
                top1.append(float(sims[r, ti[0]]))            # top-1 raw cosine
                if para in probe_paras:
                    rec = [(rk + 1, nt_meta[j][1], nt_meta[j][2], nt_meta[j][0],
                            nt_texts[j], float(sims[r, j]), float(simc[r, j]))
                           for rk, j in enumerate(ti)]
                    para_hits.setdefault(para, []).append((seg_id, text, rec))

    dt = time.time() - t0
    print(f"\nwrote {out_path}  ({len(segs)} rows)  in {dt:.1f}s  model={MODEL}",
          file=sys.stderr)

    # ---- top-1 score distributions (raw cos vs centered cos_c) + suggested floor ----
    a = np.array(top1); ac = np.array(top1c)
    print("\n=== top-1 distribution (across all sentences) ===", file=sys.stderr)
    print(f"  raw cos : median={np.median(a):.3f}  p90={np.percentile(a,90):.3f}  max={a.max():.3f}",
          file=sys.stderr)
    print(f"  cos_c   : median={np.median(ac):.3f}  p90={np.percentile(ac,90):.3f}  "
          f"p95={np.percentile(ac,95):.3f}  max={ac.max():.3f}  "
          f"(thresholdable; p90≈{np.percentile(ac,90):.3f})", file=sys.stderr)

    # ---- probe table ----
    print("\n=== PROBE TABLE (gold book+chapter reached in some sentence's top-5?) ===",
          file=sys.stderr)
    print(f"{'probe':32s} | {'gold':9s} | best dense hit at paragraph                 | result",
          file=sys.stderr)
    print("-" * 110, file=sys.stderr)
    n_them_hit = n_them = n_ctrl_hit = n_ctrl = 0
    for label, para, gcode, gchap, note in PROBES:
        is_them = label.startswith("THEMATIC")
        if is_them: n_them += 1
        else: n_ctrl += 1
        gold = f"{gcode} {gchap}"
        sents = para_hits.get(para, [])
        # best (rank, cos, cos_c, ref) among sentences whose top-5 hits gold book+chapter
        best = None  # (rank, cos, cos_c, ref)
        for seg_id, text, rec in sents:
            for rank, code, chap, ref, vtext, cos, cosc in rec:
                if code == gcode and chap == gchap:
                    cand = (rank, cos, cosc, ref)
                    if best is None or rank < best[0] or (rank == best[0] and cosc > best[2]):
                        best = cand
        if best is not None:
            hit = "HIT"
            if is_them: n_them_hit += 1
            else: n_ctrl_hit += 1
            desc = f"{best[3]:14s} rank={best[0]} cos={best[1]:.3f} cos_c={best[2]:.3f}"
        else:
            hit = "MISS"
            # show the single overall-best centered score in this paragraph for orientation
            allrec = [(cosc, ref) for _, _, rec in sents
                      for (_, _, _, ref, _, _, cosc) in rec]
            if allrec:
                bc, br = max(allrec)
                desc = f"(no gold; best in para {br} cos_c={bc:.3f})"
            else:
                desc = "(paragraph not found / no sentences)"
        nt = f" [{note}]" if note else ""
        print(f"{label:32s} | {gold:9s} | {desc:43s} | {hit}{nt}", file=sys.stderr)

    print("-" * 110, file=sys.stderr)
    print(f"THEMATIC (lexically-missed) reached in top-5: {n_them_hit}/{n_them}",
          file=sys.stderr)
    print(f"CONTROL  (lexically-found)  reached in top-5: {n_ctrl_hit}/{n_ctrl}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
