"""Word-bigram EMBEDDING retrieval for CnP->NT (Burns et al. 2021, survey §3).

Plain dense (cnp_dense.py) embeds whole verses / whole sentences, so a short reordered
echo ("иерусалим новый" <- "Новый Иерусалим") drowns in the surrounding sentence vector.
This method works at the PHRASE level instead: it represents every consecutive
content-lemma BIGRAM by the (order-invariant) mean of its two word embeddings, and scores a
verse by the best matching bigram pair. Because the bigram vector is the MEAN of the two
lemma vectors it is reorder-tolerant, and because the lemma vectors are dense it is
synonym-tolerant — exactly the Burns et al. "word-embedding n-gram" matcher.

  * lemma vectors : every DISTINCT content lemma from NT verses + CnP sentence segments,
                    embedded with paraphrase-multilingual-MiniLM-L12-v2 (the model
                    cnp_dense.py uses), normalize_embeddings=True, cached lemma->vector.
  * bigram vector : L2-normalized mean of the two adjacent content-lemma vectors.
  * retrieval     : verse score = max cosine over (segment-bigram x verse-bigram);
                    per segment aggregate to per-verse max, take top-5 verses.
                    The (segment-bigram x verse-bigram) product is chunked to bound memory.

    python scripts/cnp_bigram_embed.py

Writes results/crime_and_punishment/cnp_bigram_top5.jsonl and prints a dense-vs-bigram reach report to stderr.
Content lemmas via cnp_match._content; lemma sources are the .lemma_cache jsonl files.
Needs sentence-transformers, torch, numpy.
"""
from __future__ import annotations
import os, sys, json, time, hashlib, re
from collections import defaultdict
import numpy as np

from cnp_match import _content
from cnp_eval_commentary import load_gold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)            # release root
EXP = ROOT
CACHE = os.path.join(ROOT, "cache", "lemmas")
EMB_CACHE = os.path.join(ROOT, "cache", "embeddings")

MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
BATCH = 64
TOPK = 5
BIGRAM_CHUNK = 2000                                     # segment-bigrams per matmul chunk


# ---- corpora -----------------------------------------------------------------
def load_verses():
    """[(verse_id, [content lemmas]), ...] from the NT lemma cache."""
    out = []
    with open(os.path.join(CACHE, "nt1823_lemmas.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append((r["id"], _content(r["lemmas"])))
    return out


def load_segments():
    """[(seg_id, [content lemmas]), ...] — sentence-level seg_ids only (/s, not /c)."""
    out = []
    with open(os.path.join(CACHE, "cnp_segments_lemmas.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "/s" in r["id"] and "/c" not in r["id"]:
                out.append((r["id"], _content(r["lemmas"])))
    return out


def bigrams(seq):
    """Consecutive content-lemma bigrams as (a, b) tuples."""
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


# ---- lemma -> vector ---------------------------------------------------------
def embed_lemmas(model, lemmas):
    """Embed each DISTINCT content lemma (normalized); cache lemma->vector map."""
    lemmas = sorted(set(lemmas))
    sig = hashlib.md5(
        (MODEL + "|lemmas|" + str(len(lemmas)) + "|" +
         hashlib.md5("".join(lemmas[::97]).encode("utf-8")).hexdigest()).encode("utf-8")
    ).hexdigest()[:16]
    vpath = os.path.join(EMB_CACHE, f"cnp_bigram_lemmavecs_{sig}.npy")
    kpath = os.path.join(EMB_CACHE, f"cnp_bigram_lemmakeys_{sig}.json")
    if os.path.exists(vpath) and os.path.exists(kpath):
        print(f"  (cached {len(lemmas)} lemma vectors)", file=sys.stderr)
        keys = json.load(open(kpath, encoding="utf-8"))
        vecs = np.asarray(np.load(vpath, mmap_mode="r"), dtype=np.float32)
        return {w: vecs[i] for i, w in enumerate(keys)}
    print(f"  encoding {len(lemmas)} distinct lemmas on CPU…", file=sys.stderr)
    vecs = model.encode(lemmas, batch_size=BATCH, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float16)
    os.makedirs(EMB_CACHE, exist_ok=True)
    np.save(vpath, vecs)
    json.dump(lemmas, open(kpath, "w", encoding="utf-8"), ensure_ascii=False)
    vecs = np.asarray(vecs, dtype=np.float32)
    return {w: vecs[i] for i, w in enumerate(lemmas)}


def bigram_matrix(items, lemvec):
    """Build (M, d) L2-normalized mean-of-pair bigram matrix + owner ids per row.

    items: list of (owner_id, [content lemmas]).
    Returns (mat float32 (M,d), owner np.int32 (M,), owner_ids list).
    """
    rows, owners = [], []
    for oi, (_, seq) in enumerate(items):
        for a, b in bigrams(seq):
            va, vb = lemvec.get(a), lemvec.get(b)
            if va is None or vb is None:
                continue
            rows.append((va + vb) * 0.5)
            owners.append(oi)
    if not rows:
        return np.zeros((0, 1), np.float32), np.zeros((0,), np.int32)
    mat = np.asarray(rows, dtype=np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return mat, np.asarray(owners, dtype=np.int32)


# ---- evaluation helpers (mirror cnp_baselines.py) ----------------------------
BOOK = {"матфея": "Мф", "марка": "Мк", "луки": "Лк", "иоанна": "Ин"}


def book_sigla(s):
    s = (s or "").lower()
    if "откровение" in s:
        return "Откр"
    for k, v in BOOK.items():
        if k in s:
            return v
    return "Ин" if s.strip() == "евангелие" else None


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[«»\"“”']", " ", t)).strip()


def bc(ref):
    m = re.match(r"(.+?) (\d+):", ref)
    return (m.group(1), int(m.group(2))) if m else (ref, 0)


def by_para(ids, tops):
    d = defaultdict(set)
    for sid, top in zip(ids, tops):
        d[sid.split("/")[0]].update(bc(v) for v in top[:TOPK])
    return d


def load_top(path):
    ids, tops = [], []
    with open(path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            ids.append(r["seg_id"])
            tops.append([h["nt_ref"] if isinstance(h, dict) else h for h in r["top"]])
    return ids, tops


# ---- main --------------------------------------------------------------------
def main():
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    print("loading corpora…", file=sys.stderr)
    verses = load_verses()
    segs = load_segments()
    print(f"  NT verses={len(verses)}  CnP sentences={len(segs)}", file=sys.stderr)

    print(f"loading {MODEL}…", file=sys.stderr)
    try:
        model = SentenceTransformer(MODEL)
    except Exception as e:
        print(f"BLOCKED: could not load model {MODEL}: {e}", file=sys.stderr)
        sys.exit(2)

    # distinct content lemmas across both corpora
    all_lemmas = set()
    for _, seq in verses:
        all_lemmas.update(seq)
    for _, seq in segs:
        all_lemmas.update(seq)
    lemvec = embed_lemmas(model, all_lemmas)
    print(f"  lemma vectors: {len(lemvec)}", file=sys.stderr)

    # verse-bigram matrix
    vmat, vown = bigram_matrix(verses, lemvec)          # (Vb, d), owner -> verse idx
    print(f"  verse-bigrams: {vmat.shape[0]}", file=sys.stderr)
    vmatT = vmat.T                                       # (d, Vb)
    n_verse = len(verses)

    out_path = os.path.join(EXP, "results/crime_and_punishment/cnp_bigram_top5.jsonl")
    top1 = []

    # collect everything for top5 + reach
    seg_ids, seg_tops = [], []
    n_seg_bigrams = 0

    with open(out_path, "w", encoding="utf-8") as out:
        # process segment by segment, but chunk its bigrams x verse-bigrams
        for sid, seq in segs:
            para = sid.split("/")[0]
            smat, sown = bigram_matrix([(sid, seq)], lemvec)   # (Sb, d)
            sb = smat.shape[0]
            n_seg_bigrams += sb
            if sb == 0 or vmat.shape[0] == 0:
                out.write(json.dumps({"seg_id": sid, "paragraph_ref": para, "top": []},
                                     ensure_ascii=False) + "\n")
                seg_ids.append(sid); seg_tops.append([])
                continue
            # per-verse max cosine, accumulated over segment-bigram chunks
            best = np.full(n_verse, -1.0, dtype=np.float32)
            for c0 in range(0, sb, BIGRAM_CHUNK):
                sims = smat[c0:c0 + BIGRAM_CHUNK] @ vmatT       # (chunk, Vb)
                cmax = sims.max(axis=0)                          # best over these seg-bigrams
                np.maximum.at(best, vown, cmax)                 # scatter-max into per-verse
            ti = np.argpartition(-best, TOPK)[:TOPK]
            ti = ti[np.argsort(-best[ti])]
            tops = [{"nt_ref": verses[j][0], "cos": round(float(best[j]), 4)} for j in ti]
            out.write(json.dumps({"seg_id": sid, "paragraph_ref": para, "top": tops},
                                 ensure_ascii=False) + "\n")
            seg_ids.append(sid)
            seg_tops.append([t["nt_ref"] for t in tops])
            top1.append(float(best[ti[0]]))

    dt = time.time() - t0
    print(f"\nwrote {out_path}  ({len(segs)} rows)  in {dt:.1f}s  model={MODEL}",
          file=sys.stderr)
    print(f"counts: lemmas={len(lemvec)}  verse-bigrams={vmat.shape[0]}  "
          f"segment-bigrams={n_seg_bigrams}", file=sys.stderr)

    a = np.array(top1)
    if len(a):
        print("\n=== top-1 cosine distribution (per sentence, max bigram cos) ===",
              file=sys.stderr)
        print(f"  n={len(a)}  min={a.min():.3f}  median={np.median(a):.3f}  "
              f"p90={np.percentile(a,90):.3f}  max={a.max():.3f}", file=sys.stderr)

    # ---- dense vs bigram reach over the РВБ gold (mirror cnp_baselines.py) ----
    P_bi = by_para(seg_ids, seg_tops)
    di, dt2 = load_top(os.path.join(EXP, "results/crime_and_punishment/cnp_dense_top5.jsonl"))
    P_dense = by_para(di, dt2)

    paras = [(p["ref"], norm(p["text"])) for p in
             (json.loads(l) for l in open(os.path.join(EXP, "data/crime_and_punishment/crime_and_punishment.jsonl"),
                                          encoding="utf-8"))]

    def find_para(frag):
        for key in sorted([s.strip() for s in re.split(r"~|\.\.\.|…", norm(frag))],
                          key=len, reverse=True):
            if len(key) >= 6:
                for ref, txt in paras:
                    if key[:30] in txt:
                        return ref
        fk = norm(frag)[:18]
        return next((ref for ref, txt in paras if fk and fk in txt), None)

    gold = load_gold()   # the frozen gold, (bk, ch, gv) parsed from nt_ref
    tot = defaultdict(int)
    print("\n=== dense vs bigram-embed reach over РВБ gold (top-5, chapter-level) ===",
          file=sys.stderr)
    print(f"{'gold':9} {'para':14} {'dense':6}{'bigram':7}{'union':6} fragment", file=sys.stderr)
    for g in gold:
        bk, ch = g["bk"], g["ch"]
        if not bk or ch is None:
            continue
        pr = find_para(g["fragment"])
        if not pr:
            continue
        key = (bk, ch)
        hd = key in P_dense.get(pr, set())
        hb = key in P_bi.get(pr, set())
        hu = hd or hb
        tot["dense"] += hd; tot["bigram"] += hb; tot["union"] += hu; tot["n"] += 1
        b = lambda x: "Y" if x else "·"
        print(f"{bk+' '+str(ch):9} {pr:14} {b(hd):6}{b(hb):7}{b(hu):6} {g['fragment'][:30]}",
              file=sys.stderr)
    n = tot["n"]
    print(f"\nreach over {n} gold notes (top-{TOPK}, chapter-level):"
          f"\n  dense {tot['dense']}/{n} · bigram-embed {tot['bigram']}/{n} · "
          f"UNION {tot['union']}/{n}", file=sys.stderr)

    # ---- KEY PROBES: does the reorder/synonym-tolerant method reach these? ----
    # (label, seg_id-or-paragraph_ref, gold book code, gold chapter)
    PROBES = [
        ("«Новый Иерусалим» (reorder echo)", "ч3.гл5:p1855", "Откр", 21),
        ("«всё тайное становится явным»",    "ч1.гл2:p63",   "Мф",   10),
        ("CONTROL Lazarus «воскресение»",    "ч4.гл4:p2440", "Ин",   11),
        ("CONTROL «царствие твое» (Мф)",     "ч1.гл2:p83",   "Мф",    6),
        ("CONTROL «царствие твое» (Лк)",     "ч1.гл2:p83",   "Лк",   11),
    ]
    # re-read our own output to recover per-(ref) cosines per paragraph
    para_refcos = defaultdict(list)
    with open(out_path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            for h in r["top"]:
                para_refcos[r["paragraph_ref"]].append((h["nt_ref"], h["cos"]))

    print("\n=== KEY PROBES (bigram-embed top-5; is gold book+chapter reached?) ===",
          file=sys.stderr)
    for label, para, gcode, gchap in PROBES:
        hits = [(ref, cos) for ref, cos in para_refcos.get(para, [])
                if bc(ref) == (gcode, gchap)]
        if hits:
            ref, cos = max(hits, key=lambda x: x[1])
            print(f"  HIT  {label:36s} -> {gcode} {gchap}: {ref}  cos={cos:.4f}",
                  file=sys.stderr)
        else:
            allh = para_refcos.get(para, [])
            if allh:
                br, bcos = max(allh, key=lambda x: x[1])
                print(f"  MISS {label:36s} -> {gcode} {gchap} not in top-5 "
                      f"(best in para: {br} cos={bcos:.4f})", file=sys.stderr)
            else:
                print(f"  MISS {label:36s} -> paragraph {para} not found", file=sys.stderr)


if __name__ == "__main__":
    main()
