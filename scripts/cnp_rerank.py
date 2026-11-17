"""Cross-encoder reranker for the CnP->NT Bible-reference pipeline — the "retrieve -> rerank"
pattern the SOTA survey names as the deployable, no-LLM win.

We already RETRIEVE candidates with several methods (Smith-Waterman lexical, dense bi-encoder,
BM25, Tesserae); we have never RERANKED. A cross-encoder scores a (query, candidate) pair with
full cross-attention -> better precision than bi-encoder/lexical scores. It runs on a SHORTLIST
(union of the methods' top candidates, restricted to candidate-bearing segments) so it only
scores a few thousand pairs on CPU, not the whole corpus.

  python scripts/cnp_rerank.py
    -> results/crime_and_punishment/cnp_rerank_top.jsonl
    -> SW vs union vs RERANK gold-reach table (mirrors cnp_baselines.py)
    -> cross-encoder scores for REAL vs NOISE probe pairs + the «Новый Иерусалим» top-5
"""
import json, re, sys, time
from collections import defaultdict
import ngram_baseline as nb
from cnp_eval_commentary import load_gold

HERE = nb.HERE                 # release root, trailing slash
TOPK = 5
SW_FLOOR = 6.0                 # keep a segment if it has an SW candidate with score >= this …
DENSE_FLOOR = 0.78             # … OR its dense top-1 cosine >= this
WCHAR_FLOOR = 55.0             # wchar is a SOURCE only above its floor (matches cnp_wchar wiring)
MAX_PAIRS = 40000
MODEL = "BAAI/bge-reranker-base"
FALLBACK = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
BOOK = {"матфея": "Мф", "марка": "Мк", "луки": "Лк", "иоанна": "Ин"}


# ---------- load the four retrieval shortlists, the segment texts, the verse texts ----------
def load_inputs():
    seg_text = {}
    para_of = {}
    for l in open(HERE + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["level"] == "sentence":               # /s but not /c
            seg_text[r["id"]] = r["text"]
            para_of[r["id"]] = r["paragraph_ref"]

    verse_text = {}
    for l in open(HERE + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8"):
        r = json.loads(l)
        verse_text[r["ref"]] = r["text"]

    # SW: results/crime_and_punishment/cnp_match_features.jsonl, sentence-level seg_ids only, best score per (seg, ref)
    sw_best = defaultdict(dict)                     # seg -> {ref: score}
    for l in open(HERE + "results/crime_and_punishment/cnp_match_features.jsonl", encoding="utf-8"):
        r = json.loads(l)
        sid = r["seg_id"]
        if "/s" in sid and "/c" not in sid:
            d = sw_best[sid]
            d[r["nt_ref"]] = max(d.get(r["nt_ref"], 0.0), r["score"])

    # Dense: results/crime_and_punishment/cnp_dense_top5.jsonl (top[].nt_ref + cos)
    dense = {}                                      # seg -> [(ref, cos), ...]
    for l in open(HERE + "results/crime_and_punishment/cnp_dense_top5.jsonl", encoding="utf-8"):
        r = json.loads(l)
        dense[r["seg_id"]] = [(h["nt_ref"], h["cos"]) for h in r["top"]]

    # BM25 / Tesserae: top[] = bare refs (bm25 is now {nt_ref, score} dicts → extract the ref)
    def load_bare(path):
        d = {}
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            d[r["seg_id"]] = [t["nt_ref"] if isinstance(t, dict) else t for t in r["top"]]
        return d
    bm25 = load_bare(HERE + "results/crime_and_punishment/cnp_bm25_top5.jsonl")
    tess = load_bare(HERE + "results/crime_and_punishment/cnp_tesserae_top5.jsonl")

    # the CnP-weighted SOURCES added since this reranker was first wired — without them the
    # cross-encoder never scores the cases they uniquely retrieve (e.g. c102 «Новый Иерусалим»,
    # excluded by the SW/dense gate). Restrict to sentence-level seg_ids (rerank is sentence-only).
    def sent(sid):
        return "/s" in sid and "/c" not in sid

    nt_surf = defaultdict(set)                       # nt_surface: row-per-(seg,ref), floor already applied
    for l in open(HERE + "results/crime_and_punishment/cnp_nt_surface.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if sent(r["seg_id"]):
            nt_surf[r["seg_id"]].add(r["nt_ref"])
    wchar = defaultdict(set)                          # wchar: top[]; SOURCE only at score >= WCHAR_FLOOR
    for l in open(HERE + "results/crime_and_punishment/cnp_wchar_top5.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if sent(r["seg_id"]):
            wchar[r["seg_id"]].update(t["nt_ref"] for t in r["top"] if t["score"] >= WCHAR_FLOOR)
    wpair = defaultdict(set)                          # wpair: per-segment top-1 (ties)
    for l in open(HERE + "results/crime_and_punishment/cnp_wpair_top5.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if sent(r["seg_id"]):
            wpair[r["seg_id"]].update(t["nt_ref"] for t in r["top"])

    return seg_text, para_of, verse_text, sw_best, dense, bm25, tess, nt_surf, wchar, wpair


# ---------- build per-segment shortlist (union) restricted to candidate-bearing segments ----------
def build_shortlists(seg_text, sw_best, dense, bm25, tess, sw_floor):
    # SW/dense-gated shortlist (bm25/tess enrich). This is the part the MAX_PAIRS budget trims via
    # sw_floor; returns seg -> set(refs). The floored CnP-weighted sources are overlaid afterwards
    # (overlay_sources), OUTSIDE the budget, so raising sw_floor can always reach the cap.
    segs = set(seg_text) | set(sw_best) | set(dense)
    shortlist = {}
    for sid in segs:
        if sid not in seg_text:
            continue
        sw = sw_best.get(sid, {})
        dn = dense.get(sid, [])
        sw_ok = any(s >= sw_floor for s in sw.values())
        dense_ok = bool(dn) and dn[0][1] >= DENSE_FLOOR        # dense top-k is cos-ranked; dn[0]=max cos
        if not (sw_ok or dense_ok):
            continue
        cands = set(sw) | {r for r, _ in dn} | set(bm25.get(sid, [])) | set(tess.get(sid, []))
        if cands:
            shortlist[sid] = cands
    return shortlist


def overlay_sources(shortlist, seg_text, nt_surf, wchar, wpair):
    """Always-include the floored CnP-weighted SOURCE candidates (nt_surface, wchar≥floor, wpair),
    so the cross-encoder covers the cases they uniquely retrieve (e.g. c102 «Новый Иерусалим», which
    the SW/dense gate drops). Mutates `shortlist` (seg -> set) and returns it sorted (seg -> list)."""
    for src in (nt_surf, wchar, wpair):
        for sid, refs in src.items():
            if sid in seg_text and refs:
                shortlist.setdefault(sid, set()).update(refs)
    return {sid: sorted(c) for sid, c in shortlist.items()}


# ---------- gold reach (copied from cnp_baselines.py) ----------
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


def main():
    seg_text, para_of, verse_text, sw_best, dense, bm25, tess, nt_surf, wchar, wpair = load_inputs()

    # the source overlay is always-included (≈11k pairs); trim only the SW/dense part to the
    # remaining budget so the loop always terminates.
    n_src = len({(sid, r) for src in (nt_surf, wchar, wpair) for sid, rs in src.items()
                 if sid in seg_text for r in rs})
    sw_budget = max(MAX_PAIRS - n_src, 10000)
    sw_floor = SW_FLOOR
    core = build_shortlists(seg_text, sw_best, dense, bm25, tess, sw_floor)
    while sum(len(v) for v in core.values()) > sw_budget:
        sw_floor += 1.0
        core = build_shortlists(seg_text, sw_best, dense, bm25, tess, sw_floor)
        print(f"  SW/dense pairs>{sw_budget}; raised SW threshold to {sw_floor}", file=sys.stderr)
    shortlist = overlay_sources(core, seg_text, nt_surf, wchar, wpair)
    n_pairs = sum(len(v) for v in shortlist.values())
    print(f"shortlist: {len(shortlist)} candidate-bearing segments, {n_pairs} (segment,verse) "
          f"pairs  [SW>={sw_floor} OR dense cos>={DENSE_FLOOR}; +{n_src} from "
          f"nt_surface/wchar(>={WCHAR_FLOOR})/wpair]", file=sys.stderr)

    # ---------- cross-encoder ----------
    from sentence_transformers import CrossEncoder
    model_name = MODEL
    t = time.time()
    try:
        ce = CrossEncoder(MODEL, max_length=512)
    except Exception as e:
        print(f"  {MODEL} failed ({e}); falling back to {FALLBACK}", file=sys.stderr)
        model_name = FALLBACK
        ce = CrossEncoder(FALLBACK, max_length=512)
    print(f"loaded {model_name} in {time.time()-t:.1f}s", file=sys.stderr)

    pairs, index = [], []                          # flat list + (seg, ref) back-pointers
    for sid, refs in shortlist.items():
        for ref in refs:
            vt = verse_text.get(ref)
            if vt is None:
                continue
            pairs.append((seg_text[sid], vt))
            index.append((sid, ref))
    t = time.time()
    scores = ce.predict(pairs, batch_size=32, show_progress_bar=True)
    rt = time.time() - t
    print(f"scored {len(pairs)} pairs in {rt:.1f}s ({len(pairs)/max(rt,1e-9):.0f} pairs/s)",
          file=sys.stderr)

    ce_score = {}                                  # (seg, ref) -> score
    for (sid, ref), s in zip(index, scores):
        ce_score[(sid, ref)] = float(s)

    # ---------- write reranked top-5 ----------
    reranked = {}                                  # seg -> [(ref, score), ...] full sorted
    with open(HERE + "results/crime_and_punishment/cnp_rerank_top.jsonl", "w", encoding="utf-8") as f:
        for sid in sorted(shortlist):
            ranked = sorted(((r, ce_score[(sid, r)]) for r in shortlist[sid]
                             if (sid, r) in ce_score), key=lambda x: -x[1])
            reranked[sid] = ranked
            f.write(json.dumps({
                "seg_id": sid,
                "paragraph_ref": para_of[sid],
                "seg_text": seg_text[sid],
                "reranked": [{"nt_ref": r, "ce_score": round(s, 4),
                              "nt_text": verse_text[r]} for r, s in ranked[:TOPK]],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {HERE}results/crime_and_punishment/cnp_rerank_top.jsonl", file=sys.stderr)

    # ---------- reach: SW vs union vs RERANK, chapter-level top-5 at the note paragraph ----------
    # per-paragraph (book,chapter) sets for each method
    P_sw, P_union, P_rerank = defaultdict(set), defaultdict(set), defaultdict(set)
    for sid in shortlist:
        p = para_of[sid]
        # SW top-5 for this segment
        for r, _ in sorted(sw_best.get(sid, {}).items(), key=lambda x: -x[1])[:TOPK]:
            P_sw[p].add(bc(r))
        # union = the whole shortlist (every retrieved candidate)
        for r in shortlist[sid]:
            P_union[p].add(bc(r))
        # rerank = cross-encoder top-5
        for r, _ in reranked[sid][:TOPK]:
            P_rerank[p].add(bc(r))

    paras = [(p["ref"], norm(p["text"])) for p in
             (json.loads(l) for l in open(HERE + "data/crime_and_punishment/crime_and_punishment.jsonl", encoding="utf-8"))]

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
    print(f"\n{'gold':9} {'para':14} {'SW':4}{'union':6}{'rerank':7} fragment")
    for g in gold:
        bk, ch = g["bk"], g["ch"]
        if not bk or ch is None:
            continue
        pr = find_para(g["fragment"])
        if not pr:
            continue
        key = (bk, ch)
        h = {"SW": key in P_sw.get(pr, set()),
             "union": key in P_union.get(pr, set()),
             "rerank": key in P_rerank.get(pr, set())}
        for m, v in h.items():
            tot[m] += v
        tot["n"] += 1
        b = lambda x: "Y" if x else "·"
        print(f"{bk+' '+str(ch):9} {pr:14} {b(h['SW']):4}{b(h['union']):6}{b(h['rerank']):7} "
              f"{g['fragment'][:30]}")
    n = tot["n"]
    print(f"\nreach over {n} gold notes (top-{TOPK}, chapter-level): "
          f"SW {tot['SW']}/{n} · union {tot['union']}/{n} · rerank {tot['rerank']}/{n}")

    # ---------- probe pairs: REAL vs NOISE precision separation ----------
    def sc(sid, ref):
        return ce_score.get((sid, ref))

    def show(label, sid, ref):
        s = sc(sid, ref)
        inlist = ref in shortlist.get(sid, [])
        tag = f"{s:+.4f}" if s is not None else "(not scored — not in shortlist)"
        print(f"  {label:8} {sid:20} -> {ref:12} {tag}"
              + ("" if inlist else "  [absent from shortlist]"))

    print("\n--- probe pairs (cross-encoder score; higher = more relevant) ---")
    print(" REAL:")
    show("Lazarus", "ч4.гл4:p2440/s3", "Ин 11:25")
    show("Lord'sPr", "ч1.гл2:p83/s30", "Мф 6:10")          # «да приидет царствие твое»
    show("Лк7:47", "ч1.гл2:p83/s30", "Лк 7:47")
    # the Лк 7:47 note fragment is «Прощаются же и теперь грехи твои мнози» — also probe that sentence
    for sid in sorted(s for s in shortlist if s.startswith("ч1.гл2:p83/")):
        if sc(sid, "Лк 7:47") is not None:
            show("Лк7:47*", sid, "Лк 7:47")

    print(" NOISE:")
    # «пристально посмотрел» -> Деян 3:4, anywhere it appears in a shortlist
    found_403 = False
    for sid, refs in shortlist.items():
        if "Деян 3:4" in refs and "пристальн" in seg_text[sid].lower():
            show("Деян3:4", sid, "Деян 3:4")
            found_403 = True
            break
    if not found_403:
        # fall back: any Деян 3:4 shortlist pair
        for sid, refs in shortlist.items():
            if "Деян 3:4" in refs:
                show("Деян3:4", sid, "Деян 3:4")
                found_403 = True
                break
        if not found_403:
            print("  Деян3:4  (no «пристально посмотрел»->Деян 3:4 pair in any shortlist)")
    # Откр 22:21 repeated-word pairs (sample up to 3)
    shown = 0
    for sid, refs in shortlist.items():
        if "Откр 22:21" in refs:
            show("Откр22:21", sid, "Откр 22:21")
            shown += 1
            if shown >= 3:
                break
    if not shown:
        print("  Откр22:21 (no Откр 22:21 pair in any shortlist)")

    # ---------- KEY TEST: «Новый Иерусалим» -> does rerank surface Откр 21:2? ----------
    print("\n--- KEY TEST: «Новый Иерусалим» ч3.гл5:p1855/s1 ---")
    sid = "ч3.гл5:p1855/s1"
    sl = shortlist.get(sid)
    if sl is None:
        print(f"  {sid} is NOT candidate-bearing (dropped by the SW/dense gate); not reranked.")
        # diagnose why
        sw = sw_best.get(sid, {})
        dn = dense.get(sid, [])
        print(f"  SW best={max(sw.values()) if sw else None}; "
              f"dense top-1 cos={dn[0][1] if dn else None}")
    else:
        print(f"  seg_text: {seg_text[sid]}")
        print(f"  Откр 21:2 in shortlist? {'YES' if 'Откр 21:2' in sl else 'NO'}  "
              f"(shortlist size {len(sl)})")
        print("  reranked top-5:")
        for r, s in reranked[sid][:TOPK]:
            print(f"    {s:+.4f}  {r:12} {verse_text.get(r,'')[:60]}")
        if "Откр 21:2" in sl:
            rank = [r for r, _ in reranked[sid]].index("Откр 21:2") + 1
            print(f"  Откр 21:2 reranked at position {rank}/{len(sl)} "
                  f"(score {ce_score[(sid,'Откр 21:2')]:+.4f})")


if __name__ == "__main__":
    main()
