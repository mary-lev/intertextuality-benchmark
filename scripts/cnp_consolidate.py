"""Consolidate every CnP->Bible method output (both reference texts) into one
candidate store for the apparatus view.

  results/crime_and_punishment/cnp_connections.jsonl  — one row per (seg_id, ref, corpus) candidate
  results/crime_and_punishment/cnp_connections.csv    — flat, sortable export
  results/crime_and_punishment/cnp_gold_overlay.json  — the 16 commentary notes: detected vs RECALL GAP

    .venv/bin/python scripts/cnp_consolidate.py
"""
import json
import os
import re
from collections import defaultdict

import ngram_baseline as nb
import cnp_eval_commentary as ev

H = nb.HERE  # release root, trailing slash


def sent(seg_id):
    return re.sub(r"/c\d+$", "", seg_id)


def load_segments():
    seg = {}
    for l in open(H + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["level"] == "sentence":
            seg[r["id"]] = {"paragraph_ref": r["paragraph_ref"], "text": r["text"]}
    return seg


def load_verses(path):
    return {v["ref"]: v["text"]
            for v in (json.loads(l) for l in open(path, encoding="utf-8"))}


def run_members(ref):
    m = re.match(r"(.+? \d+):(\d+)(?:[-–—](\d+))?$", ref)
    if not m:
        return [ref]
    base, lo, hi = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
    return [f"{base}:{v}" for v in range(lo, hi + 1)]


def gold_notes():
    """The frozen structured eval gold (data/crime_and_punishment/gold.jsonl) — already resolved to
    (paragraph_ref, nt_ref, bible_text), with (bk, ch, gv) parsed. No runtime re-resolution
    from the raw commentary (find_para / book_sigla / vrange)."""
    notes = ev.load_gold()
    for n in notes:
        n["pr"] = n["paragraph_ref"]
    return notes


STOP = {"который", "которая", "которое", "которые", "этот", "тогда", "теперь",
        "очень", "когда", "чтобы", "потому", "также", "своей", "своих"}
_WORD = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


def _snum(seg_id):
    """Sentence index for ordering a paragraph's segments (ч4.гл4:p2438/s7 -> 7)."""
    m = re.search(r"/s(\d+)", seg_id)
    return int(m.group(1)) if m else 0


def _norm_anchor(s):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zа-я ]+", " ",
                  (s or "").lower().replace("ё", "е"))).strip()


def _anchor_chunks(fragment):
    """The commentator's fragment elides the middle with ~ / … — split into the
    quoted chunks so we can locate which sentence(s) of the paragraph it points at."""
    parts = re.split(r"\s*(?:~|\.{2,}|…)\s*", fragment or "")
    return [c for c in (_norm_anchor(p) for p in parts) if len(c) >= 8]


def shared_words(seg_text, verse_text):
    a = {w.lower() for w in _WORD.findall(seg_text) if len(w) >= 4}
    b = {w.lower() for w in _WORD.findall(verse_text) if len(w) >= 4}
    common = [w for w in (a & b) if w not in STOP]
    return sorted(set(common), key=len, reverse=True)[:12]


# the six rank-only NT retrievers: (method_name, file, field)
_RANK_FILES = [("bm25", "results/crime_and_punishment/cnp_bm25_top5.jsonl", "top"),
               ("tesserae", "results/crime_and_punishment/cnp_tesserae_top5.jsonl", "top"),
               ("char", "results/crime_and_punishment/cnp_char_top5.jsonl", "top"),
               ("overlap", "results/crime_and_punishment/cnp_overlap_top5.jsonl", "top"),
               ("synonym", "results/crime_and_punishment/cnp_synonym_top5.jsonl", "top"),
               ("fusion", "results/crime_and_punishment/cnp_fusion_top.jsonl", "fused"),
               ("wpair", "results/crime_and_punishment/cnp_wpair_top5.jsonl", "top")]   # distinctive word-pair enricher (reordered allusions)

# wchar = CnP-rarity-weighted char (the de-noised `char`): score = summed CnP-idf of shared 4-grams
# (NT-idf removed — measured to hurt). Score is meaningful, so it SEEDS candidates above a floor (a
# precision SOURCE), not enrich-only. Floor set for COMPLETENESS — below the gold reordered cases
# (c102 «Новый Иерусалим» 56.4, c131 «царствие божие» 110.2) and above the interjection noise (~36),
# so wchar is self-complete on the verbal gold. Residual mid-range noise lands in the weak band.
# See wiki .../retrieval-sweep/char-method-review.md (Resolution).
WCHAR_FLOOR = 55.0


def collect_nt(seg_text, nt_text):
    feats = {}          # (seg, ref) -> feature dict (keep max-score row)
    for l in open(H + "results/crime_and_punishment/cnp_match_features.jsonl", encoding="utf-8"):
        r = json.loads(l)
        s = sent(r["seg_id"])
        if s not in seg_text:
            continue
        k = (s, r["nt_ref"])
        if r["score"] > feats.get(k, {}).get("score", -1):
            feats[k] = {"score": r["score"],
                        "longest_lemma_ngram": r["longest_shared_lemma_ngram"],
                        "surface_sim": r["aligned_span_surface_similarity"],
                        "sum_idf": r["sum_idf_shared_content_lemmas"],
                        "max_idf": r["max_idf_shared_lemma"]}
    banded = {}         # (seg, ref) -> {band, score, text}
    for l in open(H + "results/crime_and_punishment/cnp_bible_refs.jsonl", encoding="utf-8"):
        r = json.loads(l)
        banded[(sent(r["seg_id"]), r["nt_ref"])] = {
            "band": r["band"], "score": r["score"], "text": r.get("nt_text")}
    tier = {}
    for l in open(H + "results/crime_and_punishment/cnp_bible_tiered.jsonl", encoding="utf-8"):
        r = json.loads(l)
        tier[(sent(r["seg_id"]), r["nt_ref"])] = r["tier"]
    ce = defaultdict(dict)      # seg -> {single_ref: ce}
    for l in open(H + "results/crime_and_punishment/cnp_rerank_top.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for h in r["reranked"]:
            ce[sent(r["seg_id"])][h["nt_ref"]] = h["ce_score"]
    cos = defaultdict(dict)
    cosc = defaultdict(dict)                 # dense centered score (cos_c) — the thresholdable signal
    for l in open(H + "results/crime_and_punishment/cnp_dense_top5.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for h in r["top"]:
            cos[sent(r["seg_id"])][h["nt_ref"]] = h.get("cos")
            cosc[sent(r["seg_id"])][h["nt_ref"]] = h.get("cos_c")
    rankhits = {}               # method -> {seg: set(single_refs)}
    for name, fn, field in _RANK_FILES:
        d = defaultdict(set)
        for l in open(H + fn, encoding="utf-8"):
            r = json.loads(l)
            for x in r.get(field, []):
                d[sent(r["seg_id"])].add(x if isinstance(x, str) else x.get("nt_ref"))
        rankhits[name] = d
    nts = defaultdict(dict)   # seg -> {single_ref: (score, shared)}  — the nt_surface source
    nts_path = H + "results/crime_and_punishment/cnp_nt_surface.jsonl"
    if os.path.exists(nts_path):
        for l in open(nts_path, encoding="utf-8"):
            r = json.loads(l)
            nts[sent(r["seg_id"])][r["nt_ref"]] = (r["score"], r.get("shared"))

    wch = defaultdict(dict)   # seg -> {single_ref: score} — the wchar (CnP-weighted char) source, floored
    wch_path = H + "results/crime_and_punishment/cnp_wchar_top5.jsonl"
    if os.path.exists(wch_path):
        for l in open(wch_path, encoding="utf-8"):
            r = json.loads(l)
            for h in r.get("top", []):
                if h["score"] >= WCHAR_FLOOR:
                    wch[sent(r["seg_id"])][h["nt_ref"]] = h["score"]

    # Rule A: candidate SOURCES = SW-banded ∪ tier ∪ nt_surface ∪ wchar. The raw SW pool (feats),
    # dense (cos), reranked (ce), and the rank-only retrievers (rankhits) are loaded only to ENRICH
    # these candidates with scores/provenance — never to seed new candidates.
    # This keeps the store to the display-worthy spine, not the ~112k raw seed pool.
    #
    # rerank is an ENRICHER, not a source (2026-06-25): it is recall-neutral by construction — it can
    # only reorder candidates other methods already retrieved, so it must never SEED a card. Seeding
    # from `ce` laundered ~14k enricher-only candidates into the store, including bge-base's
    # short-phrase false positives («Слава богу!»→1Кор 16:23 at ce 1.0). ce still tags+scores the real
    # source candidates below (line `if cescore is not None`), so the apparatus ce slider is unchanged.
    keys = set(banded) | set(tier)
    for seg, refs in nts.items():            # nt_surface is a candidate source
        for ref in refs:
            keys.add((seg, ref))
    for seg, refs in wch.items():            # wchar (floored CnP-weighted char) is a candidate source
        for ref in refs:
            keys.add((seg, ref))

    out = {}
    for (seg, ref) in keys:
        if seg not in seg_text:
            continue
        mem = run_members(ref)
        cescore = max([ce[seg][m] for m in mem if m in ce.get(seg, {})], default=None)
        cosscore = max([cos[seg][m] for m in mem if m in cos.get(seg, {})], default=None)
        coscscore = max([cosc[seg][m] for m in mem if m in cosc.get(seg, {})], default=None)
        ntsscore = max([nts[seg][m][0] for m in mem if m in nts.get(seg, {})], default=None)
        wchscore = max([wch[seg][m] for m in mem if m in wch.get(seg, {})], default=None)
        methods = []
        if (seg, ref) in feats or (seg, ref) in banded:
            methods.append("sw")
        if cescore is not None:
            methods.append("rerank")
        if cosscore is not None:
            methods.append("dense")
        if ntsscore is not None:
            methods.append("nt_surface")
        if wchscore is not None:
            methods.append("wchar")
        for name, d in rankhits.items():
            if any(m in d.get(seg, set()) for m in mem):
                methods.append(name)
        f = feats.get((seg, ref), {})
        b = banded.get((seg, ref))
        # verse_text from the canonical corpus (single verses); the banded SW text is
        # only a fallback for merged run-refs (e.g. "Ин 11:19-22") absent from nt_text.
        # (Banded text is a stale snapshot — it held the old 22KB Откр 22:21.)
        vtext = nt_text.get(ref) or (b or {}).get("text") or ""
        out[(seg, ref)] = {
            "corpus": "NT", "seg_id": seg, "ref": ref, "verse_text": vtext,
            "is_run": len(mem) > 1, "run_members": mem,
            "scores": {"ce": cescore, "sw": (b or f).get("score"),
                       "dense_cos": cosscore, "dense_cos_c": coscscore,
                       "nt_surface_idf": ntsscore, "wchar": wchscore,
                       "longest_lemma_ngram": f.get("longest_lemma_ngram"),
                       "surface_sim": f.get("surface_sim"),
                       "sum_idf": f.get("sum_idf"), "max_idf": f.get("max_idf")},
            "methods": methods, "consensus": len(methods),
            "band": (b or {}).get("band", "weak"), "tier": tier.get((seg, ref)),
            "shared_words": shared_words(seg_text[seg]["text"], vtext)}
    return out


def collect_cs(seg_text, cs_text):
    # Rule A: the surface matcher is the candidate SOURCE; the lemma-SW pool
    # (cnp_match_cs_raw) only ENRICHES surface candidates with a cs_sw score.
    rows = {}
    for l in open(H + "results/crime_and_punishment/cnp_cs_surface.jsonl", encoding="utf-8"):
        r = json.loads(l)
        s = sent(r["seg_id"])
        if s not in seg_text:
            continue
        k = (s, r["cs_ref"])
        cur = rows.setdefault(k, {"cs_sw": None, "cs_surface_idf": None,
                                  "methods": [], "shared": None})
        if r["score"] > (cur["cs_surface_idf"] or -1):
            cur["cs_surface_idf"] = r["score"]
            cur["shared"] = r.get("shared")
        if "cs_surface" not in cur["methods"]:
            cur["methods"].append("cs_surface")
    for l in open(H + "results/crime_and_punishment/cnp_match_cs_raw.jsonl", encoding="utf-8"):
        r = json.loads(l)
        k = (sent(r["seg_id"]), r["nt_ref"])
        if k not in rows:
            continue
        if r["score"] > (rows[k]["cs_sw"] or -1):
            rows[k]["cs_sw"] = r["score"]
        if "cs_sw" not in rows[k]["methods"]:
            rows[k]["methods"].append("cs_sw")

    out = {}
    for (seg, ref), v in rows.items():
        mem = run_members(ref)
        vtext = cs_text.get(ref, "")
        out[(seg, ref)] = {
            "corpus": "CS", "seg_id": seg, "ref": ref, "verse_text": vtext,
            "is_run": len(mem) > 1, "run_members": mem,
            "scores": {"ce": None, "sw": None, "dense_cos": None,
                       "longest_lemma_ngram": None, "surface_sim": None,
                       "sum_idf": None, "max_idf": None,
                       "cs_sw": v["cs_sw"], "cs_surface_idf": v["cs_surface_idf"]},
            "methods": v["methods"], "consensus": len(v["methods"]),
            "band": "weak", "tier": None,
            "shared_words": v["shared"] or shared_words(seg_text[seg]["text"], vtext)}
    return out


import csv as _csv

_CHAP_ORDER = {}  # chapter key -> first appearance order, filled in build_connections


def flag_gold(cands, notes):
    for c in cands.values():
        pi = ev.pidx(c["seg_id"])
        hit = None
        for n in notes:
            if n["bk"] is None or n["pr"] is None or n["type"].startswith("meta"):
                continue
            if abs(ev.pidx(n["pr"]) - pi) > 3:
                continue
            if any(ev.matches(m, n["bk"], n["ch"], n["gv"]) for m in c["run_members"]):
                hit = n
                break
        c["gold"] = ({"is_gold": True, "gold_id": hit["id"], "gold_type": hit["type"]}
                     if hit else {"is_gold": False, "gold_id": None, "gold_type": None})


def build_connections():
    seg = load_segments()
    nt = load_verses(H + "data/crime_and_punishment/nt1823.jsonl")
    cs = load_verses(H + "data/crime_and_punishment/cs_nt.jsonl")
    notes = gold_notes()
    cands = {}
    for k, v in collect_nt(seg, nt).items():
        cands[(k[0], k[1], "NT")] = v
    for k, v in collect_cs(seg, cs).items():
        cands[(k[0], k[1], "CS")] = v
    flag_gold(cands, notes)
    rows = []
    for c in cands.values():
        pr = seg[c["seg_id"]]["paragraph_ref"]
        chap = pr.split(":")[0]
        _CHAP_ORDER.setdefault(chap, ev.pidx(c["seg_id"]))
        c["paragraph_ref"] = pr
        c["chapter"] = chap
        c["seg_text"] = seg[c["seg_id"]]["text"]
        rows.append(c)
    rows.sort(key=lambda r: (_CHAP_ORDER[r["chapter"]], ev.pidx(r["seg_id"]), r["ref"]))
    return rows


def write_outputs(rows):
    with open(H + "results/crime_and_punishment/cnp_connections.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    score_keys = sorted({k for r in rows for k in r["scores"]})
    cols = (["seg_id", "paragraph_ref", "chapter", "corpus", "ref", "band", "tier",
             "consensus", "is_gold", "gold_id"] + score_keys + ["methods"])
    with open(H + "results/crime_and_punishment/cnp_connections.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for r in rows:
            row = [r["seg_id"], r["paragraph_ref"], r["chapter"], r["corpus"], r["ref"],
                   r["band"], r["tier"] or "", r["consensus"],
                   r["gold"]["is_gold"], r["gold"]["gold_id"] or ""]
            row += [r["scores"].get(k, "") for k in score_keys]
            row += ["|".join(r["methods"])]
            w.writerow(row)


# enrich-only retrievers whose reaches rule A keeps OUT of the store; (method, file, field)
_RAW_TOPK = [("dense", "results/crime_and_punishment/cnp_dense_top5.jsonl", "top"), ("bm25", "results/crime_and_punishment/cnp_bm25_top5.jsonl", "top"),
             ("tesserae", "results/crime_and_punishment/cnp_tesserae_top5.jsonl", "top"), ("char", "results/crime_and_punishment/cnp_char_top5.jsonl", "top"),
             ("overlap", "results/crime_and_punishment/cnp_overlap_top5.jsonl", "top"), ("synonym", "results/crime_and_punishment/cnp_synonym_top5.jsonl", "top"),
             ("fusion", "results/crime_and_punishment/cnp_fusion_top.jsonl", "fused"), ("wpair", "results/crime_and_punishment/cnp_wpair_top5.jsonl", "top")]


def _load_raw_topk():
    """Per enrich-only retriever: [(paragraph_idx, [refs])] from its raw top-5 file — so the gold
    overlay can see reaches that rule A keeps out of the store (e.g. dense/char on thematic notes)."""
    raw = {}
    for meth, fn, field in _RAW_TOPK:
        path = H + fn
        if not os.path.exists(path):
            continue
        raw[meth] = [(ev.pidx(r["seg_id"]),
                      [x if isinstance(x, str) else x.get("nt_ref") for x in r.get(field, [])])
                     for r in (json.loads(l) for l in open(path, encoding="utf-8"))]
    return raw


def _content_words(text):
    return {w for w in re.findall(r"[а-я]+", (text or "").lower().replace("ё", "е")) if len(w) >= 4}


def _quote_span(n, para_segs, hit_ids):
    """The novel side of a gold note: the sentence(s) that make up the quotation — which for the
    running Lazarus readings span several paragraphs interleaved with narration. Bound the region by
    the commentator's anchor (fragment start ~ end); inside it keep the reading sentences (anchor /
    detected / sharing a content word with the cited verse) and drop the interjected narration."""
    pr = n["pr"]
    if pr not in para_segs:
        return []
    pi = ev.pidx(pr)
    window = sorted(((ev.pidx(p), _snum(sid), sid, txt)
                     for p, segs in para_segs.items() if abs(ev.pidx(p) - pi) <= 3
                     for sid, txt in segs),
                    key=lambda x: (x[0], x[1]))
    chunks = _anchor_chunks(n.get("fragment"))
    frag_norm = _norm_anchor(n.get("fragment"))

    def is_anchor(txt):
        nt = _norm_anchor(txt)
        return bool(nt) and (any(c in nt for c in chunks)
                             or (len(nt) >= 12 and nt in frag_norm))
    apos = [i for i, (_, _, _, txt) in enumerate(window) if is_anchor(txt)]
    if apos:
        span = window[apos[0]:apos[-1] + 1]
    else:                       # no anchor located — fall back to detected sentences in the own paragraph
        own = {s for s, _ in para_segs[pr]}
        det = [i for i, (_, _, sid, _) in enumerate(window) if sid in own and sid in hit_ids]
        if not det:
            return [{"t": t, "hit": False} for _, t in para_segs[pr]]
        span = window[det[0]:det[-1] + 1]
    vwords = _content_words(n.get("bible_text") or "")
    return [{"t": txt,
             "hit": (sid in hit_ids or is_anchor(txt) or bool(_content_words(txt) & vwords))}
            for (_, _, sid, txt) in span]


def gold_overlay(rows, notes, seg=None):
    raw = _load_raw_topk()
    if seg is None:
        seg = load_segments()
    para_segs = defaultdict(list)            # paragraph_ref -> [(seg_id, text), …] in reading order
    for sid, s in seg.items():
        para_segs[s["paragraph_ref"]].append((sid, s["text"]))
    for pr in para_segs:
        para_segs[pr].sort(key=lambda x: _snum(x[0]))
    out = []
    for n in notes:
        e = {"id": n["id"], "bk": n["bk"], "ch": n["ch"], "type": n["type"], "pr": n["pr"],
             "ref_str": n["nt_ref"] or "—", "fragment": n.get("fragment", ""),
             "gold_text": n["bible_text"],   # frozen from the structured gold (full range, no cap)
             "status": "out-of-scope", "detected_by": [], "methods": []}
        # Check EVERY note with a resolvable verse against both the store (source methods) and the
        # raw retriever top-5s (enrich-only methods) — so we see whether ANY method reaches it,
        # including the thematic/meta notes dense/char can hit. Quotations get detected/gap;
        # thematic/meta keep their out-of-scope *scope* label but carry the methods that reached them.
        hit_ids = set()
        if n["bk"] and n["pr"] is not None and (n["ch"] is not None or n["gv"]):
            pi = ev.pidx(n["pr"])
            hitrows = [r for r in rows
                       if abs(ev.pidx(r["seg_id"]) - pi) <= 3
                       and any(ev.matches(m, n["bk"], n["ch"], n["gv"]) for m in r["run_members"])]
            hit_ids = {r["seg_id"] for r in hitrows}
            store_methods = {m for r in hitrows for m in r["methods"]}
            raw_methods = {meth for meth, lst in raw.items()
                           if any(abs(p - pi) <= 3
                                  and any(ev.matches(r, n["bk"], n["ch"], n["gv"]) for r in refs)
                                  for p, refs in lst)}
            e["detected_by"] = sorted({(r["ref"], r["corpus"]) for r in hitrows})
            e["methods"] = sorted(store_methods | raw_methods)   # store sources ∪ raw retrievers
            if n["type"] in ("quotation", "allusion"):   # in-scope verbal subset
                e["status"] = "detected" if e["methods"] else "gap"
        # Novel side = the quotation span. Running quotes cross several paragraphs interleaved with
        # narration; _quote_span bounds the region by the commentator's anchor and keeps the reading
        # sentences (the apparatus then shows the hit sentences, dropping the narration).
        e["paragraph"] = _quote_span(n, para_segs, hit_ids)
        out.append(e)
    return out


def main():
    rows = build_connections()
    write_outputs(rows)
    notes = gold_notes()
    overlay = gold_overlay(rows, notes)
    with open(H + "results/crime_and_punishment/cnp_gold_overlay.json", "w", encoding="utf-8") as f:
        json.dump(overlay, f, ensure_ascii=False, indent=2)
    import sys
    gaps = sum(1 for e in overlay if e["status"] == "gap")
    print(f"{len(rows)} candidates -> results/crime_and_punishment/cnp_connections.jsonl; "
          f"{len(overlay)} gold notes ({gaps} recall gaps) -> results/crime_and_punishment/cnp_gold_overlay.json",
          file=sys.stderr)


if __name__ == "__main__":
    main()
