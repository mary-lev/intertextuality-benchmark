"""RRF fusion of the lexical (Smith-Waterman alignment) and dense (line-to-line) retrievers,
per CnP sentence segment. Tests whether combining catches gold that neither alone surfaces —
the hybrid question from EXPERIMENT.md §4 (there, RRF lifted R@50 0.49->0.64).

Reciprocal Rank Fusion: for a verse v under method m with rank r_m(v),
    rrf(v) = sum_m 1 / (K + r_m(v)),  K=60.
Inputs (per sentence segment):
  * lexical : results/crime_and_punishment/cnp_match_features.jsonl  (top-3 verses by SW alignment score)
  * dense   : results/crime_and_punishment/cnp_dense_top5.jsonl       (top-5 verses by cosine)

    python scripts/cnp_fusion.py   # -> results/crime_and_punishment/cnp_fusion_top.jsonl + reach comparison vs gold

Evaluates reach against the РВБ commentary gold: for each biblical note, does the method surface
the gold book+chapter (top-k) at the note's paragraph? Compares lexical / dense / fused.
"""
import json, re, sys
from collections import defaultdict
import ngram_baseline as nb
from cnp_eval_commentary import load_gold

K = 60
TOPK = 5

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


def rank_lists():
    lex = defaultdict(list)
    for l in open(nb.HERE + "results/crime_and_punishment/cnp_match_features.jsonl", encoding="utf-8"):
        r = json.loads(l)
        lex[r["seg_id"]].append((r["nt_ref"], r["score"]))
    for s in lex:
        lex[s] = [v for v, _ in sorted(lex[s], key=lambda x: -x[1])]
    dense = {}
    for l in open(nb.HERE + "results/crime_and_punishment/cnp_dense_top5.jsonl", encoding="utf-8"):
        r = json.loads(l)
        dense[r["seg_id"]] = [h["nt_ref"] for h in r["top"]]
    return lex, dense


def fuse(lex_list, dense_list):
    score = defaultdict(float)
    for r, v in enumerate(lex_list, 1):
        score[v] += 1.0 / (K + r)
    for r, v in enumerate(dense_list, 1):
        score[v] += 1.0 / (K + r)
    return sorted(score, key=lambda v: -score[v])


def bc(ref):
    m = re.match(r"(.+?) (\d+):", ref)
    return (m.group(1), int(m.group(2))) if m else (ref, 0)


def main():
    lex, dense = rank_lists()
    segs = sorted(set(lex) | set(dense))
    # only sentence-level segments are in dense; fuse on those (line-to-line unit)
    fused = {}
    with open(nb.HERE + "results/crime_and_punishment/cnp_fusion_top.jsonl", "w", encoding="utf-8") as f:
        for s in segs:
            ll, dl = lex.get(s, []), dense.get(s, [])
            fu = fuse(ll, dl)[:TOPK]
            fused[s] = fu
            f.write(json.dumps({"seg_id": s, "paragraph_ref": s.split("/")[0],
                                "fused": fu, "lex": ll[:TOPK], "dense": dl[:TOPK]},
                               ensure_ascii=False) + "\n")

    # ---- reach comparison vs the commentary gold ----
    paras = [(p["ref"], norm(p["text"])) for p in
             (json.loads(l) for l in open(nb.HERE + "data/crime_and_punishment/crime_and_punishment.jsonl", encoding="utf-8"))]

    def find_para(frag):
        for key in sorted([s.strip() for s in re.split(r"~|\.\.\.|…", norm(frag))], key=len, reverse=True):
            if len(key) >= 6:
                for ref, txt in paras:
                    if key[:30] in txt:
                        return ref
        fk = norm(frag)[:18]
        return next((ref for ref, txt in paras if fk and fk in txt), None)

    # index method candidates by paragraph -> set of (book,chapter) in top-k across its sentences
    def by_para(method):
        d = defaultdict(set)
        for s, verses in method.items():
            d[s.split("/")[0]].update(bc(v) for v in verses[:TOPK])
        return d
    P_lex = by_para({s: lex.get(s, []) for s in segs})
    P_dense = by_para({s: dense.get(s, []) for s in segs})
    P_fuse = by_para(fused)

    gold = load_gold()   # the frozen gold, (bk, ch, gv) parsed from nt_ref
    rows, tally = [], defaultdict(lambda: [0, 0, 0])
    for g in gold:
        bk, ch = g["bk"], g["ch"]
        if not bk or ch is None:
            continue
        pr = find_para(g["fragment"])
        if not pr:
            continue
        key = (bk, ch)
        hit_lex = key in P_lex.get(pr, set())
        hit_dense = key in P_dense.get(pr, set())
        hit_fuse = key in P_fuse.get(pr, set())
        hit_union = hit_lex or hit_dense
        rows.append((f"{bk} {ch}", pr, hit_lex, hit_dense, hit_fuse, hit_union, g["fragment"][:34]))
        tally["all"][0] += hit_lex; tally["all"][1] += hit_dense; tally["all"][2] += hit_fuse
        tally["union"][0] += hit_union

    print(f"RRF fusion (K={K}, top-{TOPK}) -> {nb.HERE}results/crime_and_punishment/cnp_fusion_top.jsonl\n", file=sys.stderr)
    print(f"{'gold':10} {'para':14} {'lex':4} {'dense':5} {'rrf':4} {'union':5}  fragment")
    for ref, pr, hl, hd, hf, hu, frag in rows:
        b = lambda x: "Y" if x else "·"
        print(f"{ref:10} {pr:14} {b(hl):4} {b(hd):5} {b(hf):4} {b(hu):5}  {frag}")
    n = len(rows)
    s, u = tally["all"], tally["union"][0]
    print(f"\nreach over {n} chapter-resolvable gold notes (top-{TOPK}, chapter-level — incl. some "
          f"same-chapter noise):\n  lexical {s[0]}/{n} · dense {s[1]}/{n} · RRF-fused {s[2]}/{n} · "
          f"UNION(lex∪dense) {u}/{n}\n"
          f"  → RRF averaging dilutes the weak dense thematic signal; the UNION is the right hybrid "
          f"(dense adds the no-shared-word notes lexical misses).")


if __name__ == "__main__":
    main()
