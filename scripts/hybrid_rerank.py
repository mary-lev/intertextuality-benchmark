"""Phase 4c — end-to-end: hybrid (RRF) shortlist → Claude reranker.

Same reranker as passim_rerank, but the shortlist comes from the higher-recall hybrid front-end
(hybrid_shortlist.csv) instead of Passim alone. Measures whether a better shortlist + the same
Claude rerank beats Passim→rerank.

    python scripts/hybrid_rerank.py [MODEL]      # default claude-opus-4-8

Needs: anthropic + ANTHROPIC_API_KEY, pymystem3, lemma cache, hybrid_shortlist.csv.
"""
from __future__ import annotations
import sys, csv, json, random, re
from collections import defaultdict
import ngram_baseline as nb
from rank_metrics import recall_at_k, mrr as rm_mrr
from tesserae_baseline import content_lemmas
from passim_ranks import load_lines, load_meta
from passim_rerank import load_env_key, window, PROMPT

csv.field_size_limit(10**8)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-4-8"
SHORTLIST = sys.argv[2] if len(sys.argv) > 2 else "hybrid_shortlist.csv"


def main():
    import anthropic
    from pymystem3 import Mystem
    client = anthropic.Anthropic(api_key=load_env_key())
    print("loading…", file=sys.stderr)
    poems, _ = load_lines()
    meta = load_meta()
    flat = {pid: " ".join(w for l in lem for w in l) for pid, (lem, _) in poems.items()}
    dups = nb.dup_groups(flat)
    gold = {g["conn_id"]: g for g in nb.load_gold(dups)}
    mys = Mystem()

    frag_text = {}
    for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/gold.csv", encoding="utf-8")):
        f = (r["analyst_commented_span"] or r["commented_fragment"]).strip()   # analyst-first, as in load_gold
        if f:
            frag_text[r["conn_id"]] = f

    shortlists = {}
    for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/" + SHORTLIST, encoding="utf-8")):
        shortlists[r["conn_id"]] = [p for p in r["shortlist"].split(";") if p]

    pre_ranks, post_ranks, rows, score_rows = [], [], [], []
    tok_in = tok_out = 0                              # exact API usage accounting
    for gi, (cid, cands) in enumerate(shortlists.items()):
        g = gold.get(cid)
        if not g or not cands:
            continue
        pre = next((i for i, pid in enumerate(cands, 1) if pid in g["hit"]), None)
        qset = set(content_lemmas("".join(mys.lemmatize(g["q_frag"]))))
        # shuffle presentation order (position-bias mitigation); shuf[shown_idx] = original_idx.
        # Seeded with the conn_id STRING (stable across processes; str hash() is salted).
        shuf = list(range(len(cands)))
        random.Random("rerank:" + str(cid)).shuffle(shuf)
        blocks = []
        for shown_idx, orig_idx in enumerate(shuf):
            pid = cands[orig_idx]
            lem_lines, surf_lines = poems.get(pid, ([], []))
            bi, bov = 0, -1
            for li, ll in enumerate(lem_lines):
                ov = len(qset & set(ll))
                if ov > bov:
                    bov, bi = ov, li
            ctx = " / ".join(window(surf_lines, bi)) if surf_lines else ""
            blocks.append(f'{shown_idx}. [{meta.get(pid, pid)}] {ctx}')
        prompt = PROMPT.format(frag=frag_text.get(cid, g["q_frag"]), k=len(cands),
                               cands="\n".join(blocks))
        try:
            resp = client.messages.create(model=MODEL, max_tokens=1024,
                                          messages=[{"role": "user", "content": prompt}])
            tok_in += resp.usage.input_tokens; tok_out += resp.usage.output_tokens
            arr = json.loads(re.search(r"\[.*\]", resp.content[0].text, re.S).group(0))
            cscore_shown = {int(o["id"]): float(o["score"]) for o in arr}
        except Exception as e:
            print(f"  conn {cid}: rerank failed ({e})", file=sys.stderr)
            with open(nb.HERE + "results/russian_poetry/hybrid_rerank_failures.jsonl", "a", encoding="utf-8") as ff:
                ff.write(json.dumps({"conn_id": cid, "error": str(e)}, ensure_ascii=False) + "\n")
            cscore_shown = {shown_idx: (len(cands) - shown_idx) for shown_idx in range(len(cands))}
        # map shown-order scores back to original shortlist indices
        cscore = {shuf[shown_idx]: sc for shown_idx, sc in cscore_shown.items()}
        order = sorted(range(len(cands)), key=lambda idx: -cscore.get(idx, -1))
        reranked = [cands[idx] for idx in order]
        post = next((i for i, pid in enumerate(reranked, 1) if pid in g["hit"]), None)
        for idx, pid in enumerate(cands):                       # per-candidate scores for PR/F1
            score_rows.append({"conn_id": cid, "cand_id": pid,
                               "claude_score": cscore.get(idx, ""),
                               "is_gold": int(pid in g["hit"])})
        pre_ranks.append(pre); post_ranks.append(post)
        rows.append({"conn_id": cid, "band": g["band"], "gold_in_shortlist": int(pre is not None),
                     "fused_rank": pre or "", "rerank_rank": post or ""})
        if (gi + 1) % 10 == 0:
            print(f"  {gi+1}/{len(shortlists)}", file=sys.stderr)

    rk = recall_at_k
    mrr = rm_mrr
    K = max((len(c) for c in shortlists.values()), default=10)
    print(f"\n### hybrid shortlist → {MODEL}  (n={len(pre_ranks)}, fused top-{K})")
    print(f"  {'':16s} R@1    R@5    R@{K:<3d} MRR")
    print(f"  Fused (pre)      {rk(pre_ranks,1):.2f}   {rk(pre_ranks,5):.2f}   {rk(pre_ranks,K):.2f}   {mrr(pre_ranks):.3f}")
    print(f"  Claude rerank    {rk(post_ranks,1):.2f}   {rk(post_ranks,5):.2f}   {rk(post_ranks,K):.2f}   {mrr(post_ranks):.3f}")
    both = [(a, b) for a, b in zip(pre_ranks, post_ranks) if a is not None]
    if both:
        pin = [a for a, _ in both]; qin = [b for _, b in both]
        print(f"\n  among {len(both)} queries with gold in fused top-{K}:")
        print(f"  Fused    R@1 {rk(pin,1):.2f}  MRR {mrr(pin):.3f}")
        print(f"  rerank   R@1 {rk(qin,1):.2f}  MRR {mrr(qin):.3f}")

    with open(nb.HERE + "results/russian_poetry/hybrid_rerank_results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conn_id", "band", "gold_in_shortlist", "fused_rank", "rerank_rank"])
        w.writeheader(); w.writerows(rows)
    with open(nb.HERE + "results/russian_poetry/rerank_scores.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conn_id", "cand_id", "claude_score", "is_gold"])
        w.writeheader(); w.writerows(score_rows)
    print(f"wrote hybrid_rerank_results.csv + rerank_scores.csv ({len(score_rows)} rows)", file=sys.stderr)
    cost = tok_in / 1e6 * 5 + tok_out / 1e6 * 25       # claude-opus-4-8 rate ($5/M in, $25/M out)
    print(f"API usage: {tok_in:,} input + {tok_out:,} output tokens over {len(pre_ranks)} calls "
          f"≈ ${cost:.2f} (@ $5/M in, $25/M out)", file=sys.stderr)


if __name__ == "__main__":
    main()
