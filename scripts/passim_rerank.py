"""Phase 4 — retrieve-and-rerank: Passim top-k → Claude reranker.

The SOTA deployable pattern (Loci Similes 2026): a strong first-stage retriever proposes a
shortlist, a stronger judge reranks it, a human reviews the top-k. Here Passim (our best
retriever) proposes the top-K candidates; Claude scores each candidate passage 0–10 for how
likely it is the *actual source* the fragment draws on; we rerank by that score and compare
Recall@k / MRR / precision@1 before vs after.

Claude is used strictly as a reranker over a bounded shortlist (the survey's recommended,
hallucination-bounded use), never as an autonomous detector. One API call per query (the K
candidates judged together). Scoped to the 105 fragment-bearing connections (S1/S2).

    python scripts/passim_rerank.py [K] [MODEL]      # default K=10, claude-opus-4-8

Needs: anthropic SDK + ANTHROPIC_API_KEY (loaded from .env), pymystem3, the lemma cache.
"""
from __future__ import annotations
import os, sys, csv, json, math, random, re
from collections import defaultdict
import ngram_baseline as nb
from rank_metrics import recall_at_k, mrr as rm_mrr
from tesserae_baseline import content_lemmas, DF_CAP, MIN_SHARED
from passim_baseline import sw
from passim_ranks import load_lines, load_meta

csv.field_size_limit(10**8)
K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
MODEL = sys.argv[2] if len(sys.argv) > 2 else "claude-opus-4-8"


def load_env_key():
    for line in open(nb.HERE + ".env", encoding="utf-8"):
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("ANTHROPIC_API_KEY")


def window(surf_lines, bi, r=1):
    return [l for l in surf_lines[max(0, bi - r):bi + r + 1] if l.strip()]


PROMPT = """You are verifying intertextual connections in Russian poetry. A commentator marked \
the FRAGMENT below (from one poem) as drawing on an earlier source poem — by quotation, \
paraphrase, a shared distinctive image/phrase, or a clear echo.

FRAGMENT:
«{frag}»

Below are {k} candidate source passages retrieved from a large poetry corpus. For EACH, score \
0–10 how likely it is the ACTUAL source this fragment draws on. Judge shared wording, imagery, \
and phrasing — NOT mere shared topic. A high score needs a real textual/imagistic link, not a \
common theme.

CANDIDATES:
{cands}

Return ONLY a JSON array, one object per candidate, no prose:
[{{"id": <number>, "score": <0-10>}}, ...]"""


def main():
    import anthropic
    key = load_env_key()
    if not key:
        sys.exit("no ANTHROPIC_API_KEY in .env")
    client = anthropic.Anthropic(api_key=key)

    from pymystem3 import Mystem
    print("loading lemma corpus…", file=sys.stderr)
    poems, df = load_lines()
    Npoems = len(poems)
    idf = {w: math.log(Npoems / d) for w, d in df.items()}
    seqs = {pid: [w for l in lem for w in l] for pid, (lem, _) in poems.items()}
    flat = {pid: " ".join(s) for pid, s in seqs.items()}
    dups = nb.dup_groups(flat)
    gold = [g for g in nb.load_gold(dups) if g["stratum"] in ("S1", "S2")]
    meta = load_meta()
    inv = defaultdict(set)
    for pid, s in seqs.items():
        for w in set(s):
            inv[w].add(pid)
    m = Mystem()

    # surface fragment text per conn (for the prompt) — from connections_full
    frag_text = {}
    for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/gold.csv", encoding="utf-8")):
        f = (r["analyst_commented_span"] or r["commented_fragment"]).strip()   # analyst-first, as in load_gold
        if f:
            frag_text[r["conn_id"]] = f

    pre_ranks, post_ranks, rows = [], [], []
    for gi, g in enumerate(gold):
        Q = content_lemmas("".join(m.lemmatize(g["q_frag"])))
        qset = set(Q)
        cand = defaultdict(int)
        for w in qset:
            if w in inv and len(inv[w]) <= DF_CAP:
                for pid in inv[w]:
                    cand[pid] += 1
        cands = [pid for pid, c in cand.items() if c >= MIN_SHARED and pid not in g["self"]]
        scored = sorted(((sw(Q, seqs[pid], idf), pid) for pid in cands), key=lambda t: -t[0])[:K]
        if not scored:
            pre_ranks.append(None); post_ranks.append(None); continue

        # Passim rank of gold within this shortlist
        pre = next((i for i, (_, pid) in enumerate(scored, 1) if pid in g["hit"]), None)

        # build candidate blocks (matched-line window + author/title); shuffle presentation
        # order so the model can't learn "the retriever's rank-1 is usually right" (position
        # bias) -- shuf[shown_idx] = original_idx undoes the shuffle when parsing back.
        # Seeded with the conn_id STRING (stable across processes; str hash() is salted).
        shuf = list(range(len(scored)))
        random.Random("rerank:" + str(g["conn_id"])).shuffle(shuf)
        blocks = []
        for shown_idx, orig_idx in enumerate(shuf):
            s, pid = scored[orig_idx]
            lem_lines, surf_lines = poems[pid]
            bi, bov = 0, -1
            for li, ll in enumerate(lem_lines):
                ov = len(qset & set(ll))
                if ov > bov:
                    bov, bi = ov, li
            ctx = " / ".join(window(surf_lines, bi))
            blocks.append(f'{shown_idx}. [{meta.get(pid, pid)}] {ctx}')
        prompt = PROMPT.format(frag=frag_text.get(g["conn_id"], g["q_frag"]), k=len(scored),
                               cands="\n".join(blocks))
        try:
            resp = client.messages.create(model=MODEL, max_tokens=1024,
                                          messages=[{"role": "user", "content": prompt}])
            txt = resp.content[0].text
            arr = json.loads(re.search(r"\[.*\]", txt, re.S).group(0))
            cscore_shown = {int(o["id"]): float(o["score"]) for o in arr}
        except Exception as e:
            print(f"  conn {g['conn_id']}: rerank failed ({e}); keeping Passim order", file=sys.stderr)
            with open(nb.HERE + "results/russian_poetry/rerank_failures.jsonl", "a", encoding="utf-8") as ff:
                ff.write(json.dumps({"conn_id": g["conn_id"], "error": str(e)}, ensure_ascii=False) + "\n")
            cscore_shown = {shown_idx: (len(scored) - shown_idx) for shown_idx in range(len(scored))}

        # map shown-order scores back to original retriever-order indices
        cscore = {shuf[shown_idx]: sc for shown_idx, sc in cscore_shown.items()}
        order = sorted(range(len(scored)), key=lambda idx: -cscore.get(idx, -1))
        reranked = [scored[idx][1] for idx in order]
        post = next((i for i, pid in enumerate(reranked, 1) if pid in g["hit"]), None)

        pre_ranks.append(pre); post_ranks.append(post)
        rows.append({"conn_id": g["conn_id"], "band": g["band"],
                     "gold_in_topK": int(pre is not None), "passim_rank": pre or "",
                     "rerank_rank": post or ""})
        if (gi + 1) % 10 == 0:
            print(f"  {gi+1}/{len(gold)}", file=sys.stderr)

    rk = recall_at_k
    mrr = rm_mrr
    print(f"\n### retrieve-and-rerank — Passim top-{K} → {MODEL}  (n={len(pre_ranks)})")
    print(f"  {'':16s} R@1    R@5    R@10   MRR")
    print(f"  Passim (pre)     {rk(pre_ranks,1):.2f}   {rk(pre_ranks,5):.2f}   {rk(pre_ranks,K):.2f}   {mrr(pre_ranks):.3f}")
    print(f"  Claude rerank    {rk(post_ranks,1):.2f}   {rk(post_ranks,5):.2f}   {rk(post_ranks,K):.2f}   {mrr(post_ranks):.3f}")
    # within-shortlist view (only queries whose gold Passim surfaced in top-K)
    both = [(a, b) for a, b in zip(pre_ranks, post_ranks) if a is not None]
    if both:
        pre_in = [a for a, _ in both]; post_in = [b for _, b in both]
        print(f"\n  among {len(both)} queries with gold in Passim top-{K}:")
        print(f"  Passim   R@1 {rk(pre_in,1):.2f}  MRR {mrr(pre_in):.3f}")
        print(f"  rerank   R@1 {rk(post_in,1):.2f}  MRR {mrr(post_in):.3f}")

    with open(nb.HERE + "results/russian_poetry/rerank_results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conn_id", "band", "gold_in_topK", "passim_rank", "rerank_rank"])
        w.writeheader(); w.writerows(rows)
    print("wrote results/russian_poetry/rerank_results.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
