"""Profile-invariance verification (paper §5, robustness).

Hypothesis: configuration variants shift absolute numbers but not the TYPOLOGY-LEVEL
pattern (the band profile). If confirmed, tuning that cannot move a profile cannot
have manufactured it — the in-sample disclosure's counterweight. Fallback on
disconfirmation: per-configuration reporting, naming the config-sensitive cells.

Three pinned variant arms (not an open-ended sweep):
  A  dense encoder    MiniLM vs E5-base dense leg inside the deployable fusion
                      (paired shortlists on disk; zero recompute)
  B  fusion depth     4-method vs +wpair (5) vs +wpair+lc (6, deployable)
                      (fusion_variants.csv from one hybrid_baseline run)
  C  ngram gram type  word-2gram vs char-4gram exact-surface probe
                      (word2 ranks from factorial_ranks.csv; char4 computed here)

For every variant pair x band cell: paired R@10 (+R@1) with exact McNemar; verdict per
arm = does the verbatim > close > loose ordering hold in every variant, and which
cells (if any) differ beyond noise (p < 0.05).

    python scripts/invariance.py     # from the release root; after hybrid_baseline run

Deterministic; no API calls. Writes results/russian_poetry/invariance.csv + prints the report.
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict

import ngram_baseline as nb
from rank_metrics import recall_at_k, mcnemar_exact

csv.field_size_limit(10 ** 8)
BANDS = ("verbatim", "close", "loose")


def load_col(path, col, key="conn_id"):
    out = {}
    for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/" + path, encoding="utf-8")):
        v = str(r[col]).strip()
        out[r[key]] = int(v) if v.isdigit() else None
    return out


def bands_of(path="factorial_ranks.csv"):
    return {r["conn_id"]: r["band"]
            for r in csv.DictReader(open(nb.HERE + "results/russian_poetry/" + path, encoding="utf-8"))}


def compare(name, va, vb, ranks_a, ranks_b, band, out):
    """Per-band paired comparison of two variants sharing conn_ids."""
    shared = sorted(set(ranks_a) & set(ranks_b))
    prof = {}
    for b in BANDS + ("ALL",):
        cids = [c for c in shared if b == "ALL" or band.get(c) == b]
        if not cids:
            continue
        ra = [ranks_a[c] for c in cids]
        rb = [ranks_b[c] for c in cids]
        row = {"arm": name, "variant_a": va, "variant_b": vb, "cell": b, "n": len(cids)}
        for k in (1, 10):
            mc = mcnemar_exact(ra, rb, k)
            row[f"R@{k}_a"] = round(recall_at_k(ra, k), 3)
            row[f"R@{k}_b"] = round(recall_at_k(rb, k), 3)
            row[f"R@{k}_p"] = mc["p_value"]
        out.append(row)
        prof[b] = (row["R@10_a"], row["R@10_b"])
    ok_a = all(prof[b1][0] >= prof[b2][0] for b1, b2 in zip(BANDS, BANDS[1:]) if b1 in prof and b2 in prof)
    ok_b = all(prof[b1][1] >= prof[b2][1] for b1, b2 in zip(BANDS, BANDS[1:]) if b1 in prof and b2 in prof)
    flips = [r["cell"] for r in out if r["arm"] == name and r["cell"] != "ALL"
             and min(r["R@1_p"], r["R@10_p"]) < 0.05]
    print(f"\n[{name}] {va} vs {vb} — band ordering holds: "
          f"{'YES' if ok_a and ok_b else 'NO'}; cells beyond noise (p<0.05): {flips or 'none'}")
    for r in [r for r in out if r["arm"] == name]:
        print(f"  {r['cell']:9s} n={r['n']:3d}  R@10 {r['R@10_a']:.2f} vs {r['R@10_b']:.2f} "
              f"(p={r['R@10_p']})   R@1 {r['R@1_a']:.2f} vs {r['R@1_b']:.2f} (p={r['R@1_p']})")


def main():
    band = bands_of()
    out = []

    # --- Arm A: dense encoder inside the deployable fusion ---
    a = load_col("hybrid_shortlist.csv", "gold_rank_fused")
    b = load_col("hybrid_shortlist_multilingual-e5-base.csv", "gold_rank_fused")
    compare("A:encoder", "fusedD-MiniLM", "fusedD-E5base", a, b, band, out)

    # --- Arm B: fusion depth ---
    f4 = load_col("fusion_variants.csv", "fused4")
    f5 = load_col("fusion_variants.csv", "fused5")
    fD = load_col("fusion_variants.csv", "fusedD")
    compare("B:fusion45", "fused4", "fused5(+wpair)", f4, f5, band, out)
    compare("B:fusion5D", "fused5", "fusedD(+lc)", f5, fD, band, out)

    # --- Arm C: ngram gram type (word2 from the factorial; char4 computed here) ---
    w2 = load_col("factorial_ranks.csv", "ngram")
    print("\nbuilding char-4gram index…", file=sys.stderr)
    corpus = nb.load_corpus()
    dups = nb.dup_groups(corpus)
    gold = nb.load_gold(dups)
    idx, idf = nb.build_index(corpus, lambda t: nb.char_ngrams(t, 4))
    c4 = {}
    for g in gold:
        scores = nb.rank(nb.char_ngrams(g["q_frag"], 4), idx, idf)
        for s in g["self"]:
            scores.pop(s, None)
        c4[g["conn_id"]] = nb.best_rank(scores, g["hit"])
    compare("C:gram", "word-2gram", "char-4gram", w2, c4, band, out)
    with open(nb.HERE + "results/russian_poetry/char4_ranks.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)          # per-relation ranks of the variant, for detectability_table.py
        w.writerow(["conn_id", "char4"])
        w.writerows([cid, r or ""] for cid, r in c4.items())

    with open(nb.HERE + "results/russian_poetry/invariance.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("\nwrote results/russian_poetry/invariance.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
