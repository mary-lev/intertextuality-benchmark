"""Shared rank-metrics core: R@k/MRR + bootstrap CIs + paired McNemar significance.

Consolidates the highest-traffic inline recall@k/MRR duplicates scattered across
results/russian_poetry/*.py (project-review-2026-07.md §3 item 5: ~11 inline copies found) and adds
the uncertainty quantification §2.2 found completely absent (zero CIs/bootstrap/
significance tests anywhere in the codebase as of 2026-07-11).

Ranks convention (matches ngram_baseline.py/author_oracle.py): a list of `int | None`,
one 1-based rank per query/connection, `None` when the gold item never scored at all.

    python -m pytest scripts/test_rank_metrics.py -v

Stdlib only.
"""
from __future__ import annotations
import math
import random


def recall_at_k(ranks: list, k: int) -> float:
    n = len(ranks) or 1
    return sum(1 for r in ranks if r and r <= k) / n


def mrr(ranks: list) -> float:
    n = len(ranks) or 1
    return sum(1 / r for r in ranks if r) / n


def metrics(ranks: list, ks: tuple = (1, 5, 10, 50)) -> dict:
    """Same shape as ngram_baseline.metrics: {"R@1":.., ..., "MRR":.., "n":..}."""
    m = {f"R@{k}": recall_at_k(ranks, k) for k in ks}
    m["MRR"] = mrr(ranks)
    m["n"] = len(ranks)
    return m


def bootstrap_ci(ranks: list, stat_fn, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI for `stat_fn(ranks)`, e.g. `lambda rs: recall_at_k(rs, 10)`.

    Resamples queries (connections) with replacement -- the standard IR bootstrap unit
    is the query, not an individual score. Returns (point_estimate, lo, hi). Seeded with
    a private `random.Random` instance (never touches global random state), so repeated
    calls with the same seed are byte-identical -- required by this project's existing
    determinism discipline (see ngram_baseline.best_rank's pid tiebreak).
    """
    n = len(ranks)
    point = stat_fn(ranks)
    if n == 0:
        return point, point, point
    rng = random.Random(seed)
    boots = sorted(stat_fn([ranks[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo_i = int(n_boot * (alpha / 2))
    hi_i = min(n_boot - 1, int(n_boot * (1 - alpha / 2)))
    return point, boots[lo_i], boots[hi_i]


def mcnemar_exact(ranks_a: list, ranks_b: list, k: int) -> dict:
    """Exact (binomial) McNemar test comparing two methods' hit/miss at R@k on the SAME
    paired queries -- ranks_a[i] and ranks_b[i] must be the same connection.

    Returns discordant-pair counts (b = A-hit/B-miss, c = A-miss/B-hit) and the
    two-sided exact p-value from Binomial(n=b+c, p=0.5).
    """
    if len(ranks_a) != len(ranks_b):
        raise ValueError("ranks_a and ranks_b must be paired: same length, same order")
    hits_a = [bool(r and r <= k) for r in ranks_a]
    hits_b = [bool(r and r <= k) for r in ranks_b]
    b = sum(1 for ha, hb in zip(hits_a, hits_b) if ha and not hb)
    c = sum(1 for ha, hb in zip(hits_a, hits_b) if hb and not ha)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"b": b, "c": c, "n_discordant": n, "p_value": round(p, 4)}
