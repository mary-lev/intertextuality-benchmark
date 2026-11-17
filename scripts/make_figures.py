"""Paper figures 1-3 (PDF, grayscale-print-safe) from the committed artifacts.

F1 factorial: small multiples, one panel per probe, cells on x, R@10 + CI whiskers
   (single-hue marks — the reading is the monotone gradient WITHIN each panel).
F2 prior ladder: rungs on x (median pool as secondary tick line), R@10 lines for
   ALL / S1 / S1-loose + the cardinality-matched null (ALL) as neutral dashed.
F3 granularity: 4 query levels on x, R@10 for BM25 vs alignment (Passim-style SW) + CI whiskers.

Identity is never color-alone: distinct markers + linestyles per series, direct
labels at line ends, no legend boxes where a panel title or end-label suffices.
Palette: validated slots #2a78d6 (blue) / #eb6834 (orange) / #1baf7a (aqua) +
neutral #6b7280 for the null reference.

    python scripts/make_figures.py    # from the release root -> figures/
"""
from __future__ import annotations
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ngram_baseline as nb

csv.field_size_limit(10 ** 8)
OUT = os.path.join(nb.HERE, "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280"
INK, MUTED = "#1f2430", "#6b7280"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.titlesize": 9,
    "grid.color": "#d8dbe0", "grid.linewidth": 0.5,
})


def rd(path):
    return list(csv.DictReader(open(os.path.join(nb.HERE, "results", "russian_poetry", path), encoding="utf-8")))


# ---------------------------------------------------------------- Figure 1
def fig1():
    prof = rd("factorial_profile.csv")
    probes = [("ngram", "surface (word 2-gram)"), ("passim", "alignment (SW lemma)"),
              ("bm25", "BM25"), ("wpair", "collocation (wpair)"),
              ("dense", "dense (MiniLM line)"), ("rerank", "LLM rerank (top-10)")]
    cells = [("S1/verbatim", "verbatim\n(11)"), ("S1/close", "close\n(31)"),
             ("S1/loose", "loose\n(58)"), ("S4", "whole-poem\n(50)")]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.6), sharey=True)
    for ax, (p, title) in zip(axes.flat, probes):
        xs, ys, lo, hi = [], [], [], []
        for i, (cell, _) in enumerate(cells):
            row = next((r for r in prof if r["probe"] == p and r["cell"] == cell), None)
            if row is None:
                continue
            xs.append(i); ys.append(float(row["R@10"]))
            lo.append(float(row["R@10"]) - float(row["R@10_lo"]))
            hi.append(float(row["R@10_hi"]) - float(row["R@10"]))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=BLUE, ms=4.5,
                    elinewidth=1.1, capsize=2.5, zorder=3)
        ax.plot(xs, ys, "-", color=BLUE, lw=1.2, alpha=0.55, zorder=2)
        if p == "rerank":
            ax.text(2.98, 0.03, "S4 not applicable", ha="right", fontsize=7,
                    color=MUTED, style="italic")
        ax.set_title(title, pad=3)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([c[1] for c in cells], fontsize=6.8)
        ax.set_ylim(-0.04, 1.06)
        ax.grid(True, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("attested R@10")
    fig.tight_layout(w_pad=1.0, h_pad=1.4)
    fig.savefig(os.path.join(OUT, "fig1_factorial.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_factorial.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig2():
    lad = rd("prior_ladder.csv")
    ctl = rd("prior_ladder_control.csv")
    rungs = ["none", "period±50", "period±25", "period±10", "author"]
    pools = {"none": "45,451", "period±50": "15,788", "period±25": "9,360",
             "period±10": "4,509", "author": "507"}

    def series(cell):
        out = []
        for rg in rungs:
            row = next(r for r in lad if r["rung"] == rg and r["cell"] == cell)
            out.append((float(row["R@10"]), float(row["R@10_lo"]), float(row["R@10_hi"])))
        return out

    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    # end-labels spread manually: the four series converge at the author rung
    styles = [("ALL", "all relations (146)", BLUE, "o", "-", 0.615),
              ("S1", "localized spans (89)", AQUA, "s", "-", 0.79),
              ("S1/loose", "loose band (50)", ORANGE, "^", "-", 0.52)]
    x = range(len(rungs))
    for cell, label, color, mk, ls, laby in styles:
        vals = series(cell)
        ys = [v[0] for v in vals]
        ax.errorbar(x, ys, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt=mk, ls=ls, color=color, ms=4.5, lw=1.4, elinewidth=0.9,
                    capsize=2, zorder=3)
        ax.annotate(label, (x[-1] + 0.12, laby), fontsize=7.5, color=color,
                    va="center")
    # cardinality-matched null (ALL): informed rungs only
    nx, ny = [], []
    for i, rg in enumerate(rungs[1:], 1):
        row = next(r for r in ctl if r["rung"] == rg)
        nx.append(i); ny.append(float(row["exact_R@10"]))
    ax.plot(nx, ny, ls=(0, (4, 2)), marker="D", ms=3.6, color=NEUTRAL, lw=1.3, zorder=2)
    ax.annotate("size-matched random\npools (expectation)", (nx[-1] + 0.12, 0.70),
                fontsize=7.5, color=NEUTRAL, va="center")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r}\n{pools[r]}" for r in rungs], fontsize=7.5)
    ax.set_xlabel("candidate restriction (median pool size)", labelpad=2)
    ax.set_ylabel("attested R@10")
    ax.set_xlim(-0.3, 6.4)
    ax.set_ylim(0, 0.85)
    ax.grid(True, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_ladder.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_ladder.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig3():
    rows = rd("granularity.csv")
    # x in order of median query length (n=98: 5 / 21 / 25 / 71 tokens), as in §5
    levels = [("fragment", "fragment\n(5 tok)"), ("context2", "context ±2\n(21 tok)"),
              ("stanza", "stanza\n(25 tok)"), ("poem", "whole poem\n(71 tok)")]
    from rank_metrics import recall_at_k, bootstrap_ci
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for ranker, label, color, mk in (("bm25", "BM25", ORANGE, "o"),
                                     ("passim", "alignment", BLUE, "s")):
        ys, lo, hi = [], [], []
        for lvl, _ in levels:
            ranks = [int(r[f"{ranker}_{lvl}"]) if str(r[f"{ranker}_{lvl}"]).strip() else None
                     for r in rows]
            pt, l, h = bootstrap_ci(ranks, lambda rs: recall_at_k(rs, 10), seed=0)
            ys.append(pt); lo.append(pt - l); hi.append(h - pt)
        x = range(len(levels))
        ax.errorbar(x, ys, yerr=[lo, hi], fmt=mk, ls="-", color=color, ms=4.5,
                    lw=1.4, elinewidth=0.9, capsize=2, zorder=3)
        ax.annotate(label, (len(levels) - 1 + 0.09, ys[-1]), fontsize=7.5,
                    color=color, va="center")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([l[1] for l in levels], fontsize=7.5)
    ax.set_xlabel("query granularity (median tokens)", labelpad=2)
    ax.set_ylabel("attested R@10")
    ax.set_xlim(-0.25, 4.6)
    ax.set_ylim(0, 0.55)
    ax.grid(True, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_granularity.pdf"))
    fig.savefig(os.path.join(OUT, "fig3_granularity.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("wrote fig1-fig3 (pdf+png) ->", OUT)
