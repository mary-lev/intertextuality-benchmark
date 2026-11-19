"""Paper figures 1-2 (PDF, grayscale-print-safe) from the committed artifacts.

F1 which relations each probe finds: one column per banded relation (verbatim / close /
   loose), one row per probe and one for the rank fusion; filled = referent in the top ten.
F2 two panels. Left, candidate restriction: rungs on x (median candidate-set size), R@10 for all
   relations and the loose band, with the size-matched random-set expectation (dashed, same
   colour as the series it controls). Right, query context: 4 query levels on x, R@10 for
   alignment and BM25 + CI whiskers.

Identity is never color-alone: distinct markers + linestyles per series, direct
labels at line ends, no legend boxes where a panel title or end-label suffices.
Palette: validated slots #2a78d6 (blue) / #eb6834 (orange) + neutral #6b7280.

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
    """One column per banded relation, one row per probe; filled = referent in the top ten."""
    from matplotlib.patches import Rectangle

    def top10(v):
        v = (v or "").strip()
        return v not in ("", "NA", "None") and float(v) <= 10

    fr = {r["conn_id"]: r for r in rd("factorial_ranks.csv")}
    e5 = {r["conn_id"]: r["dense_e5"] for r in rd("dense_e5_ranks.csv")}
    fused = {r["conn_id"]: r["gold_rank_fused"] for r in rd("hybrid_shortlist.csv")}
    probes = [("surface", lambda k: fr[k]["ngram"]), ("alignment", lambda k: fr[k]["passim"]),
              ("BM25", lambda k: fr[k]["bm25"]), ("collocation", lambda k: fr[k]["wpair"]),
              ("MiniLM", lambda k: fr[k]["dense"]), ("E5", lambda k: e5.get(k))]
    bands = ["verbatim", "close", "loose"]
    rels = [k for k, r in fr.items() if r["stratum"] == "S1" and r["band"] in bands]
    found = {k: [top10(get(k)) for _, get in probes] for k in rels}
    # within a band: relations found by more probes first, then by which probes
    rels.sort(key=lambda k: (bands.index(fr[k]["band"]), -sum(found[k]), [not f for f in found[k]]))

    fig, ax = plt.subplots(figsize=(6.3, 2.1))
    n = len(probes)
    for x, k in enumerate(rels):
        cells = found[k] + [top10(fused.get(k))]
        for y, on in enumerate(cells):
            yy = -y - (0.4 if y == n else 0)
            color = (BLUE if y < n else INK) if on else "#eceef1"
            ax.add_patch(Rectangle((x, yy), 0.84, 0.84, color=color, lw=0))
    x0 = 0
    for b in bands:
        m = sum(fr[k]["band"] == b for k in rels)
        ax.text(x0 + m / 2, 1.0, f"{b}\n({m})", ha="center", va="bottom", fontsize=7.3,
                linespacing=1.0)
        if x0:
            ax.plot([x0 - 0.08] * 2, [-n - 0.45, 0.9], color=MUTED, lw=0.7)
        x0 += m
    ax.set_xlim(-0.4, len(rels) + 0.2)
    ax.set_ylim(-n - 0.55, 2.2)
    ax.set_yticks([-y + 0.42 - (0.4 if y == n else 0) for y in range(n + 1)])
    ax.set_yticklabels([p for p, _ in probes] + ["fusion"], fontsize=7.8)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig1_relations.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_relations.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
def fig2():
    """Left: candidate restriction against size-matched random sets. Right: query context."""
    from rank_metrics import recall_at_k, bootstrap_ci
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.75), gridspec_kw={"width_ratios": [1.3, 1]})

    # left: restriction
    lad = rd("prior_ladder.csv")
    ctl = rd("prior_ladder_control.csv")
    rungs = ["none", "period±50", "period±25", "period±10", "author"]
    pools = {"none": "45,451", "period±50": "15,788", "period±25": "9,360",
             "period±10": "4,509", "author": "507"}
    ticks = {"none": "none", "period±50": "±50", "period±25": "±25",
             "period±10": "±10", "author": "author"}
    x = list(range(len(rungs)))
    for cell, label, color, mk, laby in (("ALL", "all relations (146)", BLUE, "o", 0.60),
                                         ("S1/loose", "loose band (50)", ORANGE, "^", 0.51)):
        vals = []
        for rg in rungs:
            row = next(r for r in lad if r["rung"] == rg and r["cell"] == cell)
            vals.append((float(row["R@10"]), float(row["R@10_lo"]), float(row["R@10_hi"])))
        ys = [v[0] for v in vals]
        ax.errorbar(x, ys, yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                    fmt=mk, ls="-", color=color, ms=4, lw=1.3, elinewidth=0.8, capsize=2, zorder=3)
        ax.annotate(label, (x[-1] + 0.15, laby), fontsize=7, color=color, va="center")
    # size-matched random sets that contain the referent: exact expectation, all relations
    nx, ny = [], []
    for i, rg in enumerate(rungs[1:], 1):
        row = next(r for r in ctl if r["rung"] == rg)
        nx.append(i); ny.append(float(row["exact_R@10"]))
    ax.plot(nx, ny, ls=(0, (3.5, 2)), marker="D", ms=3, color=BLUE, lw=1.1, alpha=0.75, zorder=2)
    ax.annotate("random sets of the\nsame size (expectation)", (nx[-1] + 0.15, 0.71),
                fontsize=7, color=BLUE, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ticks[r]}\n{pools[r]}" for r in rungs], fontsize=6.8)
    ax.set_xlabel("period window (years) or author", labelpad=2)
    ax.set_ylabel("attested R@10")
    ax.set_xlim(-0.3, 6.6)
    ax.set_ylim(0, 0.85)
    ax.set_title("restricting the candidates", pad=4)

    # right: query context
    rows = rd("granularity.csv")
    # x in order of median query length (n=98: 5 / 21 / 25.5 / 71 tokens), as in §5
    levels = [("fragment", "fragment\n(5)"), ("context2", "±2 lines\n(21)"),
              ("stanza", "stanza\n(25.5)"), ("poem", "poem\n(71)")]
    for ranker, label, color, mk in (("bm25", "BM25", ORANGE, "o"), ("passim", "alignment", BLUE, "s")):
        ys, lo, hi = [], [], []
        for lvl, _ in levels:
            ranks = [int(r[f"{ranker}_{lvl}"]) if str(r[f"{ranker}_{lvl}"]).strip() else None
                     for r in rows]
            pt, l, h = bootstrap_ci(ranks, lambda rs: recall_at_k(rs, 10), seed=0)
            ys.append(pt); lo.append(pt - l); hi.append(h - pt)
        bx.errorbar(range(len(levels)), ys, yerr=[lo, hi], fmt=mk, ls="-", color=color, ms=4,
                    lw=1.3, elinewidth=0.8, capsize=2, zorder=3)
        bx.annotate(label, (len(levels) - 1 + 0.2, ys[-1]), fontsize=7, color=color, va="center")
    bx.set_xticks(range(len(levels)))
    bx.set_xticklabels([l[1] for l in levels], fontsize=6.8)
    bx.set_xlabel("query (median tokens)", labelpad=2)
    bx.set_xlim(-0.35, 4.4)
    bx.set_ylim(0, 0.55)
    bx.set_title("widening the query", pad=4)

    for a in (ax, bx):
        a.grid(True, axis="y")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(OUT, "fig2_restriction_context.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_restriction_context.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2()
    print("wrote fig1-fig2 (pdf+png) ->", OUT)
