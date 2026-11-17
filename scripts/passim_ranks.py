"""Dump top-5 Passim matches per comment, with the aligned line. md + CSV.

The Passim analogue of dump_ranks / line_dense's dumps. For each fragment-bearing connection
(S1/S2): rank candidate poems by lemma Smith–Waterman score, show the top-5 with score, hit-flag,
the gold's rank, and the candidate's best-matching surface line (the line sharing the most query
lemmas). Writes passim_top5.md + passim_top5.csv.

    python scripts/passim_ranks.py        # needs cache/lemmas/poetree_lemmas.csv
"""
from __future__ import annotations
import os, sys, csv, math
from collections import defaultdict
import ngram_baseline as nb
from tesserae_baseline import content_lemmas, LEMMAS, DF_CAP, MIN_SHARED
from passim_baseline import sw

csv.field_size_limit(10**8)
TOPK = 5


def load_lines():
    """poem_id -> (lemma line-lists, aligned surface lines), + poem-level lemma df."""
    surf = {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        surf[p["poetree_poem_id"]] = p["full_text"].split("\n")
    for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/added_referents.csv", encoding="utf-8")):
        surf[r["extra_id"]] = r["full_text"].split("\n")
    poems, df = {}, defaultdict(int)
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        raw_lem = r["lemma_text"].split("\n")
        raw_surf = surf.get(r["poem_id"], raw_lem)
        lem_lines, surf_lines = [], []
        for i, ll in enumerate(raw_lem):
            toks = content_lemmas(ll)
            if toks:
                lem_lines.append(toks)
                surf_lines.append(raw_surf[i].strip() if i < len(raw_surf) else " ".join(toks))
        poems[r["poem_id"]] = (lem_lines, surf_lines)
        for l in lem_lines:
            for w in set(l):
                df[w] += 1
    return poems, df


def load_meta():
    meta = {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        meta[p["poetree_poem_id"]] = f'{p["author_name"]} — {p["title"]}'.strip(" —")
    for r in csv.DictReader(open(nb.HERE + "data/russian_poetry/added_referents.csv", encoding="utf-8")):
        meta[r["extra_id"]] = f'{r["ref_author"]} — {r["ref_title"]}'.strip(" —")
    return meta


def main():
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
    print(f"corpus {Npoems} | fragment connections {len(gold)}", file=sys.stderr)

    md, rows, byb = [], [], defaultdict(list)
    md.append("# Top-5 Passim matches per comment — lemma Smith–Waterman alignment\n")
    md.append("Candidate poems ranked by rarity-weighted local alignment of the query lemmas. "
              "Each shows the SW score, ✓ if it is the gold referent, and the poem's "
              "best-matching surface line (most shared query lemmas).\n")
    for g in gold:
        byb[g["stratum"]].append(g)

    for stratum in ("S1", "S2"):
        gs = sorted(byb[stratum], key=lambda x: int(x["conn_id"]))
        if not gs:
            continue
        md.append(f"\n## {stratum}  (n={len(gs)})\n")
        for g in gs:
            Q = content_lemmas("".join(m.lemmatize(g["q_frag"])))
            qset = set(Q)
            cand = defaultdict(int)
            for w in qset:
                if w in inv and len(inv[w]) <= DF_CAP:
                    for pid in inv[w]:
                        cand[pid] += 1
            cands = [pid for pid, c in cand.items() if c >= MIN_SHARED and pid not in g["self"]]
            scored = sorted(((sw(Q, seqs[pid], idf), pid) for pid in cands), key=lambda t: -t[0])
            grank = next((i for i, (_, pid) in enumerate(scored, 1) if pid in g["hit"]), None)

            md.append(f"### conn {g['conn_id']}  ·  gold rank {grank if grank else '—'}"
                      f"{' · ' + g['band'] if g['band'] else ''}")
            md.append(f"- **query:** `{g['q_frag'][:160]}`")
            gold_names = " / ".join(meta.get(h, h) for h in sorted(g["hit"]) if h in meta) or "—"
            md.append(f"- **gold referent:** {gold_names}")
            for rank, (s, pid) in enumerate(scored[:TOPK], 1):
                lem_lines, surf_lines = poems[pid]
                bi, bov = 0, -1
                for li, ll in enumerate(lem_lines):
                    ov = len(qset & set(ll))
                    if ov > bov:
                        bov, bi = ov, li
                line = surf_lines[bi] if surf_lines else ""
                flag = "✓" if pid in g["hit"] else " "
                md.append(f"  {rank}. [{flag}] **{s:.2f}**  {meta.get(pid, pid)}  `({pid})`")
                md.append(f"       ↳ {line[:200]}")
                rows.append({
                    "conn_id": g["conn_id"], "stratum": g["stratum"], "band": g["band"],
                    "gold_rank": grank or "", "rank": rank, "score": f"{s:.4f}",
                    "is_gold": int(pid in g["hit"]), "cand_id": pid,
                    "cand_label": meta.get(pid, pid), "matched_line": line,
                })
            if not scored:
                md.append("  *(no candidates)*")
            md.append("")

    open(nb.HERE + "results/russian_poetry/passim_top5.md", "w", encoding="utf-8").write("\n".join(md))
    with open(nb.HERE + "results/russian_poetry/passim_top5.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["conn_id", "stratum", "band", "gold_rank", "rank",
                                          "score", "is_gold", "cand_id", "cand_label", "matched_line"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote passim_top5.md + .csv ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
