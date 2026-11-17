"""Audit for paper §6 (Crime and Punishment) and §7 (Dante): recomputes every number the
two sections quote from the shipped gold, texts and result files. Prints; changes nothing.

    python scripts/audit_s6_s7.py     # from the release root
"""
from __future__ import annotations
import csv, json, random, re
from collections import Counter, defaultdict

import ngram_baseline as nb

csv.field_size_limit(10 ** 8)


def jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def toks(s):
    return re.findall(r"[а-яё]+", s.lower().replace("ё", "е"))


def verses(ref):
    m = re.match(r"(\S+) (\d+):(\d+)(?:-(\d+))?", ref)
    bk, ch, a = m.group(1), m.group(2), int(m.group(3))
    return [f"{bk} {ch}:{v}" for v in range(a, int(m.group(4) or a) + 1)]


def section6():
    print("== §6 Crime and Punishment ==")
    gold = {r["id"]: r for r in jsonl(nb.HERE + "data/crime_and_punishment/gold.jsonl")}
    print("  relation types:", dict(Counter(r["type"] for r in gold.values())))
    reach = json.load(open(nb.HERE + "results/crime_and_punishment/cnp_gold_reach.json", encoding="utf-8"))
    scope = [r for r in reach if r["in_scope"]]
    print(f"  in scope {len(scope)}; reached by some method {sum(bool(r['methods']) for r in scope)}; "
          f"by rank fusion {sum('fusion' in r['methods'] for r in scope)}; "
          f"missed: {[r['nt_ref'] for r in scope if not r['methods']]}")
    for r in scope:
        print(f"    {r['id']:5s} {r['nt_ref']:12s} {', '.join(r['methods']) or '-'}")

    nov = {r["ref"]: r["text"] for r in jsonl(nb.HERE + "data/crime_and_punishment/crime_and_punishment.jsonl")}
    nt = {r["ref"]: r["text"] for r in jsonl(nb.HERE + "data/crime_and_punishment/nt1823.jsonl")}
    cs = {r["ref"]: r["text"] for r in jsonl(nb.HERE + "data/crime_and_punishment/cs_nt.jsonl")}

    def pid(ref):
        return int(re.search(r"p(\d+)$", ref).group(1))

    def quoted(text):
        segs = re.findall(r"«[^»]*»?", text)
        if not segs and "»" in text:            # a quotation continued from an earlier paragraph
            segs = [text[:text.index("»")]]
        return " ".join(segs)

    def cover(text, vs, corpus):
        a, b = toks(text), set(toks(" ".join(corpus.get(v, "") for v in vs)))
        return sum(w in b for w in a) / max(1, len(a))

    # The passage of a note is its quotation, from the paragraph where the commentator's
    # fragment begins (at most three paragraphs back) to the paragraph the note is attached to.
    print("  Sonya's reading: share of quoted words found in the cited verses (1823 / Church Slavonic)")
    shares = []
    for gid in ("c125", "c126", "c127", "c129"):
        g = gold[gid]
        anchor = g["paragraph_ref"]
        head = " ".join(toks(g["fragment"].split("~")[0])[:3])
        pre = anchor.rsplit("p", 1)[0] + "p"
        start = next(i for i in range(pid(anchor) - 3, pid(anchor) + 1)
                     if head in " ".join(toks(nov.get(f"{pre}{i}", ""))))
        text = " ".join(quoted(nov.get(f"{pre}{i}", "")) for i in range(start, pid(anchor) + 1))
        a, b = cover(text, verses(g["nt_ref"]), nt), cover(text, verses(g["nt_ref"]), cs)
        shares.append((a, b))
        print(f"    {gid} {g['nt_ref']:12s} p{start}-p{pid(anchor)}: {a:.2f} / {b:.2f}")
    print(f"    range 1823 {min(s[0] for s in shares):.2f}-{max(s[0] for s in shares):.2f}, "
          f"Church Slavonic {min(s[1] for s in shares):.2f}-{max(s[1] for s in shares):.2f}")

    for ref in ("Ин 19:5", "Мк 10:14", "Лк 18:16", "Мф 19:14", "Мк 9:25"):
        print(f"  {ref:9s} 1823: {nt.get(ref)}\n            CS:   {cs.get(ref)}")
    for p, t in nov.items():
        for pat in ("Сих есть царствие", "немым и глухим"):
            if pat in t:
                i = t.index(pat)
                print(f"  novel {p}: …{t[max(0, i - 40):i + 60]}…")


def top10(v):
    v = (v or "").strip()
    return v not in ("", "NA", "None") and float(v) <= 10


def r10(rows, col):
    return sum(top10(r[col]) for r in rows) / len(rows)


def boot_gain(rows, a, b, draws=2000, seed=0):
    """Paired difference R@10(a) - R@10(b), resampling passages (query_id), 95% interval."""
    by = defaultdict(list)
    for r in rows:
        by[r["query_id"]].append(r)
    keys = list(by)

    def diff(sample):
        rs = [r for k in sample for r in by[k]]
        return r10(rs, a) - r10(rs, b)
    rng = random.Random(seed)
    bs = sorted(diff([rng.choice(keys) for _ in keys]) for _ in range(draws))
    return diff(keys), bs[int(0.025 * draws)], bs[int(0.975 * draws) - 1]


def section7():
    print("\n== §7 Dante ==")
    rows = list(csv.DictReader(open(nb.HERE + "results/dante/dante_factorial_ranks_open.csv", encoding="utf-8")))
    w = [r for r in rows if r["band"] in ("close", "loose", "verbatim")]
    multi = sum(1 for v in Counter(r["query_id"] for r in w).values() if v > 1)
    print(f"  relations {len(rows)}, passages {len({r['query_id'] for r in rows})}; with a located anchor "
          f"{len(w)} from {len({r['query_id'] for r in w})} passages ({multi} with several)")
    print("  bands:", dict(Counter(r["band"] for r in w)))
    probes = ["ngram", "char4", "sw", "bm25", "wpair", "dense", "dense_e5"]
    print("  R@10, widened anchor / anchor alone / whole passage:")
    for p in probes:
        vals = [f"{r10(w, f'{q}_{p}'):.2f}" if f"{q}_{p}" in w[0] else "  - " for q in ("wide", "lemma", "whole")]
        print(f"    {p:9s} " + " / ".join(vals))
    for a, b in (("lemma_char4", "wide_char4"), ("lemma_sw", "wide_sw"), ("wide_sw", "whole_sw"),
                 ("wide_char4", "whole_char4"), ("wide_dense_e5", "whole_dense_e5"),
                 ("wide_char4", "wide_sw"), ("wide_sw", "wide_dense"), ("wide_dense_e5", "wide_dense")):
        d, lo, hi = boot_gain(w, a, b)
        print(f"  {a} - {b}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for band in ("close", "loose"):
        sub = [r for r in w if r["band"] == band]
        print(f"  {band} (n={len(sub)}): " + ", ".join(f"{p} {r10(sub, 'wide_' + p):.2f}" for p in probes))
    sub = [r for r in w if r["referent_corpus"] != "verified"]
    print(f"  {len(sub)} relations citing poems outside the 60-poem subset: "
          + ", ".join(f"{p} {r10(sub, 'wide_' + p):.2f}" for p in ("sw", "char4", "bm25")))
    closed = [r for r in csv.DictReader(open(nb.HERE + "results/dante/dante_factorial_ranks.csv", encoding="utf-8"))
              if r["band"] in ("close", "loose", "verbatim")]
    opened = [r for r in w if r["referent_corpus"] == "verified"]
    print(f"  60-poem subset (n={len(opened)} open / {len(closed)} closed), R@10 open -> closed:")
    for p in probes:
        col = "wide_" + p
        if col in closed[0]:
            print(f"    {p:9s} {r10(opened, col):.2f} -> {r10(closed, col):.2f}")


if __name__ == "__main__":
    section6()
    section7()
