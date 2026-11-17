"""Render the CnP->NT links as a readable side-by-side document: for every link show
the novel text and the FULL New-Testament verse text (expanding verse runs), so the
actual correspondence is visible.

    python scripts/cnp_view.py    # -> results/crime_and_punishment/cnp_bible_matches.md
"""
import json, re, sys
import ngram_baseline as nb


def load_verses():
    """ref -> text, plus an index (book,chapter,verse)->text for expanding runs."""
    by_ref, by_bcv = {}, {}
    for line in open(nb.HERE + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8"):
        v = json.loads(line)
        by_ref[v["ref"]] = v["text"]
        m = re.match(r"(.+?) (\d+):(\d+)$", v["ref"])
        if m:
            by_bcv[(m.group(1), int(m.group(2)), int(m.group(3)))] = v["text"]
    return by_ref, by_bcv


def expand(ref, by_bcv):
    """'Ин 11:23-26' -> [(ref, text), ...] for each verse in the run."""
    m = re.match(r"(.+?) (\d+):(\d+)(?:-(\d+))?$", ref)
    if not m:
        return [(ref, "")]
    bk, ch, v0 = m.group(1), int(m.group(2)), int(m.group(3))
    v1 = int(m.group(4)) if m.group(4) else v0
    out = []
    for v in range(v0, v1 + 1):
        out.append((f"{bk} {ch}:{v}", by_bcv.get((bk, ch, v), "")))
    return out


def main():
    links = [json.loads(x) for x in open(nb.HERE + "results/crime_and_punishment/cnp_bible_refs.jsonl", encoding="utf-8")]
    _, by_bcv = load_verses()
    out = nb.HERE + "results/crime_and_punishment/cnp_bible_matches.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# CnP → 1823 NT — matches, side by side\n\n")
        f.write("Each entry: the novel text (the matched segment) and the full NT verse "
                "text it aligned to. Sorted by alignment score within each band. "
                "Full data: `results/crime_and_punishment/cnp_bible_refs.csv`. Method: `CNP_BIBLE_METHOD.md`.\n\n")
        for band in ("verbatim", "close"):
            rows = sorted([l for l in links if l["band"] == band],
                          key=lambda x: -x["score"])
            f.write(f"\n## {band.upper()} band — {len(rows)} matches\n\n")
            for l in rows:
                verses = expand(l["nt_ref"], by_bcv)
                nt_text = " ".join(t for _, t in verses)
                f.write(f"### {l['nt_ref']}  ·  {l['paragraph_ref']}  ·  score {l['score']}\n\n")
                f.write(f"- **Novel:** {l['cnp_span']}\n")
                f.write(f"- **NT 1823 ({l['nt_ref']}):** {nt_text}\n\n")
    n = len(links)
    print(f"{n} matches -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
