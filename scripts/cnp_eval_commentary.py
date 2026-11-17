"""The Crime and Punishment gold and verse matching, shared by the C&P scripts: the frozen gold
with (book, chapter, verses) parsed from each nt_ref, the verse-overlap test and the
paragraph index of a segment id.
"""
import json, re, sys
import ngram_baseline as nb


def pidx(pref):
    m = re.search(r"p(\d+)", pref)
    return int(m.group(1)) if m else -1


def our_vrange(nt_ref):
    m = re.match(r".+? \d+:(\d+)(?:-(\d+))?", nt_ref)
    return (int(m.group(1)), int(m.group(2) or m.group(1))) if m else None


def parse_ref(nt_ref):
    """'Ин 11:19-22' -> ('Ин', 11, (19,22)); 'Ин 11' -> ('Ин', 11, None); '' / None -> (None,)*3."""
    if not nt_ref:
        return (None, None, None)
    m = re.match(r"(.+?) (\d+)(?::(\d+)(?:[-–—](\d+))?)?$", nt_ref)
    if not m:
        return (None, None, None)
    bk, ch = m.group(1), int(m.group(2))
    if m.group(3) is None:
        return (bk, ch, None)
    lo = int(m.group(3))
    return (bk, ch, (lo, int(m.group(4) or lo)))


def load_gold():
    """The frozen structured eval gold (data/crime_and_punishment/gold.jsonl), with (bk, ch, gv) parsed
    from each note's nt_ref. This replaces runtime re-resolution from the raw commentary."""
    out = []
    for l in open(nb.HERE + "data/crime_and_punishment/gold.jsonl", encoding="utf-8"):
        r = json.loads(l)
        r["bk"], r["ch"], r["gv"] = parse_ref(r["nt_ref"])
        out.append(r)
    return out


def matches(nt_ref, bk, ch, gv):
    """Honest match: same book; same chapter (if gold gives one); and the verse ranges OVERLAP
    (if gold gives verses). A same-chapter candidate to an unrelated verse does NOT count."""
    m = re.match(r"(.+?) (\d+):", nt_ref)
    if not m or m.group(1) != bk:
        return False
    if ch is not None and int(m.group(2)) != ch:
        return False
    if gv is None:
        return True
    ov = our_vrange(nt_ref)
    return ov is not None and ov[0] <= gv[1] and gv[0] <= ov[1]
