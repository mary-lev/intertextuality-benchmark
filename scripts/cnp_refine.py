"""Band, threshold, dedup and verse-run merge over raw segment->verse links."""
import re
from collections import defaultdict
from difflib import SequenceMatcher

VERBATIM = 0.85     # surface ratio >= -> verbatim
CLOSE = 0.45        # >= and < VERBATIM -> close; below -> dropped


def _norm(s):
    return re.sub(r"[^а-яa-z0-9 ]", "", s.lower().replace("ё", "е")).strip()


def band_of(span_surface, verse_surface):
    r = SequenceMatcher(None, _norm(span_surface), _norm(verse_surface)).ratio()
    if r >= VERBATIM:
        return "verbatim"
    if r >= CLOSE:
        return "close"
    return None


def dedup_overlapping(links):
    """Among links to the SAME verse from the same paragraph, keep the highest score."""
    best = {}
    for l in links:
        k = (l["paragraph_ref"], l["nt_ref"])
        if k not in best or l["score"] > best[k]["score"]:
            best[k] = l
    return list(best.values())


def _parse_ref(ref):
    m = re.match(r"(.+?) (\d+):(\d+)$", ref)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (ref, 0, 0)


def merge_runs(links):
    """Merge same-paragraph links to consecutive verses of one book/chapter into a run."""
    by_par = defaultdict(list)
    for l in links:
        by_par[l["paragraph_ref"]].append(l)
    out = []
    for par, ls in by_par.items():
        ls = sorted(ls, key=lambda x: _parse_ref(x["nt_ref"]))
        i = 0
        while i < len(ls):
            bk, ch, v0 = _parse_ref(ls[i]["nt_ref"])
            j, vN = i, v0
            while j + 1 < len(ls):
                bk2, ch2, vn = _parse_ref(ls[j + 1]["nt_ref"])
                if bk2 == bk and ch2 == ch and vn == vN + 1:
                    j += 1; vN = vn
                else:
                    break
            ref = f"{bk} {ch}:{v0}" if v0 == vN else f"{bk} {ch}:{v0}-{vN}"
            best = max(ls[i:j + 1], key=lambda x: x["score"])
            out.append({**best, "nt_ref": ref})
            i = j + 1
    return out
