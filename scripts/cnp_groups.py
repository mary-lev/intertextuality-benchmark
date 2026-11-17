"""Detect GROUP citations — sustained passages where a contiguous stretch of the novel
quotes many verses of one NT book+chapter in reading order (e.g. Sonya's Lazarus reading
of John 11). Such a group is both the natural unit to display AND a strong precision
signal: coincidental noise never forms an ordered, novel-contiguous, multi-verse chain.

A group = a maximal run of links, ordered by novel reading position, such that
  * all share one (book, chapter),
  * consecutive members are within GAP paragraphs in the novel,
  * the run covers >= MIN_VERSES distinct verses across >= MIN_MEMBERS links.

    python scripts/cnp_groups.py    # -> results/crime_and_punishment/cnp_bible_groups.md
"""
import json, re, sys
from collections import defaultdict
import ngram_baseline as nb

GAP = 6           # max paragraph-index gap between consecutive members
MIN_MEMBERS = 3   # links needed to call it a group
MIN_VERSES = 3    # distinct verses the group must cover
MIN_PARAS = 2     # distinct novel paragraphs — a real sustained citation is not one
                  # phrase matching a repeated verse (drops single-paragraph artifacts)


def pidx(paragraph_ref):
    m = re.search(r"p(\d+)$", paragraph_ref)
    return int(m.group(1)) if m else -1


def parse_ref(ref):
    m = re.match(r"(.+?) (\d+):(\d+)(?:-(\d+))?$", ref)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4) or m.group(3))


def load_verse_index():
    by_bcv = {}
    for line in open(nb.HERE + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8"):
        v = json.loads(line)
        m = re.match(r"(.+?) (\d+):(\d+)$", v["ref"])
        if m:
            by_bcv[(m.group(1), int(m.group(2)), int(m.group(3)))] = v["text"]
    return by_bcv


def find_groups(links):
    """Return (groups, used_link_ids). Each group: dict with members + summary."""
    # bucket links by (book, chapter)
    buckets = defaultdict(list)
    enriched = []
    for l in links:
        pr = parse_ref(l["nt_ref"])
        if not pr:
            continue
        e = {**l, "_p": pidx(l["paragraph_ref"]), "_bk": pr[0], "_ch": pr[1],
             "_vf": pr[2], "_vt": pr[3]}
        enriched.append(e)
        buckets[(pr[0], pr[1])].append(e)

    groups, used = [], set()
    for (bk, ch), ls in buckets.items():
        ls.sort(key=lambda x: (x["_p"], x["_vf"]))
        i = 0
        while i < len(ls):
            run = [ls[i]]
            j = i + 1
            while j < len(ls) and ls[j]["_p"] - run[-1]["_p"] <= GAP:
                run.append(ls[j]); j += 1
            verses = set()
            for m in run:
                verses.update(range(m["_vf"], m["_vt"] + 1))
            n_paras = len({m["paragraph_ref"] for m in run})
            if len(run) >= MIN_MEMBERS and len(verses) >= MIN_VERSES and n_paras >= MIN_PARAS:
                groups.append({
                    "book": bk, "chapter": ch,
                    "v_min": min(verses), "v_max": max(verses),
                    "verses": sorted(verses),
                    "p_first": run[0]["paragraph_ref"], "p_last": run[-1]["paragraph_ref"],
                    "n_members": len(run), "n_verses": len(verses),
                    "score": round(sum(m["score"] for m in run), 1),
                    "members": run,
                })
                for m in run:
                    used.add((m["paragraph_ref"], m["nt_ref"]))
            i = j
    groups.sort(key=lambda g: -g["score"])
    return groups, used


def main():
    links = [json.loads(x) for x in open(nb.HERE + "results/crime_and_punishment/cnp_bible_refs.jsonl", encoding="utf-8")]
    by_bcv = load_verse_index()
    groups, used = find_groups(links)
    singletons = [l for l in links if (l["paragraph_ref"], l["nt_ref"]) not in used]

    out = nb.HERE + "results/crime_and_punishment/cnp_bible_groups.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# CnP → 1823 NT — GROUP citations\n\n")
        f.write(f"A *group* is a sustained passage: a novel-contiguous run of matches "
                f"(≤{GAP} paragraphs apart) all in one NT book+chapter, covering "
                f"≥{MIN_VERSES} verses across ≥{MIN_MEMBERS} links. These are extended "
                f"quotations; the long tail of one-off matches is left in "
                f"`results/crime_and_punishment/cnp_bible_matches.md`.\n\n")
        f.write(f"**{len(groups)} group citation(s) detected.**\n")
        for g in groups:
            vr = f"{g['book']} {g['chapter']}:{g['v_min']}-{g['v_max']}"
            f.write(f"\n---\n\n## {vr}  (group score {g['score']})\n\n")
            f.write(f"- **Novel:** {g['p_first']} – {g['p_last']}  "
                    f"({g['n_members']} matched segments)\n")
            f.write(f"- **Source:** {g['book']} {g['chapter']}, "
                    f"{g['n_verses']} verses matched: "
                    f"{', '.join(str(v) for v in g['verses'])}\n\n")
            f.write("**Matched scripture (in order):**\n\n")
            for v in g["verses"]:
                t = by_bcv.get((g["book"], g["chapter"], v), "")
                f.write(f"> **{v}.** {t}\n>\n")
            f.write("\n**Member links:**\n\n| verse | novel ¶ | band | score |\n|---|---|---|---:|\n")
            for m in sorted(g["members"], key=lambda x: (x["_p"], x["_vf"])):
                f.write(f"| {m['nt_ref']} | {m['paragraph_ref']} | {m['band']} | {m['score']} |\n")
        f.write(f"\n---\n\n*{len(singletons)} one-off (non-group) links remain in "
                f"`results/crime_and_punishment/cnp_bible_matches.md`.*\n")
    print(f"{len(groups)} groups, {len(singletons)} singletons -> {out}", file=sys.stderr)
    for g in groups:
        print(f"  GROUP {g['book']} {g['chapter']}:{g['v_min']}-{g['v_max']}  "
              f"{g['p_first']}–{g['p_last']}  ({g['n_members']} links, "
              f"{g['n_verses']} verses, score {g['score']})", file=sys.stderr)


if __name__ == "__main__":
    main()
