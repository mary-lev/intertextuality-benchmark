"""Evaluate our CnP->NT detections against the frozen commentary gold.

For every note in the frozen gold, locate the novel paragraph (via the
note's fragment), classify the note type, and report what our pipeline found there — at which level
(tier A/B/C, the v2 365-link close band, or only the raw candidate pool) and with which verse/run.
Writes a full per-note results doc.

    python scripts/cnp_eval_commentary.py   # -> results/crime_and_punishment/commentary-eval.results.md
"""
import json, re, sys
import ngram_baseline as nb

DOC = nb.HERE + "results/crime_and_punishment/commentary-eval.results.md"
RANK = {"A": 0, "B": 1, "C": 2}

BOOK = {"матфея": "Мф", "марка": "Мк", "луки": "Лк", "иоанна": "Ин"}


def book_sigla(s):
    s = (s or "").lower()
    if "откровение" in s:
        return "Откр"
    for k, v in BOOK.items():
        if k in s:
            return v
    if s.strip() == "евангелие":
        return "Ин"          # bare "Евангелие" notes here are all the John-11/14 Lazarus context
    return None              # "Новый завет" / "Библия" -> meta


def note_type(frag, bk, ch):
    """Hand-tuned over the 16 notes. Four labels; quotation + allusion are the in-scope verbal subset:
      meta/narrative — the book is mentioned, not quoted;
      thematic       — no/low shared wording (paraphrase/motif), lexically unreachable by design;
      allusion       — a verbal single-reference allusion: a shared *distinctive* phrase, possibly
                       reordered, recoverable order-independently (char-4gram) though not by lemma-SW;
      quotation      — running / near-verbatim verbal reuse."""
    f = frag
    if any(w in f for w in ("Новый завет в русском", "Под подушкой", "Сидел в мое время",
                            "Дочли до 32-го", "за чтением вечной книги")):
        return "meta/narrative"
    if bk is None:
        return "meta/narrative"
    if "Новый Иерусалим" in f:        # «Новый Иерусалим» reorders Откр 21:2 «Иерусалим новый» — a shared
        return "allusion"             # distinctive bigram, recovered order-independently (char-4gram)
    if any(w in f for w in ("тайное становится явным", "Лазаря петь",
                            "Спастись во всем мире")):
        return "thematic"
    return "quotation"


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[«»\"“”']", " ", t)).strip()


def pidx(pref):
    m = re.search(r"p(\d+)", pref)
    return int(m.group(1)) if m else -1


def load():
    paras = [(p["ref"], norm(p["text"])) for p in
             (json.loads(l) for l in open(nb.HERE + "data/crime_and_punishment/crime_and_punishment.jsonl", encoding="utf-8"))]
    tiered = [json.loads(l) for l in open(nb.HERE + "results/crime_and_punishment/cnp_bible_tiered.jsonl", encoding="utf-8")]
    band365 = [json.loads(l) for l in open(nb.HERE + "results/crime_and_punishment/cnp_bible_refs.jsonl", encoding="utf-8")]
    runs = {}
    for l in open(nb.HERE + "results/crime_and_punishment/cnp_match_distinct.jsonl", encoding="utf-8"):
        r = json.loads(l)
        runs[(r["seg_id"].split("/")[0], r["nt_ref"])] = r["run"]
    pool = []
    for l in open(nb.HERE + "results/crime_and_punishment/cnp_match_features.jsonl", encoding="utf-8"):
        r = json.loads(l)
        pool.append((r["seg_id"].split("/")[0], r["nt_ref"], r["score"]))
    return paras, tiered, band365, runs, pool


def find_para(frag, paras):
    for key in sorted([s.strip() for s in re.split(r"~|\.\.\.|…", norm(frag))], key=len, reverse=True):
        if len(key) < 6:
            continue
        for ref, txt in paras:
            if key[:30] in txt:
                return ref
    fk = norm(frag)[:18]
    for ref, txt in paras:
        if fk and fk in txt:
            return ref
    return None


def vrange(verses):
    """Parse gold verses ('19—22', '47—48', None) -> (lo, hi) or None for whole-chapter."""
    if not verses:
        return None
    nums = re.findall(r"\d+", str(verses).replace("—", "-").replace("–", "-"))
    return (int(nums[0]), int(nums[-1])) if nums else None


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


def substantive_run(run):
    return run and len([w for w in run.split() if len(w) >= 4]) >= 2


def find_hit(pr, bk, ch, gv, tiered, band365, runs, pool):
    """Best honest detection near pr matching book+chapter+verse-overlap, or None."""
    if pr is None or bk is None:
        return None
    pi = pidx(pr)
    near = lambda p: abs(pidx(p) - pi) <= 3
    t = [r for r in tiered if near(r["paragraph_ref"]) and matches(r["nt_ref"], bk, ch, gv)]
    if t:
        b = min(t, key=lambda r: RANK.get(r["tier"], 9))
        return ("Tier " + b["tier"], b["nt_ref"], runs.get((b["paragraph_ref"], b["nt_ref"]), ""))
    s = [r for r in band365 if near(r["paragraph_ref"]) and matches(r["nt_ref"], bk, ch, gv)]
    if s:
        b = max(s, key=lambda r: r.get("score", 0))
        return ("v2 close-band (365)", b["nt_ref"], runs.get((b["paragraph_ref"], b["nt_ref"]), ""))
    # candidate pool: only a real detection if the verse overlaps AND the shared run is substantive
    c = [(p, nr, sc) for (p, nr, sc) in pool
         if near(p) and matches(nr, bk, ch, gv) and substantive_run(runs.get((p, nr), ""))]
    if c:
        b = max(c, key=lambda x: x[2])
        return ("candidate pool only", b[1], runs.get((b[0], b[1]), ""))
    return None


def main():
    paras, tiered, band365, runs, pool = load()
    gold = load_gold()      # frozen structured gold: resolved (paragraph_ref, nt_ref, type)

    results = []
    for i, g in enumerate(gold, 1):
        bk, ch, gv, typ, pr = g["bk"], g["ch"], g["gv"], g["type"], g["paragraph_ref"]
        hit = None if typ.startswith("meta") else find_hit(pr, bk, ch, gv, tiered, band365, runs, pool)
        results.append((i, g, typ, pr, hit))

    # ---- verdict per note ----
    def verdict(typ, hit):
        if typ.startswith("meta"):
            return "n/a — book mentioned, not quoted (out of scope)"
        if hit:
            lvl = hit[0]
            if lvl.startswith("Tier"):
                return f"FOUND — {lvl}"
            return f"DETECTED (below tier) — {lvl}"
        return "not found" + (
            " — thematic/paraphrase, lexically unreachable" if typ == "thematic"
            else " — reordered allusion; recovered by char-4gram (enrich-only), not by the SW pipeline"
            if typ == "allusion" else "")

    reach = [r for r in results if r[2] in ("quotation", "allusion")]  # in-scope verbal subset
    found_reach = [r for r in reach if r[4]]
    tier_reach = [r for r in reach if r[4] and r[4][0].startswith("Tier")]

    with open(DOC, "w", encoding="utf-8") as f:
        f.write("# Evaluation vs. the РВБ commentary gold (partial ground truth)\n\n")
        f.write("Evaluated against the **frozen structured gold** `data/crime_and_punishment/gold.jsonl` "
                "(resolved paragraph_ref + nt_ref); c131/c196 carry verified corrections.\n\n")
        f.write(f"Gold: `data/crime_and_punishment/gold.jsonl` — {len(gold)} biblical notes. The commentary is "
                f"only *partly* ground truth: many notes are **thematic** (no shared wording) or "
                f"**meta** (the book is mentioned, not quoted), which a verbal-matching pipeline "
                f"cannot and should not detect. Recall is reported against the **verbal-quotation** "
                f"subset.\n\n")
        f.write("## Headline\n\n")
        f.write(f"- verbal in gold (quotation + allusion): **{len(reach)}** → detected somewhere: "
                f"**{len(found_reach)}/{len(reach)}**, of which in confident tiers A/B/C: "
                f"**{len(tier_reach)}/{len(reach)}** (the Lazarus reading).\n")
        f.write(f"- thematic (unreachable by lexical method): "
                f"**{sum(1 for r in results if r[2]=='thematic')}**.\n")
        f.write(f"- meta / narrative (not quotations): "
                f"**{sum(1 for r in results if r[2].startswith('meta'))}**.\n\n")

        f.write("## Summary table\n\n| # | id | gold reference | type | paragraph | result |\n")
        f.write("|--:|---|---|---|---|---|\n")
        for i, g, typ, pr, hit in results:
            ref = g["nt_ref"] or g["source_book"] or "?"
            ours = f" → `{hit[1]}`" if hit else ""
            f.write(f"| {i} | {g['id']} | {ref} | {typ} | {pr or '—'} | {verdict(typ, hit)}{ours} |\n")

        f.write("\n## Per-note detail\n")
        for i, g, typ, pr, hit in results:
            f.write(f"\n### {i}. {g['id']} — {g['nt_ref'] or g['source_book'] or '?'}  ·  *{typ}*\n\n")
            f.write(f"- **Novel fragment:** {g['fragment']}\n")
            f.write(f"- **Attested in:** {g.get('source', '')}\n")
            f.write(f"- **Mapped paragraph:** {pr or 'not located'}\n")
            f.write(f"- **What we found:** {verdict(typ, hit)}")
            if hit:
                f.write(f" — our match `{hit[1]}`" + (f", shared run «{hit[2]}»" if hit[2] else ""))
            f.write("\n")

        f.write("\n## Caveats on specific gold records\n\n")
        f.write("- **#2 `Ин 19:5` «Се человек!» (Marmeladov):** RESOLVED — the 1823 *Russian* NT "
                "renders `Ин 19:5` as «вот человек» (се→вот), which is why the Russian-NT lexical "
                "methods can't seed a candidate on «человек» alone (478/7952 verses, > DF_CAP). The "
                "Church-Slavonic NT (`data/crime_and_punishment/cs_nt.jsonl`) now exists and `cnp_match_cs_surface.py` (floor 7, "
                "top-1, terminal tiebreak) closes this exact gap against CS `Ин 19:5`.\n")
        f.write("- **#13 `Мф 19:14` «Сих есть царствие божие» (Raskolnikov):** the gold's book "
                "attribution does not match our source — 1823 `Мф 19:14` reads «царствие **небесное**»; "
                "«царствие божие» actually matches `Мк 10:14`/`Лк 18:16`. We *do* detect «царствие "
                "божие» as a deferred 2-gram (to `Лк 18:29`); it is demoted because «Сих»→«таковых» is "
                "Church-Slavonic and «царствие божие» also recurs in Marmeladov's speech "
                "(`ч1.гл2:p73`). So 'not found' here = gold mis-attribution + correct self-freq "
                "demotion, not a missed quotation.\n")

    print(f"{len(found_reach)}/{len(reach)} reachable quotations detected "
          f"({len(tier_reach)} in tiers) -> {DOC}", file=sys.stderr)


if __name__ == "__main__":
    main()
