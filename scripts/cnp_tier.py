"""Evidence-tier assembly using ONLY the two texts we have (NT + CnP) — no external corpus.

Poetree was dropped: it is *poetry*, the wrong background for *prose* phraseology (it scored
ordinary prose phrases like «изъявить желание» as distinctive). With only the Gospel and the novel,
the legitimate signals are:
  * shared-run LENGTH (needs no background) — a long contiguous shared run is a quotation;
  * CnP SELF-FREQUENCY (`cnp_self_freq`) — a phrase recurring in the novel is ordinary Dostoevsky
    prose, not a unique biblical borrowing.

Tiers we can assign reliably from these:
  A  sustained group citation   (novel-contiguous run of B/C links, same book+chapter)
  B  direct quotation           (aligned-span surface sim >= 0.90 AND surface run >= 4 tokens)
  C  close verbal quotation      (lemma run >= 3, >= 2 substantive words, not recurring in CnP)
Deferred (NOT claimed as references — need an external 19th-c PROSE corpus to classify):
  D2 weak 2-gram overlaps        (lemma run == 2, unique in CnP) — biblical formula vs ordinary
                                  prose 2-gram cannot be told apart from two texts alone.
  E  ordinary overlap / recurring authorial phrase / 1-lemma coincidence.

    python scripts/cnp_tier.py    # -> cnp_bible_tiered.{csv,jsonl} + results/crime_and_punishment/cnp_bible_tiered_report.md
"""
import csv, json, sys
from collections import defaultdict
import ngram_baseline as nb
from cnp_refine import merge_runs
from cnp_groups import find_groups, load_verse_index
from cnp_view import expand

# function words / pronouns / prepositions / particles excluded when counting "substantive" run words
FUNCTION = set("""
я ты он она оно мы вы они себя свой мой твой наш ваш его ее их этот тот сей такой весь самый
который кто что чей где когда там тут вот это
от при пред для из изо у о об обо до по за над под про без через над меж между ради
не ни же бы ли да и а но или то либо как так уже еще только даже ведь разве хоть хотя бы
друг быть мочь стать сам один два
""".split())


def substantive(run_tokens):
    return [w for w in run_tokens if w not in FUNCTION and len(w) >= 3]


def tier_of(r):
    ss = r["aligned_span_surface_similarity"]; sn = r["longest_shared_surface_ngram"]
    ln = r["longest_shared_lemma_ngram"]; self = r["cnp_self_freq"]
    run = r["run"].split()
    if ss >= 0.90 and sn >= 4:
        return "B"
    if ln >= 3 and self <= 1 and len(substantive(run)) >= 2:
        return "C"
    if ln == 2 and self <= 1:
        return "D2"
    return "E"


RANK = {"A": 0, "B": 1, "C": 2}


def main():
    dist = [json.loads(x) for x in open(nb.HERE + "results/crime_and_punishment/cnp_match_distinct.jsonl", encoding="utf-8")]
    score = {(f["seg_id"], f["nt_ref"]): f["score"] for f in
             (json.loads(x) for x in open(nb.HERE + "results/crime_and_punishment/cnp_match_features.jsonl", encoding="utf-8"))}
    segtext = {s["id"]: s["text"] for s in
               (json.loads(x) for x in open(nb.HERE + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8"))}
    versetext = {v["ref"]: v["text"] for v in
                 (json.loads(x) for x in open(nb.HERE + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8"))}
    by_bcv = load_verse_index()

    refs, deferred, ndef = [], 0, defaultdict(int)
    for r in dist:
        t = tier_of(r)
        if t == "D2":
            deferred += 1
            continue
        if t == "E":
            continue
        refs.append({"seg_id": r["seg_id"], "paragraph_ref": r["seg_id"].split("/")[0],
                     "nt_ref": r["nt_ref"], "tier": t,
                     "score": score.get((r["seg_id"], r["nt_ref"]), 0.0),
                     "run": r["run"], "self": r["cnp_self_freq"],
                     "cnp_span": segtext.get(r["seg_id"], ""),
                     "nt_text": versetext.get(r["nt_ref"], "")})

    # dedup per (paragraph, verse): keep best tier, then best score
    best = {}
    for l in refs:
        k = (l["paragraph_ref"], l["nt_ref"])
        if k not in best or (RANK[l["tier"]], -l["score"]) < (RANK[best[k]["tier"]], -best[k]["score"]):
            best[k] = l
    links = list(best.values())

    runs = merge_runs(links)                 # merge consecutive verses in a paragraph
    groups, used = find_groups(runs)         # sustained passages -> tier A
    for r in runs:
        if (r["paragraph_ref"], r["nt_ref"]) in used:
            r["tier"] = "A"
    # A standalone "direct/close quotation" that recurs in the novel is an ordinary authorial
    # phrase, not a biblical borrowing (e.g. «до сей пора», «до тот пора когда»). Drop non-grouped
    # B/C with self-freq > 1. (Tier-A members are exempt: a sustained quotation inflates its own
    # self-frequency, so the gate would wrongly remove the Lazarus reading.)
    runs = [r for r in runs if r["tier"] == "A" or r["self"] <= 1]
    runs.sort(key=lambda x: (RANK.get(x["tier"], 9), -x["score"]))

    # ---- outputs ----
    jl = nb.HERE + "results/crime_and_punishment/cnp_bible_tiered.jsonl"
    with open(jl, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cs = nb.HERE + "results/crime_and_punishment/cnp_bible_tiered.csv"
    cols = ["tier", "paragraph_ref", "nt_ref", "score", "self", "run", "cnp_span", "nt_text"]
    with open(cs, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in cols})

    import collections
    tc = collections.Counter(r["tier"] for r in runs)
    md = nb.HERE + "results/crime_and_punishment/cnp_bible_tiered_report.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# CnP → 1823 NT — evidence tiers (NT + CnP only, no external corpus)\n\n")
        f.write(f"Reliable references: **A {tc.get('A',0)} · B {tc.get('B',0)} · C {tc.get('C',0)}** "
                f"(after dedup/merge). Deferred (2-gram candidates needing a prose control corpus): "
                f"**{deferred}**.\n\n")
        f.write("Background note: with only the Gospel and the novel, references rest on shared-run "
                "length + CnP self-frequency; the distinctiveness of short biblical *phraseology* "
                "(vs ordinary 19th-c prose) cannot be measured until a prose control corpus is added.\n")
        for t, title in (("A", "Group citations"), ("B", "Direct quotations"),
                         ("C", "Close verbal quotations")):
            rows = [r for r in runs if r["tier"] == t]
            f.write(f"\n## Tier {t} — {title} ({len(rows)})\n\n")
            for r in rows:
                nt = " ".join(tx for _, tx in expand(r["nt_ref"], by_bcv))
                f.write(f"### {r['nt_ref']}  ·  {r['paragraph_ref']}  ·  score {round(r['score'],1)}\n\n")
                f.write(f"- **Novel:** {r['cnp_span']}\n- **NT 1823 ({r['nt_ref']}):** {nt}\n\n")
        f.write(f"\n## Deferred — 2-gram phraseology candidates ({deferred})\n\n"
                "Shared two-word lemma runs, unique in the novel, that match an NT verse. With only "
                "two texts these cannot be told apart from ordinary prose collocations (e.g. «рука "
                "человек», «улица переулок»). They become analysable once a 19th-c prose background "
                "exists (then they feed the phraseological-influence measurement, not the reference "
                "list). Not counted as references here.\n")

    print(f"tiers A/B/C = {tc.get('A',0)}/{tc.get('B',0)}/{tc.get('C',0)} references | "
          f"{deferred} deferred 2-gram candidates -> {cs}, {jl}, {md}", file=sys.stderr)


if __name__ == "__main__":
    main()
