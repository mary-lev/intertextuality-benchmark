"""Feature-based band (supersedes the whole-segment SequenceMatcher ratio of cnp_refine).

The old band compared the WHOLE segment to the WHOLE verse, penalizing a short quotation
embedded in long prose and scoring 2-lemma coincidences the same as real matches. This bands
each link from features computed on the Smith-Waterman ALIGNED SPAN (see cnp_features.py):

  verbatim : aligned-span surface similarity >= V_SURF_SIM AND a >=V_SURF_NGRAM exact run
  close    : a >=C_LEMMA_NGRAM contiguous shared-lemma run
             OR (>=C_NSHARE shared content lemmas AND sum-idf >= C_SUMIDF)
  else     : dropped

Writes the refs table (now carrying the 7 features) + drives cnp_report_full for the report.

    python scripts/cnp_report_v2.py     # then: python scripts/cnp_report_full.py
"""
import csv, json, sys
import ngram_baseline as nb
from cnp_refine import dedup_overlapping, merge_runs

# --- feature-based band thresholds (tunable) ---
V_SURF_SIM = 0.90
V_SURF_NGRAM = 4
C_LEMMA_NGRAM = 3
C_NSHARE = 4
C_SUMIDF = 18.0

FEAT_KEYS = ["aligned_span_surface_similarity", "aligned_span_lemma_similarity",
             "longest_shared_surface_ngram", "longest_shared_lemma_ngram",
             "number_of_shared_content_lemmas", "sum_idf_shared_content_lemmas",
             "max_idf_shared_lemma"]


def band_of_features(f):
    if (f["aligned_span_surface_similarity"] >= V_SURF_SIM
            and f["longest_shared_surface_ngram"] >= V_SURF_NGRAM):
        return "verbatim"
    if (f["longest_shared_lemma_ngram"] >= C_LEMMA_NGRAM
            or (f["number_of_shared_content_lemmas"] >= C_NSHARE
                and f["sum_idf_shared_content_lemmas"] >= C_SUMIDF)):
        return "close"
    return None


def main():
    feats = [json.loads(x) for x in open(nb.HERE + "results/crime_and_punishment/cnp_match_features.jsonl", encoding="utf-8")]
    segtext = {s["id"]: s["text"] for s in
               (json.loads(x) for x in open(nb.HERE + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8"))}
    versetext = {v["ref"]: v["text"] for v in
                 (json.loads(x) for x in open(nb.HERE + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8"))}

    links = []
    for f in feats:
        band = band_of_features(f)
        if not band:
            continue
        seg = f["seg_id"]
        l = {"seg_id": seg, "paragraph_ref": seg.split("/")[0],
             "cnp_span": segtext.get(seg, ""), "nt_ref": f["nt_ref"],
             "nt_text": versetext.get(f["nt_ref"], ""), "band": band, "score": f["score"]}
        for k in FEAT_KEYS:
            l[k] = f[k]
        links.append(l)

    n_before = len(links)
    links = dedup_overlapping(links)
    runs = merge_runs(links)
    runs.sort(key=lambda x: -x["score"])

    jl = nb.HERE + "results/crime_and_punishment/cnp_bible_refs.jsonl"
    with open(jl, "w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cs = nb.HERE + "results/crime_and_punishment/cnp_bible_refs.csv"
    cols = ["paragraph_ref", "nt_ref", "band", "score"] + FEAT_KEYS + ["cnp_span", "nt_text"]
    with open(cs, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in cols})

    nv = sum(1 for r in runs if r["band"] == "verbatim")
    nc = sum(1 for r in runs if r["band"] == "close")
    print(f"feature band: {n_before} links pass -> {len(runs)} after dedup/merge "
          f"(verbatim {nv}, close {nc}) -> {cs}, {jl}\n"
          f"now run cnp_report_full.py for the grouped report", file=sys.stderr)


if __name__ == "__main__":
    main()
