"""Segment CnP paragraphs into sentences and clauses.

    python scripts/cnp_segment.py    # data/crime_and_punishment/crime_and_punishment.jsonl -> results/crime_and_punishment/cnp_segments.jsonl
"""
import json, re, sys
import ngram_baseline as nb

ABBR = ("т. е.", "т.е.", "и т. д.", "и т.д.", "и пр.", "т. п.", "до н. э.")
_PLACEHOLD = {a: a.replace(". ", ". ").replace(".", "․") for a in ABBR}

SENT_END = re.compile(r"([.!?…]+|\.{3})(\s+|$)")


def split_sentences(text):
    t = text
    for a, ph in _PLACEHOLD.items():
        t = t.replace(a, ph)
    out, start = [], 0
    for m in SENT_END.finditer(t):
        seg = t[start:m.end()].strip()
        if seg:
            out.append(seg)
        start = m.end()
    tail = t[start:].strip()
    if tail:
        out.append(tail)
    # restore placeholders
    fix = {v: k for k, v in _PLACEHOLD.items()}
    return [re.sub("|".join(map(re.escape, fix)), lambda x: fix[x.group()], s).strip()
            for s in out]


CLAUSE_SPLIT = re.compile(r"\s*[;:,—]\s*|\s+[-–]\s+")


def split_clauses(sentence):
    parts = [p.strip(" .!?…«»\"") for p in CLAUSE_SPLIT.split(sentence)]
    return [p for p in parts if len(p) >= 3]


def main():
    src = nb.HERE + "data/crime_and_punishment/crime_and_punishment.jsonl"
    out = nb.HERE + "results/crime_and_punishment/cnp_segments.jsonl"
    n_par = n_sent = n_cl = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            r = json.loads(line); n_par += 1
            for si, sent in enumerate(split_sentences(r["text"]), 1):
                sid = f"{r['ref']}/s{si}"
                f.write(json.dumps({"id": sid, "level": "sentence",
                                    "paragraph_ref": r["ref"], "text": sent},
                                   ensure_ascii=False) + "\n")
                n_sent += 1
                for ci, cl in enumerate(split_clauses(sent), 1):
                    f.write(json.dumps({"id": f"{sid}/c{ci}", "level": "clause",
                                        "paragraph_ref": r["ref"], "text": cl},
                                       ensure_ascii=False) + "\n")
                    n_cl += 1
    print(f"{n_par} paragraphs -> {n_sent} sentences, {n_cl} clauses -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
