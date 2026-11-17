"""Lemmatize CnP segments and NT verses with Mystem (cached).

    python scripts/cnp_lemmatize.py
"""
import json, os, re, sys
import ngram_baseline as nb
from lemmatize_corpus import lemmatize_texts

CACHE = nb.HERE + "cache/lemmas/"


def normalize(s):
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip()


def _lemmatize_file(m, ids, texts, out_path):
    lem = lemmatize_texts(m, [normalize(t) for t in texts])
    with open(out_path, "w", encoding="utf-8") as f:
        for i, l in zip(ids, lem):
            f.write(json.dumps({"id": i, "lemmas": re.sub(r"\s+", " ", l).strip()},
                               ensure_ascii=False) + "\n")
    print(f"  {len(ids)} -> {out_path}", file=sys.stderr)


def main():
    from pymystem3 import Mystem
    os.makedirs(CACHE, exist_ok=True)
    m = Mystem()

    seg = [json.loads(x) for x in open(nb.HERE + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8")]
    _lemmatize_file(m, [s["id"] for s in seg], [s["text"] for s in seg],
                    CACHE + "cnp_segments_lemmas.jsonl")

    nt = [json.loads(x) for x in open(nb.HERE + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8")]
    _lemmatize_file(m, [v["ref"] for v in nt], [v["text"] for v in nt],
                    CACHE + "nt1823_lemmas.jsonl")


if __name__ == "__main__":
    main()
