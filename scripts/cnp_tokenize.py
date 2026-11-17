"""Word-aligned content-token lists for CnP segments and NT verses.

For every segment and every verse, emit parallel lists `surf[]` (original surface word)
and `lem[]` (Mystem lemma), so lemma index i ↔ surface word i. Only **content tokens**
are kept — same rule as cnp_match._content (lemma not in STOP, len>1, alphabetic).

These let the feature step (cnp_features.py) recover the surface words of an aligned
lemma span, instead of re-scoring whole-segment vs whole-verse.

    python scripts/cnp_tokenize.py
        -> .lemma_cache/cnp_segments_tokens.jsonl
        -> .lemma_cache/nt1823_tokens.jsonl

Cached, git-ignored (live under .lemma_cache/). Regenerable.
"""
import json, os, re, sys
import ngram_baseline as nb
from cnp_lemmatize import normalize
from cnp_match import STOP

CACHE = nb.HERE + "cache/lemmas/"
SENT = "\nЩЩЩ\n"          # sentinel line; Mystem passes it through as its own token(s)
CH = 800                  # texts per analyze() call


def _is_content(lex):
    return lex not in STOP and len(lex) > 1 and lex.isalpha()


def _tokens_from_analysis(tokens):
    """Walk a Mystem analyze() token stream -> (surf[], lem[]) of content tokens."""
    surf, lem = [], []
    for t in tokens:
        an = t.get("analysis")
        if not an:
            continue
        lex = an[0].get("lex", "")
        if _is_content(lex):
            surf.append(t["text"].strip())
            lem.append(lex)
    return surf, lem


def _analyze_per_text(m, texts):
    return [_tokens_from_analysis(m.analyze(normalize(t))) for t in texts]


def tokenize_texts(m, texts):
    """Return list of (surf, lem) per text. Batched via sentinel; per-text fallback."""
    out = []
    for i in range(0, len(texts), CH):
        chunk = [normalize(t) for t in texts[i:i + CH]]
        joined = SENT.join(chunk)
        stream = m.analyze(joined)
        # split the token stream on tokens whose text contains the sentinel marker
        groups, cur = [], []
        for t in stream:
            if "ЩЩЩ" in t.get("text", ""):
                groups.append(cur)
                cur = []
            else:
                cur.append(t)
        groups.append(cur)
        if len(groups) != len(chunk):              # sentinel mangled -> per-text
            out.extend(_analyze_per_text(m, texts[i:i + CH]))
        else:
            out.extend(_tokens_from_analysis(g) for g in groups)
        print(f"  {min(i + CH, len(texts))}/{len(texts)}", file=sys.stderr)
    return out


def _write(ids, pairs, out_path):
    bad = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (surf, lem) in zip(ids, pairs):
            if len(surf) != len(lem):
                bad += 1
            f.write(json.dumps({"id": i, "surf": surf, "lem": lem}, ensure_ascii=False) + "\n")
    print(f"  {len(ids)} rows -> {out_path} (len mismatches: {bad})", file=sys.stderr)
    return bad


def main():
    from pymystem3 import Mystem
    os.makedirs(CACHE, exist_ok=True)
    m = Mystem()

    seg = [json.loads(x) for x in open(nb.HERE + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8")]
    seg_pairs = tokenize_texts(m, [s["text"] for s in seg])
    b1 = _write([s["id"] for s in seg], seg_pairs, CACHE + "cnp_segments_tokens.jsonl")

    nt = [json.loads(x) for x in open(nb.HERE + "data/crime_and_punishment/nt1823.jsonl", encoding="utf-8")]
    nt_pairs = tokenize_texts(m, [v["text"] for v in nt])
    b2 = _write([v["ref"] for v in nt], nt_pairs, CACHE + "nt1823_tokens.jsonl")

    assert b1 == 0 and b2 == 0, "surf/lem length mismatch in some rows"
    print(f"OK: {len(seg)} segments + {len(nt)} verses, all len(surf)==len(lem)", file=sys.stderr)


if __name__ == "__main__":
    main()
