"""Shared no-lemmatizer surface matcher: shared *rare* content words, idf-weighted,
NO lemmatization, NO Smith-Waterman. Reused for the Russian NT and the CS NT.
"""
import json
import math
import re
from collections import defaultdict

import ngram_baseline as nb

H = nb.HERE
DF_CAP = 400

# A proper Russian FUNCTION-WORD stoplist (служебные части речи) — the small 46-word list let
# «было/для/до/тех/пор/когда/чтоб/можно/меня/есть/так/все» through as "content", which formed
# spurious distinctive pairs in `wpair` and noise words in the surface matchers. We deliberately
# KEEP «се» (the Church-Slavonic «Се человек» register key) out of the list.
_PREP = ("в во на с со к ко о об обо от ото из изо у за под подо над надо при про для до без через "
         "сквозь между меж среди около возле подле против напротив вдоль мимо после перед пред "
         "передо ради кроме вместо насчет благодаря согласно вопреки по")
_CONJ = ("и а но или либо да что чтобы чтоб как когда пока если хотя хоть потому поэтому оттого "
         "затем зато также тоже причем притом ибо дабы чем нежели будто словно точно ежели коли раз")
_PART = ("не ни же бы б ли ль вот вон ведь уж уже еще ещё только лишь даже разве неужели именно ка "
         "мол якобы пусть пускай")
_PRON = ("я ты он она оно мы вы они меня тебя его ее её них нас вас им ими мне тебе ему ей ним ней "
         "нем нём себя себе собой свой своя свое свои мой моя мое мои твой твоя твое твои наш наша "
         "наше наши ваш ваша ваше ваши этот эта это эти тот та то те этого этом этой этих того той "
         "тех таком такой такая такое такие сам сама само сами самый который которая которое которые "
         "кто чей весь вся всё все всего всех всем всём кого чего чем чём кому чему ком")
_BYT = "быть был была было были есть будет будут буду будешь будем будете бывает"
_MOD = "можно нужно надо нельзя"
_ADV = ("так там тут здесь теперь тогда сейчас потом затем уже ещё очень совсем вообще впрочем итак "
        "где куда откуда зачем почему отчего пор")
_CS = "аз аще бо еже иже убо яко сей сие сия"   # Church-Slavonic function words (NOT «се»)
STOP = set(" ".join((_PREP, _CONJ, _PART, _PRON, _BYT, _MOD, _ADV, _CS)).split())


def toks(s):
    s = re.sub(r"[^а-яё ]", " ", s.lower().replace("ё", "е"))
    return [w for w in s.split() if len(w) > 1 and w not in STOP]


def surface_match(ref_file, ref_key, out_file, *, floor, topk, df_cap=DF_CAP, min_shared=2,
                  score_idf=None, proximity_idf=None, terminal_tiebreak=False):
    """score_idf: optional {word: weight}. If given, the SHARED-word score is summed over it
    instead of the reference (NT/CS) idf — used by nt_surface to weight by CnP-rarity (a word
    common in the novel, e.g. «крайней мере», is down-weighted). Candidate generation (df_cap,
    inverted index) is unchanged. None → reference idf (the original behaviour; cs_surface).

    proximity_idf: optional {word: weight} for a CONTIGUITY re-rank bonus. The rarity gate stays
    on the summed score (`floor`, unchanged — no new filter); the emitted score gains
    `+ longest contiguous run of shared words in the verse, summed over proximity_idf`. So a
    real contiguous quotation («приидет царствие твое») outranks a scattered-word coincidence of
    the same summed rarity. None → no bonus.

    terminal_tiebreak: when True, break exact SCORE ties by preferring the verse where the matched
    words sit at the END of the verse (fewest trailing content tokens after the last shared word).
    A short standalone CnP quote («Се человек!») then resolves to the verse where the phrase is a
    terminal clause (CS Ин 19:5 «…се, человек.») over the canonically-earlier verses that merely
    contain it mid-sentence (Мф 11:19 «се, человек ядца…»). Only reorders verses of EQUAL score —
    no effect when scores differ. Used by cs_surface to close the c8 «Се человек» gap."""
    refs = [json.loads(x) for x in open(H + ref_file, encoding="utf-8")]
    for v in refs:
        v["set"] = set(toks(v["text"]))
        if proximity_idf is not None or terminal_tiebreak:
            v["toks"] = toks(v["text"])          # ordered tokens (contiguity bonus / terminal tiebreak)
    df = defaultdict(int)
    for v in refs:
        for w in v["set"]:
            df[w] += 1
    N = len(refs)
    idf = {w: math.log(N / d) for w, d in df.items()}
    wt = score_idf if score_idf is not None else idf      # the SCORE weight
    inv = defaultdict(list)
    for vi, v in enumerate(refs):
        for w in v["set"]:
            inv[w].append(vi)

    rows = []
    # process every segment level (sentence + clause), like the original CS matcher;
    # a clause id (…/sN/cM) counts for its sentence. Clause ids
    # are where short embedded quotes (e.g. «Се человек») surface.
    for s in (json.loads(x) for x in open(H + "results/crime_and_punishment/cnp_segments.jsonl", encoding="utf-8")):
        Q = set(toks(s["text"]))
        if len(Q) < min_shared:
            continue
        cand = set()
        for w in Q:
            if len(inv.get(w, ())) <= df_cap:
                cand.update(inv[w])
        hits = []
        for vi in cand:
            shared = Q & refs[vi]["set"]
            if len(shared) < min_shared:
                continue
            base = sum(wt.get(w, 0.0) for w in shared)
            if base < floor:                          # rarity gate (unchanged — no new filter)
                continue
            score = base
            if proximity_idf is not None:             # + contiguity bonus (re-rank only)
                best = cur = 0.0
                for w in refs[vi]["toks"]:
                    cur = cur + proximity_idf.get(w, 0.0) if w in shared else 0.0
                    if cur > best:
                        best = cur
                score = base + best
            tb = 0
            if terminal_tiebreak:                 # higher = matched phrase nearer the verse end
                vt = refs[vi]["toks"]
                last = max((i for i, w in enumerate(vt) if w in shared), default=len(vt) - 1)
                tb = -(len(vt) - 1 - last)        # 0 when the match is terminal, negative otherwise
            hits.append((round(score, 2), tb, refs[vi]["ref"],
                         sorted(shared, key=lambda w: -wt.get(w, 0.0))))
        if not hits:
            continue
        hits.sort(key=lambda x: (-x[0], -x[1]))   # score desc, then most-terminal first (tb=0 wins)
        for sc, _tb, ref, sh in hits[:topk]:
            rows.append({"seg_id": s["id"], ref_key: ref, "score": sc, "shared": sh})

    rows.sort(key=lambda r: -r["score"])
    with open(H + out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)
