"""Corpus helpers for scripts/hybrid_rerank.py: the verse lines of every poem and its
author and title.
"""
from __future__ import annotations
import os, sys, csv, math
from collections import defaultdict
import ngram_baseline as nb
from tesserae_baseline import content_lemmas, LEMMAS, DF_CAP, MIN_SHARED


csv.field_size_limit(10**8)


def load_lines():
    """poem_id -> (lemma line-lists, aligned surface lines), + poem-level lemma df."""
    surf = {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        surf[p["poetree_poem_id"]] = p["full_text"].split("\n")
    for r in nb.load_added_referents():
        surf[r["extra_id"]] = r["full_text"].split("\n")
    poems, df = {}, defaultdict(int)
    for r in csv.DictReader(open(LEMMAS, encoding="utf-8")):
        raw_lem = r["lemma_text"].split("\n")
        raw_surf = surf.get(r["poem_id"], raw_lem)
        lem_lines, surf_lines = [], []
        for i, ll in enumerate(raw_lem):
            toks = content_lemmas(ll)
            if toks:
                lem_lines.append(toks)
                surf_lines.append(raw_surf[i].strip() if i < len(raw_surf) else " ".join(toks))
        poems[r["poem_id"]] = (lem_lines, surf_lines)
        for l in lem_lines:
            for w in set(l):
                df[w] += 1
    return poems, df


def load_meta():
    meta = {}
    for p in csv.DictReader(open(nb.HERE + "data/poetree/poetree_full_poems.csv", encoding="utf-8")):
        meta[p["poetree_poem_id"]] = f'{p["author_name"]} — {p["title"]}'.strip(" —")
    for r in nb.load_added_referents():
        meta[r["extra_id"]] = f'{r["ref_author"]} — {r["ref_title"]}'.strip(" —")
    return meta
