"""Reranker helpers for scripts/hybrid_rerank.py: the prompt, the API key (from .env or the
environment) and the line window shown around a matched line.
"""
from __future__ import annotations
import os, sys, csv, json, math, random, re
import ngram_baseline as nb


csv.field_size_limit(10**8)


def load_env_key():
    for line in (open(nb.HERE + ".env", encoding="utf-8") if os.path.exists(nb.HERE + ".env") else []):
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("ANTHROPIC_API_KEY")


def window(surf_lines, bi, r=1):
    return [l for l in surf_lines[max(0, bi - r):bi + r + 1] if l.strip()]


PROMPT = """You are verifying intertextual connections in Russian poetry. A commentator marked \
the FRAGMENT below (from one poem) as drawing on an earlier source poem — by quotation, \
paraphrase, a shared distinctive image/phrase, or a clear echo.

FRAGMENT:
«{frag}»

Below are {k} candidate source passages retrieved from a large poetry corpus. For EACH, score \
0–10 how likely it is the ACTUAL source this fragment draws on. Judge shared wording, imagery, \
and phrasing — NOT mere shared topic. A high score needs a real textual/imagistic link, not a \
common theme.

CANDIDATES:
{cands}

Return ONLY a JSON array, one object per candidate, no prose:
[{{"id": <number>, "score": <0-10>}}, ...]"""
