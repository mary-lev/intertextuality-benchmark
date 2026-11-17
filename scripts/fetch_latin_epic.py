"""Fetch the Latin epic parallels used by latin_factorial.py into data/latin_epic/.

Two public sources, pinned by revision and checked by SHA-256:

  * the gold — Burns et al. 2021 (NAACL), `vf_intertext_dataset_1_0.csv`: 945 parallels
    between Valerius Flaccus, Argonautica 1 and Vergil / Lucan / Ovid / Statius, each with
    its references to the commentaries of Kleywegt, Zissos and Spaltenstein. The upstream
    repository states no licence, so the file is fetched, not redistributed.
  * the texts — the PoetryMTEB packaging `LatinEpicIntertextualityRetrieval` (CC BY 4.0):
    the 39,675 target verse lines and the Valerius Flaccus query lines.

    python scripts/fetch_latin_epic.py     # from the release root; stdlib only
"""
from __future__ import annotations
import hashlib, os, sys, urllib.request

import ngram_baseline as nb

DEST = nb.HERE + "data/latin_epic/"
HF = ("https://huggingface.co/datasets/PoetryMTEB/LatinEpicIntertextualityRetrieval/resolve/"
      "27f4fb7a1485f2fd6dffb5f3a2d529b0fc799e95/")
GH = ("https://raw.githubusercontent.com/QuantitativeCriticismLab/"
      "NAACL-HLT-2021-Latin-Intertextuality/fb2235e086cf08d49df51791a5cef6b8b6cbcbf0/"
      "latin-embeddings-search/data/datasets/")
FILES = [  # (url, local name, sha256)
    (GH + "vf_intertext_dataset_1_0.csv", "vf_intertext_dataset_1_0.csv",
     "ab0e0e191e9f5ef9de42dd6b733ff5dd4706993333f5c98f5e16646ee26aa71e"),
    (HF + "corpus/test-00000-of-00001.parquet", "corpus.parquet",
     "a414db005252b51ce7e388a0195c5bafebe0b44d9ad8f3f9a9787c4340935d84"),
    (HF + "queries/test-00000-of-00001.parquet", "queries.parquet",
     "de0f16ec464686060cd1b63de69fdfc2e708a7a312c73a9c320acae8ad1bdd10"),
]


def main():
    os.makedirs(DEST, exist_ok=True)
    for url, name, sha in FILES:
        path = DEST + name
        if not os.path.exists(path):
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req) as r, open(path, "wb") as f:
                f.write(r.read())
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if got != sha:
            sys.exit(f"{name}: checksum mismatch ({got})")
        print(f"ok  {name}")


if __name__ == "__main__":
    main()
