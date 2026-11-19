# Attested, Not Exhaustive: Constructing and Evaluating Benchmarks for Literary Intertextuality Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24309021.svg)](https://doi.org/10.5281/zenodo.24309021)

Data, code and results for the paper *Attested, Not Exhaustive: Constructing and
Evaluating Benchmarks for Literary Intertextuality Detection*.

## Layout

```
data/           the benchmark: one gold file per set, plus the texts it is searched against
scripts/        retrieval probes, experiments, figures
results/        the tables the paper's numbers and figures are read from
figures/        the paper's figures
```

## Data

| Path | Content |
|---|---|
| `data/russian_poetry/gold.csv` | **The Russian poetry gold: 286 verified relations in one file**, with the full text of the commented poem and, where it is a poem in the corpus, of the referent. |
| `data/russian_poetry/idiom_candidates.csv` | The 216 idioms of the shared-idiom comparison: entries of the Russian Wiktionary category «Фразеологизмы/ru» with three or more words found in 2 to 99 corpus poems, not reviewed by a reader. |
| `data/crime_and_punishment/gold.jsonl` | *Crime and Punishment* gold: 48 New Testament relations from two commentary layers: 17 from Fridlender and Kogan (`c…` ids) and 31 that only Tikhomirov records (`t…` ids). The thresholds were adjusted on nine of the eligible rows (`c8`, `c19`, `c102`, `c125`–`c127`, `c129`, `c131`, `t1`); the other 19 eligible rows are a held-out check (paper §6). Five rows carry a corrected verse pointer: `original_nt_ref` keeps the pointer before correction and `correction_note` says what was wrong (one error of the printed apparatus, three of our parsing, one narrowing of scope). `note` gives each row's commentary note as printed (page, lemma, note). |
| `data/crime_and_punishment/crime_and_punishment.jsonl`, `nt1823.jsonl`, `cs_nt.jsonl` | The novel (3,758 paragraphs), the 1823 Russian New Testament (7,952 verses) and the Church Slavonic New Testament (7,824 verses; Ephesians 2–6 missing), all in modern orthography. |
| `data/dante/gold.csv`, `data/dante/poems_index.json` | Dante–Duecento set: 478 relations over 196 passages, one row per commentary citation (the anchor in Dante's text, i.e. the words the note comments on → cited verses of a poem), from the DanteSources citation network (Bartalesi et al. 2015); the poem texts are from Biblioteca Italiana (`poems_index.json`, 929 records), cut into four-line chunks by `scripts/dante_factorial.py`. Column `referent_corpus` separates the 369 relations citing the 60-poem subset (431 chunks) from the 109 citing other poems. The headline search space is 8,959 four-line chunks of 904 poems. |
| `data/manzoni/` | Nardi's 41 references in *I promessi sposi* (Introduction and chapter 1; 32 of his 249 notes), with the role he gives each (`nardi_intertesti.tsv`), and, for the eight names checked by hand, how many of the 39 other commentaries of *Leggo Manzoni* name them and how many mean the same reference (`nardi_intertesti_confronto.tsv`). At don Abbondio's «testimonio consolante della coscienza» (`c1_12438–12442`), 23 of the 40 commentaries gloss the passage; Nardi and Bianchi name the Dante tercet (*Inf.* XXVIII 115–117), and Viti-Sapegno quotes it at an adjacent anchor (`c1_12424–12426`). Unscored. |

### Columns of `data/russian_poetry/gold.csv`

| Column | Meaning |
|---|---|
| `conn_id`, `edition` | Relation id; the commented poet's edition (seven editions). |
| `commented_author`, `commented_work`, `commented_text_id`, `commented_text` | The commented poem: author, title or first line, Poetree id, full text. Relation 60 lists two ids, the poem and a second redaction of it in Poetree; a query's own poem is never a candidate. |
| `commented_fragment` | The place in the commented poem the commentary points at; empty when the relation concerns the whole poem. |
| `commented_lines`, `commented_stanza_lines` | The lines of the commented poem that the fragment (or the annotator's span) occupies, and the lines of the stanza(s) containing them, used by the query-context analysis of §5 (stanzas from PoeTree's stanza structure). Line numbers count the non-empty lines of `commented_text` from 0, separated by `;`. |
| `ref_author`, `ref_title`, `ref_citation` | The referent as the commentary names it, with the words it quotes, if any. |
| `referent_text_id`, `referent_source`, `referent_status`, `referent_text` | The referent in the reference corpus: id, origin (`poetree` or `extra`), link status, full text. The 10 `extra` referents are poems PoeTree lacks; the experiments add them to the reference corpus from this file. |
| `referent_year`, `referent_text_source` | For the 10 `extra` referents only: year of composition and the edition the text comes from. The period windows of §5 read PoeTree metadata only, so these referents are outside them. |
| `referent_span` | The referent passage, where the commentary gives one. |
| `analyst_commented_span`, `analyst_referent_span` | Spans added by the annotator where the commentary names a relation without quoting it. |
| `category` | `poetry` (168), `foreign` (55), `prose` (30), `cultural` (23), `non_textual` (10). Only `poetry` relations enter the retrieval task. |
| `commentary`, `note`, `review_note` | The commentary sentence that attests the relation, and annotator notes. |
| `provenance`, `verified` | `extraction` (247, LLM-extracted then verified) or `curated` (39, entered from print); every row is human-verified. |

The relations were taken from the commentaries of the critical editions listed in Appendix A of the
paper; the editions are in copyright and are not included.

Of the 168 poem-to-poem relations, 158 are evaluable: the other ten name a
poet's manner or an untitled text, not an identifiable poem. The reference corpus is
45,452 poems: the 45,442 distinct poems of the Poetree snapshot and the 10 added
referents.

### How to use the gold

The gold records **scholar-attested** relations. It is a pool of attested
positives, not an exhaustive relevance judgment:

- A retrieved text that is absent from the gold is unattested, not wrong.
- Recall of attested relations (R@k, MRR) is well defined; report it with
  bootstrap intervals (`scripts/rank_metrics.py`), since n is between 100 and 160.
- Precision and F1 against the gold are not: they would count every unattested
  hit as an error. Precision needs human adjudication of the retrieved candidates.

## Reproducing the results

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
./reproduce.sh            # figures from the shipped result tables; no corpus needed
./reproduce.sh --full     # recompute the result tables from the corpus first
```

`--full` needs two files that are not in this repository:

```
data/poetree/poetree_full_poems.csv     the Russian part of PoeTree (the paper used a snapshot of 45,442 distinct poems)
cache/lemmas/poetree_lemmas.csv         its lemmatised form (build it with scripts/lemmatize_corpus.py)
```

PoeTree (Plecháč et al. 2025) is not redistributed here; obtain it from its authors.

These scripts are deterministic, and `--full` rewrites the result tables byte
for byte. It takes several hours on a CPU; the granularity study alone takes about
three. It makes no paid API calls: the LLM reranker's outputs are shipped
(`results/russian_poetry/hybrid_rerank_results.csv`), and `hybrid_rerank.py` is
only needed to regenerate them with your own key. One shipped reranker result predates a correction to the gold:
for relation 43 the reranker saw the fragment with a variant reading from the
edition's apparatus («К земле, беспомощный, прижат!»), while the gold and the
other probes use the main text, «К земле беспомощно прижат!». With the corrected fragment the fusion
shortlist for relation 43 differs in two of its ten candidates (the attested
referent stays second); the shipped rerank rank (8) was computed on the earlier
shortlist and was not re-run. The dense probe downloads a
sentence encoder on first use.

| Paper | Script | Output |
|---|---|---|
| §5 profile (Table 5) and Fig. 1 | `scripts/factorial.py` | `results/russian_poetry/factorial_ranks.csv`, `factorial_profile.csv` |
| §5 dense probe on its own, either encoder | `scripts/line_dense.py` (set `EMB_MODEL=intfloat/multilingual-e5-base` for the larger encoder) | `line_dense_top5.csv`, `line_dense_top5_multilingual-e5-base.csv` |
| §5 E5-base row of the profile table (all scored relations, same query as the other probes; needs the E5 line cache written by `line_dense.py`) | `scripts/dense_e5_ranks.py` | `dense_e5_ranks.csv` |
| §5 and appendix, scale of verbal distance (post hoc description of the 100 banded relations; `wording_class` value `verbatim` is the paper's *repeated*) | `scripts/lexical_evidence.py` | `lexical_evidence.csv` |
| §5 fusion on whole-poem relations | `SCOPE=whole scripts/hybrid_baseline.py` | `fusion_variants_wholepoem.csv`, `hybrid_shortlist_wholepoem.csv` |
| §8 discovery pilot: candidates outside the gold (three Mandelstam poems, windows of 1, 2 and 4 lines, earlier poems only; other poems with `POEMS=`) | `scripts/window_search.py` | `window_search.csv` (top 10) |
| §5 candidate restriction and control, Fig. 2 left | `scripts/prior_ladder.py`, `prior_ladder_control.py` | `prior_ladder.csv`, `prior_ladder_control.csv` |
| §5 query context, Fig. 2 right | `scripts/granularity.py` | `granularity.csv` |
| §5 and appendix, shared idioms | `scripts/idiom_control.py lexical`, then `dense` (MiniLM; with `EMB_MODEL=intfloat/multilingual-e5-base` for E5), then `summary` | `idiom_control_lexical.csv`, `idiom_control_dense_minilm.csv`, `idiom_control_dense_e5.csv` |
| §5 configuration stability (Table 7) | `scripts/invariance.py` | `invariance.csv`, `char4_ranks.csv` (per-relation ranks of the character 4-gram variant, Table 6) |
| §5 and appendix numbers quoted in the text | `scripts/audit_s5.py` | printed |
| §6 and §7 numbers quoted in the text | `scripts/audit_s6_s7.py` | printed |
| §6 *Crime and Punishment*, Table 8 | `scripts/cnp_rebuild.sh` | `results/crime_and_punishment/cnp_gold_reach.json` |
| §7 Dante transfer (three query definitions: anchor widened to its line or sentence, anchor alone, whole passage; seven probes, including the character 4-gram surface variant) | `CORPUS=open scripts/dante_factorial.py` (headline: all 478 relations, 8,959 chunks); `scripts/dante_factorial.py` (the 369 relations citing the 60-poem subset, on its 431 chunks) | `results/dante/dante_factorial_ranks_open.csv`; `results/dante/dante_factorial_ranks.csv` |
| Figures | `scripts/make_figures.py` | `figures/` |

All scripts are run from the repository root, for example
`.venv/bin/python scripts/prior_ladder.py`.

### *Crime and Punishment*

`scripts/cnp_rebuild.sh` runs the whole set in order: it splits the novel into
sentences, runs each method against the 1823 New Testament (and the surface and
alignment methods also against the Church Slavonic text), and writes
`results/crime_and_punishment/cnp_gold_reach.json` (`scripts/cnp_gold_reach.py`).
That file lists, for each of the 48 gold relations, the methods whose own output
pairs a sentence of the quoted passage with one of the cited verses; Table 8 of
the paper is read from it. Method names, with the probe of the paper's Table 3:
`sw` alignment, `cs_sw` alignment against the Church Slavonic text, `nt_surface`
and `cs_surface` surface (words), `char` and `wchar` surface (character
4-grams), `bm25` BM25, `tesserae` and `wpair` collocation, `dense` dense,
`rerank` reranker, `fusion` rank fusion. The two encoder steps (dense, reranker)
take about 25 minutes each on a CPU, with local models only.

The per-method outputs that are small are in the repository. The larger
intermediate files (`cnp_segments.jsonl`, `cnp_match_*.jsonl`,
`cnp_dense_top5.jsonl`, `cnp_fusion_top.jsonl`, `cnp_rerank_top.jsonl`; about
110 MB) are not; the script regenerates them.

## Use of generative AI

Claude Opus 4.8 proposed 247 of the 286 Russian relations, each verified by the
author against the printed edition, and reranks the fused shortlists of the
Russian set (paper §§3 and 5). The author used Claude Code (Anthropic) to write
and audit the code in this repository and to check the paper's statements
against the data and sources. The author reviewed all code, results and text.

## Licence

See `LICENSE.md`.

## Citation

Please cite the paper (see `CITATION.cff`). Every release is archived on Zenodo:
https://doi.org/10.5281/zenodo.24309021 (all versions; each version has its own DOI).
