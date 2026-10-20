# From Bones to Rocks — LREC 2026

Code and **scores-only** experiment results for [*From Bones to Rocks: A Systematic Evaluation of Specialized Definition Generation for Portuguese*](https://aclanthology.org/2026.lrec-1.302.pdf), Rafael Oleques Nunes, Dennis Giovani Balreira, and Joel Luís Carbonera, LREC 2026, pp. 3794–3804.

This is a focused public export of the LREC experiments, assembled from the authors' earlier `definition_generation_lrec` project. It is **not** a redistribution of the lexical resources, corpora, aligned term–context–definition instances, or full generated definitions. The existing research directories were left untouched. The previous project's `adversarial` results, Falcon runs, 16-shot runs, unrelated notebooks and scraping experiments have not been copied here.

## Resources used in the paper (download from their owners)

| Domain | Definition source | Context corpus |
|---|---|---|
| Medicine | [Medical Subject Headings (MeSH)](https://meshb.nlm.nih.gov/) and the multilingual [DeCS/MeSH](https://decs.bvsalud.org/en/) initiative. The paper used the **2018** edition; obtain the appropriate edition from the providers, subject to their terms. | [BETE](https://github.com/pavalucas/Bete), a Portuguese diabetes-domain NER/RE dataset. |
| Geology | [SIGEP geological glossary](https://sigep.eco.br/glossario/index.html). The paper reports the edition published **11 May 2025**; the live site may have changed. | [GeoCorpus-3 (GeoCorpus-V3)](https://github.com/bsconsoli/GeoCorpus-V3). Do not substitute GeoCorpus-2 for version 3. |

The paper reports 28,939 MeSH definitions / 121,318 terms and 1,514 SIGEP definitions / 2,026 terms. These counts identify the studied editions, **not** the number of aligned evaluation examples. Obtain and process the underlying resources yourself under their respective terms; this repository does not download or validate them automatically. The historical aligned folds and exact resource snapshots are not included, so the full inference experiment cannot be replayed from this public checkout alone.

## What is included

- `folds.py`: prompt construction and semantic-neighbor retrieval for locally supplied IID folds; the prompts request a Portuguese definition with zero or 1/2/4/8 examples, with or without sentence context. This is research code and requires the original input format and an embedding model.
- `prompt.py` and `models/DecoderResponse.py`: inference with Ollama and BERTScore (`bert-base-multilingual-cased`), BLEU, METEOR evaluation. The script expects **private** `definitions_bete/iid/fold_N/test.json` and `definitions_geocorpus/iid/fold_N/test.json` files with a `scope` column and prompt columns. Run only after preparing data lawfully; its raw outputs go into ignored `local_raw_results/`, never into the public `results/` directory. Model IDs in the historical score directories correspond to `gaia:4b`, `gemma3:12b`, `llama3.1:8b`, `mistral:7b`, `phi3:14b`, and `qwen3:14b`.
- `results/`: **600** score-only JSON files (six models × two domains × ten prompt conditions × five IID folds). They contain `n_items` and `scores` only: per-item BERTScore precision/recall/F1 and aggregate BLEU/METEOR. We removed `true_definitions` and `pred_definitions` from the original result files. Anonymized per-item metric vectors are retained for statistical analysis. Portable directory names replace `:` in Ollama model IDs with `-`.
- `summarize_results.py`: standard-library validation and mean / sample standard deviation across the five folds. `view_analysis.py`, `visualizations.py`, and `generate_tex.py` are the original exploratory analysis scripts, retained for provenance; they may require adaptation for portable model-directory names and their optional dependencies. Do **not** treat generated summaries as necessarily identical to the paper's rounding and presentation.

## Quick verification (no corpus download required)

```bash
python summarize_results.py
python summarize_results.py --csv summary.csv
```

The first command checks the expected result layout, parses all score files, and verifies that BERTScore item counts match `n_items`. The optional CSV is ignored by Git. The score-only bundle is sufficient for score analysis, but cannot regenerate definitions or verify the original reference text without the external resources.

## Reproducing inference locally

Use a dedicated Python environment with `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `tqdm`, `evaluate`, `bert-score`, `nltk`, `langchain-core`, `langchain-ollama`, and the dependencies of your chosen analysis scripts. Install [Ollama](https://ollama.com/) and obtain the models under their licenses. Download and align the two definition sources and two corpora, then prepare five private IID folds in the directory structure described above. `folds.py` expects training/test records with `term`, `scope`, and `sentences_containing_term`. Verify the dataset edition, alignment and fold methodology against [the paper](https://aclanthology.org/2026.lrec-1.302.pdf) before interpreting a reproduction. Never commit local raw inputs, reference definitions, prompt texts, or raw model responses.

## Citation

> Nunes, Rafael Oleques; Balreira, Dennis Giovani; Carbonera, Joel Luís. 2026. *From Bones to Rocks: A Systematic Evaluation of Specialized Definition Generation for Portuguese*. In Proceedings of LREC 2026, pp. 3794–3804. [Paper](https://aclanthology.org/2026.lrec-1.302.pdf).

Please cite the original resource owners as appropriate when using their dictionaries and corpora. No license is asserted here for third-party data or for the paper text.
