# Explain Difficulty

Fit IRT to a response matrix, then discover and select hypotheses explaining item difficulty.
The code has two packages: `irt` and `hypotheses`.

## Install

Python 3.10 or newer, from the repository root:

```bash
python -m pip install -e '.[irt,hypotheses]'
```

## 1. Fit IRT

Provide `data/responses.npy`: a NumPy matrix with **questions in rows and models in
columns**. Use `1` for correct, `0` for incorrect, and `-1` or `NaN` for missing.
Missing entries are excluded from fitting. All-missing rows or columns are rejected.

```bash
python -m irt.fit --responses data/responses.npy --epochs 500 --lr 0.1 --seed 0 --output output/irt_results.json
```

- `--responses`: input response matrix.
- `--epochs`, `--lr`: number of fitting epochs and initial learning rate.
- `--seed`: random seed.
- `--output`: JSON containing item difficulties, model abilities and posterior SDs.

The model is hierarchical 1PL, fitted with py-irt on CPU:
`P(correct) = sigmoid(ability - difficulty)`. Larger difficulty means harder.
Check convergence when choosing the number of epochs.

By default, item ID `"0"` is matrix row 0, `"1"` is row 1, and so on. To use your own
IDs, add `--item-ids data/item_ids.json`; this must be a JSON list in matrix row order.
`--model-ids data/model_ids.json` similarly supplies IDs in column order.

## 2. Discover hypotheses

### 2.1 Configure the API key

```bash
export GEMINI_API_KEY="your-api-key"
```

Only generation, refinement and matching need the API. IRT, preprocessing and Lasso
selection run locally. Do not commit your key.

### 2.2 Prepare questions and splits

Provide a JSON list (`data/items.json`), CSV, or JSONL file. Each record needs
`item_id` and `question`; `answer` and `explanation` are optional reference fields.
When present, both are included in generation and matching.

```json
[
  {"item_id": "0", "question": "What is 2 + 3?", "answer": "5", "explanation": "Add the two quantities."},
  {"item_id": "1", "question": "What is 8 - 2?", "answer": "6", "explanation": "Subtract 2 from 8."}
]
```

For CSV, use the same field names as column headers. Include **every question from
the response matrix**, with the same IDs used in step 1. Records may be shuffled:
the code joins by ID and restores matrix order, not file row order. Without custom
IDs, `item_id: "7"` therefore means row 7 of `responses.npy` (the eighth question).

```bash
python -m hypotheses.discover prepare --items data/items.json --difficulty output/irt_results.json --responses data/responses.npy --v-size 400 --t-size 200 --seed 0 --output .cache/prepared.json
```

- `--items`: question file; change the path to use CSV or JSONL.
- `--difficulty`: IRT result from step 1, joined to questions by `item_id`.
- `--responses`: verifies that the original `.npy` file matches the IRT fit's checksum and shape.
- `--v-size`, `--t-size`: numbers of items assigned to V (selection) and T (held out).
  The remainder form G (generation). Splits are disjoint and difficulty-stratified.
- `--seed`: split seed; `--output`: prepared questions, difficulties and splits.

Choose sizes appropriate for your dataset: G needs at least 200 questions and V
needs at least 14 for the default selection settings. To reuse existing splits,
add `--splits data/splits.json` containing `{"G": [...], "V": [...], "T": [...]}`
with item-ID lists; this overrides the split sizes and must cover all input items.

### 2.3 Generate, refine and match

```bash
python -m hypotheses.discover generate --input .cache/prepared.json --group-rounds 30 --pair-rounds 50 --min-gap 4.0 --propose-model gemini-3.1-pro-preview --match-model gemini-3.7-flash --output output/hypotheses_results.json --execute
```

- `--input`: prepared file from step 2.2, already aligned to the response matrix.
- `--group-rounds`: rounds showing 20 harder and 20 easier G items per round.
- `--pair-rounds`: rounds showing five hard/easy G pairs per round.
- `--min-gap`: minimum IRT difficulty gap within each pair.
- `--propose-model`, `--match-model`: API models for generation/refinement and matching.
- `--execute`: enables paid API calls. Omit it to validate sampling without calling the API.
- `--output`: refined hypotheses and a binary item-by-hypothesis matching matrix.

This runs **30 group + 50 pair rounds** by default, followed by exact-text and strict
semantic deduplication, semantic-family refinement, and fresh matching against all
questions. Generation uses G only; matching does not expose difficulty to the judge.
The default cosine threshold is 0.85 (`--threshold`). The local embedding model is
`Qwen/Qwen3-Embedding-4B` (`--embedding-model`); its weights download on first use
and require sufficient memory. Model names can be changed to ones your account supports.

If the requested sampling plan is infeasible, the command stops before API calls.
Validated responses are cached in `.cache/hypotheses/`; rerun an interrupted command
to reuse them. Keep the cache to avoid repeat calls. Retries may incur extra charges.
IRT, preparation and generation do not overwrite existing outputs.

### 2.4 Select hypotheses with stability Lasso

```bash
python -m hypotheses.discover select --input output/hypotheses_results.json --top-k 20 --repeats 100 --sample-fraction 0.75 --cv-folds 5 --seed 0
```

- `--input`: hypotheses, matches and difficulties saved in step 2.3.
- `--top-k`: maximum number of hypotheses to retain; fixed by you, not tuned on T.
- `--repeats`: number of Lasso fits on random subsamples of V.
- `--sample-fraction`: fraction of V sampled without replacement per fit.
- `--cv-folds`: folds used to choose the regularization weight within each subsample.
- `--seed`: subsampling and cross-validation seed.

Each fit chooses its own Lasso alpha by cross-validation over 25 log-spaced values
from `1e-4` to `1`. Features are standardized inside each fold. Hypotheses are ranked
by nonzero-coefficient frequency, then mean absolute standardized coefficient.
At most `--top-k` hypotheses with nonzero frequency are retained. Only V is used.

Selection adds `selected`, `rank` and `selection_frequency` to each hypothesis in
the **same result JSON**, preserving hypothesis order and matching-matrix alignment.
The previous file is backed up in `.cache/hypotheses/`. Add `--output new_result.json`
to save separately. You can rerun selection with different settings without API calls.

The two main outputs are `output/irt_results.json` and `output/hypotheses_results.json`.
The latter includes explicit row item IDs, column hypothesis IDs, splits and selection
settings. Temporary preparation data and API caches stay in `.cache/` (ignored by Git).
Do not run concurrent commands against the same output or cache directory.

Prompts are in `hypotheses/prompts.py`. This release provides the consolidated 30+50
workflow; historical run settings and external model outputs can differ.

## Tests

```bash
python -m pip install -e '.[irt,hypotheses,test]'
python -m pytest -q
```

Tests use synthetic data and mocked API responses; no paid API calls are made.
