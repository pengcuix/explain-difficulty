# Explain Difficulty

**From model responses to interpretable hypotheses about question difficulty.**

Fit item response theory (IRT) to a response matrix, then discover question
properties through group and pair contrasts. Two packages, two main outputs.

[Installation](#installation) · [Input formats](#input-formats) · [Data](data/README.md) · [Fit IRT](#1-fit-irt) · [Discover hypotheses](#2-discover-hypotheses)

## Installation

Python **3.10+**. Install from source:

```bash
git clone --branch main https://github.com/pengcuix/explain-difficulty.git
cd explain-difficulty
python -m pip install -e '.[irt,hypotheses]'
```

<details>
<summary>Install only one package</summary>

Use `python -m pip install -e '.[irt]'` for IRT only, or
`python -m pip install -e '.[hypotheses]'` for hypothesis discovery only.

</details>

## Input formats

Original question texts and model names are available in [`data/`](data/README.md),
with their response-matrix row and column mappings. Their original field names
differ from the command-line input format below; see the data notes before use.

Include every question from the response matrix. Records are joined by `item_id`,
not file position, so question records may be shuffled.

<details>
<summary>JSON / JSONL</summary>

```json
[
  {
    "item_id": "0",
    "question": "What is 2 + 3?",
    "answer": "5",
    "explanation": "Add the two quantities."
  },
  {
    "item_id": "1",
    "question": "What is 8 - 2?",
    "answer": "6"
  }
]
```

For JSONL, put one object per line without the outer list.

</details>

<details>
<summary>CSV</summary>

```csv
item_id,question,answer,explanation
0,What is 2 + 3?,5,Add the two quantities.
1,What is 8 - 2?,6,Subtract 2 from 8.
```

Use standard CSV quoting for commas or line breaks inside a field.

</details>

`answer` and `explanation` are optional reference fields; when supplied, both are
included in generation and matching. Without custom IDs, `"7"` refers to matrix
row 7 (the eighth question). CSV IDs are read as strings.

## 1. Fit IRT

Start with `responses.npy`: **rows are questions, columns are models**.
Entries are `1` (correct), `0` (incorrect), and `-1` or `NaN` (missing).

```bash
python -m irt.fit --responses responses.npy --epochs 500 --lr 0.1 --seed 0
```

`--responses` specifies the matrix; `--epochs` and `--lr` control fitting;
`--seed` fixes the random seed.

**Output:** `output/irt_results.json` — item difficulties, model abilities and
posterior standard deviations. Larger difficulty means harder.

<details>
<summary>Model, item IDs and output options</summary>

- Hierarchical **1PL**, fitted with py-irt on CPU:
  `P(correct) = sigmoid(ability - difficulty)`.
- Missing responses are excluded. All-missing rows or columns are rejected.
- Default item IDs are zero-based row indices: `"0"`, `"1"`, …
  For custom IDs, pass `--item-ids item_ids.json`, a JSON list in matrix row order.
  `--model-ids model_ids.json` supplies IDs in column order.
- Use `--output path.json` to change the destination. Existing fits are not overwritten.
- Check convergence when choosing the number of epochs. Posterior SD measures
  uncertainty in an estimate, not the spread of difficulties across questions.

</details>

## 2. Discover hypotheses

### 2.1 Configure the API key

```bash
export GEMINI_API_KEY="your-api-key"
```

The API is used for generation, refinement and matching.
IRT, data preparation and Lasso selection run locally. Do not commit your key.

### 2.2 Prepare questions and splits

Provide `items.json`, `items.csv` or `items.jsonl` with `item_id` and `question`
fields ([format examples](#input-formats)). IDs must match the matrix from step 1.

```bash
python -m hypotheses.discover prepare --items items.json --difficulty output/irt_results.json --responses responses.npy --v-size 400 --t-size 200
```

`--items` supplies question text; `--difficulty` supplies fitted IRT parameters;
`--responses` verifies the original matrix. `--v-size` and `--t-size` set the
selection and test sizes; the remaining questions form the generation split.

**Output:** `.cache/prepared.json` — aligned questions, difficulties and G/V/T splits.

<details>
<summary>Split sizes and existing splits</summary>

Splits are disjoint and difficulty-stratified. **G** is used for generation,
**V** for Lasso selection, and **T** is held out from both.

G needs at least **200** questions; V needs at least **14** with the default
selection settings. Adjust the split sizes to your dataset. The default seed is 0;
change it with `--seed`.

To reuse a split, add `--splits splits.json` containing
`{"G": [...], "V": [...], "T": [...]}` with item-ID lists covering every question.
This overrides the split sizes. Use `--output` to change the prepared-file path.

</details>

### 2.3 Generate, refine and match

```bash
python -m hypotheses.discover generate --input .cache/prepared.json --group-rounds 30 --pair-rounds 50 --min-gap 4.0 --execute
```

`--group-rounds` shows 20 harder and 20 easier questions per round;
`--pair-rounds` shows five hard/easy pairs per round.
`--min-gap` sets the minimum pairwise difficulty gap.

**Output:** `output/hypotheses_results.json` — refined hypotheses and their
binary matching matrix, with explicit item and hypothesis IDs.

> Omit `--execute` to validate the sampling plan without making paid API calls.

<details>
<summary>Models, refinement and caching</summary>

- `--propose-model`: generation/refinement model; default `gemini-3.1-pro-preview`.
- `--match-model`: matching model; default `gemini-3.7-flash`.
- `--embedding-model`: local embedding model; default `Qwen/Qwen3-Embedding-4B`.
  Weights download on first use and require sufficient disk space and memory.
- `--threshold`: cosine-similarity threshold for candidate semantic comparisons;
  default **0.85**. Model names can be changed to ones your account supports.
- Exact-text deduplication is followed by strict semantic deduplication,
  semantic-family refinement and fresh matching against all questions.
  The judge does not see IRT difficulty. Prompts are in
  [`hypotheses/prompts.py`](hypotheses/prompts.py).
- Validated API responses are cached in `.cache/hypotheses/`.
  Rerun an interrupted command to reuse them; retries can incur additional charges.
- An infeasible sampling plan stops before API calls. Use `--output` for a new
  result path; generation does not overwrite existing results.

</details>

### 2.4 Select with stability Lasso

```bash
python -m hypotheses.discover select --input output/hypotheses_results.json --top-k 20 --repeats 100 --sample-fraction 0.75 --cv-folds 5
```

`--top-k` limits the number retained; `--repeats` controls the number of fits.
Each fit samples `--sample-fraction` of V and uses `--cv-folds` to choose its
Lasso regularization weight.

**Output:** selection flags, ranks and frequencies are added to the **same**
hypothesis result file. This step makes no API calls.

<details>
<summary>Selection rule and rerunning selection</summary>

Each fit chooses alpha from 25 log-spaced values between `1e-4` and `1`.
Standardization is fitted inside each CV fold. Rank by nonzero-coefficient frequency,
then mean absolute standardized coefficient, then original order.
Retain at most `--top-k` hypotheses with nonzero frequency.

Only V is used. Choose `--top-k` yourself; it is not optimized on T.
The default seed is 0 (`--seed`).

The result adds `selected`, `rank` and `selection_frequency` without changing
hypothesis order or matrix alignment. The previous JSON is backed up in
`.cache/hypotheses/`. Use `--output new_result.json` to save separately.
Selection can be rerun without regenerating or rematching hypotheses.

</details>

## Outputs

- **`output/irt_results.json`** — IRT parameters and input provenance.
- **`output/hypotheses_results.json`** — hypotheses, matches, splits and selection.

Prepared data, cached API responses and selection backups stay in `.cache/`,
which is ignored by Git. Do not run concurrent commands against the same output
or cache directory.

<details>
<summary>Development and tests</summary>

```bash
python -m pip install -e '.[irt,hypotheses,test]'
python -m pytest -q
```

Tests use synthetic data and mocked API responses; no paid API calls are made.
Source packages: [`irt/`](irt/) and [`hypotheses/`](hypotheses/).

This is the consolidated 30+50 workflow. Historical run settings and external
model outputs can differ.

</details>
