# Explain Difficulty

Two packages: estimate question difficulty from model responses, then discover
natural-language hypotheses explaining that difficulty.

## Install

Python 3.10 or newer:

```bash
python -m pip install -e '.[irt,hypotheses]'
```

Install `.[irt]` or `.[hypotheses]` separately if only one stage is needed.
The hypothesis package uses the Gemini API and a local sentence embedding model.
Set `GEMINI_API_KEY` in your environment; never put credentials in input files.
The default embedding model is `Qwen/Qwen3-Embedding-4B`, downloaded on first use.
It requires sufficient disk space and memory; use `--embedding-model` to override.
Model identifiers are configurable because API availability can change.

## 1. Fit IRT

Input: a NumPy `.npy` matrix with **questions in rows, models in columns**.
Entries are `0` (incorrect), `1` (correct), and `-1` or `NaN` (missing).
Missing responses are excluded, not treated as incorrect. All-missing rows/columns
and non-binary observed responses are rejected. Optional ID files are JSON lists
in matrix order. Without them, IDs are zero-based index strings.

```bash
fit-irt --responses responses.npy \
  --item-ids item_ids.json --model-ids model_ids.json
```

Output: **`output/irt_results.json`**, with `items` (ID, difficulty, posterior SD),
`models` (ID, ability, posterior SD), fit settings, and input checksum.
The hierarchical 1PL model uses `P(correct) = sigmoid(ability - difficulty)`.
We use py-irt's stochastic variational inference, default 500 epochs, CPU, seed 0.
Means and SDs come from the same final variational iterate. Posterior SD is not
the standard deviation of difficulties across questions. Larger difficulty is
harder; the scale is relative to the fitted response panel. Fit separate benchmarks
separately unless a common scale is intentionally desired. Inspect fit quality
and sensitivity to epochs; a finite fit alone does not establish convergence.

## 2. Discover hypotheses

Input `items.jsonl`: one JSON object per line with a unique `item_id` and `question`.
Optional `answer` and `explanation` fields are included in generation and matching.
Use complete reference answers/explanations when available; the package does not
generate them. IDs are normalized to strings and joined by ID, never by position.

```json
{"item_id": "0", "question": "What is 2 + 3?", "answer": "5", "explanation": "Add the two quantities."}
```

Difficulty can come from the IRT output or an external JSON list of
`{"item_id": "0", "difficulty": 1.2}` records on the same scale.

```bash
# Validate input and sampling before making any paid calls.
discover-hypotheses --items items.jsonl --difficulty output/irt_results.json

# Run; completed valid API responses are reused automatically after interruption.
discover-hypotheses --items items.jsonl --difficulty output/irt_results.json --execute
```

Default protocol:

1. Difficulty-stratified, disjoint G/V/T splits, with V=400 and T=200; G is the
   remainder. Override sizes or pass `--splits splits.json` containing G/V/T ID
   lists covering exactly the input items. Group sampling requires at least 20
   G items in each of ten rank buckets (at least 200 G items).
2. **30 group rounds**, each showing 20 hard and 20 easy items. Bucket gaps are
   at least 3; bucket-pair weights are `gap^2/(1+previous_count)`, and within-bucket
   item weights are `1/(1+previous_count)`; no within-round replacement.
3. **50 pair rounds**, each showing five disjoint hard/easy pairs, minimum IRT
   gap 4.0 (`--min-gap`). Pairs are not topic-matched or bucket-sampled. Pair
   weights penalize previous item use with exponent 2 and pair use with exponent 1.
   Sampling plans are determined before generation. Total: **1,700 item
   presentations**, not necessarily 1,700 distinct questions. If no valid plan
   exists, execution stops rather than silently lowering the difficulty gap.
4. Exact-text deduplication, then strict semantic deduplication over all
   embedding-cosine pairs above 0.85, with joint component review retaining
   unchanged representatives. The survivors undergo semantic-family pair review
   and joint refinement using the same cosine threshold. Distinct criteria remain
   separate; redundant classes keep an original or receive one rewritten criterion.
   Similarity/transitivity alone never deletes a hypothesis. No item labels or test
   metrics guide refinement.
5. Every final hypothesis is freshly matched against every input question in
   chunks of 40, producing strictly binary labels. Difficulty is hidden from
   matching. Rewritten hypotheses are not assigned their predecessors' labels.
6. Stability selection on **V only**: 100 subsamples of 75% of V, five-fold CV
   choosing Lasso alpha from 25 log-spaced values between 1e-4 and 1, with scaling
   fitted inside each fold. Rank by selection frequency, breaking ties by mean
   absolute standardized coefficient, then original order. Select the first 20
   with nonzero frequency (`--top-k`). G/T labels are exported but not used for
   selection. Top-k is fixed by the user, not optimized on T.

Output: **`output/hypotheses_results.json`** contains refined hypothesis texts,
ranking, selection flags, raw/distinct/refined counts, splits, settings, and a
binary matching matrix with explicit row item IDs and column hypothesis IDs.
Raw counts include repeated generated statements. `--debug` additionally embeds
raw candidates, joint refinement decisions, and the sampling plan in that file.
No separate intermediate result files are produced.

Validated API responses live in **`.cache/hypotheses/`**, excluded from Git.
Rerun the same command to reuse them. Requests are keyed by full prompt and model
settings, so changed inputs do not reuse stale matches. Keep this cache until a
run finishes; removing it may cause additional API charges. Retries can incur
charges, including when a server processes a request but the client loses its
response. Output files are never overwritten; choose a new `--output` path.
Do not run concurrent processes against the same cache/output directory.

Default models: `gemini-3.1-pro-preview` for generation/refinement and
`gemini-3.7-flash` for matching, temperature 0, low thinking. Override with
`--propose-model` / `--match-model`. All prompts are in `hypotheses/prompts.py`.
There is no automatic model fallback. API calls are serial, validated, and retried
up to three times. A dry run reports generation calls and presentation counts;
total matching/refinement cost depends on the generated pool and is not estimated.

## Scope and reproducibility

This is the compact **30+50 method implementation**, not a byte-for-byte reproduction
of every historical paper run. Historical experiments differed in round budgets,
candidate-pool provenance and selection settings. This release retains text-only
strict deduplication followed by semantic-family refinement, without the historical
manual recovery/re-audit entry points; prompts are consolidated for arbitrary datasets.
The sampling seed is reproducible, but external model outputs can vary even at
temperature zero. Baselines, question rewriting, plotting, benchmark data, cached
API responses, and private infrastructure scripts are not distributed.

## Tests

```bash
python -m pip install -e '.[irt,hypotheses,test]'
python -m pytest -q
```

Tests use a small synthetic response matrix and mocked generation/refinement/
matching responses; they do not invoke paid APIs or download embedding weights.
