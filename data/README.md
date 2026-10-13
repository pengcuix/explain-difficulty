# Original data

These files retain their original names and contents. They cover the full
exported benchmark collection, not only the three datasets used in the paper.

| File | Contents | Size |
| --- | --- | --- |
| [`all_response_matrix_q_by_m.npy` (Google Drive)](https://drive.google.com/file/d/1IDf2ekLvq6U9o28Yiu8dXmxj_vMCRpHo/view?usp=drivesdk) | 50,265 × 8,577 response matrix | 1,724,491,748 bytes |
| [`all_benchmarks_dataset_new.json`](all_benchmarks_dataset_new.json) | 50,265 question records | 39,621,075 bytes |
| [`column_to_llm.json`](column_to_llm.json) | 8,577 model names, indexed by matrix column | 400,032 bytes |

Use the **Download** button on the Google Drive page for the response matrix;
the two JSON files are included in this repository. Save the downloaded matrix
in this directory without changing its filename.

## Matrix alignment

The corresponding response matrix is `all_response_matrix_q_by_m.npy`:

- Shape: **50,265 questions × 8,577 models**; dtype: `float32`.
- Original file size: **1,724,491,748 bytes** (about 1.61 GiB); it is not stored in Git.
- Row `i` corresponds to question record `i`, whose `id` is also `i`.
- Column `j` corresponds to `column_to_llm.json[str(j)]`. The mapping is a JSON
  object, not a list; order columns numerically, not lexicographically.
- Values are `1` for correct, `0` for incorrect and `-1` for missing responses,
  except that TruthfulQA contains fractional MC2 scores. Preserve these values;
  do not cast the entire matrix to an integer type.

## Question fields and preprocessing

Question text is in `problem`; `source` identifies the benchmark or subtask.
The original `subject`, `level` and `solution` fields may contain the literal
string `"None"`; this is missing metadata, not a reference answer.

For the hypothesis-discovery input format, map `id` to a string `item_id` and
`problem` to `question`. Do not pass placeholder `solution` values as answers.
For `--model-ids`, convert the column mapping to a JSON list in numeric column
order. This preserves model identities while matching the command-line schema.

Select the desired benchmark rows before fitting IRT and remove model columns
with no observed responses in that subset. Keep item IDs and model names aligned
with the selected rows and columns. The current binary IRT fitter accepts only
`0`, `1` and missing values, so the unfiltered matrix containing fractional
TruthfulQA scores is not a valid input to `irt.fit`.
