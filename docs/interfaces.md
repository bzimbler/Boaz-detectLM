# Interfaces and output reference

## Python interfaces

| Symbol | Inputs | Return or effect |
| --- | --- | --- |
| `PrepareSentenceContext(engine='spacy', context_policy=None, context=None)` | Call with document text. `engine='regex'` is also implemented; `context_policy='previous_sentence'` uses the prior parsed sentence, and explicit `context` overrides the policy. | Dictionary of equal-length lists: `text` (sentence strings), `length` (character counts), `context` (strings or `None`), `tag` (current HTML-like tag or `None`), and `number_in_par` (running count of non-tag sentences). Requires `en_core_web_sm` for the default engine. |
| `PerplexityEvaluator(model, tokenizer)` | Call with a sentence and optional context. | Model loss as a log-perplexity response. Context tokens are intended to be masked from scored labels, but the default mask value is incompatible with the causal model loss; context scoring is currently blocked. |
| `DetectLM(sentence_detection_function, survival_function_per_length, min_len=1, max_len=100, HC_type='stbl', length_limit_policy='truncate', ignore_first_sentence=False)` | Call with matching lists of sentence strings and contexts. The survival function is called as `(whitespace_word_length, response)`. | Dictionary with `sentences` DataFrame, `HC`, `fisher`, and `fisher_pvalue`. Sentences below `min_len` are ignored; overlong sentences follow `truncate`, `ignore`, or `max_available`. |

The `sentences` DataFrame contains `sentence`, `response`, `pvalue`, `context`, `comment`, and `mask`. Ignored sentence responses and p-values are `NaN`. When there are no valid scored sentences, HC and Fisher values are `NaN` and masks are missing. Otherwise `mask` is true when a sentence p-value is at or below the threshold returned by HC. It identifies candidate discoveries, not proven human authorship.

`PrepareSentenceContext(context_policy='previous_sentence')` supplies a nonempty context after the first parsed sentence. `PerplexityEvaluator` uses `ignore_index=-1` for context labels, while the causal model loss expects `-100`; do not use the detector CLIs with `--context` until this is corrected and verified. The [workflow](workflow.md#current-limits) records this limit.

HC is the combined Higher Criticism statistic; `fisher` is Fisher's combined statistic and `fisher_pvalue` is its returned p-value. Their interpretation depends on suitable null calibration. The scripts do not produce a validated yes/no authorship verdict. `text_detect.py` adds `tag` to the DataFrame, writes it with the DataFrame index to `out.csv`, and prints document statistics.

## Atomic response CSV

`many_atomic_detections.py` writes a CSV with a pandas index plus these columns:

| Column | Meaning |
| --- | --- |
| `num` | Sentence number within its source document, starting at 1. |
| `length` | Character count of the parsed sentence, **not** the word count used during inference. |
| `response` | Model loss returned by `PerplexityEvaluator` for that sentence and context. |
| `context_length` | Whitespace word count of context, or 0 without context. |
| `name` | Source document ID or file path. |

The CSV is written incrementally by the atomic workflow. It is not known to be directly compatible with inference because of the length-unit mismatch. See [workflow limits](workflow.md#current-limits).

## Batch JSON

`many_text_detect.py` reads a JSON object keyed by document ID. Every processed entry must have `text` (document string) and `eps` (theoretical edit rate). `generate_mixtures.py` also writes `source` and `appx_no_sents`, which the detector does not require. A processed result is keyed by the same document ID:

| Field | Meaning |
| --- | --- |
| `no_discoveries` | Number of true values in the HC sentence mask. |
| `empirical_edit_rate` | Fraction of parsed sentences tagged `<edit>`. |
| `theoretical_edit_rate` | Input `eps`. |
| `true_discoveries` | Masked sentences tagged `<edit>`. |
| `false_discoveries` | Masked sentences tagged `not edit`. |
| `recall` | `true_discoveries` divided by the number of tagged edits; `NaN` when there are no tagged edits. |
| `FDP` | `false_discoveries / no_discoveries`. |
| `TDP` | `true_discoveries / no_discoveries`. |
| `hc` | Document HC statistic. |
| `fisher_pval` | Fisher's returned p-value. |

For a document with no valid scored sentences (`HC` is `NaN`), the script skips the document. For a processed document with **zero HC discoveries**, `FDP` and `TDP` divide by zero before a result is written. The script raises an error; because it writes results after each processed document, its output may be empty or contain only earlier documents. Do not rely on complete batch output in that case.

The JSON output reports aggregate fields only; inspect the `DetectLM` DataFrame or single-document CSV for per-sentence response, p-value, and mask. See [setup](setup.md) for calibration prerequisites.
