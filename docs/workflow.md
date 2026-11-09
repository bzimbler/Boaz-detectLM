# Detector workflow

## 1. Collect null responses

The intended calibration input is sentence log-perplexity measured on model-generated text using the same model and context policy as inference. `many_atomic_detections.py` can read supported external datasets (`wiki`, `wiki-long`, `news`, `news-long`, `abstracts`) or `.txt` files from a path prefix, parse sentences, and write an atomic CSV. Its `--context` flag selects previous-sentence context; without it the parser uses no context. Its columns are described in [interfaces](interfaces.md#atomic-response-csv).

This is currently **not a reproducible fresh-setup step**: the CLI passes a fixed embedded credential during model loading and has no user authentication argument. A separate code change and verification are needed before giving a runnable command for new users. External datasets and model weights are not bundled.

## 2. Fit a survival function

The detection CLIs read the selected null CSV and call `fit_per_length_survival_function` from `fit_survival_function.py`. That function fits the upper tail of sentence response by recorded length; `number-of-interpolation-points` controls its response grid. There is no separate fitting CLI or saved fitted model. The fitted function is constructed when a detector starts. If `ignore-first-sentence` is enabled, rows with `num <= 1` are removed first.

The resulting sentence p-values are meaningful only when null data is compatible with scored sentences. The code does not establish that compatibility for newly generated atomic output; see [current limits](#current-limits).

## 3. Run inference with compatible null data

With a validated null CSV at the configured path, the single-document interface is:

```sh
python text_detect.py -i example.txt -conf conf.yml
python text_detect.py -i example.txt -conf conf.yml --context
```

Use the second form only when the selected context null data matches previous-sentence scoring. Input must be a `.txt` file. The script writes `out.csv` in the working directory, prints HC and Fisher statistics, and prints edit-tag diagnostics. Those diagnostics can be undefined for untagged or zero-edit input; they are not an accuracy estimate. `out.csv` is overwritten on each invocation.

For batch input, this is a synthetic JSON shape:

```json
{
  "synthetic-1": {
    "text": "A short synthetic document. Another synthetic sentence follows.",
    "eps": 0.0
  }
}
```

With compatible null data, `python many_text_detect.py -i examples.json -o results.json -conf conf.yml` reads that mapping. Add `--context` only with matching previous-sentence calibration. If `-o` is omitted, the script chooses `results/results_<input-stem>.json`; the parent directory must exist. The script may prompt when the input path is not a substring of the selected null-data path. See [batch output](interfaces.md#batch-json) and its zero-discovery failure. This example specifies input format; it does not imply the sample will produce completed output.

`generate_mixtures.py` can produce the mapping from supported dataset names (`wiki`, `wiki-long`, `news`, `news-long`). It accepts `-i`, `-o`, `-eps`, `-n`, `-no-sentences`, and `--random`. Generated entries also include `source` and `appx_no_sents`; batch detection reads `text` and `eps`.

## Current limits

- No null CSV is checked in; `results/` is ignored. The default configuration does not establish a fresh working inference path.
- `PrepareSentenceContext` records character counts as `length` in atomic CSV data, while `DetectLM` chooses its survival function using whitespace word counts. Resolve this mismatch or verify a compatible source before treating results as calibrated.
- The atomic CLI cannot be presented as a fresh setup command until its fixed credential handling is replaced and verified.
- A batch document with zero HC discoveries divides by zero before its result is saved, potentially leaving JSON absent or incomplete. A document with no valid scored sentences is skipped instead.
- Model and dataset availability, runtime dependencies, and complete inference have not been verified for a clean checkout. See [setup](setup.md) and [interface details](interfaces.md).
