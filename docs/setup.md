# Setup and prerequisites

These instructions describe the environment needed on a machine with model and dataset access. They are not a verified end-to-end run for this checkout.

1. Use a separate Python environment and install declared dependencies with `python -m pip install -r requirements.txt`. The requirements include the `en_core_web_sm` spaCy model, which the default sentence parser loads by name. Detection CLIs also import `yaml`, but `PyYAML` is not explicitly declared in `requirements.txt`; ensure it is available before using them.
2. Arrange access to the causal model selected by `language-model-name` in `conf.yml`. Hugging Face model and tokenizer loading can require network access, permission, and substantial memory. The checked-in selection is `gpt2-xl`; another model's null responses are not interchangeable.
3. Supply compatible null-response CSV data locally and set the appropriate `context-null-data-file` or `no-context-null-data-file` path in `conf.yml`. The path is read as a glob. The checked-in paths point under ignored `results/`, so a clean checkout has no bundled null data. The CSV needs `length` and `response` (or `logloss`); when `ignore-first-sentence` is true it also needs `num`.
4. Choose the context policy: omitting `--context` uses no context. The flag selects previous-sentence context, but inference with it is currently blocked by a context-label masking bug. Null data, model, sentence parsing, length units, and inference context policy must agree. Read the [workflow](workflow.md) before interpreting results.

`conf.yml` also sets `ignore-first-sentence`, `number-of-interpolation-points`, `min-tokens-per-sentence`, `max-tokens-per-sentence`, and `hc-type`. The detectors use these for filtering, survival-function fitting, sentence length limits, and the HC variant. `text_detect.py` ignores the first sentence only with `--context`; `many_text_detect.py` does so when `ignore-first-sentence` is true or `--context` is supplied.

External datasets used by `generate_mixtures.py` and `many_atomic_detections.py` need network access and any applicable permissions. The atomic CLI currently contains a fixed model-loading credential with no CLI authentication option; it is blocked as a reproducible fresh-setup command. This guide provides no command to run it and no workaround using the embedded value.
