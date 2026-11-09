# DetectLM

DetectLM scores sentences against a causal language model and combines calibrated sentence p-values into document-level Higher Criticism (HC) and Fisher statistics. It is research code for studying mixed human and model authorship, not a ready-to-use authorship verdict. Meaningful results require null data from the same model and context policy as the tested document.

Start with [setup](docs/setup.md), then read the [detector workflow](docs/workflow.md) and [interfaces and output reference](docs/interfaces.md).

| Entry point | Purpose | Current prerequisite or limit |
| --- | --- | --- |
| `text_detect.py` | Score one `.txt` document; write `out.csv` and print document statistics. | Requires compatible null CSV data at the path selected by `conf.yml`. None is checked in. `--context` inference is blocked by the context-label masking bug. |
| `many_text_detect.py` | Score a JSON map of documents with `text` and `eps` fields; write JSON results. | Requires compatible null data. `--context` inference is blocked by the same masking bug. A processed document with zero HC discoveries raises division by zero and may leave output incomplete. |
| `many_atomic_detections.py` | Collect sentence responses for null-data research. | **Blocked as a reproducible fresh-setup CLI:** model loading uses a fixed embedded credential and the CLI has no way to supply the user's own authentication. |
| `generate_mixtures.py` | Generate tagged mixed documents from supported external datasets. | Requires dataset access; its JSON can be input to the batch detector. |
| `fit_survival_function.py` | Library functions for fitting response survival functions by length. | Imported by the detectors; not a standalone fitting CLI. |

The default `conf.yml` points to ignored `results/` files absent from this checkout. Atomic output records sentence **character** length while inference uses **whitespace word** length. Resolve that mismatch or verify a compatible data source before claiming a reliable fresh null-data-to-inference run. See [workflow limits](docs/workflow.md#current-limits).
