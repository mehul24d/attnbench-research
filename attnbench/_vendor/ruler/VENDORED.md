# Vendored from NVIDIA/RULER

Source: https://github.com/NVIDIA/RULER
Commit: `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`
Fetched: 2026-09-02
License: Apache 2.0 (original headers preserved in each file)

## What's here and how faithful it is

| File | Source file | Status |
|---|---|---|
| `scoring.py` | `scripts/eval/synthetic/constants.py` | **Verbatim.** No argparse, no globals, no external deps in the original -- copied unmodified. |
| `niah.py` | `scripts/data/synthetic/niah.py` | **Adapted.** Same needle/haystack algorithm, restructured from a CLI script (`argparse.parse_args()` at import time, module-level `args`/RNG globals) into plain functions with explicit parameters and a local `random.Random(seed)`. |
| `variable_tracking.py` | `scripts/data/synthetic/variable_tracking.py` | **Adapted**, same reasons. Few-shot/ICL example machinery (`add_fewshot`) dropped -- an orthogonal CLI convenience feature, not part of core task construction. |

## Why "unmodified, import-path fixes only" (the original plan) didn't hold

The generator scripts execute `argparse.parse_args()` when *imported*, not
just when run as `__main__` -- there is no way to `import` them without a
real command line. They also read config from a module-level `args`
Namespace and seed the global `random`/`numpy.random` state once at import
time, rather than taking parameters. None of that survives becoming a
library call.

## What's deliberately not implemented (and why)

- **`haystack_mode="essay"`** (niah.py, variable_tracking.py): RULER's
  "essay" haystack needs NLTK (`sent_tokenize`) and a separately-downloaded
  Paul Graham essay corpus (`json/PaulGrahamEssays.json`, fetched by a
  script that isn't part of this repo, not a vendorable data file). Both
  generator functions accept `haystack_mode` and raise `NotImplementedError`
  on `"essay"` rather than silently falling back to `"noise"` -- the seam is
  there (see `attnbench/accuracy/ruler.py`), just not wired. If GPU compute
  frees up later, running one context length in essay mode is a cheap
  calibration point for how much the noise-haystack substitution shifted
  results, turning this from a limitation into a controlled comparison.
- **`type_needle="words"`** (niah.py): needs the `wonderwords` package's
  noun/adjective word lists. Only `"numbers"` and `"uuids"` are implemented
  (both stdlib-only). Raises `NotImplementedError` on `"words"`.
- **`common_words_extraction` (CWE)**: the entire task needs `wonderwords`
  for its word pool -- not implemented at all yet, not just narrowed.
- **`qa` (QA)**: needs a real downloaded QA dataset (SQuAD/HotpotQA context
  passages) that RULER's own scripts fetch separately. Not implemented.

## Consequence for comparability

Every result produced by this vendored code uses RULER's **task
construction algorithm** with a **noise-haystack substitution** in place of
real prose, plus (for NIAH) numeric/UUID needles rather than word needles.
This is a real methodological difference from RULER's own published setup,
not just a fidelity footnote -- filler text is a measurably easier haystack
to search than real prose, so absolute accuracy numbers from this harness
will likely read higher than published RULER numbers and are **not
comparable to them**. They remain valid for what this study actually needs:
comparing attention backends against each other on identical inputs. See
`schema.AccuracyResult.haystack_mode` -- it's a first-class field precisely
so this substitution travels with the data, not just this file.
