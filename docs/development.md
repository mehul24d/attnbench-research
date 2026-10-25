# Development notes

Notes for people working on the code. Several numbers here are recomputed by
`tests/test_doc_derived_numbers.py`, which fails when they drift: the suite
counts, the skip breakdown, the line count of `docs/limitations.md` and the
incident count. Update them in the same commit as the change that moves them.
These notes moved here from `README.md` on 19 Oct 2026, when the README was
rewritten for a public reader.

## Running the tests

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,eval,learned]"
.venv/bin/python -m pytest tests/ -q     # 3668 passed, 98 skipped, no GPU
```

The `[dev,eval]` extras are required: four test modules import `transformers`
at collection time, so a bare `pip install -e .` cannot collect the suite. The
`kernels` extra is separate because those kernels compile from source and are
architecture-gated, and the harness must import on a machine where a kernel is
unavailable. The `learned` extra installs LightGBM, pinned at 4.6.0, for the
learned baseline (T2.10); without it one more test skips.

The 98 skips are the honest part, and they split six ways: **15** need CUDA,
**65** read banked result files that `results/` keeps out of git, **7** are
scripts `test_script_call_sites.py` has nothing to check because they import
nothing from `attnbench`, **7** are the same scripts skipped again by
`test_import_resolution.py`, **2** need the pinned RULER data directory
(`scripts/fetch_ruler_data.py`, found through `$ATTNBENCH_RULER_DATA` or
`~/.cache/attnbench/ruler_data`), and **2** need the pinned Llama tokenizer in
the local Hugging Face cache. Each skip names the file or device it wanted.

Those counts are for a fresh clone on a machine without the RULER data or
the tokenizer.
**With `results/` present the suite reads 3737 passed, 29 skipped**, because
the 65 banked-file tests run instead of skipping and, on a machine with the
RULER data directory and the tokenizer, those four tests run too (with `results/` present and those two hidden it reads
3733 passed, 33 skipped). All three totals are 3766, which is
what `pytest --collect-only` reports, and
`test_the_suite_counts_match_the_collected_suite` asserts it.

To measure the fresh-clone count on a machine that has the RULER data and the
tokenizer, point `ATTNBENCH_RULER_DATA` and `HF_HOME` at empty directories. An empty string is not the same
as unset.

## Earlier work in this repository

The repository started as a benchmark of sparse attention speedups at matched
accuracy, with oracle masks, on L4, A100 and H100 machines. That work is
banked, commit-stamped evidence that motivates the current study; it is not
the study's confirmatory evidence. Its audit history is kept and annotated in
place, never rewritten:

- [`docs/claims.md`](claims.md): one row per claim the earlier measurements
  support, with the paraphrase each must not become.
- [`docs/limitations.md`](limitations.md) is over 2,700 lines: what the
  earlier measurements cannot answer, and the mask-rule eras that must not be
  pooled.
- [`docs/silent_failure_patterns.md`](silent_failure_patterns.md):
  56 confirmed incidents of a plausible number produced by machinery that
  looked like it was working, each with how it was detected.
- [`docs/withdrawn_figures.md`](withdrawn_figures.md) and
  [`docs/audit_register.md`](audit_register.md): withdrawn figures and audit
  dispositions. `tests/test_no_stale_figures.py` fails if a withdrawn figure
  appears unmarked in `README.md`, these notes, the other top-level documents
  in `docs/`, the brief's source or the code.
