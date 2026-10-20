# Development notes

Notes for people working on the code. Several numbers here are recomputed by
`tests/test_doc_derived_numbers.py`, which fails when they drift: the suite
counts, the skip breakdown, the line count of `docs/limitations.md` and the
incident count. Update them in the same commit as the change that moves them.
These notes moved here from `README.md` on 19 Oct 2026, when the README was
rewritten for a public reader.

## Running the tests

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,eval]"
.venv/bin/python -m pytest tests/ -q     # 1418 passed, 68 skipped, no GPU
```

The `[dev,eval]` extras are required: four test modules import `transformers`
at collection time, so a bare `pip install -e .` cannot collect the suite. The
`kernels` extra is separate because those kernels compile from source and are
architecture-gated, and the harness must import on a machine where a kernel is
unavailable.

The 68 skips are the honest part, and they split five ways: **14** need CUDA,
**45** read banked result files that `results/` keeps out of git, **4** are
scripts `test_script_call_sites.py` has nothing to check because they import
nothing from `attnbench`, **4** are the same scripts skipped again by
`test_import_resolution.py`, and **1** needs the pinned RULER data directory
(`scripts/fetch_ruler_data.py`, found through `$ATTNBENCH_RULER_DATA` or
`~/.cache/attnbench/ruler_data`). Each skip names the file or device it wanted.

Those counts are for a fresh clone on a machine without the RULER data.
**With `results/` present the suite reads 1464 passed, 22 skipped**, because
the 45 banked-file tests run instead of skipping and, on a machine with the
RULER data directory, the data test runs too. Both totals are 1486, which is
what `pytest --collect-only` reports, and
`test_the_suite_counts_match_the_collected_suite` asserts it. At this commit,
with the full `results/` tree present, one of those tests fails:
`tests/test_eras.py::test_the_register_agrees_with_the_documented_era_table`,
because an era listed in `docs/limitations.md` is not yet registered in
`attnbench/analysis/eras.py`.

To measure the fresh-clone count on a machine that has the RULER data, point
`ATTNBENCH_RULER_DATA` at an empty directory. An empty string is not the same
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
