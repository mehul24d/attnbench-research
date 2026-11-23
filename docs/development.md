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
.venv/bin/python -m pytest tests/ -q     # 7016 passed, 187 skipped, no GPU
```

The `[dev,eval]` extras are required: four test modules import `transformers`
at collection time, so a bare `pip install -e .` cannot collect the suite. The
`kernels` extra is separate because those kernels compile from source and are
architecture-gated, and the harness must import on a machine where a kernel is
unavailable. The `learned` extra installs LightGBM, pinned at 4.6.0, for the
learned baseline (T2.10); without it one more test skips.

torch must be below 2.14 for the CPU suite, and the `[dev]` extra pins it so
(`torch<2.14`, the adviser's ruling 4, second set of 12 Nov 2026). Three compiled-variant tests
(`test_compiled_variants.py`'s XAttention report and both settings of
`test_shakedown6_rulings.py`'s declared kernel exclusion) read the declared
graph break by Dynamo's message, and torch 2.14 rewords it at the same place
(`block_sparse.py` in `kernel_interface`): "Call to `torch.compiler.disable()`"
for "Attempted to call function marked as skipped". Without the pin, `torch>=2.6`
resolves to 2.14.1 today, and those three fail. They pass on 2.13.0 and on the GPU
images' 2.9.1+cu129, where the declared wording was read on the L4. On a newer
torch the check refuses the variant, so it fails closed (Appendix A, 12 Nov
2026). The counts below were measured with torch 2.13.0 and the `[eval]`
extra's `wonderwords`, which one RULER test needs.

The 187 skips are the honest part, and they split eleven ways: **56** need CUDA
or the GPU image's x-attention and block-sparse-attn, or the pinned x-attention clone
(`$ATTNBENCH_XATTN_DIR`, for the tests behind the CPU boundary), **86** read banked result
files that `results/` keeps out of git, **17** are scripts
`test_script_call_sites.py` has nothing to check because they import nothing
from `attnbench`, **17** are the same scripts skipped again by
`test_import_resolution.py`, **3** need the pinned RULER data directory
(`scripts/fetch_ruler_data.py`, found through `$ATTNBENCH_RULER_DATA` or
`~/.cache/attnbench/ruler_data`; one of them wants a pinned build; a fourth
skips without `wonderwords`, which the `[eval]` extra installs), **2** need the pinned Llama tokenizer in the local Hugging
Face cache, **1** needs the primary model's pinned `config.json` there, and **1** is the pre-launch CPU pass (`tests/test_prelaunch_cpu_pass.py`), which runs only from `scripts/prelaunch_cpu_pass.sh`, **1** needs P4's local scoring data (the H100 diagnostic's Q4 recursion test, through the `H100_DIAG_Q4_*` variables), and **2** need a clean Python 3.10.12 (the environment build's pre-compile steps, through `$ATTNBENCH_CLEAN_PY310` or `uv python find 3.10.12`), and **1** needs P1's banked T3.2d prediction under the home directory (the P3 transfer prediction's check against its source). Each skip names the file or device it wanted.

Those counts are for a fresh clone on a machine without the RULER data or
the tokenizer. Three more skip than the `-r3` counts said (182 fresh, 97 hidden):
`f38a625` gives the same 185, skip for skip, on the machine that measured these, so
the change is the environment's, not the code's (13 Nov 2026). At `lock-2026-11-14-r2` the
Mac has the RULER data and the tokenizer in `~/.cache`, so the fresh count is measured with
`HOME` pointed at an empty directory (14 Nov 2026).
**With `results/` present the suite reads 7107 passed, 96 skipped**, because
the 86 banked-file tests run instead of skipping (all but three do: the third, `test_script_usage_paths`, skips while the Mac's `results/` has no `p1_training/`) and, on a machine with the
RULER data directory and the tokenizer, those tests run too (with `results/`
present and those two hidden it reads 7065 passed, 102 skipped). All three
totals are 7203 (the rulings after the rehearsal add 17 that pass with no data, derived from the collection, not re-measured on a fresh clone: 8 for the AWS bootstrap script, 8 for the preflight's accepted L4 hosts and 1 script-parametrized check of the bootstrap script, 23 Nov 2026; the g6.xlarge rehearsal adds 30 that pass with no data, derived from the collection, not re-measured on a fresh clone: 23 for the rehearsal's checks and instance script, 2 for the launcher's REH settings and 5 script-parametrized checks of the two new scripts, 23 Nov 2026; the env-build follow-ups add 14, derived from the collection, not re-measured on a fresh clone: 4 script-parametrized checks of the P3 prediction script that pass with no data, 1 that passes with no data for the teardown's stopped instance with an empty DNS, and 9 for the P3 transfer prediction, 8 that pass with no data and 1 that skips without P1's banked prediction under the home directory, counted here as skipped, 23 Nov 2026; the env-build merge with main, 23 Nov 2026, adds main's 15 below; the interpreter wave adds 36 that pass with no data, derived from the collection, not re-measured on a fresh clone: 32 for the bare-interpreter scan of the build and instance scripts and 4 for the dry run's metadata steps and the two-interpreter image list, 22 Nov 2026; the dry-run wave adds 9, derived from the collection, not re-measured on a fresh clone: 7 that pass with no data for the memory guard's /proc/meminfo read, the real-error reporting and --stop-before-compile, and 2 that run the pre-compile steps on a clean Python 3.10.12 and skip without one, counted here as skipped, 22 Nov 2026; the image probe adds 20 that pass with no data, derived from the collection, not re-measured on a fresh clone: 8 for the launcher's PROBE pairing, 11 net for the build script's cuobjdump search on fixtures from the probe's inventory and 1 script-parametrized check of the inventory script, 22 Nov 2026; the environment build's fifth wave adds 7 that pass with no data: 5 for the script's FA3 manifest, nvcc, cubin and import checks and 2 for the card's launcher settings, 21 Nov 2026; the AWS environment build adds 44 that pass with no data: 39 for its script and list helper, 4 script-parametrized checks of the helper and 1 launcher-tag check of the script, 21 Nov 2026; the 22 Nov 2026 residual split adds 15 that pass with no data: its 10 and 5 file-parametrized checks; the P6 host-effect rule adds 21 that pass with no data: its 17 and 4 script-parametrized checks, 21 Nov 2026; the AWS launch-policy and launcher tests add 9 that pass with no data: 8 for rule 12's Sydney lift, PR #91, and 1 for the ENV build target, 21 Nov 2026; the exploratory wave adds 47 that pass with no data: its 37 and 10 file-parametrized checks, 17 Nov 2026; the H100 diagnostic adds 73: its 68, one of which skips without P4's local data, and 5 file-parametrized checks, 17 Nov 2026, six of the 68 added with the card's 17 Nov rule edits and three with the rerun's launch-table fix), which is
what `pytest --collect-only` reports, and
`test_the_suite_counts_match_the_collected_suite` asserts it. T4.3's 36 reproduction tests all pass, none marked (`docs/t43/import_edits.md`).

**Known flaky test (16 Nov 2026):** `tests/test_shakedown3_cpu.py::test_the_chain_check_holds_the_composites_against_the_isolated_replay`
can fail in a full-suite run under load and passes when run alone. It is off the scoring path, so it is recorded here and left unchanged (Appendix A, 16 Nov 2026).

**Known flaky test (17 Nov 2026):** `tests/test_session_exit.py::test_a_hang_up_stops_the_session_then_syncs_and_halts`
timed out once (its 60 s wait) in a full-suite run. Rerun alone five times under artificial CPU load (one busy loop per core, load average 61 to 131), all ten of the file's tests passed every time (adviser's rulings of 17 Nov 2026, item 2). It is recorded here and left unchanged.

On 16 Nov 2026 (scoring wave) the fresh count was re-measured at 6650 passed and 184 skipped, plus one test added afterwards. The run had no `wonderwords`, so the extra skip is its RULER test. The 31 new tests (the scorer's, P4's training file's and the per-script checks of the three new files) all pass. The two with-`results/` counts add them and were not re-measured.

To measure the fresh-clone count on a machine that has the RULER data and the
tokenizer, point `ATTNBENCH_RULER_DATA` and `HF_HOME` at empty directories. An empty string is not the same
as unset.

## The CPU-boundary rehearsal

Before a shakedown is launched, the T3.1 rehearsal (steps 1 to 8) runs on a
CPU with the real arms, backends, recorder, null pass, replay and pricing,
on the two-layer random model, behind `attnbench/cpu_boundary.py`, which
stubs only what needs a GPU and lists each stub. It needs the x-attention
clone outside the repo, and runs best under the L4's transformers (4.46.0)
in a venv over the system torch:

```bash
export ATTNBENCH_XATTN_DIR=/path/to/x-attention       # at XATTN_COMMIT
python3 scripts/t31_session.py --cpu-boundary --session /tmp/t31_cpu
python3 scripts/t228_steps.py density --cpu-boundary --out /tmp/density --length 1024
```

`t227_steps.py` and `t228_steps.py` take `--cpu-boundary` for one step.
`tests/test_shakedown3_cpu.py` holds Shakedown 3's failures, reproduced on
the real classes, and the rulings of 1 Nov 2026 tested on them. Behind the
boundary the replay runs on `cpu_boundary.GraphLikeBackend`, whose capture
writes nothing, as a CUDA graph's does; and the chain check is recorded but
not gated, because a CPU kernel runs inside its call.
`tests/test_shakedown4_cpu.py` holds Shakedown 4's failures and the later
rulings of 1 Nov 2026: the empty-slice `fill_`, alignment by call instance,
the stand-in's shape-read prediction and the GPU session's `launch_check`
against it, and compiled XAttention's mask as an output of its compiled
function. Behind the boundary every profiler records shapes, which the
stand-in reads. It cannot tell a host tensor from a device one, since it
keeps every tensor on the CPU, so host-side work outside a host-compute
region gets kernels here that the L4 never launches. A GPU recording checks
the prediction and names any such difference.
`tests/test_shakedown5_cpu.py` holds Shakedown 5's: the boolean
`index_put_` that launches beside its children in XAttention's chunked
path, the launch check's attribution against a brute-force one, and
XAttention's threshold row built outside a compiled graph, and its kernel
interface called outside Dynamo.
`tests/test_shakedown6_cpu.py` holds the rulings after Shakedown 5: the
bare timer (`attnbench/components/bare_timer.py`) on a recording behind the
boundary, its chain gate and fallback label, the declared breaks as a
compiled variant's code path, and the recompile gate. Behind the boundary a
bare call runs its CPU compute, which the null pass skips, so the CPU's
instrumented/bare ratios mean nothing; `pricing_zoo` takes
`--cpu-boundary` for the same reason it is a dry run, not a measurement.
`tests/test_shakedown6_rulings.py` holds the rulings after Shakedown 6:
once-only host pricing on Shakedown 6's own XAttention sequence (the
committed fixture `tests/fixtures/s6_xattention_2048_layer0.json.gz` is its
first 187 operations; with `results/` present the whole sequence is tested
too), with its conservation test; the kernel interface outside Dynamo for
every block-sparse arm, behind the boundary, where `t227_steps.py
graph-breaks-one --cpu-boundary` now runs; the per-layer residual; and the
null pass's speed-ups. `scripts/t31_projection.py` prints T3.1's projected
step durations from Shakedown 6's banked anchors.

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
