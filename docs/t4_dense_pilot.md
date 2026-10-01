# T4 dense-only pilot

Status: pre-registered before any new T4 candidate-task accuracy rows.

## Probe result (2026-10-01)

The five-example-per-task probe ran on an NVIDIA L4 at both planned bands,
using commit `822a4dd`, with 50 dense-only rows and clean provenance. Mean
dense scores were:

| task | 16384 | 32768 | selection outcome |
|---|---:|---:|---|
| `niah_multikey_1` | 100 | 100 | exclude: ceiling |
| `niah_multivalue` | 80 | 70 | retain |
| `niah_multiquery` | 65 | 35 | retain at 16384 |
| `qa_1` | 80 | 60 | retain |
| `qa_2` | 0 | 0 | exclude: no useful dense baseline |

The preregistered 40--90 rule therefore selects `niah_multivalue`,
`niah_multiquery`, and `qa_1` for the 50-example pilot. These probe results
are task-selection evidence, not the final accuracy claim; the selected-task
pilot must still run before any sparse arm is interpreted.

The capped L4 session ran for 41 minutes at the ledger rate and was torn down
with no compute resources left. The result parquet is kept under
`results/t4_dense_probe_20261001/` locally, while the session diagnostics and
cost record are under `results/t4_dense_probe_session_20261001/`.

## Selected pilot result (2026-10-01)

The selected dense-only pilot ran 50 examples per task and band, for 300
rows total, on an NVIDIA L4 at commit `40118f7`. Exact 95% Clopper--Pearson
intervals for the dense baseline were:

| task | 16384 | 32768 |
|---|---:|---:|
| `niah_multivalue` | 16/50 = 32.0% [19.5, 46.7] | 17/50 = 34.0% [21.2, 48.8] |
| `niah_multiquery` | 13/50 = 26.0% [14.6, 40.3] | 3/50 = 6.0% [1.3, 16.5] |
| `qa_1` | 30/50 = 60.0% [45.2, 73.6] | 20/50 = 40.0% [26.4, 54.8] |

The intervals confirm that the selected tasks are non-ceiling under the fixed
dense baseline. This closes the dense-task-selection gate for T4, but does
not establish sparse non-inferiority: the oracle and deployable estimator
arms, exact non-inferiority analysis, and any measured accuracy loss remain
the next experiment.

The selected-pilot result parquet is under
`results/t4_selected_dense_pilot_20261001/`; the session diagnostics and cost
record are under `results/t4_selected_pilot_session_20261001/`. The session
ran for 43 minutes at the ledger rate, estimated at `INR 57`, and left no
compute resources running.

The pilot tests whether the five newly wired RULER candidate tasks are useful
for the T4 accuracy study before any sparse arm is run. It measures the dense
`sdpa_flash` reference only, on Qwen2.5-1.5B-Instruct, at 16384 and 32768
tokens. The default probe is five examples per task and band; the planned
pilot is 50 examples per task and band after the probe passes.

The fixed candidate set is:

`niah_multikey_1`, `niah_multivalue`, `niah_multiquery`, `qa_1`, `qa_2`.

The selection rule is fixed before sparse results exist: retain a task for the
main T4 study only if its dense score is strictly between 40 and 90 inclusive
at at least one of the two bands. Report every candidate, including candidates
that select no band. A task is never removed because a sparse result is poor.

Run the cheap probe with:

```bash
python scripts/run_accuracy.py --t4-dense-pilot --dry-run
```

The dry run checks planning only. The real probe and the 50-example pilot use
the same command without `--dry-run`, with a separate output directory and the
pinned RULER data directory. No sparse arm is permitted in either run.

The pilot does not establish the final accuracy claim. It only determines
which candidate tasks satisfy the pre-registered dense-baseline rule. The
selected tasks must then be run with the dense reference, the oracle and the
deployable estimators under a separate analysis plan, including exact
non-inferiority intervals.

The selected dense-only pilot command is:

```bash
python scripts/run_accuracy.py --t4-dense-pilot \
	--t4-pilot-tasks niah_multivalue,niah_multiquery,qa_1 \
	--n-per-length 50
```
## Planning regression in `40118f7` (found and fixed 2026-10-01)

`40118f7` added `--t4-pilot-tasks` as a second `if` whose `else` reset the
task list to the whole grid. From that commit until the fix, `--tasks` and a
bare `--t4-dense-pilot` were silently overwritten: `--tasks vt` planned 19500
cells instead of 6500, and the probe command above planned the Stage 3 tasks
(`niah_single`, `niah_multikey`, `vt`) rather than the five candidates. The
pilot's n was also written twice, the second write (5) always winning, so
`--t4-pilot-tasks` without `--n-per-length` planned 5, not 50.

Neither banked result is affected. The probe ran at `822a4dd`, before the
change, and its 50 rows name the five candidates. The selected pilot passed
`--t4-pilot-tasks` and `--n-per-length 50` explicitly, which planned the
intended 300 cells. Selection now lives in `select_tasks` and
`select_seq_lens` in `scripts/run_accuracy.py`, and
`tests/test_run_accuracy_selection.py` pins both banked commands to the row
counts they produced.
