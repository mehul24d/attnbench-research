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