# T4 dense-only pilot

Status: pre-registered before any new T4 candidate-task accuracy rows.

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