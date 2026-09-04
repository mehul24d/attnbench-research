# Silent failures: the dominant failure mode in this project

Every serious error in this study so far has had the same shape. Not a crash,
not a wrong answer that looked wrong — **a plausible number, produced by
machinery that appeared to be working, with no error raised anywhere.**

Nobody is going to tamper with these results. The entire realistic threat
model is self-inflicted, and this file is the record of it, kept because nine
instances in three days is no longer a coincidence.

---

## The nine

### 1. A correctness oracle computing a different function than the kernel

`masks.to_dense_bool` expanded the block grid without applying causality
*inside* the diagonal block — which a block grid structurally cannot express.
A query could attend to up to `block_size - 1` strictly future keys: measured
at seq_len=256/block_size=64, **8064 (query, key) pairs, 63 future keys for
the worst-case query**.

Meanwhile `backends/block_sparse.py` passes `is_causal=mask.causal` to the
real kernel, which masks exactly those positions. So the oracle and the kernel
under test computed different things *by construction*. The correctness gate
would have failed and looked like BSA's fault.

**Found by**: implementing a second backend and having to state the causality
convention explicitly. Not by any test.

### 2. A metric reporting JIT compilation as batching benefit

`batching_speedup` reported `gla 4.28x — batching helps` for a workload where
batching does nothing. The probe had no warmup pass, so each backend's first
call paid CUDA context setup and, for GLA, a full Triton JIT compile. Because
the cold call is always the smallest batch, and the smallest batch is the
ratio's numerator, the one-time cost was reported as a batching speedup.

Two *identical* batch=1 GLA calls in one process: **5.394 s and 1.384 s**.

**Found by**: the raw totals being internally impossible — GLA's batch=2 run
took *less total wall time* than batch=1 while doing twice the work. The
summary line was wrong; the underlying data was fine.

### 3. A guard that passed without ever running

`provenance.git_commit` recorded the literal string `"HEAD"` on every row.
`git rev-parse HEAD` fails in a repository with no commits — but **git echoes
the unresolved argument to stdout before writing its error to stderr**, so a
wrapper capturing stdout receives `"HEAD"` and treats non-empty output as
success.

This is the worst of the seven, because the integrity layer being built on top
of it — segment commit consistency, Stage 1 pass verification — would have
compared `"HEAD"` against `"HEAD"`, agreed, and certified nothing. A missing
field would have been *safer*: it fails loudly.

Compounding it, the enclosing repository was rooted at the user's **home
directory** (an accidental clone of an unrelated project), so `git_dirty` was
`True` from thousands of unrelated files and git resolved differently
depending on the process's working directory.

**Found by**: checking whether a requested feature would actually work before
building it.

### 4. A test whose edit never applied

While verifying the AST call-site checker, the file it was supposed to detect
was "broken" by a `str.replace` that silently did not match. The test run
reported **9 passed**, which was read as "the checker works" when it actually
meant "the experiment was never performed."

**Found by**: the result being suspicious — a checker that reports success on
a file that should fail is either broken or untested, and both need
investigation. The fix was to assert `EDIT APPLIED: True` before running.

### 5. Thirteen green tests exercising nothing

Adding commit-consistency checks to `cross_arch.load_segments` broke 13
existing tests. Every one of them had been passing while carrying no
`git_commit` in its fixtures — so every one had been exercising the
*unverified* path, not the path production uses.

The tests were green. They were also testing nothing about provenance. This
is instance 3's shape at the test layer.

**Found by**: making the production path stricter and watching what broke.

### 6. A retry loop that did not stop on success

An ad-hoc zone-retry created an instance in `asia-south1-c` and then went on
to attempt `-b` and `-a` as well. Both were stocked out, so exactly one
instance existed — **luck, not design**. With capacity in those zones, three
GPU instances would have been created and billed simultaneously.

The only thing that would have prevented it was a `GPUS_ALL_REGIONS=1` quota,
which is not a plan.

**Found by**: reading back what the loop had actually done, after it finished.

### 7. A stale cache baked into a machine image

The corrected 16K anchor reported `scoring_pass: 0.007 s, 9041 TFLOPS` for a
phase that genuinely takes 12.1 s. It was a **score-cache hit**: the v3
machine image had been captured from the session-3 instance *after* its 16K
anchor ran, so `results/accuracy/score_cache` rode along inside the image and
the timed pass loaded a 7 MB file.

**This one has a different shape from the other six.** Those were all defects
in code or process that existed in one place and could, in principle, be found
by reading. This is *contamination that survives teardown and travels between
sessions* — the instance that created it was deleted hours earlier, and the
evidence propagated through an artifact nobody thinks of as containing state.
A machine image is mentally a "clean environment"; it is in fact a disk
snapshot, and a cache is exactly the kind of thing whose purpose is to make
expensive work free.

Caught only by the "internally impossible data" heuristic — 9041 TFLOPS is
absurd on an L4 whose dense peak is ~121. **At a plausible magnitude it would
have been invisible**, and the Stage 3 estimate built on it would simply have
been hours low with nothing to indicate a problem. The 32K run in the same
session was unaffected purely by luck: no 32K entry happened to exist.

**Fixed structurally, not procedurally.** Clearing derived caches is now a
step in the instance startup script (`scripts/gcp_launch_compile_session.sh`),
which runs on *every* boot including boots from an image, rather than a
runbook line someone has to remember. Anything that can be forgotten will be,
and this one does not announce itself when it fires. The loop is deliberately
narrow — it removes only recomputable intermediates, never measurement
outputs — and `tests/test_launch_startup_script.py` renders the startup script
and *executes* the loop against a fake home layout to confirm both halves.

**Found by**: a number being impossible rather than merely surprising.

### 8. A gate that passed by running code the sweep never runs

Stage 1 certified `flex` block-sparse **72/72**, with real numerical agreement
(`max_abs_err` 0.013–0.021, `check_kind=masked_exact`) — not nulls, not
vacuous rows. Twelve minutes later on the same instance, at the same commit,
Stage 2 failed **72/72** of those cells at verified-identical geometry, and so
did three fresh processes on an idle GPU.

Both results were correct. They were measuring two different implementations
under one backend name.

From `stage1.log`:

```
W0903 18:41:46 torch/_dynamo/convert_frame.py:1358] [0/8]
    torch._dynamo hit config.recompile_limit (8)
    function: 'flex_attention'
    last reason: 0/7: tensor 'key' requires_grad mismatch
torch/nn/attention/flex_attention.py:1687: UserWarning:
    flex_attention called without torch.compile() - this will be slow
```

Dynamo recompiles per distinct guard set. The probe pushes hundreds of configs
through one process — here the recompiles were driven partly by `requires_grad`
differing between `fwd` and `fwd_bwd` cells, not only by shape — so it blew
past the default limit of 8. Past the limit **dynamo stops compiling and runs
the function eagerly**, and eager `flex_attention` materialises the score
matrix, which handles any block size on any card. The sweep, whose ≤4096 band
produced only 6 distinct flex shapes, never reached the limit, compiled every
cell, and hit two hard constraints the eager path does not have:

| block_size | compiled outcome on sm_89, head_dim=128 |
|---|---|
| 64 | `ValueError: Q and KV block size must be divisible by BLOCK_M and BLOCK_N` — 64 % 128 ≠ 0 |
| 128 | `No valid triton configs. OutOfMemoryError: Required: 114688  Hardware limit: 101376` |

Neither is perturbable by cache state or memory fragmentation, which were the
first two hypotheses: 64 % 128 does not become 0, and 114688 does not fall
below 101376. Both follow from inductor picking exactly **one** candidate
config when `max_autotune` is off (BLOCK_M=BLOCK_N=128 at head_dim=128) — and
raising rather than skipping *because* there is only one
(`torch/_inductor/kernel/flex/flex_attention.py`, the `if len(configs) == 1:
raise` branch).

**This is a new shape, and the worst-behaved one yet.** The other seven were
defects — something was wrong and produced a wrong number. Here nothing was
wrong. `torch.compile`'s fallback exists precisely to keep programs *working*
when compilation cannot proceed, and it succeeded: the answers it returned
were numerically correct, which is exactly why the gate passed and nothing
looked amiss. The fallback silently converts what looks like a performance
property into a **semantic swap**, and a correctness gate cannot tell the
difference, because by design there is no numerical difference to see.

Three aggravating properties worth naming separately:

* **The warning was in the log and nobody read it.** Torch announced the
  eager path in plain English. The gate's output said PASS, and the PASS was
  the artefact that travelled into the pass table.
* **One log line covered 72 rows.** Python's default `once` warning filter
  dedups by (message, category, module, lineno), so the number of warnings in
  the log carries no information about how many results were affected.
* **The signal fires once, at the crossing; contamination is permanent.**
  Per-call detection would have flagged the single cell that happened to cross
  the threshold and cleared every cell after it — the exact inverse of the
  truth. Detection has to be sticky at process scope.

The direction it happened to fire in was the lucky one. Had the sweep been the
process that exhausted the limit — and with more `seq_len` values per process,
segments 2 and 3 plausibly would have — flex would have produced *timing rows*
from eager score-materialising attention: plausible numbers, wrong
implementation, no error anywhere. That is instance 2's shape with no
internally-impossible total to catch it.

**Fixed in both directions** (`attnbench/compile_guard.py`). Prevention:
`configure()` raises the recompile ceiling far above any segment's distinct
shape count, so ordinary variety never trips the fallback. Detection:
`guard()` fails closed anyway, because prevention-by-tuned-constant is exactly
what a future grid quietly outgrows. A fallback during a timed cell yields
`status="compile_fallback"` rather than a latency; a fallback during a
correctness check **voids the verdict** — `passed` goes to False while
`check_kind` and the measured error are preserved, since what was attempted is
still the honest label and the number is still real. It is the *licence* that
is withdrawn, not the measurement.

**Found by**: the two results being mutually impossible, and refusing to
attribute it to environment noise. The cheap hypotheses (inductor cache state,
memory fragmentation) were both wrong; reading the two error strings against
torch's own source is what settled it, and the mechanism then reproduced on
CPU in seconds (`tests/test_compile_guard.py`) with no GPU involved.

### 9. Regression tests that pass against the bug they were written for

Three times in one day (2026-09-04), a test written *specifically* to catch a
defect just observed was then run against that defect reintroduced — and
passed.

| the test | why it passed anyway |
|---|---|
| unbounded mask cache OOMs the probe | it summed tensors in the instance `__dict__`; the cache was a **`dict`**, and a dict is not a Tensor |
| `pgrep -f` matches the matcher | it only ever exercised **self**-match, which a second filter already handled; the ancestor path was never reached |
| ...the ancestor test that replaced it | on macOS `pgrep -f` **cannot see an ancestor's command line at all**, so the scenario was unconstructable and the test was vacuous *by platform* |

Each was caught by the same move: break the fix, re-run, and require the test
to go red. None would have been caught by reading the test, and all three
would have shipped as green coverage of nothing.

The third is the most instructive, because the test was *correct* — it was the
platform that made it empty. `ps -o args=` showed the ancestor's command line
plainly while `pgrep -f` returned nothing for it, so the exact failure that
cost two sessions on Linux is not reproducible on the workstation. That is
instance 5's shape one level up: not "green tests exercising nothing", but a
green test whose emptiness depends on where it runs. The fix was to stop
testing the platform's matcher and test **our** filter instead, by stubbing
`pgrep` to return a chosen pid list — deterministic everywhere.

**Found by**: never trusting a new guard until it has been watched failing.
This is now the single most productive habit in the project, and its cost is
about ninety seconds per guard.

---


## The general hazards, stated once

**Non-empty stdout is not success.** Instance 3 is a specific case of a
general trap: a shell wrapper that returns `stdout.strip() or None` treats any
output as a result. Tools that echo their argument, print usage, or emit
partial output on failure will all defeat it. Check the exit code, and
validate the *shape* of what came back — a commit must look like a SHA, a
version must look like a version. `provenance._git_state` now does both.

**A check nobody has watched fail is a guess.** This is the through-line of
instances 1, 3, 4 and 5. Every guard in this project is now deliberately
broken once, observed failing, and restored — the memory guard, the AST
call-site checker, `BLOCK_SIZE` pinning, sub-block causality, the canary,
the zone-retry `break`. When breaking a guard does *not* make its test fail,
the test is the thing that's broken.

**Green tests can test nothing.** Instance 5. When tightening a production
path, check what breaks: nothing breaking is evidence the tests were not
covering it, not evidence the change was safe.

**Artifacts carry state you did not intend to ship.** Instance 7. A machine
image is a disk snapshot, not a clean environment; caches, temp files and
derived intermediates travel inside it and outlive the instance that made
them. Clear recomputable state on boot, as a scripted step.

**A field that admits ignorance beats one that invents an answer.** `None` is
honest; `"HEAD"` was not. Prefer failing closed at the point of *use* (the
join refuses unverified segments) over failing at the point of *capture*
(which would take down a measurement run for an unrelated environment
problem).

**Watch for internally impossible data.** Instances 2 and 7 were both caught
this way, and nothing else would have caught either: batch=2 taking less total
time than batch=1, and a scoring pass reporting 9041 TFLOPS on a card whose
dense peak is ~121. Summaries hide this; raw totals
show it. When a metric surprises you, check whether the underlying numbers
are consistent with *any* explanation before believing the metric's.

**A gate must exercise the path the thing it licenses will run — and process
state can change which path that is.** Instance 8. Stage 1 and Stage 2 called
the same method on the same object at the same commit and got different
implementations, because one process had compiled eight times before and the
other had not. "Same code, same machine, same commit" is therefore *not*
sufficient to conclude "same computation": accumulated JIT, compile-cache and
autotune state are inputs too, and they are invisible in every result column.
Where a runtime can silently substitute an implementation, the harness has to
assert which one ran, not infer it from the answer being right.

**A fallback designed to preserve correctness will defeat a correctness
gate.** Also instance 8, stated as its own trap because it generalises past
`torch.compile`: cuDNN/cuBLAS algorithm fallbacks, SDPA's dispatch to the math
backend, and any `try: fast_path except: slow_path` in a dependency all have
this shape. The output is right, so nothing downstream can notice, and the
property that actually changed — which kernel ran, and therefore what the
latency means — is not one the gate was ever looking at.

**A process-matching command must never match itself or its ancestors.** Two
sessions lost, in opposite directions: `pkill -f 'pip install'` killed its own
invoking SSH command and took the build with it (2026-09-03), and `pgrep -f
run_probe.py` matched the polling command, so a probe already dead of an Xid 31
MMU fault reported RUNNING for six more minutes (2026-09-04). Both were
"documented" -- the `[n]vcc` bracket trick appears twice in the compile-session
runbook -- and both happened anyway, because a rule that must be remembered at
every call site will be forgotten at one of them. It is now
`scripts/procmatch.sh`, which walks the full ancestor chain rather than only
excluding `$$`; the bracket trick defeats self-match but not the `bash -c` that
`gcloud compute ssh` spawns around the whole command.

**Warning counts do not measure blast radius.** Python's default `once` filter
dedups by (message, category, module, lineno). One line in `stage1.log`
covered 72 contaminated rows. When a warning is the evidence, capture it with
`simplefilter("always")` and count the affected *operations*, never the log
lines.

**Distinguish "the check passed" from "the check ran."** An empty canary that
reports no drift, a parametrised test that collected zero cases, a pass set
filtered to nothing — all report success. Where a check can be vacuous, assert
it is not: `test_every_script_is_actually_covered`,
`test_an_empty_canary_raises_instead_of_passing`.
