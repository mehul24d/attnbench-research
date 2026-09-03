# Silent failures: the dominant failure mode in this project

Every serious error in this study so far has had the same shape. Not a crash,
not a wrong answer that looked wrong — **a plausible number, produced by
machinery that appeared to be working, with no error raised anywhere.**

Nobody is going to tamper with these results. The entire realistic threat
model is self-inflicted, and this file is the record of it, kept because six
instances in two days is no longer a coincidence.

---

## The six

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

This is the worst of the six, because the integrity layer being built on top
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

**A field that admits ignorance beats one that invents an answer.** `None` is
honest; `"HEAD"` was not. Prefer failing closed at the point of *use* (the
join refuses unverified segments) over failing at the point of *capture*
(which would take down a measurement run for an unrelated environment
problem).

**Watch for internally impossible data.** Instance 2 was caught because
batch=2 took less total time than batch=1. Summaries hide this; raw totals
show it. When a metric surprises you, check whether the underlying numbers
are consistent with *any* explanation before believing the metric's.

**Distinguish "the check passed" from "the check ran."** An empty canary that
reports no drift, a parametrised test that collected zero cases, a pass set
filtered to nothing — all report success. Where a check can be vacuous, assert
it is not: `test_every_script_is_actually_covered`,
`test_an_empty_canary_raises_instead_of_passing`.
