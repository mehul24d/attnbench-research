# Silent failures: the dominant failure mode in this project

Every serious error in this study so far has had the same shape. Not a crash,
not a wrong answer that looked wrong — **a plausible number, produced by
machinery that appeared to be working, with no error raised anywhere.**

Nobody is going to tamper with these results. The entire realistic threat
model is self-inflicted, and this file is the record of it, kept because
seventeen instances in seven days is no longer a coincidence.

It stood at seventeen when that sentence was written. It stands at
**fifty**. The original sentence is kept rather than updated because the
rate is the point: the count went on growing under a discipline built
specifically to stop it growing.

**#45 is the first one found from outside.** Every instance before it was
found by the author, usually while working on something adjacent. #45 was
found by an adversarial audit that had this file in hand and was explicitly
looking for the instance it did not contain — and it found one that had
survived the full test suite, a diagnostic written to test the exact property
it broke, and a structural argument standing in for the measurement at the
band where it mattered most. **A catalogue of one's own mistakes is not a
substitute for someone else reading the code**, and the three near-misses in
#45 are all the same failure: a fixture that could not express the defect.

---

## The instances

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

#### Recurrence, 2026-09-16: the same image, a measurement output this time

The H100 session found `results/probe/versions.json` on a fresh instance,
before any phase had run, stamped:

    gpu_name: NVIDIA L4

Stage 0 resumes on `(backend, config_key)`. Run it on an H100 against that
tree and every config the L4 already probed is **skipped**, and the H100's
`probe.parquet` inherits L4 capability rows under an H100 provenance stamp.
Whether an H100 supports a config is exactly what Stage 0 exists to establish,
so the inherited answer is not merely stale, it is about a different machine.

**Why the existing fix did not catch it.** The startup-script loop from the
first occurrence is deliberately narrow: "it removes only recomputable
intermediates, never measurement outputs." That narrowness is right -- a boot
script that deletes results is a worse failure than the one it prevents -- and
it is exactly why this got through. `probe.parquet` *is* a measurement output.
The loop looked at it and correctly left it alone. The guard was calibrated to
the substance of the first instance (a cache) rather than to its shape
(anything on the image's disk that a resume key can match), so the second
instance walked straight past it.

There is no version of the clearing loop that fixes this without becoming
dangerous. The remedy has to be at the other end: **refuse to start** when
`results/` is non-empty before any phase has run, and make the operator look.
`scripts/gcp_preflight_instance.sh` does that, and refuses on *any* image with
the problem rather than purging this one, so the next image that ships state
is caught by the same check rather than by the next incident.

**Found by**: syncing per phase. The first sync listed the objects it had
written back, and the listing contained `results/stage3_s1/`,
`results/accuracy/score_cache/` and a `probe/` tree from a session that had
not happened yet. A sync that reported only success would have shown nothing.

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

### 10. A provenance flag set correctly and read by nobody

2026-09-04. The first complete Stage 1 in the project — 4200 Stage 0 rows and
376 correctness rows — was stamped `git_commit=b6ed63b`, a commit **18 behind**
the code that produced it.

Cause: the deploy untarred source over the instance's existing checkout. That
replaced `attnbench/`, `scripts/`, `tests/` and `docs/`, and left `.git`
describing the machine image. `provenance.capture()` then reported the image's
commit, correctly, for a repository that no longer held the code being run.

The stamp was wrong in the worst available way — **plausible**. `b6ed63b` is a
real commit in this repo's history, so nothing was malformed, no field was
empty, and the format validation added after instance 3 passed cleanly. The
Stage 2 gate compares `git_commit` against the current commit, so this table
would have been rejected as WRONG COMMIT and re-run — or, had the sweep been
launched from a matching checkout, accepted outright.

**What makes this instance different from the other nine: the mechanism
worked.** `git_dirty=True` was recorded on every one of those rows. The stamp
announced its own unreliability, accurately, and nothing was listening.
`load_stage1_pass_set` read `git_commit` and no other field.

An audit of the rest of the stamp found the same shape elsewhere. Of 23
`Provenance` fields, exactly **four** are consulted by any gate (`git_commit`,
`git_dirty`, `host`, `gpu_name`). The rest are recorded and never read —
including `clocks_locked`, despite `provenance.py`'s own docstring saying "an
unlocked run must be flagged in the results". It is flagged. Nothing refuses to
use it.

**Found by**: noticing that the recorded commit was not the commit deployed,
while cross-checking results against the local repo. Nothing in the pipeline
would have raised it. The contradiction was visible in the data — the table
contains 84 `illegal_memory_access` rows, and the code that writes that status
did not exist at `b6ed63b` — which is the "internally impossible data" heuristic
again, and again the only thing that worked.

**Fixed by**: `GATED_FIELDS`/`RECORDED_FIELDS` in `provenance.py`, so every
field must be classified as read-by-a-gate or deliberately-decorative and a new
field forces that decision; a test asserting each GATED field is genuinely read
somewhere, because a declaration that drifts from reality is the same failure
one level up; `Stage1ProvenanceError` refusing dirty passes in
`load_stage1_pass_set`; and `scripts/gcp_deploy_source.sh`, which deploys a git
bundle rather than a source tarball and then asks the *instance* what commit it
is at.

---

### 11. A heredoc executing its own comments on the launching machine

The launch script builds the instance startup script with an **unquoted**
heredoc, so that `$CAP_MINUTES` expands and the agreed deadline is baked in.
That expansion is not selective. It runs command substitution over the whole
body — comment lines included, because a heredoc body is not shell source and
a leading `#` protects nothing.

One comment documented the restart-recovery procedure and quoted the command
in backticks. So every launch ran `sudo shutdown -h HH:MM` **on the operator's
laptop** and pasted its output — empty — into the script. The instance shipped
with the line reading `by hand: .` and the procedure silently deleted.

It was harmless only by luck, twice over: `HH:MM` is not a valid time, and
`sudo` had no tty. A comment quoting any runnable command would have run it,
once per launch, with the operator's privileges, saying nothing.

**What makes this instance different**: the damage was to the *instructions*,
not the data. No result row is wrong because of it. It belongs here anyway,
because the mechanism is indifferent to what it deletes — the same heredoc
also carries `shutdown -h +$CAP_MINUTES`, the cap that exists so a forgotten
instance cannot bill for a week. A substitution that swallowed that line would
disarm the backstop and leave a script that still looks correct in the repo.

**Found by**: reading a live instance's deployed metadata and comparing it
against the source that produced it, while investigating an unexpected
instance for an unrelated reason. Reading the script alone would never show
it: the bug is invisible in the source and visible only in the output. That is
the same "compare the artefact against its source" move that caught instance
10's stale provenance stamp, and it is now two for two.

**Also found in the same read**: the launcher's default source image was still
`v3`, and the unexpected instance had booted it — an image predating the
instance-10 deploy fix. A default that has gone stale does not announce
itself; it just quietly hands you last week's machine.

**Fixed by**: single quotes in that comment, plus a test that renders the
heredoc and asserts the body contains no backticks and no `$(`, and a second
test asserting the recovery text actually arrives — because "no backticks"
alone would pass if someone deleted the line instead of fixing it. A third
test pins the set of launch-time expansions to exactly `{CAP_MINUTES}`, so
the next variable someone interpolates has to be a decision. The image default
is now v4 with a test naming the version.

*(A fourth defect surfaced while fixing this one and is worth recording as a
near miss: the new accelerator-override flag used `"${ARR[@]}"` on a possibly
empty array, which under `set -u` in bash 3.2 — what macOS ships — is an
unbound-variable abort. It failed closed, killing the launch before gcloud was
called, and it would have done so on every G2 launch, not just the A2 one the
flag was added for. Caught by the existing zone-retry tests within a minute,
which is what those tests are for.)*

---
---

### 12. A green test that created a billable GPU instance on every run

`test_a_single_zone_stockout_does_not_advise_retrying_the_list` ran the real
`gcp_launch_compile_session.sh` with the **real `PATH`** and **no
`GCP_PROJECT`**, fed `launch\n` to its confirmation prompt, and set
`GCP_ZONE_FALLBACKS=asia-southeast1-c`. The script therefore resolved the live
project from `gcloud config get-value project` and created `test-instance`,
a `g2-standard-8` + L4, in GCP. Four times. **₹218.**

It passed every time. All three of its assertions read `SCRIPT.read_text()`:

```python
src = SCRIPT.read_text()
assert "SINGLE-ZONE target" in src
```

**The test could not fail.** It asserted that a string appears in a file that
was never modified, so it reported nothing about the process it had spawned.
An assertion against the file under test is not a test of that file; it is a
copy of it. Every other test in the same file stubbed `gcloud` onto `PATH` --
this one was written without the stub, and nothing distinguishes the two until
the invoice arrives.

It also carried a comment rationalising a failure that never happened:

```python
# The script may exit earlier (no project, image check) in a sandbox
```

That is the `EDIT APPLIED: True` shape from instance 3 -- an assumption about
what occurred, written down in the position where an observation belongs. The
sibling deploy test carried the same fiction ("fails later, at the gcloud call,
which is not available here"); the interception log showed it really attempting
`gcloud compute scp` to a host called `inst`.

**What makes this instance different from the other eleven: it cost money, and
the investigation was the failure.** A full day went into finding the source of
these instances. Their appearance was correlated 4-for-4 with commits, 1-3
minutes before each -- because the full suite runs immediately before every
commit. Two of them were created *during* the investigation, by test runs made
while investigating, and were then reported as fresh evidence of an unknown
external cause. Two peer sessions were asked to account for themselves. An
elaborate 2-hour-cadence theory was built on two timestamps that happened to
fall two hours apart, and a clean interval was twice read as proof the source
was gone.

**Found by**: shadowing `gcloud` with a logging shim and running the suite. The
ancestry chain ended at `pytest -> bash gcp_launch_compile_session.sh
test-instance` in one line. The question that would have found it on the first
try -- *what runs `gcloud` on this machine besides me?* -- was never asked, and
the suite that answers it was run repeatedly throughout.

A near-miss worth recording: the transcript was searched for
`Created [https...test-instance` and returned zero hits, which was read as
exoneration. It only meant the create was not in *this* process's output. It
was in a subprocess's.

**Fixed by**: routing the test through `_run` like its siblings, and asserting
behaviour rather than source text; and `tests/conftest.py`, a session-scoped
autouse fixture that puts refusing shims for every network-capable CLI
(`gcloud`, `gsutil`, `aws`, `ssh`, `scp`, `curl`, `wget`, `rsync`, `kubectl`,
...) ahead of everything on `PATH`. A test that stubs a tool itself still wins,
so reaching production is now deliberate rather than available by omission.
A static check additionally fails the suite if any test hands a cloud-facing
script an unmodified `os.environ["PATH"]`.


### 13. A median that moved in the opposite direction to every cell inside it

Two findings came out of the 2026-09-05 A100 session that were not findings.
**"GLA's cost FALLS from 4096 to 8192"** -- physically impossible for an
attention kernel -- and **"fa2 is 15x faster"**, the same mechanism with the
sign reversed.

Nothing was mismeasured. Every latency in the dataset is correct. On the L4,
per cell at `batch=1`:

```
seq_len      1024   2048   4096    8192    16384    32768
per cell     0.96   2.56   5.08   10.97    20.55    41.90   ms   (2x per doubling)
marginal     5.02   9.93  19.81   10.97    20.55   101.20   ms
                                  ^^^^^ falls 45%
```

GLA is almost exactly linear, which is what a linear-attention kernel should
be. The marginal median inverts at 8192 because batches 4 and 16 OOM'd out of
that band, so its composition changed from `{1,4,16}` to `{1}` and the median
switched to a cheaper population. Simpson's paradox, in a results table.

**Why this is structural here rather than unlucky.** The attrition that
unbalances a band is OOM; OOM correlates with sequence length and batch size;
sequence length is the independent variable in most of the study's claims. The
composition is therefore unbalanced *as a function of the thing being
measured*, which is precisely the condition under which a marginal inverts.
Every long-context table in this project is in scope by default.

It looked like a result rather than a mistake, which is why it nearly shipped.
There was no error, no warning, no failed check -- just a plausible number from
arithmetic that ran correctly on correct inputs and answered a different
question than the one asked.

The fix is `attnbench/analysis/composition.py`. `aggregate()` refuses by
default across groups whose facet composition differs at all, names the missing
levels, and offers `matched_subset()` as the repair rather than only the
complaint. It cannot return a bare number: every exit carries `n`,
`facet_levels`, `facet_counts` and `composition_tvd`, so a table built from it
cannot lose the provenance of its own averages. A guard that offers no
alternative gets bypassed under time pressure.

### 14. A divide-by-zero guard mistaken for a resolution floor

Three places in this codebase divide by a quantity that can be arbitrarily
small. Two had `clamp_min(1e-8)` / `max(..., 1e-12)`; the third had nothing.
None of them was wrong in a way that raised, and two produced numbers that
looked like results.

**`gates.check_correctness`.** `abs_err / expected.abs().clamp_min(1e-8)`. On a
block-sparse config the mask zeroes most of `expected` by construction, so
this divided a real bf16 error by a number carrying no information. Live A100
rows: `max_rel_err = 3.3e+04` beside `max_abs_err = 1.3e-02`. Pass/fail keys
off absolute error, so no verdict was ever wrong -- but the column would have
produced nonsense the moment anything plotted or aggregated it.

**`diagnostic_agreement._rel`.** `max(abs(a), abs(b), 1e-12)`. 1e-10 against
3e-10 scored as a 67% disagreement between two numbers that are zero to every
tolerance in the project.

**`cross_arch.Speedup.speedup`.** No floor at all, and a flat 5% materiality
bar that had been added the day before. Against a floor measured from the
study's own data -- the L4 rented three times, 75 pairs measured on more than
one host, cross-host ratio spread up to **27.2%** below 3 ms -- **0 of 19
flips clear it, where the flat bar certified 4.** Two of the four were at
seq_len=1024, where the two L4 hosts straddle parity *by themselves*: `fa2` at
1.031 and 0.811, `sdpa_cudnn` at 0.895 and 1.004. The reported "flip" was
between an average of those two and the A100.

**Why the pattern hid.** The median cross-host spread is 1.8-5.0% in every
band. Almost every ratio is fine, so an unfloored analysis looks healthy and
spot-checking confirms it. The tail is what disqualifies a claim, and a tail
does not show up in the cells you happen to look at.

**Why it hit the headline specifically.** A flip requires one side near parity
by definition, and near parity is where the margin is smallest relative to the
noise. So the test the study most wants to pass is the one its instrument
resolves worst. The crossover claim survived the same standard only because
FA2/GLA ratios run from 0.11 to 3.7 and sit far from parity -- with the single
exception of the matched 8192 cell, which missed by 0.007 and was the cell the
headline had been resting on.

### 15. A cost model that divided a bandwidth-bound phase by a compute-bound throughput

Stage 3 scores generated text, so a row is a prefill plus `k` greedy decode
steps. Once the KV cache existed,
`docs/stage3_generation_decision.md` priced the remaining decode tax as the
FLOPs ratio `k / seq_len` — 19.3 steps against 8192 tokens, **0.24%** — and
concluded the 13.35 h estimate survived intact.

The ratio is arithmetically correct. It is a *time* ratio only if both phases
run at the same throughput, and they cannot:

| | FLOPs per byte of weight read | regime |
|---|---|---|
| prefill, 8192 tokens | ~8192 | compute-bound |
| decode, batch 1 | 2 | **bandwidth-bound** |

A batch-1 decode step reads all **3.09 GB** of Qwen2.5-1.5B's bf16 weights to
produce one token. On an L4 that is a **~13 ms floor**. Priced at the
prefill's measured 42.2 TFLOPS, the same 3.09 GFLOP step "takes" **0.074 ms**.
**177×.**

Corrected, the decode term is **1.5–6.6 h on a 13.35 h grid (11–49%)**, not
under 1%. And it is worst where prefill is cheapest — the weight read does not
depend on context, so at the 2048 band decode costs **more than the prefill it
follows** (+70% at the floor, +328% at the worst case).

Nothing raised. The estimate was internally consistent, built on real measured
TFLOPS, and produced a plausible number that would have been discovered as a
session overrunning its window at hour four.

**What makes this the third of its kind.** The unit matched every time and the
regime did not:

- attention-**kernel** TFLOPS read as whole-**model** TFLOPS — a 42% phantom
  speedup (recorded in `configs/accuracy/stage3_grid.yaml`);
- GLA's 1.7 "TFLOPS" against FA2's 61.8, concluding GLA is slow when it is
  faster in wall clock and simply issues ~30× fewer FLOPs;
- and now prefill TFLOPS applied to decode.

**Fixed by:** modelling decode from bytes rather than FLOPs
(`timing_probe.decode_memory_traffic_bytes` / `decode_seconds`), with the trap
written at the top of that section; `decode_steps_by_task` made a **required**
argument of `total_grid_flops_by_category`, so the unit of work has to be
stated rather than defaulted; and `scripts/time_one_accuracy_example.py` now
**measures** a decode step against the bandwidth floor before the grid
commits, which collapses a five-hour bracket in two minutes.


### 16. A clock lock that reported success while the clocks stayed unlocked

Found in Phase 1 of the 2026-09-06 Stage 3 session, on the first instance in
this project where root actually works — so the first time the claim could be
checked against reality rather than assumed to be failing.

`provenance.lock_clocks()` ended:

```python
res = _sh(["nvidia-smi", "-lgc", f"{sm_mhz},{sm_mhz}"])
return res is not None
```

`_sh` returns `stdout.strip() or None` and **never looks at the exit code**.
Run without root, `nvidia-smi -lgc` exits 4 and prints

```
The current user does not have permission to change clocks for GPU 00000000:00:03.0.
```

to **stdout**. Non-empty stdout, so `_sh` returns a string, so `lock_clocks()`
returns `True`.

Measured directly on the L4 rather than reasoned about: reset to unlocked,
called `lock_clocks()` unprivileged, got `True` back, and read the clock at
**2040 MHz** — the unlocked maximum, not the 1734 MHz requested.

**Third instance of stdout being read as an outcome**, and the family is now
unmistakable:

| | stdout | truth |
|---|---|---|
| `git rev-parse HEAD` (#3) | echoes `"HEAD"` | failure |
| `--query-compute-apps` | prints nothing | success (GPU is clean) |
| `nvidia-smi -lgc` (#16) | prints a permission error | failure |

`_sh_result` was written for exactly this and already existed; `lock_clocks`
simply never used it.

**This one is the worst of the three by direction.** The other two produced a
blocked run and a confusing error. This reports a *control* as established
when it is absent, so every row it stamps overstates how well the run was
controlled — and it would only ever be believed, never questioned, because
"clocks locked" is the answer everyone wants.

**Second finding, in the same place:** `provenance.capture(clocks_locked=False)`
takes the flag as a **parameter**, not an observation, and no caller in the
project has ever passed it. So every result row ever written says
`clocks_locked=False` regardless of the machine's actual state. Harmless while
the lock genuinely never worked; wrong the moment it does. `run_accuracy.py`
now attempts the lock and stamps the returned outcome.

**Fixed by:** `lock_clocks` using `_sh_result` (exit status, not stdout
truthiness), three tests driving a fake `nvidia-smi` that reproduces the real
one's exit-4-with-stdout behaviour, and `--lock-clocks` on `run_accuracy.py`
passing the measured outcome into every row's provenance.


### 17. A forget gate made of random numbers, and 900 rows that meant nothing

Stage 3 segment 1 measured GLA against 300 distinct 2048-token prompts per
task and wrote 900 rows scoring **0.0 on every task**. Nothing raised. Every
row had fluent text, a real `latency_ms`, a real `stop_reason`, a valid
provenance stamp and `clocks_locked=True`.

**The tell was not the garbage text, it was input-dependence:**

| | distinct predictions / 300 prompts |
|---|---|
| `sdpa_flash` (niah_single / multikey / vt) | **300 / 300 / 300** |
| `gla` | **15 / 81 / 27** |

A model that merely *could not retrieve* would still say something different
for a different prompt. Near-constant output means the context is not
reaching the computation. And the first generated token — which comes from
the **prefill** logits, before any decode step — was already near-constant,
which eliminated the KV-state handoff as a cause before any code was read.

**The cause.** GLA's forget gate has no analogue in the `(q, k, v)` triple
`AttentionBackend.forward` carries, so `linear.py` synthesized it:

```python
self._gate = F.logsigmoid(torch.randn(k.shape, generator=seeded_from_cfg_key))
```

`logsigmoid(randn)` averages **-0.81 per step**, so the recurrent state is
multiplied by 0.45 at every token — an effective memory horizon of **1.24
tokens**. At seq_len 2048 the prompt survives at `exp(-1654)`, which is zero.
The model answered from the last token or two of a prompt whose final ~20
tokens are the same template in all 300 examples, and the gate is cached on
`cfg.key()` so every example got the *identical* random decay. Hence a
handful of attractor outputs.

**Nothing here was a mistake when it was written.** For Stage 2 a synthesized
gate is exactly right: a kernel's throughput cannot depend on the values in
its gate tensor, only its shape and dtype. The module docstring said so
plainly. What was missing was a boundary — the timing-only path was reachable
by omission from an accuracy runner, and it produced output-shaped output.

**There is no correct gate to supply.** Qwen2.5 has no gate projection; its
weights were never trained with one. So this is not a bug with a fix, it is a
scope boundary that was never enforced, and the honest conclusion is that
**GLA may have no accuracy arm at all** — its results may be timing-only.

**Fixed by:** `gate_source` with no safe default — `"learned"` refuses,
`"synthetic"` must be asked for by name, and the refusal happens *before* the
optional `fla` import so it fires on any machine. Stage 2's call sites now say
`gate_source="synthetic"` at the point of use.

**And by the cheapest test in this repo**,
`tests/test_output_depends_on_input.py`: every backend's output must differ
for two different inputs, and — the specific shape of this failure — rewriting
only the **first half** of the context must still change the **last**
position's output. GLA's output did vary; it varied with the last token or
two. A long-context benchmark whose backend cannot see the start of its own
prompt is measuring nothing, and one assertion catches it.


## The general hazards, stated once

**A verification that runs downstream of the spend is diagnostic, not
preventive.** Both are worth having and they are not interchangeable, so the
question to ask of any check is *what does it still cost me if this fires*.
`load_stage1_pass_set` refusing a dirty stamp before Stage 2 begins, the
`run_phase.sh` clean-tree refusal, `gcp_verify_gcs_writable.sh` and the
cold-cache guard are guards: they fire while failing is still free. The dense
canary is a guard by construction — it is chained with `&&` ahead of the
sparse arms, so a failure measures no sparse row. `run_s1a_comparison.py`
refusing when the new and banked decode kernels disagree is a **check**: by
the time it can fire, the GPU hours are spent and the only question left is
whether the data is interpretable. A check that reads like a guard is how a
session gets paid for twice.

**A near-miss worth the same space as an instance: a session script is an
era's configuration, and copying it forward copies the era.** Planning the
2026-09-20 S7 re-measure, the decode pin was carried over from
`s1a_run_band.sh`, which pins `sdpa_math` because the 2048–8192 bands decoded
through it. The 16384 rows stamp `decode_backend=sdpa_flash` — they were
measured after `DENSE_DECODE_BACKEND` moved on 2026-09-08. Had it shipped,
the decode kernel would have changed alongside the variable under test, on the
one band the session existed to measure, and the −6.0 point shift it found
would have been unattributable between two causes.

It was caught by reading the banked rows' own `decode_backend` column before
writing the runner, which cost two minutes. **The near-miss is recorded
because the catch was not structural** — no gate would have fired. The
comparison script does refuse when the new and banked sparse decode kernels
disagree, so it would have surfaced *after* the measurement was paid for,
which is a check and not a guard. The general shape: a script that encodes
"what was true for that run" reads like a script that encodes "how we do
this", and the difference is invisible at the call site.


**A measurement is wrong wherever it is written; a status is wrong only after
the moment it described.** Both go stale, and the difference decides what to
do about them. `16 of 31 matched points are dominated` is false on every page
that carries it, today and next year, so it stays in
`docs/withdrawn_figures.md` under permanent watch -- and a correction that
paraphrases it rather than quoting it removes the string the watch is keyed
on, which is a loss of provenance disguised as tidying. *"The re-run is
approved as audit item S1a"* was true when written and stopped being true when
the re-run happened; there is no page on which it is now a false claim about
the world, only pages on which it is out of date. Those entries retire.

The consequence is a working rule for anything that tracks the currency of
claims: **watch the quantities, date the statuses.** Entries that keep
watching must have something to watch, so a withdrawal note quotes the
withdrawn sentence verbatim; entries that retire must say when they stopped
applying, so a status carries the date it was true on. Mixing the two
produces the two failures this project has now seen — a guard that goes dead
because the string it watched was paraphrased away, and a status block
asserting a pending action that completed five days earlier, in a document
that had itself been corrected in the same session.

### A tolerance bounds noise, not bias that fits inside it

**Standing rule, promoted from #27 on 2026-09-08.** Any check of the form
"is each cell within X%" is blind to a systematic error smaller than X. It
will report every cell passing while the model is wrong in all of them, and
it will do so consistently, which reads as robustness.

So wherever a check produces residuals across N cells, **test the signs, not
only the magnitudes**:

    from attnbench.accuracy.phase_timing import bias_warning
    warn = bias_warning([r.residual_ms for r in results])   # None if balanced

Same-sign residuals across all cells are **bias by default** until shown
otherwise. The null is symmetric, the test is a one-line binomial, and it
costs nothing to run. 24 of 24 the same sign is p ~ 1.2e-7; that is not a
pattern to describe in prose and move past, which is exactly what happened
on 2026-09-07 (the residuals were called "a small fixed per-call overhead"
and left there).

Applies to every tolerance-based check in this codebase, not only the Stage 5
reconciliation: `gates.TOL` comparisons, `canary.CanaryDrift`,
`cross_arch.Speedup`, `RECONCILE_TOLERANCE`. Each answers "is this cell
acceptable" and none of them answers "is this set centred".

### A filtered command tells you nothing unless you also kept its exit status

**Standing rule, 2026-09-08, after this shape appeared four times in one
session.** `cmd | grep PATTERN` and `cmd | head -n` both discard the exit
status *and* discard the error text. What is left is an absence, and an
absence is consistent with three different worlds: the thing succeeded and
had nothing to report, the thing failed, or the thing never ran.

The four, all mine, all in tooling written to check something else:

| what was filtered | what the silence meant | what it was read as |
|---|---|---|
| `run_probe.py \| tail -20` | argparse exit 2, gate never ran | gate passed (#18) |
| precondition fixtures `\| grep FAIL` | crash on a missing path | preconditions held (#25) |
| `instances describe` empty output | transient API failure | instance deleted |
| `run_decision_map \| grep -E ...` | KeyError traceback, no file written | map unchanged |

The last is the sharpest: the old output file still existed, so the
comparison ran happily and reported *0 cells changed* — comparing the stale
file to itself. A failed write is invisible to anything that reads the path
afterwards.

**The rule:** redirect to a file and capture `$?`, or `set -o pipefail`, any
time the command's success is part of what you are concluding. Reserve
`| grep` for reading, never for deciding. And when a comparison reports "no
change", check that the thing being compared was actually regenerated.

### Cheap instruments can be undiagnosable, not merely noisy

Also from #27, and the more transferable half. The two-point decode estimator
was replaced because it amplified prefill noise. But the reason it let a bug
live for the life of the stage is different and worse: **a line through two
points has no free parameter left over to check against anything.** Its
intercept is whatever the arithmetic requires, so it always agreed with
itself.

Three points cost one more measurement and buy a residual — something the
model predicts that can be compared to something measured independently.
When choosing an instrument, ask not only how precise it is but **what it
would look like if it were wrong**. An instrument with no answer to that
question cannot report its own failure.



**A divide-by-zero guard is not a resolution floor, and they are the same line
of code.** Instance 14. `clamp_min(1e-8)` keeps the arithmetic finite and says
nothing about whether the answer means anything; a floor excludes the cases the
instrument cannot resolve. Reviewing one as the other is easy because the fixed
version looks almost identical. The question that separates them: *if the
denominator were at this bound, would I believe the ratio?* A backstop test now
fails the suite on any bare epsilon in a division under `attnbench/`.

**Set the floor from measurement, not from a round number.** The first bar for
flip materiality was 5% -- defensible-sounding, borrowed from a tolerance
measured for a different purpose, and wrong by a factor of five in the
permissive direction. The right number was already in the data: the same card
rented three times, measuring itself. Where a study has repeated measurements
of anything, that is the resolution, and no argument about what the tolerance
"should" be beats it.

**Arithmetic that runs correctly on correct inputs can still answer a different
question.** Instance 13. Every guard in this project until then watched for
wrong *inputs* -- a stale commit, a mismatched host, a contaminated row. None
watched the *combination step*, which is where the error lived. When a summary
number contradicts a mechanism you are confident about (an attention kernel
getting cheaper with length), the number is not surprising evidence; it is a
composition question, and the first thing to look at is what changed about the
population between the two groups being compared, not the measurements inside
them.

**A number is only as comparable as the cells it was formed from.** The
cross-architecture crossover result rests on 11 (seq_len, batch) cells measured
on *both* cards. The unrestricted table has 17, and reading the winner off it
gives the same answer -- by luck, since the A100 covers three batches at 8192
where the L4 covers one. Getting the right answer from an invalid comparison is
the worst outcome available: it certifies the method.

**Check the constraints that apply, not the ones you just learned.** On
2026-09-05 a CPU machine family was chosen as a workaround after verifying it
against the two constraints that had failed that morning -- NVME capability and
`pd-balanced` support -- while the third, `guestAccelerators`, went unchecked
even though it was the *first* property that had failed that day. Three of four
constraints verified reads as diligence and fails identically to none. The
answer is not more care; it is a preflight that enumerates every inherited
property every time, which is now in the launcher.

**Per-item protection fails by omission.** Instance 12. Every test in that
file stubbed `gcloud` except one, and one is enough. When a safety measure has
to be remembered per use, the question is not whether it will be forgotten but
what it costs the first time it is. Put it at the boundary -- a session
fixture, a conftest, a default -- so that skipping it is an act rather than an
oversight.

**An assertion against the source of the thing under test is not a test.**
Instance 12 asserted that a string appears in a shell script, having just run
that shell script. The run was decorative. If a test spawns a process, it must
assert on what the process did; otherwise delete the spawn, because it is pure
side effect.

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

**A fix that relocates a symptom looks exactly like progress.** Stage 1 died
at the 8192 band on three consecutive rented sessions: first in a reference
backend's forward, then in the comparison arithmetic, then in `make_inputs` at
the start of the *next* check. Each site was patched, and after each patch the
run got further — 4116 rows, then 4450, then 4513. Monotonic improvement, a
plausible story at every step, and three fixes that were all wrong, because
the defect was in none of those places. `timing.measure` had released the
allocator's reserve in a `finally` since it was written and `check_for_family`
never had.

Two rules fall out of it. **Read the sequence, not the step:** three fixes
that each move a failure later rather than resolving it are evidence about
the diagnosis, and after the second one the right move is to stop and ask what
the sites have in common — here, that they were simply wherever the
accumulated reserve happened to run out. And **the obvious mechanism was the
wrong one:** "leaked tensors" would have sent the fix hunting for a dangling
reference that does not exist. Every check's tensors *are* freed by
refcount. Freeing them returns the blocks to PyTorch's caching allocator,
which keeps them **reserved** from the driver and hands them back only to
allocations that fit; successive checks at different shapes fragment that
reserve until a 1 GiB contiguous request fails with the card nominally
empty. The last crash reported **47 MiB free against 22.03 GiB total** — a
reading that is impossible under the leak theory and diagnostic under the
right one. Internally impossible data again, one paragraph up.

**A flag set correctly and read by nobody fails exactly as a missing flag
does.** Instance 10. It is worse than a missing field in one respect: the field
is there, so an auditor reading the schema concludes the risk is covered. The
question to ask of every recorded field is not "is this captured correctly" but
"what refuses to run when it says something is wrong" — and if the answer is
nothing, the field is documentation, not a check. Say so where it is declared,
rather than letting it look like a guard.

**A stamp can be well-formed, validated, internally consistent, and still
describe different code.** Also instance 10, and the reason a commit check
could not catch it: the commit was real and matched exactly. Format validation
answers "is this a SHA"; it cannot answer "does this SHA describe what ran".
Only something that observes the *working tree* can, which is why `git_dirty`
exists — and why it has to be read. Deploy by moving history (a clone, a pull,
a bundle), never by copying source over a checkout: source without its history
takes the identity of whatever it landed on.

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

**A ratio between two regimes is not a ratio.** Instance 15, and the third of
its shape. Two quantities sharing a unit — TFLOPS, tokens, bytes — invites
dividing one by the other, and the division is meaningless when the two sit in
different regimes: kernel vs whole model, compute-bound vs bandwidth-bound,
one backend's issued FLOPs vs another's. Before dividing, ask what *limits*
each side. If the answers differ, the ratio has no physical meaning however
well the units cancel.

**"Confirm the inputs are still valid" is the arithmetic question, not the
modelling one.** The 13.35 h estimate was re-checked thoroughly and every
input held; what had changed was the *unit of work* — a row stopped being one
forward pass. Re-checking an estimate means asking what one unit is, not only
whether the numbers behind it are current.

**Read the exit code. Three times now, stdout has been the wrong witness.**
Instances 3 and 16 and the exclusivity check. A command's stdout answers "what
did it say", never "did it work" — and the two diverge exactly where a tool
reports a problem on stdout instead of stderr, which is common. `_sh` cannot
answer the second question; `_sh_result` exists for it. Any new `_sh` call site
whose result is used as a *verdict* rather than as *data* is this bug again.

**Check that the output depends on the input.** Instance 17. It is one
assertion, it costs nothing, and it separates "this backend is bad at the
task" from "this backend is not reading the task" — which are indistinguishable
in any score, because both produce a low number. Where the claim is about long
context, strengthen it: changing the *distant* past must change the answer.
A near-constant output across varied inputs is not a weak result, it is an
absent one.

**A path that is correct for one stage is not thereby safe for another.** The
synthesized gate was right for timing and catastrophic for accuracy, and
nothing in between said so. Where a component is valid only under a stage's
assumptions, name the assumption in its constructor and make the default
refuse — a docstring cannot fail a run.

## 18. A pipeline's exit status is the last command's, so `| tail` eats the verdict

**2026-09-07, Stage 3 S1b, found within twenty minutes of predicting it.**

The session wrapper ran the Stage 1 gate as:

```bash
python3 -u scripts/check_stage1_against_diagnostic.py --correctness ... | tail -20
```

(That script was removed once Stage 2's segments were all measured. The line
above is the transcript of what was run, not a command to run now.)

`--correctness` is a *positional* argument. argparse exited **2**. `tail`
exited 0, so the pipeline exited 0, the chain continued, and the next line
printed `PHASE 2 DONE rc=$?` — where `$?` was also `tail`'s. **The Stage 1
gate did not run and nothing said so.** Had success been inferred from the
chain continuing, seven hours of bands would have been measured with the gate
never having executed.

Fourth instance of *stdout is not an outcome*, and the first in shell rather
than Python. The earlier three: `git rev-parse HEAD` echoing "HEAD",
`--query-compute-apps` printing nothing on success, `nvidia-smi -lgc`
printing its permission error to stdout.

Found only because the log was read deliberately. The compensating control —
"no news is not good news, read the verdict out of the log" — was adopted
*before* the failure, because the bug was spotted by reading the wrapper.

**Fix:** `set -o pipefail`, and prefer `cmd > file; rc=$?` over `cmd | tail`
wherever the exit status carries a decision.

## 19. A test of a safety mechanism, run against the live instance of it

**2026-09-07. The hard cap was unarmed for about ninety seconds.**

Verifying that the self-teardown could actually fire — passwordless `sudo`
was a real question, and a self-teardown that cannot execute is a fiction —
the check was:

```bash
sudo -n shutdown -h +99   # can we schedule?
sudo -n shutdown -c       # undo the test
```

`shutdown -c` cancels **every** scheduled shutdown. The 570-minute hard cap
armed by the startup script at boot lives in the same single global slot, and
went with it. Caught by checking `/run/systemd/shutdown/scheduled` in the
same command, and re-armed at the original wall-clock deadline (40 s early,
the safe direction).

Second instance of this exact shape: the `lock_clocks` test escaping to real
hardware because `sudo` resets PATH to `secure_path`, bypassing a PATH-based
fake and changing a live GPU's clocks.

The generalisation is not "don't test safety mechanisms" — the check was
correct to run and established a real fact. It is that **a mechanism with one
global slot cannot be exercised without occupying that slot.** Test the
authorisation (`sudo -n true`) rather than the effect, or snapshot and restore
the slot in the same breath.

## 20. Grouping analysis cells on a column that identifies examples, not cells

**2026-09-07, Stage 4's first run against real data.**

`run_matched_analysis` grouped on `context_length`. Stage 3 records that as
the **exact tokenized length**, which was itself a deliberate correctness fix
(docs/limitations.md, "Context lengths are exact token counts") — so it
varies per example: **224 distinct values** across three bands, 1972–2062
around 2048, 8010–8196 around 8192.

So each "cell" was one token count. **Mean 7 paired examples instead of 300,
minimum 1.** `paired_bootstrap_diff_ci` accepts n=1 without complaint — a
one-element resample has zero variance and a lower bound equal to the point
estimate, so every such cell reports a confident result. The script printed a
288-column table of sparsity budgets and a plausible `oracle_sensitive` list.
Nothing raised, nothing warned.

**Why the tests did not catch it.** Every fixture in `test_matched.py` used
`context_length=2048` *exactly*, so grouping on the raw column was correct
under the test's premise and wrong under reality's. Same shape as the
`position_ids` catch: the test asserted a mechanism against a premise that
does not hold in production. The regression test now asserts the property
that actually has to hold — rows spread across a band's real token counts
form **one** cell — and it fails against the old code.

**Fix:** `band_for(context_length, grid.seq_lens)`, nearest-band with the
assignment *asserted* unambiguous (refuses anything beyond 25% of a band)
rather than assumed, applied inside `run_matched_analysis` so a caller cannot
forget it.

**The generalisable rule:** before grouping, check that the group key has the
cardinality you expect. `df.groupby(keys).size().min()` is one line and would
have caught this instantly. A column that identifies an *example* cannot
identify a *cell*.

## 21. "Tested on CPU" that tests the arithmetic and not the interfaces

**2026-09-07. Two failures in one 7-minute session, both mine, both the same
shape.**

`scripts/run_phase_timing.py` failed twice on a rented L4:

1. `compute_importance_scores(..., cache_dir=None)` — `None` is not a
   supported "don't cache" sentinel; `score_cache._path_for` calls `Path()`
   on it.
2. `masks.mask_for(cfg, importance_scores=scores[0])` — one index short.
   `scores[layer]` is `(n_heads_kv, n_blocks, n_blocks)`; the function wants
   one head's `(n_blocks, n_blocks)`.

`tests/test_phase_timing.py` was green through both. It covers the timing
protocol and the reconciliation arithmetic, which are correct, and touches
neither `SwappableAttentionModel` nor `masks`. **The claim "tested on CPU"
was narrower than it sounded**, and the part it did not cover is the part
that only fails where it costs money.

**A mock would not have helped.** What broke was a signature, and a mock
encodes the author's belief about the signature rather than the signature
itself — it would have agreed with the bug.

**The fix, and it generalises.** The measured body moved out of the script
into `phase_timing.measure_band`, and a CPU test drives *that function*
against a real 2-layer `LlamaForCausalLM` at hidden size 32. One body, run by
both the script and the test. A test that merely called the same methods in
its own code would re-encode the same assumption; the point is that there is
one body.

`tests/test_phase_timing_interfaces.py` also pins each failure individually —
the cache-entry count proves every scoring rep is a real miss (a shared id
would time a disk read and report ~0 ms), and the shapes are asserted both
ways round.

**Remaining exposure, audited not assumed.** Of the repo's scripts, the ones
holding GPU-path logic no test reaches were `time_one_accuracy_example.py`
(398 lines — and it *did* fail on hardware this session with `KeyError:
'decode'`), `decide_gla_arm.py`, `flex_session_recheck.py`,
`probe_batch_scaling.py` and `flex_kernel_options_probe.py`. The pattern to
apply to each is the same: move the body into a module and drive it from a
toy-model test.

**Updated 2026-09-12.** Two of those five — `flex_session_recheck.py` and
`flex_kernel_options_probe.py` — were removed with the 2026-09-04 flex
diagnostic they existed for, so the exposure is now three scripts, not five.
Deleting a script is a legitimate way to close this debt and a slightly
dishonest one to count: nothing was made testable, the untestable thing
stopped existing. The remaining three are still debt.

## 22. A regression test and a coverage test are different properties

**Recorded 2026-09-07, after being caught by the gap twice.**

A **regression** test pins a failure you found. A **coverage** test asserts
the test exercises what the code actually does. Having the first is not
having the second, and this project has now shipped that confusion twice:

- `test_position_ids_are_load_bearing` asserted a mechanism against a premise
  that did not hold, so it passed vacuously.
- `test_phase_timing_interfaces.py` was written as "the test that would have
  caught both failures for free." It did. Its arm list was
  `[("sdpa_math", None)]` — **dense only** — while the script ran dense plus
  three sparsities. The third failure lived in the sparse arm and reached a
  rented instance.

**The root cause is one this project refuses everywhere else.** The test's
arm list and the script's arm list were two sources of truth for the same
fact. `gate_source` is read off the backend instance, `backend_role` is
derived from the config, the GLA verdict is read from its own JSON — the
principle was applied to the data and not to the tests.

**The asymmetry that let it survive:** a single-source-of-truth violation in
DATA produces visibly wrong rows. In TESTS it produces a green suite. The
same bug is loud in one place and silent in the other, which is why the test
layer was the one that went uncorrected.

Fixed structurally: `phase_timing.arms_for()` is the one list, and
`test_the_test_covers_every_arm_the_script_runs` asserts the test's coverage
against it. **Generalise it — anywhere a test enumerates cases that
production code also enumerates, check the two lists against each other
rather than maintaining them in parallel.**

### The detection method, which is the transferable part

**A vacuous test is one where breaking the thing it guards does not turn it
red.** That is checkable on demand, in about a minute, and it is what caught
`position_ids`, the determinism tests, and both halves of this entry:

1. Break the guarded property deliberately — revert the fix, drop a case
   from a list, feed a degenerate input.
2. Run the test.
3. If it still passes, the test asserts nothing about that property.

Applied here twice in one sitting, and it earned its place both times:

- Reintroducing the exact gap (arm list back to dense-only) turned
  `test_the_test_covers_every_arm_the_script_runs` red. It guards what it
  claims to.
- A first draft of the decode-backend test tried to trigger the refusal
  through a deliberately-wrong factory and **did not raise** — the CPU
  stand-in for block_sparse has a decode path, so the test could not fail.
  The check said the test was wrong, not the code, before it was committed.
  It was rewritten to assert the contract against the real registry, where
  `supports_decode()` is a classmethod and needs no CUDA.

Make this the standard move on any test guarding a property that matters,
not a habit that happens to have been applied when someone remembered.


## 23. A provenance stamp that overwrites what the harness measured

**2026-09-07, Stage 5.** The run locked clocks successfully. `lock_clocks()`
returned True, the script printed `clocks locked -- stamped on every row`,
and `measure_band` threaded `clocks_locked=True` onto all 27
`PhaseMeasurement` rows. Every row in the written parquet says
**`clocks_locked=False`**.

The last five lines of the script:

```python
prov = provenance.capture().to_dict()     # clocks_locked DEFAULTS to False
df = pd.DataFrame([r.to_dict() for r in rows])
for k, v in prov.items():
    df[k] = v                             # measured True -> stamped False
df.to_parquet(out / "phases.parquet", index=False)
```

`capture()` takes `clocks_locked` as a parameter defaulting to `False`. Called
bare, it does not observe the GPU — it *asserts* the default. The merge loop
then assigns every stamp key over the frame, so the assertion silently
replaced the measurement.

### Why nothing caught it

`clocks_locked` is a **GATED** field: `cross_arch.Speedup` and
`canary.CanaryDrift` both read it, precisely because an unlocked run's
variance is easy to mistake for a real effect. So the wrong value is the kind
things act on.

Three columns of the same file contradicted it, and no check compares them:

| column | value | what it implies |
|---|---|---|
| `persistence_mode` | `Enabled` | set only by `_smi(["-pm","1"])` inside `lock_clocks()` — so the sudo escalation worked |
| `sm_clock_mhz` | `1740` | nearest supported step above the requested `int(2040*0.85)=1734`, held across a 19-minute run |
| `clocks_locked` | `False` | contradicts both |

### The direction matters, and it is the opposite of the last one

Instance 3 of "stdout is not an outcome" was `lock_clocks()` returning True on
a lock that never happened — a control reported as ESTABLISHED when absent,
which overstates how well a run was controlled. This one is the mirror: a
control reported as ABSENT when established, which understates it. Both come
from the same root — **`clocks_locked` had two sources of truth**, the
measurement and the stamp — and the stamp ran last.

Neither direction is the safe one. An overstated control invites trust that
was not earned; an understated one invites a re-run that costs money and
changes no number, or gets the data discarded by a gate that was right to be
suspicious.

### The asymmetry that let it survive — a fix in a call site is not a fix

`scripts/run_accuracy.py` had **already fixed this exact bug** in its own
body, months of project-time earlier, with a comment naming the failure mode:

> Clocks: attempt, then stamp what HAPPENED, never what was asked for. Every
> row in this project so far carries clocks_locked=False because nothing has
> ever passed the flag — capture() takes it as a parameter defaulting to
> False and does not observe it. So a run whose clocks ARE pinned would be
> recorded as unpinned, and the fact would be lost.

Stage 5 was written afterwards and reproduced the bug from scratch. The fix
was correct, local, and therefore invisible to the next harness. **A lesson
that lives in a call site protects that call site only.** It has to live in
the thing every call site uses, or the second harness re-derives the bug —
which is the same single-source-of-truth asymmetry as #22, one layer up.

### The fix

`provenance.stamp_onto(df, stamp)` fills in fields the rows do NOT have and
**raises** where the rows already carry one that disagrees. The rows own any
field they measured; a stamp fills in the rest. A caller that measured a
field passes it to `capture()` too, at which point the two agree and the
guard never fires — so raising costs a correct run nothing, and per-band
output is written before it, so a raise cannot destroy a completed
measurement.

### Detection method, per #22

Reverting `stamp_onto` to the naive `for k, v in stamp.items(): df[k] = v`
turns `test_stamp_does_not_overwrite_a_measured_field` red with
`DID NOT RAISE`. Restored, it passes. The guard guards what it claims to.

The generalisable check: **anywhere a wholesale stamp, default, or config
merge is applied over rows a harness produced, ask which fields both sides
have an opinion about.** Those are exactly the fields where the merge order
decides the truth, and merge order is not a fact anyone reviews.

### The audit this prompted, and what it found

If a fix in a call site is not a fix, the question is which other rules live
in call sites. Auditing for it turned up one more, and it is a *decision*
rather than a mechanism, which makes it worse:

**The GLA exclusion had four definitions.** The pre-registered DROP verdict
(`docs/gla_arm_decision.md`) decides which rows every accuracy analysis may
read. It was restated in:

| site | form | justification given |
|---|---|---|
| `run_pareto.py` | `EXCLUDE_BACKENDS = ("gla",)` | none |
| `run_matched_analysis.py` | `EXCLUDE_BACKENDS = ("gla",)` | `results/stage3_s1/INVALID_ROWS.md` |
| `run_decode_confound.py` | `EXCLUDE_BACKENDS = ("gla",)` | none |
| `run_phase_timing.py` | `df.backend != "gla"` | none |

Only `run_decision_map.py` imported it from somewhere else. Two different
justifications, and the fourth is an inline literal that **does not answer a
grep for the constant's name** — so an audit of "where is this rule applied"
misses it, which is how it survived three earlier passes over these files.

Consolidated to `grid_configs.ACCURACY_EXCLUDED_BACKENDS`, next to the arm
decision that produced it. `tests/test_shared_project_rules.py` asserts there
is exactly one definition, that no script filters by an inline literal, and
that the shared constant carries the verdict it came from — a bare
authoritative tuple with no explanation is worse than four local copies.

**The generalisable form.** A rule with N definitions cannot be revisited; it
can only be re-found. And the cost is not symmetric with the number of
copies: it is set by the *least greppable* one, because that is the copy an
audit misses. Prefer one definition; where a second is genuinely needed,
assert the two against each other (as `decode_confound._dominates` does
against `pareto._dominates`) rather than maintaining them in parallel by
hope.


## 24. A confound that is absent from the clean case and present only where the answer lives

**2026-09-07, Stage 5 -> Stage 6.** Correcting the decode-kernel confound
alone gave **0 of 31 operating points dominated** and a best speedup of
**1.28×**, up from a measured 16/31 and 1.06×. Both numbers are artifacts,
and the correction was one step from being reported.

The second confound: on `vt` the arms do not generate the same number of
tokens — dense 35.2–38.8, sparse 27.7–34.6 — so their mean end-to-end
latencies were never comparable. An arm that stops earlier finishes sooner
for reasons that have nothing to do with how fast attention is. This was in
the **measured** result, before any correction.

### Why it hid, which is the transferable part

`niah_single` is clean. Every arm generates **exactly 14.0** tokens, because
the task's answer is a fixed-length 7-digit number. So the confound is
strictly absent from the task with the clean, easy-to-check numbers, and
present only on `vt` — **the task where all the surviving operating points
were**.

That is the dangerous shape. A confound uniformly distributed across
conditions is usually visible as noise; a confound *anticorrelated with the
clean case* is invisible in exactly the place a reader would look to
sanity-check. Checking `niah_single` and finding 14.0 across every arm is
positive evidence, and it is positive evidence about the wrong task.

**The check that finds it: for any quantity being compared as a mean, ask
what varies per row inside each group.** Here it was the divisor — the
comparison was ms-per-generation over generations of different lengths.
`n_generated` was on every row the whole time (it is in the reconciliation
identity Stage 5 checks against), and nothing had ever grouped by it.

### Two confounds, both invisible end-to-end

Neither the decode kernel nor the generation length is recoverable from a
single wall-clock total. Both were recorded per row; both needed the phases
measured *apart* before anything compared them. That is the case for Stage 5
made twice in one measurement.

### And the correction moves points in both directions

Normalizing both confounds: 12/31 dominated, best speedup 1.057×. Seven
points leave the dominated set, all at 8192 — a **length-dependence result**,
not merely a correction: sparsity's prefill saving only becomes visible at
the longest band measured. Three `vt` points at 2048 *enter* it, having been
on the frontier only by generating fewer tokens.

Keep both directions in the output. A correction that could only ever free
points is a correction nobody checked.


## 25. Verifying a guard with a fixture the guard never read

**2026-09-07, while checking `repair_stage5_stamp.py`.** The repair flips a
GATED provenance field under six preconditions, so the preconditions matter
more than the repair. Verifying them meant corrupting one input at a time and
confirming a refusal — the #22 detection method applied to a script.

The verification harness set `SP=<scratchpad>` in the shell, then wrote its
corrupted fixtures from a Python heredoc reading `os.environ.get('SP','/tmp')`.
**`SP` was never exported**, so Python fell back to `/tmp` while the shell
loop read `$SP/...` from the scratchpad. Every fixture path was missing,
`pd.read_parquet` raised, and the output was filtered through
`grep -E "FAIL|REFUSING"`.

Result: **no output, for every case.** Which is precisely what a working set
of preconditions that never fire would also look like.

It got worse: the loop reported `exit=$?` *after a pipeline ending in `head`*,
so it printed `exit=0` for all four cases — the exit status of `head`, not of
the script. Entry #18 in this file, reproduced inside the tooling written to
verify entry #23.

Fifth instance of *stdout is not an outcome*, and the second in shell. The
distinguishing feature here is that the silence was **structurally
indistinguishable from success**: a refusal prints `REFUSING`, a pass prints
nothing matching the filter, and a crash prints nothing matching the filter.
Two of those three are the good case and one is a broken test, and the filter
could not tell them apart.

**Fix:** rerun with absolute paths, `rc=$?` captured directly off the script,
and *both* the failing and the passing case asserted in the same output —
four corrupted fixtures at `exit=1`, the genuine artefact at `exit=0`. A
verification that only ever shows refusals cannot distinguish "refuses
correctly" from "refuses always".

**The general rule, which #22 states for tests and this states for
verification scripts:** when the signal for "guard fired" is a line of output,
absence of that line is not evidence the guard passed. Assert the positive
case in the same run, or the harness has one failure mode indistinguishable
from the outcome it is checking for.


## 26. A difference estimator amplifies the noise of the quantity it cancels

**2026-09-08.** Stage 5 estimates the decode step as a two-point slope
(`K_LO, K_HI = 1, 8`):

    decode_step = (T(8 tokens) - T(1 token)) / 7

Differencing cancels prefill, which is the point: at 8192 prefill is ~700 ms
and a decode step ~32 ms, so any estimator that did not cancel it would be
measuring prefill. But it cancels prefill's **value**, not prefill's
**noise**. `T(1)` and `T(8)` are separate timed calls, each executing its own
prefill, so prefill variance enters both independently and lands in the slope
divided by 7 -- amplified relative to the ~32 ms quantity being estimated.

Measured, across two clock-locked L4 sessions on different hosts:

| band | dense decode, run 1 | run 2 | change |
|---|---|---|---|
| 2048 | 34.80 ms/token | 34.96 | +0.5% |
| 4096 | 34.60 ms/token | 34.58 | −0.1% |
| 8192 | 34.16 ms/token | **31.25** | **−8.5%** |

The dense arm did not change between runs. 2048 and 4096 reproduce to 0.5%;
8192 moves 8.5% -- and in the *opposite direction* to dense prefill at the
same band, which rose 3.5%. That opposition is the tell: uniform host
variance moves both the same way.

The arithmetic closes exactly:

    1-token endpoint  699.9 -> 730.4 ms   (+30.5, carries the whole prefill)
    8-token endpoint  939.0 -> 949.1 ms   (+10.1)
    dense prefill     694.9 -> 719.0 ms   (+24.1)
    predicted slope shift from prefill alone  -24.1/7 = -3.44 ms/token
    observed slope shift                                -2.91 ms/token

**Sensitivity, stated generally:** a 1% prefill error at 8192 (7.2 ms)
becomes 1.03 ms/token of decode error, which is **3.2%** of the estimate. The
amplification is `prefill / (decode x (K_HI - K_LO))`, so it grows with
context length -- worst exactly where the measurement matters most.

### What it does and does not invalidate

The conclusion it was used for -- that changing `DENSE_DECODE_BACKEND`
removed the decode penalty -- is a **within-run** comparison, where both arms
were measured in the same session under the same clocks, so this noise does
not accumulate across the comparison. That conclusion stands. What it bounds
is the *precision*: the 8192 residuals of 0.5% / 3.4% / 4.3% sit inside the
estimator's own ~3% at that band, so "removed" means "indistinguishable from
dense to within ~3-4%", not "equal". The pre-registered 5% threshold was
passed, and it was passed by less margin than the raw numbers suggest.

**Cross-run** comparisons of a decode step at 8192 carry ~8% -- larger than
the canary's 6% host-to-host figure, and for a reason that has nothing to do
with hosts.

**Fix for a future run:** more points and a least-squares fit (1, 2, 4, 8,
16), or better, use the independently measured prefill from the same run
rather than differencing two calls that each re-execute it. Two points is the
minimum that can produce a slope, and the minimum is what makes the noise
term maximal.


## 27. A tolerance loose enough to absorb a systematic bias reports CLOSES

**2026-09-08.** The reconciliation identity was wrong by one decode step for
the entire life of Stage 5:

    end_to_end ~= prefill + n_generated * decode_step        # what it did
    end_to_end ~= prefill + (n_generated - 1) * decode_step  # what is true

The prefill forward emits the logits for the **first generated token**, so n
tokens cost one prefill plus n−1 decode steps. `run_measured(...,
logits_to_keep=1)` measures exactly that prefill, and HF generation works the
same way. Confirmed directly: at 2048, prefill alone is 141.2 ms and
generating one token costs 141.9 ms.

### The reconciliation could not catch it, by construction

One decode step is ~35 ms against totals of 400–2400 ms: **1.5–9%**, inside
the 10% tolerance, in the **same direction every time**. So every cell
reported `CLOSES` and the identity was wrong in all of them.

`closes` is not evidence that an identity is right. It is evidence it is not
*badly* wrong, and those are different claims. A tolerance is a bound on
noise; it says nothing about bias that fits inside it.

### The signature was in the data and I explained it away

The 2026-09-07 run produced **24 positive residuals out of 24**, mean +50.3
ms. A sign test on that is p ≈ 6e-8. I saw the pattern, described it as
*"consistent with a small fixed per-call overhead the two-term identity
doesn't model"*, and moved on. That sentence is even the right shape — a
fixed per-call term is exactly what one unmodelled decode step is — but
naming a residual is not the same as pursuing it, and the check that would
have settled it cost nothing.

**Same-sign residuals are a bias, not noise.** Any reconciliation reporting
N cells should test the sign distribution, not only whether each cell is
inside tolerance. That is a cheap test with a known null.

### What actually found it

The OLS decode fit's intercept cross-check (#26), on its first hardware run.
`T(k) = intercept + k * slope` makes the fitted intercept an independent
estimate of prefill, and it came in **39–46 ms below** the measured prefill
at every band and every arm — a constant offset, not a proportional one,
which reads as −27.7% at 2048 (small prefill) and −2.8% at 16384 (large
prefill). Deficit divided by slope clustered at **1.00**, which names the bug
precisely: one decode step.

The general lesson is about **what kind of question a check asks**. The
reconciliation asked "do these three numbers roughly agree", and a biased
identity agrees roughly. The intercept check asked "does the model's own
free parameter match a quantity measured independently", which has no
tolerance to hide in. Two checks over the same data are not redundant when
they fail differently.

And it only exists because the estimator has three or more points. The
two-point version had no free parameter to check: two points fit a line
exactly, so its intercept was whatever the arithmetic required. **The cheap
design was not merely noisier — it could not have found this.**

### Impact

The decode *slope* is unaffected: it is the marginal cost and was estimated
correctly. What moves is every implied total, and with it the reconciliation
residuals — which is where the off-by-one actually bit: the 24-of-24
same-sign residuals in `results/stage5/reconciliation.parquet` were the
off-by-one and nothing else. Repaired under `(n − 1)` on 2026-09-12 they
become **10+/14−, p=0.54, mean +6.5 ms**, down from 24+/0−, p=1.2e-7, mean
+50.3 ms. Measured end-to-end speedups are untouched — those are wall-clock,
not identity.

**A correction to this section, which is instance #28's shape.** It read:
"Recomputed, block-sparse operating points dominated by dense go from 12/31
to **10/31**." That attribution is wrong. Holding the phases file fixed and
changing only the identity moves *nothing*: 12/31 under both `n` and `(n−1)`
on `results/stage5/phases.parquet`, 10/31 under both on
`results/stage5_ols/phases.parquet`. The off-by-one shifts every arm's
normalized total by the same one decode step, so it cancels in a comparison
between arms. The 12→10 swing was entirely the *phases file*, which changed
in the same regeneration — a second uncontrolled variable moving alongside
the one under test, and the impact credited to the wrong one.


## 28. A guard whose grouping makes the checked property constant

**Found 2026-09-12, four days after the guard was written — by me, in my own
code, while auditing someone else's review for exactly this shape.**

`analysis/decode_backend_guard.assert_uniform` exists to make one failure
impossible: pooling latency across rows that decoded through different
kernels (#24). It groups by `OPERATING_POINT_KEYS = ("backend", "task",
"_band", "sparsity")` and raises if a group's `decode_backend` values
disagree.

`decode_backend` is a **function of** `backend`. Grouping by `backend` makes
`decode_backend` constant within every group by construction. The guard could
only ever detect era mixing *within one arm* — rows measured before and after
the 2026-09-08 `DENSE_DECODE_BACKEND` change pooled together — and was
structurally incapable of firing on the cross-*arm* case, which is the
confound it was written for and the one the study's central correction is
about.

It reported **zero offending cells** on `results/stage3_s1b/accuracy.parquet`
— sparse arms `sdpa_math`, dense arm `sdpa_flash`, six comparison cells — and
passed. Both call sites (`run_pareto.py`, `run_decode_confound.py`) called it
before averaging and were satisfied.

### Why it is not the same as #25

#25 is a guard verified with a fixture the guard never read: the test was
wrong, the guard was fine. Here the *test* was fine — it constructed a mixed
cell and watched the guard raise — and the guard was fine for what the test
constructed. The gap is between what the guard checks and what the failure
is, and no test of the guard in isolation can show it, because the test
author picks the frame and will pick one shaped like the check.

### The detection method, which is the transferable part

For any guard, ask: **is the property being checked constant within the
grouping?** If the grouped-by key determines the checked value, the check
cannot fail, and the correct number of offending cells is zero for every
input. It is not a weak guard; it is not a guard.

Cheap mechanical version: run the guard on the real data it was written for.
Not a fixture — the actual banked set whose confound motivated it. If it
passes, either the confound is gone or the guard cannot see it, and those two
have to be told apart before the green is worth anything.

### The fix

`assert_comparable` / `cross_arm_cells` group by `COMPARISON_KEYS = ("task",
"_band")` — one *comparison*, every arm whose latency is divided by another's
— which deliberately excludes both `backend` and `sparsity`. It raises on
`stage3_s1b` (6 cells) and passes on the three post-correction sets. A test
asserts that `assert_uniform` stays blind to the same frame, so nobody reads
a green one as covering the other.

---

## 29. A correction whose input is not tied to the era of what it corrects

**Found 2026-09-12. This is the one that changed a published number.**

`decode_corrected_ms` removes the decode-kernel penalty from a measured
end-to-end total. The penalty comes from a Stage 5 `phases.parquet` passed in
by the caller. Nothing tied that file's decode era to the era of the rows.

`results/stage6/decode_corrected.parquet` was built from
`results/stage5_ols/phases.parquet` — measured after `DENSE_DECODE_BACKEND`
became `sdpa_flash`, so *both* its arms already decode through the same
kernel and the largest penalty it can supply is **0.747 ms/token** — and
applied to `results/stage3_s1b/` rows whose real penalty is **8–22
ms/token**. The column subtracted 0.3–0.7 ms where it should have subtracted
100–700 ms. It was a no-op wearing the name of a correction, and the confound
it names survived it completely.

### Why it produced a plausible number

Every guard passed. The arithmetic was right, the identity was right after
`7c1fcca`, the row count was right, and `measured_ms` travelled beside it as
designed. A near-zero correction on an already-plausible latency is
indistinguishable from a correctly-applied small correction. The only way to
see it is to ask what the penalty *should* be and compare — which nothing did,
because the penalty was not written down anywhere.

### What it cost

`normalized_ms` was built from the same mispaired file, so:

- dominated-normalized **10/31**, where the era-matched phases give **12/31**
  (`claims.md` said 12 the whole time; the file disagreed with the ledger and
  nothing compared them)
- best normalized speedup **1.069×** vs **1.059×**
- the Stage 7 decision map: **five of 27 cells** recommended `block_sparse`
  where the era-matched run recommends dense; dense 16/27 → **21/27**

### And the five cells were never resolvable anyway

All five sit at band 4096, and all five turn on sparsity's prefill saving at
4096, measured three times in three sessions:

| session | 4096, sparsity 0.75 |
|---|---|
| `stage5` | **0.00 ms** |
| `stage5_flashdecode` | +1.17 ms |
| `stage5_ols` | +2.82 ms |

against a within-session sd of 1.2–2.4 ms and a between-session drift in the
dense prefill itself of 9 ms (283.76 → 292.75). The recommendation at 4096 is
determined by which session's phases you feed it. `cross_arch` has had a
`resolution` / `resolvable` concept since Stage 2 for exactly this; the
decision map has none, and reports a 1.002× recommendation with the same face
as a 1.32× one.

### The fix

`decode_confound.correct` now requires `arms_share_decode_backend` from the
caller — obtained from `decode_backend_guard.cross_arm_cells` on the rows, not
assumed — and refuses **both** mismatched pairings: a confounded set demands a
penalty above `SAME_KERNEL_PENALTY_MS`, a matched set demands one below it.
`decode_penalty_ms` is now a column, so the size of what was removed is in the
data. The correction can, at last, detect its own error.

### The transferable rule

**A derived quantity whose input can come from more than one measurement
session must check that the input matches what it is being applied to.** The
general form of "two sources of truth": not two copies of a rule, but two
populations either of which will satisfy a function's type signature while
only one answers its question.

---

## 30. Fourteen unstamped files, and the rule that already covered them

`README.md` line 89: *"No result row is written without a provenance stamp."*
On 2026-09-12, every derived artifact in the project had none — Stages 4, 5
(reconciliation), 6, 7 and all five `cross_arch` outputs, fourteen files.

They were not wrong. They were **unverifiable**, which is the whole reason
the rule exists: the 2026-09-04 case (#10) is a set that carried a stamp that
provably could not be right and cleared a check nobody read, and an absent
stamp is the same position with less to read.

The reason it happened is worth more than the count: the rule was enforced in
`sweep.py` and `run_phase_timing.py`, the two harnesses that *measure*. Every
script written afterwards that only *reduces* already-measured rows skipped
it, because a measurement stamp — `gpu_name`, `clocks_locked`,
`sm_clock_mhz` — is obviously wrong for a laptop doing a groupby, and the
absence of a right-shaped alternative read as "the rule does not apply here."

`provenance.stamp_analysis` is that alternative: `analysis_tool`,
`analysis_git_commit`, `analysis_git_dirty`, `analysis_host`,
`analysis_timestamp`. Prefixed so it cannot collide with, or overwrite, the
measured columns the source rows carry — and so a reader can never mistake
the machine that reduced the rows for the machine that measured them.

**The rule to take:** when a discipline is skipped everywhere in a whole
class of code, the usual cause is not laziness. It is that the discipline
only had one implementation and it was the wrong shape for that class.

---

## 31. A recomputation is only a check if its inputs were checked

**Found 2026-09-12, in my own audit of someone else's review, one step after
the audit had reported the opposite.**

The review of 2026-09-10 flagged an off-by-one in `decode_corrected_ms`. My
Step-5 audit of that finding regenerated the affected artifact, diffed it
against the shipped one, and reported:

> `dominated_normalized`: 0 rows differ (10/31 → 10/31). Decision map: 0/27
> cells change. **STILL TRUSTWORTHY — regenerated, unchanged.**

Every one of those numbers was correct. The conclusion was wrong.

The regeneration re-ran the producer against **the same
`results/stage5_ols/phases.parquet` the shipped file had been built from** —
and that file was the defect. It is from the wrong decode era: both its arms
already decode through `sdpa_flash`, so the largest penalty it can supply is
0.747 ms/token against a real 8–22 ms/token. Re-running a correct producer on
a defective input reproduces the defective output exactly, and the diff is
clean because both sides inherit the same fault.

**A recomputation that agrees is evidence of determinism. It is evidence of
correctness only if the inputs were established independently.**

### The signal that was already there and was not consulted

`docs/claims.md` had said **12 of 31** the whole time. The parquet said 10.
The ledger and the artifact had disagreed for four days, across a published
document and a shipped file, and nothing compared them — including the audit
whose entire job was to decide which numbers were trustworthy.

That is the tell, and it is cheap: **two independent statements of one
quantity that disagree.** Not a subtle one. A grep would have found it.

### Why "regenerate and diff" feels like verification

It has every surface property of a check. It runs real code, produces real
output, and the comparison is exact rather than eyeballed. What it does not
do is vary anything the answer depends on. The producer was the only thing
under test, and the producer was not the problem.

Compare the two mechanisms that *did* work:

- The **phases-era pairing check** asks whether the input matches what it is
  applied to (#29).
- The **decision map's resolution floor** asks how much the answer moves when
  a different session's inputs are used (#28's sibling; see
  `decision.SPEEDUP_RESOLUTION_BANDS`).

Both vary the input. The regeneration held it fixed.

### The convergence, which is the reason to trust the fix

Those two mechanisms have nothing in common. One is a provenance check on
decode era; the other is a variance measurement across three rented
sessions, derived without reference to the pairing bug or even to its
existence. They identify **the same five cells at 4096**.

That matters beyond this incident. The worry about the pairing fix was that
it had substituted one arbitrary answer for another — the era-matched phases
are not obviously more correct than the OLS ones for `normalized_ms`, since
prefill is era-independent. The resolution floor answers it from a different
direction: those five cells were never resolvable by *any* choice of input,
because the spread across sessions exceeds the effect. The fix did not pick
a better arbitrary answer; it stopped reporting an answer that was not there.
Independent methods landing on the same cells is hard to manufacture.

### The detection method

For any "I regenerated it and nothing changed":

1. **Name the inputs.** Which files did the producer read?
2. **Ask what would have to be true of each** for the output to be right.
3. **Check that separately**, by something that is not the producer.
4. **Find the second statement of the same quantity** — a claims row, a
   docstring, a printed line in a run log — and diff it against the artifact.
   The cost is a grep and it catches the case where both computations share
   an upstream fault.

A regeneration that varies nothing tests the code path. It says nothing at
all about the data.

---

## 32. A test's coverage is bounded by its fixture, not by its name

**Two instances in three days, the second found while writing up the first.**

The mechanism is one sentence: **an end-to-end test whose fixture happens to
satisfy one branch's precondition exercises that branch and claims to test
the function.**

### Instance A — `test_cross_backend_rejects_non_finite_outputs` (2026-09-11)

`check_cross_backend` has two non-finite guards: one on the backend's output,
one on each reference's. The test constructed a `_NonFiniteBackend` under
test and **two `_NonFiniteBackend` references**. Deleting the backend-side
guard — the one the test is named for — left it green, because the
reference-side guard failed the result for a different reason. The named
guard was never reached.

Removing it in reality let a NaN-emitting backend through as
`passed=True, max_abs_err=0.0, "agrees with 2 independent implementations"`.

### Instance B — the reconciliation repair test (2026-09-12)

`repair_reconciliation_identity.py` has two write paths: **annotate** (the
arithmetic is already right, the file only gains columns) and **rewrite**
(the retired `n × decode_step` identity, so everything derived is
recomputed). The new test ran the CLI end to end on a copy of the real
banked file — which was *already on the corrected identity*, so it went down
annotate every time. Deleting `add_bias_columns` from the rewrite branch left
it green.

A companion test asserted the schema by looking for the string
`"add_bias_columns"` in each script's source. That also stayed green: the
call was still present in the *other* branch.

### Why the two are the same failure

In both cases the fixture was chosen to be *realistic*, and realistic meant
"satisfies the common precondition". A NaN backend naturally suggested NaN
references; a repair test naturally used the real file, which is repaired.
The fixture's plausibility is what made it narrow.

Neither is fixed by "parametrise over branches" as a slogan. The fix is to
notice that the branch condition is **a property of the fixture**, and to
build a fixture that fails it:

- healthy references, so the backend-side guard is the only thing that can
  fire;
- a stale-identity frame, so the rewrite path is the only one that can run.

### The detection method

For any test that exercises a function with more than one path:

1. **List the branch conditions** in the function under test.
2. **Evaluate each against the fixture.** Any condition the fixture always
   satisfies or always fails marks a path the test never reaches.
3. **Delete the code in the unreached path** and re-run. Green means the test
   covers a subset of what its name claims.

Step 3 is the only one that is not opinion.

### And this is the argument for the break-it check being mandatory

Instance B was written **while documenting instance A**, by someone who had
spent the previous day auditing a third party's review for exactly this
shape, and who wrote the sentence "a guard that cannot fail, in the test
layer" about it. Knowing the pattern did not prevent reproducing it a day
later on a different surface.

That is not a remark about carelessness. It is the case for the break-it
check being a step that is always run rather than a habit that is relied on:
the knowledge does not transfer to the moment, and the mechanical check does
not need it to.

Both tests now go red on deletion — instance A verified 2026-09-11, instance
B parametrised over both branches and verified on each, 2026-09-12.

## 33. A retry loop that reports only the first refusal

A request can be refused for more than one reason at once. The API answers
with one of them, and which one it picks is not necessarily the one that
matters.

The H100 zone walk in `asia-southeast1` ran 71 create attempts across four
bursts and two days. Every one returned `ZONE_RESOURCE_POOL_EXHAUSTED`, so
the loop reported, correctly and repeatedly, that the zone had no capacity.
The conclusion drawn from 71 consistent observations was "keep trying, this
is transient."

It probably was not transient. `a3-highgpu-1g` needs 26 vCPU, and the region's
`PREEMPTIBLE_CPUS` grant was 12. A Flex Start instance is billed against the
preemptible quota, so the request was very likely *also* unplaceable on quota
-- permanently, at any hour, regardless of capacity. GCP evaluates
availability before quota, so the transient refusal fired first and masked the
permanent one on all 71 attempts. The loop was measuring the shorter of two
walls and reporting it as the only wall.

**The shape.** Every earlier entry in this document is a single check failing
to detect something. This is one check *masking* another: the loop's evidence
got stronger with each retry (71 consistent observations!) while its
conclusion stayed wrong, because retrying only ever re-samples the refusal
that fires first.

**The tell.** A retry loop whose failure reason never varies is not confirming
that reason. It is confirming the *ordering* of the checks upstream of it. If
the reason is genuinely transient, a long enough walk should eventually
produce a different message -- a success, or a second refusal. An unbroken run
of one identical error is evidence of a deterministic gate, not a flaky one.

**The rule.** Before concluding "no capacity" from any number of retries,
establish independently that the request is legal at all: check the quota that
actually governs it, from the quota API, not from the error text. The refusals
cost Rs 0, which is exactly why they bought no information.

**Sub-failure, same session.** The first version of this walk parsed the
reason with `grep -o 'code: [A-Z_]*'`, which matches the structured stockout
error and nothing else. The one-line quota error has no `code:` field, so a
zone refused on quota printed an empty reason. A failure reporter that can
only parse the failure you have already seen reports every new failure as
silence. The walk in `scripts/` now exits non-zero on any error class it does
not recognise, rather than continuing with an empty string.

## 34. An anti-vacuity guard calibrated at the wrong granularity

`tests/test_timed_region_setup.py` was written after FlexAttention's
`create_block_mask` was found inside the timed region -- a 10x error in a
reported number. The author knew the test could go vacuous (most backends
need CUDA and are unrunnable on a laptop), and added
`test_something_was_actually_exercised` specifically to stop a green suite
from meaning nothing.

The guard asserts that *some* backend was exercised. The property it protects
is per-backend: *this* backend's setup is hoisted. On a CPU host, `sdpa` and
`naive` run, the guard is satisfied, and `block_sparse` -- which rebuilds its
block mask on every forward, inside the timed region, in the path of the
headline 1.321x claim -- is quietly recorded as `not runnable here` and
asserted about not at all.

So the suite was green, the anti-vacuity guard was green, and the specific
assertion that mattered had no subject.

**Connect to #28.** The decode guard grouped by
`("backend", "task", "_band", "sparsity")` when `decode_backend` is a function
of `backend`, making the checked property constant within every group. Same
shape: a guard whose granularity does not match the property it asserts. It
has now happened twice, and the tell is identical both times --

> ask what the guard iterates over, and what the assertion is about. If the
> guard's unit is coarser than the assertion's unit, the guard can pass while
> the assertion is empty.

Set-level ("some backend ran") against a per-item property ("block_sparse
ran") is the coarse case. Group-level against a within-group-constant property
is the same error viewed from the other side.

**What the git history says, and what it does not.** The test was added
2026-09-04 in `6c46bc5`, and the `to_block_sparse_attn_mask` call it should
have caught predates it (`eee2c6b`). Both were present in 14 of the 16 commits
stamped into existing result sets, including the A100 session and every L4
Stage 2/3/5 run from 2026-09-04 on. So the test shipped to hosts where
`block_sparse` was runnable and the defect was live.

Whether it ran there is **not recoverable**. No session recorded pytest
output; the only stored per-session logs are boot serial consoles. The test's
own docstring says the backends unrunnable locally "ARE exercised by the same
test on the instance, where the suite is a per-session precondition" -- and
that precondition was asserted in prose and never once written down as an
artifact. A green suite nobody can produce evidence of is not a green suite.
The remedy is cheap and is now in `run_phase.sh`: the suite's output is a
phase artifact, synced to GCS like any result.

## 35. A cheaper reproduction narrows the search space past the bug

The 32768 fault was reproduced four times. The first three did not fire, and
each non-firing was read, briefly, as information about the bug. It was
information about the reproduction.

| attempt | what it ran | result |
|---|---|---|
| 1 | `forward()` over all 72 block_sparse configs | no fault |
| 2 | `run_once` on the smallest config of each `pass_kind` | no fault |
| 3 | same, at `b=1` | no fault |
| 4 | `run_once` over all 84 configs, in probe order | **fault at [68/84]** |

Attempt 1 missed it because `probe()` does not call `forward()`. It calls
`run_once` -> `timed_call`, and `timed_call` runs `out.sum().backward()` when
`pass_kind == "fwd_bwd"`. The faulting kernel is the **backward**; a
forward-only repro cannot reach it at any config.

Attempt 2 missed it because "smallest config" meant `b=1, hkv=8` -- GQA. The
fault needs `b=16, hkv=32`, which is MHA. Every simplification that made the
repro cheap also made it miss.

**The general property.** A reproduction is a hypothesis about which
dimensions are irrelevant. Running forward-only asserts the backward does not
matter. Taking the smallest config asserts batch and head geometry do not
matter. When the repro does not fire, exactly one of two things is true: there
is no bug on that path, or one of those assertions is false -- and the second
is far more likely, because the assertions were chosen for cheapness rather
than for evidence.

> When the error location is untrustworthy, a simplified repro that does not
> fire is evidence about the simplification, not about the bug.

**The inverse of #32.** There, a plausible fixture bounded a test's coverage
below what its name claimed. Here, a plausible simplification bounded a
*search* below where the bug lived. Same mechanism -- an unexamined narrowing
presented as the full thing -- pointed at finding a defect rather than at
catching one.

**The rule.** Start from the real path and remove one dimension at a time,
confirming the fault survives each removal. Four runs in the right order cost
the same as four in the wrong order; only the last one has to be the real
path, and it might as well be the first.

### Knowing a pattern does not prevent committing it

Both wrong diagnoses in this session were made *after* the relevant lesson was
in hand.

  - cuDNN was blamed because its 84 faults at 16384 fit the standing "cuDNN
    last" rule. A fix was written, committed, and deployed before anything
    tested whether cuDNN was the cause. Excluding it changed nothing.
  - `sdpa_flash` was blamed next, on the reasoning that it ran last before the
    crash -- the exact inference an asynchronous fault invalidates, made
    minutes after an asynchronous fault had already invalidated it once.

The second is the instructive one. The mechanism was understood, stated
explicitly in the session, and applied anyway on the very next question. That
is the argument for the break-it check being **mandatory rather than
habitual**: a habit is what fails under the pull of a hypothesis that fits.
The prior that fits the evidence too neatly is precisely the one worth
attacking, and the only reliable attack is to run the experiment that would
disconfirm it -- here, one 21-second isolation run that would have named
block_sparse before a line of code was changed.

### The corollary hazard: a fault class whose epistemic status depends on timing

Same GPU fault, two outcomes, decided entirely by whether the error surfaces
at its launch site:

    synchronous  -> probe() catches it -> "illegal_memory_access" is RECORDED
                    as a capability result, and the run continues
    asynchronous -> nothing raises, the context is dead, and the NEXT CUDA
                    call anywhere in the process dies instead

`sdpa_cudnn` faults 84 times per band and exits 0. `block_sparse` faults once
and takes the process, the band, and every unbanked row with it -- with a
traceback naming `torch.cuda.empty_cache()` in a different band, for a
different backend.

For anyone probing kernels at scale this is a live hazard, because the loud
failure is the harmless one. A backend that records 84 faults looks worse in
the results table than a backend that silently poisons the context, and is
strictly better behaved.

## 36. A guard is new code, and has the same defect rate as any other code

Three guards were written on 2026-09-16 to stop three real failures. All three
were themselves defective on first use, and every defect reproduced the class
of failure the guard existed to prevent.

| guard | written to prevent | its own defect |
|---|---|---|
| `run_phase.sh` per-phase sync | losing a session's results to a self-deleting instance | `mkdir -p logs` inside the repo left `logs/` untracked, so every row it wrote was stamped `git_dirty=True` |
| `gcp_preflight_instance.sh` quarantine | inheriting an L4-stamped `results/` tree from the machine image | `mv results <stamp>` deleted two **tracked** files that live under the gitignored `results/`, dirtying the tree |
| that quarantine's postcondition check | the defect immediately above | nested heredoc escaping did not survive; it died with `DIRT: unbound variable`, and once fixed immediately caught that the quarantine directory was itself untracked **inside the repo** |

The through-line is not carelessness; each guard was correct about the hazard
it named. It is that **a guard is ordinary code written at the moment of
greatest time pressure** — usually mid-session, usually on a billing clock,
usually right after an incident — and it inherits the defect rate of anything
else written that way. A mechanism that exists to make a failure impossible is
not thereby exempt from failing.

**The specific hazard: a broken check reports success by not running.** The
postcondition check that died on `DIRT: unbound variable` had been "in place"
for two commits. It had never executed its comparison. A guard that crashes is
visible; a guard whose *check* silently never evaluates is indistinguishable
from a guard that passes — which is #25 (verifying a guard with a fixture the
guard never read) arriving from a different direction.

**What actually worked.** Not review — all three were read carefully when
written. What caught them was **running them against the real thing and
checking the property they were supposed to establish**, one layer out:

  - `load_stage1_pass_set` rejected the dirty passes on a *dry run*, before a
    single Stage 2 cell was measured.
  - the fixed postcondition check caught the quarantine directory on its first
    live invocation.

Both are downstream consumers asserting the property, not the guard asserting
itself. **Write the guard, then verify the thing the guard is for, from
outside the guard.** The quarantine directory dirtying the tree by living
inside the repo is the miniature of the whole problem: the mechanism placed
its own working state in exactly the location whose cleanliness it was
protecting.


### Instance five of #34: the attention sink, tested in neither direction

`masks.py` had a definite, load-bearing behaviour — the attention sink was an
ordinary top-k candidate, retained for 65.0% of query blocks under the oracle
and 7.5% under a random mask — and **no test named it in either direction**.
Not one asserting the sink was forced, and not one asserting it was not. The
suite was green across every session that published an accuracy number.

The tell is the one now stated at #34: ask what property the code has, then
ask which test would fail if that property flipped. Here the answer was none,
in both directions, which is the signature of an untested behaviour rather
than a tested-and-permitted one.

The regression test written for it scores the sink **last** on purpose, so a
pass proves the sink is granted *outside* the budget rather than merely
winning a top-k it would usually win. A test that used realistic scores would
have passed against the defective code too.


## 37. The hazard was documented in the function the hazard was added to

`score_cache.cache_key` hashes `(model_id, task, example_id, seq_len)`. Its
docstring has carried, since the function was written, an explicit warning
that `model_id` must actually appear in the hashed blob and not merely be a
parameter the function accepts and forgets — and `tests/` has carried a test
asserting exactly that.

On 2026-09-16 a second scorer was added: MInference's mean-pool estimator,
alongside the dense-softmax oracle. `score_source` was threaded through the
model, the runner, the CLI and the schema — and added to `cache_key`'s
signature **without being added to its hashed blob**. Both scorers produced
key `ab224efd248582d5` for the same `(model, task, example, seq_len)`.

The consequence is the worst available shape of failure for this project:

  - the two scorers emit tensors of **identical shape**, so `load`'s
    dimension assertion — the belt-and-suspenders written for precisely this
    class of collision — cannot fire;
  - the cheap arm would have loaded the **oracle's** scores, stamped itself
    `minference_meanpool`, and reported no difference between the two;
  - that null would have appeared in the experiment built specifically to
    test whether the expensive oracle is necessary, where "no difference" is
    a publishable result rather than an obvious bug.

**A false null in a falsifiability experiment does not look like a failure.
It looks like the answer.** Every other defect in this document announces
itself as a wrong number, a crash, or a missing file. This one would have
announced itself as a finding.

**The part worth keeping is not the fix.** The fix is four words in a
`json.dumps`. What is worth recording is that the author of the defect had
written the warning against it, in that function, for the adjacent field, and
had written the test that catches it for the adjacent field — and then
instantiated it anyway, in the same function, roughly two hundred lines of
diff later.

So: **understanding a failure mode confers no protection against committing
it.** The mechanisms in this document are not redundancy for cases where the
hazard is unknown. #36 says a guard is ordinary code; this says the *author's
own knowledge* is not a guard at all. The only thing that caught it was
running the two scorers and comparing the keys — the break-it check, applied
to the field that had just been added, from outside the function that owns it.

**One decision generalises beyond this bug.** `score_source` is hashed
**unconditionally**, not only when it differs from the default. A key that
omits a field for backward compatibility is a key whose meaning depends on
which caller wrote it, which is the mechanism of the collision above rather
than an exception to it. The price was ~1,073 cached tensors re-keyed to
misses, all at 2048–8192 where the scoring pass is cheap. Pay it. A cache key
should name everything that determined the value.


## 38. A liveness probe whose terminal state is unreachable

A monitor was armed on the 16384 oracle phase with the termination condition
`ALIVE=0`, where `ALIVE` came from `pgrep -c -f run_accuracy.py` run over ssh.
It would never have fired. The probe's own `bash -c` command line contains the
string being searched for, so **the count has a floor of 1**:

    $ pgrep -c -f definitely_no_such_proc_xyz.py
    1

A monitor that cannot report the end of the thing it watches is silent when
the phase finishes and silent when the phase is healthy. It would have run to
its 60-minute timeout having reported progress lines and never the boundary —
and "no completion event yet" is indistinguishable from "still running." This
is the Monitor contract's own warning (*silence is not success*) arriving
through the terminal state rather than the failure state: the coverage gap was
not an unhandled crash signature, it was **the normal ending**.

**The diagnosis was also wrong, and that is the second lesson.** The
self-match mechanism above was inferred, the textbook fix applied
(`pgrep -f "[r]un_accuracy.py"`), and the count did not change — 3 either way.
The bracket form should have excluded the probe's own shell and did not, and
the reason was never established. What established the defect was not
reasoning about the mechanism but **measuring the floor directly**, against a
process name guaranteed not to exist. Same shape as the cuDNN misdiagnosis
earlier in this project: a plausible mechanism that fits the symptom is not
thereby the mechanism, and the cheapest way to tell is usually an experiment
rather than an argument.

**The fix was to stop using a process count at all.** The phase writes its own
completion marker (`########## ORACLE 16384 rc=$?`) to a log, which is
unambiguous, carries the exit status, and cannot match itself. Coverage for
the case where that marker never arrives is a separate condition on the ssh
session's own liveness — so the two silences, *phase ended* and *transport
died*, are distinguishable rather than both absent.

**Prefer a signal the watched thing emits over a signal the watcher derives.**
A derived signal has to model the thing; an emitted one only has to be read.

---

## 39. An equivalence proof validated on synthetic continuous data, applied to quantised real data

**2026-09-17, A100 session 7, item 4.** A vectorised mask builder was checked
against the reference on 36 synthetic configs and matched bit-for-bit. Against
the real model it disagreed on the first band measured: `seq_len=8192`, layer
3, sparsity 0.5, 2 of 4096 cells.

Nothing about the shapes was wrong. **The dtype was.** `score_cache.save`
stores scores as fp16, and fp16-rounded pooled softmax probabilities are
densely *tied*:

Measured on banked oracle tensors (Qwen2.5-1.5B, `seq_len=16384`, 28 layers,
candidates only) — not on a synthetic stand-in, because the first two attempts
to characterise this from a synthetic softmax were both wrong:

| `block_size` | exactly zero | **tied with another candidate** |
|---|---|---|
| 64 | 0.0% | **17.1%** |
| **128** (this study) | 0.0% | **3.0%** |
| 256 | 0.0% | 0.4% |

**Both of my synthetic estimates were wrong, in different ways.** The first
said "~62% of cells round to exact zero" — it counted the causally-masked
upper triangle, which is zero by construction and never a selection
candidate, so it measured the causal mask. The second fixed that and said
~30% zeros over candidates — still wrong, because a synthetic
`softmax(randn*6)` has a far heavier tail than real pooled attention. The
true zero rate is **0.0%**. Nothing underflows. The mechanism is fp16
*mantissa collision*: in a row of 256 candidates the pooled probabilities
average ~0.004, and distinct values land on the same representable fp16
number.

Two wrong synthetic estimates in a row, for a quantity that took one download
and thirty seconds to measure against banked real data, is the pattern inside
the pattern. See `docs/limitations.md` for the block-size dependence, which
also runs opposite to the direction first proposed.

`torch.rand` produces essentially no ties. So the synthetic check exercised
the one regime in which the two builders provably agree, and said nothing
about the regime that actually runs. The reference breaks ties with 1e-9
jitter; the vectorised builder uses a stable argsort. Wherever a tie group
straddles the per-row budget boundary they keep different, equal-scoring
blocks.

**The wrong claim was written down before it was measured.** The builder's
docstring argued the jitter "cannot change a top-k on non-tied input.
Equivalence is asserted below on non-tied scores, **which is the case that
occurs**." The first clause is true and the last is false, and the argument
reads as sound precisely because the true part carries the false part.

**Bitwise equality was also the wrong property to demand.** The measurement
needed the two builders to give the kernel the same work and the same
ranking, which is:

1. identical per-row **active count** — block-sparse cost depends on how many
   blocks are active, not which;
2. identical per-row **kept-score multiset** — which makes a disagreement
   provably tie-breaking rather than a ranking difference.

Both hold. So the builder is valid for the latency counterfactual it exists
for, and is **not** a drop-in for accuracy: a different equal-scoring block is
still a different block, and interchangeable-by-score is not
interchangeable-by-output.

**Two generalisations worth keeping.**

A guard relaxed after it fires has to be shown to still bite. The relaxed
check is run against a negative control — a builder that inverts the ranking —
and rejects it. Without that, "relax the assertion until it passes" is
indistinguishable from deleting it.

And the generator of the test data is part of the test. A check whose inputs
come from `torch.rand` has silently assumed continuity, distinctness and full
precision. Where the real pipeline quantises, pools, or saturates, the test
data must do the same, or the check is exercising a regime that never runs.

---

## 40. Elapsed time reported from arithmetic on remembered timestamps

**2026-09-17, A100 session 10. ₹429.** After a guard refused to run item 4 at
09:20:15Z, the diagnosis that followed was entirely CPU-local: reproducing the
failure on banked tensors, measuring fp32 ulp against the reference's jitter
magnitude, writing two regression tests, running the suite three times. None
of it touched the GPU. The instance was not deleted, and it idled for **90.6
minutes** until its own in-guest shutdown fired at 10:50:51Z.

Throughout, progress updates reported elapsed time computed from *remembered*
timestamps — "boot was 07:19, a few things have happened, call it two hours" —
rather than read from the clock. Each report was plausible and each was
further behind the truth than the last. The final reported figure was 2.0 h
against an actual 3.5 h.

**The silence here is of a specific kind: a quantity that only ever moves in
one direction and is never checked.** A wrong latency number gets contradicted
by the next measurement. A wrong elapsed-time estimate is contradicted by
nothing, because nothing else in the session reads it, and it drifts
monotonically away from the truth for as long as the session lasts.

**This is the third instance of one shape, and the fix has been the same
every time.** A number maintained by restatement, authoritative-looking, with
nothing that would notice it going wrong:

| # | the number | how it drifted | the fix |
|---|---|---|---|
| **#3** | `provenance.git_commit` | recorded the literal string `"HEAD"`; a check compared `"HEAD"` against `"HEAD"`, agreed, and certified nothing | read it from `git rev-parse`, fail closed at the point of use |
| — | the running spend total in `docs/spend_ledger.md` | restated by hand through ₹902 / 910 / 918 / 942, a ~4% spread by 2026-09-04 | assembled from `session_cost.txt`, written from the instance's own boot clock |
| **#40** | elapsed time in progress reports | arithmetic on remembered timestamps, 2.0 h reported against 3.5 h actual | `scripts/gcp_session_elapsed.sh`, read from the GCE operations log |

The spend case is the sharpest precedent, because **the fix was already
applied to one half of the pair and not the other.** `session_cost.txt` has
read the instance's own clock since 2026-09-04 — but only at teardown. The
elapsed figure quoted mid-session had no source to read, so it was remembered
while the cost figure was measured. **Two numbers describing the same
interval, one read and one recalled, can disagree indefinitely and nothing
notices** — there is no consistency check between them because they are never
computed in the same place.

`scripts/gcp_session_elapsed.sh` closes that: same derivation as
`session_cost.txt`, available at any point rather than once. It reads boot
from the GCE operations log rather than the guest, so it needs no SSH and
still works when sshd is wedged — which is precisely when a session is most
likely to be quietly burning.

**Two rules, both cheap.**

**Read the clock, don't derive it.** `date -u` costs nothing and the
instance's own `gcloud compute operations list` is authoritative for boot,
guest-shutdown and delete — it is the record that survives the instance. It
was not consulted here until after deletion, at which point it immediately
gave the correct timeline.

**A diagnosis that runs on a laptop is not a reason to hold an accelerator.**
The instinct to keep the instance "in case the fix is quick" is what converts
a ten-minute fix into a ninety-minute bill. Delete on the guard failure, and
re-create when there is something to run. A boot costs ~₹100 and 25 minutes of
setup; ninety idle minutes cost ₹429. The arithmetic is not close, and it was
never done.

**What did NOT fail, and is worth separating out.** The teardown structure
held: the in-guest halt fired 58 minutes ahead of the API-level DELETE,
leaving the intended disk-recovery window, and every phase had already synced
to GCS, so deletion cost nothing but time. The money was lost to a decision,
not to a missing safeguard — which is the harder kind to add a guard for, and
the reason it is written down here instead.

---

## 41. Composed estimates of measured quantities: reliable for sign, unreliable for size

Four instances in this project, the last one measured end-to-end so the error
could be quantified rather than suspected.

| # | composition | direction | magnitude |
|---|---|---|---|
| kernel TFLOPS → whole-model | right | wrong (attention is a small share of model FLOPs) |
| prefill throughput → decode | right | wrong (decode is memory-bound, prefill compute-bound) |
| `28 × T` conversion tax added to Stage 2 cells | right | **double-counted** — `timing.measure()` takes the mask as a parameter, so the per-call conversion was already inside |
| kernel saving + mask construction → end-to-end gap | right in all 6 cells | **wrong in both directions** |

The last one is the clean experiment, because only one input changed. The same
arithmetic, fed a strictly *better* measured term — mask construction timed on
the instance's own CPU rather than a laptop's, 3.7× slower and therefore more
accurate — moved from **undershooting** the observed gap by ~2.5× to
**overshooting** it by 1.7–2.5×. Improving an input made the prediction worse.

**That is the diagnostic.** When a better input degrades a composed estimate,
the error is not in the terms; it is in the composition. Each term carries its
own measurement error and its own scope conditions, the composition multiplies
and adds them, and **nothing cross-checks the total** — there is no measured
quantity for the sum to disagree with, which is exactly why the sum survives.

**The rule.** Composed estimates of measured quantities are usable for
**sign** and not for **size**. Use them to decide whether an effect exists and
which way it points, to choose what to measure next, and to sanity-check a
direct measurement's plausibility. Do not publish their magnitudes, and do not
let a composed magnitude stand in for the measurement it is a proxy for.

The end-to-end run that replaced the composition here took **2 minutes 13
seconds** of GPU time. The composed budget it replaced had been revised across
three sessions, produced one withdrawn figure (459.6 ms) and one withdrawn
double-count, and was wrong by 2–3× in both directions at the end of it. **The
direct measurement was always cheaper than the argument about it.**

---

## 42. A reconciliation across two regimes, with the confound collinear with the variable blamed

**Diagnosed 2026-09-19, a week after it was recorded.** `results/stage5_flashdecode/`
failed its sign test at p = 0.00028: 21 of 24 residuals negative, mean
−143.7 ms, every failing cell `block_sparse`, every one at the long generation
length. The record said so precisely and offered no cause, and then drew one
conclusion anyway: that the phase model should not be trusted to transfer
across generation lengths.

**That conclusion was false.** The failing set reconciled Stage 5 phases —
sparse arms decoding through flash — against observed totals from the
original Stage 3 bands, in which every `block_sparse` row decoded through
`sdpa_math`. The residual was the decode-kernel gap: 83–98% of it per step,
reproducing even its odd shape. Reconciled against flash-decode totals, the
same phases close 12 of 12 at p = 0.146.

And in that set, **generation length and decode kernel were perfectly
collinear.** The short (n = 8) rows were Stage 5's own flash-decode runs; the
long (n ≈ 30) rows were Stage 3's math-decode runs. So "fails at long
generation" and "fails when the kernels differ" were the same observation,
and the record reached for the variable it could see.

**The mechanism of the silence.** Nothing in `run_phase_timing.py` compared
the regime of its two inputs. The observed parquet *carried* a
`decode_backend` column the whole time — the fact was recorded, and read by
no gate, which is pattern #10 and the analysis-stamp debt again in a third
form. A reconciliation across two regimes does not test a model; it measures
the difference between the regimes and attributes it to whatever else varies.

**Two things worth keeping.**

*Refusing to name a cause was right, and naming a consequence was the same
error in disguise.* The record declined to explain the bias and then used it
to limit the model's validity — which is an explanation, just phrased as a
caveat. "Undiagnosed" should bound what is claimed about the cause *and*
about what the anomaly implies.

*The structure that was recorded honestly is what solved it.* "Every failing
cell is block_sparse, every one at long generation, n = 8 closes" was enough
to eliminate two plausible mechanisms and point at the input provenance,
entirely from banked data, at no GPU cost. The fix is
`check_observed_decode_regime`, which refuses observed rows whose
`decode_backend` differs from the kernel the phase model measures, and is
tested against the real inputs that caused this.


## 43. A timestamp that records when a file was copied, read as when it was computed

Found 2026-09-19 by the single-measurement audit, item S6
([`docs/audit_register.md`](audit_register.md)). The question: could
the banked 1.5B cheap-estimator run have loaded score tensors written by code
with the "cheap scorer returns zeros" bug (fixed in `562374a`)? The cache key
carries no code version and the rows record no cache hits, so the obvious
evidence was the objects' creation times in GCS against the run windows.

Every one of the 1,706 score-cache objects is stamped 13:xx UTC. That
includes tensors computed by the oracle run, which ended at 11:32. The times
are when each object was last *synced*: `run_phase.sh` syncs the whole cache
directory at every phase exit, and each sync rewrites the objects. So the
field is uniform, looks like an answer, and carries no information about the
question being asked.

**Closed by the session log instead.** The only failed cheap attempt
(12:31:14–12:31:37Z) raised its `NameError` in `build_generate_fn`, before the
model was constructed, so it scored nothing. No cheap phase ran before the
zeros fix landed at 12:29Z. Every cheap tensor the banked run used was
therefore computed by that run.

**Same family as `sm_clock_mhz_at_capture` (commit `a157f03`)** and the
`git_commit="HEAD"` of #3: a field that names the property you want and
records something adjacent to it. Here the adjacent thing is the transport
event, not the computation. The detection is to ask what event writes the
field, not what its name says, and to check whether values that should differ
(two runs two hours apart) do.

**Rule.** Object-store metadata describes the store's own operations. Evidence
of *when content was produced* has to be written by the producer: a log line
or a field inside the file. Nothing in the score cache has one yet. If the
cache's provenance ever matters again, the fix is a small sidecar written at
`score_cache.save` (commit and score_source), not a re-read of bucket
metadata.


## 44. Two measurements agreeing, because they share an input

Found 2026-09-20, when audit item S1a withdrew the convergence claim.

The claim was that block-sparse attention's speed benefit and its accuracy
both rise with context length — *"two independent measurements agreeing on a
length dependence, neither designed to produce it."* That sentence was the
reason it was believed, and it is stated in `claims.md` as the reason.

The two measurements were a latency sweep and an accuracy sweep. They are
independent in every way people usually check: different scripts, different
output files, different sessions, different failure modes, neither written to
show a length dependence. **They ran on masks built by the same function**,
and that function omitted the attention sink. Re-measured with the sink
forced, the accuracy ladder at 0.9 reads 97 / 99 / 100 instead of 62 / 93 /
99 — no slope — while the speed ladder is unchanged. One of the two trends
was an artifact of the shared input. The agreement was never evidence about
length; it was evidence that both measurements used the same mask builder.

**The rule.** Two measurements corroborate a phenomenon only to the extent
their *inputs* are independent, not their code paths, authors or sessions. An
input every measurement shares — a mask builder, a tokenizer, a prompt
template, a generated example set, a cached tensor — is a common cause, and a
defect in it produces agreement of exactly the shape corroboration produces.
Before treating agreement as confirmation, list what both sides consumed.
Where the list is non-empty, the agreement is conditional on those inputs and
should be written down that way.

**Why this is not #42.** Pattern 42 is one comparison whose confound is
collinear with the variable being credited — a single measurement that cannot
separate two causes. This is *two* measurements, each individually sound,
whose agreement carries no information because the thing they share is the
thing that is wrong. #42 is about what one number cannot distinguish; this is
about why a second number did not help.

**What made it diagnosable, and it is worth copying.** The re-run held
everything fixed except the mask rule — same example ids, decode kernel
pinned to the original regime, per-arm score precision reproduced — and the
dense arm was required to reproduce the banked run **exactly, 300/300
examples at every band, as a gate that fails the session** (`scripts/
check_dense_canary.py`). Because the dense arm reproduced, the change in the
sparse arms could be attributed to the mask rule rather than to the machine,
the model, the library stack or the examples. An unchanged control that is
*verified per example* is what converts "the numbers moved" into "this input
moved them".

**The honest form of the retraction.** The individual measurements were
correct and stay in the file. What is withdrawn is the inference *from their
agreement*. A reader who finds two agreeing results in this study should now
check `limitations.md` for what they were both built on before treating the
second as support for the first.

---

## 45. A shared RNG stream advanced under a condition the arms do not share

Found 2026-09-20 by an external adversarial audit of the whole repository. It
is the first entry in this file found by someone who did not write the code,
and it had survived every prior audit, the 976-test suite, and a diagnostic
written specifically to test the property it broke.

`masks.importance_block_mask` builds the sparsity ladder. Its contract is
**nesting**: the 0.9 mask must be a strict subset of the 0.75 mask, which must
be a strict subset of the 0.5 mask, so that accuracy-versus-sparsity measures
sparsity rather than resampling noise. The module docstring states this as an
automatic consequence of the design — *"deriving the shuffle/ranking from an
identity that excludes `sparsity` … makes that automatic rather than something
callers arrange."*

One generator `g` serves every query-block row, supplying the jitter that
breaks exact score ties. The loop read:

```python
budget = round((1.0 - sparsity) * len(kvs))
if budget <= 0:
    continue                                     # <-- returns before the draw
scores = importance_scores[qb, kvs]
jitter = torch.rand(len(kvs), generator=g) * 1e-9
```

`budget` depends on `sparsity`. Short rows cross the `budget <= 0` threshold at
different sparsities, so an arm that skips a row's draw leaves the generator
one row behind an arm that takes it. At 2048/`block_size=128` the 0.5 stream
has consumed 14 draws by `qb=7`, the 0.75 stream 12, and the 0.9 stream none.
**From the first short row onward, no two arms of the ladder ever draw the same
jitter.** Where scores tie, the jitter *is* the decision, and three different
tie-breaks are three unrelated masks.

**Where it bit.** `_scoring_forward_chunked` zero-pads the final query block
to a full `block_size` before pooling, so when a prompt does not land on a
block boundary that row's pooled means are divided by the padding and fall out
of fp16's range. Measured on banked oracle tensors, **52.8% of that row's
candidates are exactly 0.0** at `n_blocks=65`. Its top-k is therefore decided
entirely by jitter — and 95–100% of every banked accuracy row has a ragged
final block, because `ruler.generate_examples` solves the filler count once per
budget and reuses it while needle content varies. Nesting broke in **70 of 112
layer-masks** at that block count, and in **126 of the 128 `seq_len` values**
that produce it, so it was never a seed artifact.

**The rule.** *A shared RNG stream is a shared input, and the condition under
which it advances is part of its identity.* Any branch that can skip a draw
must not depend on a variable the callers are supposed to hold constant. Draw
unconditionally, before the branch — or give each arm its own generator. The
cost of an unused draw is nothing; the cost of a shifted stream is that two
runs which agree about everything they were supposed to agree about disagree
about everything they were not.

### Why three independent checks all missed it

This is the reusable part, and all three failures have the same shape:
**a fixture whose distribution cannot express the failure.**

1. **`test_masks_determinism.py` seeded with `torch.rand`.** Continuous floats
   essentially never tie, so jitter never decided anything and the property
   under test was never exercised. `limitations.md` *already said this* —
   "torch.rand produces essentially no ties" — in a section about a different
   bug. The sentence that names the blind spot was written, filed, and not
   connected to the test that had it.
2. **`check_score_dtype_asymmetry.py` used real scores and real `seq_len`s,
   and still missed it.** It computes exactly the right quantity
   (`nest_break_single_ranking`) and reported 0 at 2048/4096/8192. It sampled
   3–9 examples per band, and the examples it drew happened to have well-filled
   final blocks. The instrument was correct; the *sample* could not reach the
   configuration where the defect lives.
3. **A structural argument stood in for a measurement at 16384.**
   `limitations.md` said 16384 "was not sampled" and that "the structural
   argument carries the claim, not a sample". The structural argument assumed
   the arms share a tie-break. That assumption was the bug. **An argument is
   only as good as the premise nobody checked, and the premise here was a
   property of the code, which is checkable.**

**The countermeasure, applied.** The fixture is now adversarial rather than
representative: `_tied_scores` builds a tensor shaped like the real failing
case (fp16 round-trip, near-empty final block), parametrised over the block
counts that actually occur — 16, 17, 32, 65, 128 — and the ragged ones are
included *because* they are ragged. Alongside it,
`test_every_sparsity_draws_the_same_jitter_for_the_same_row` asserts the
mechanism directly by recording the draw sequence, so a shifted stream is
caught even where no fixture happens to tie. All nine parametrisations were
verified to fail against the pre-fix code before the fix was kept.

### The fix that was written down, and would not have worked

`limitations.md` carried a proposed remedy for a *neighbouring* defect — the
cold-cache fp32 / warm-cache fp16 split — that read: *"return the fp16
round-trip on a miss too, so every arm ranks from one tensor."*

It is a correct fix for that problem and it would not have touched this one.
The arms did not need to share a *tensor*; they needed to share a
*tie-break*. The reproduction that found this used a single fp16 tensor for
all three sparsities and broke anyway. **A remedy filed against the symptom
you noticed does not generalise to the symptom you did not** — and a remedy
sitting unexecuted in a limitations file is especially prone to this, because
nothing ever runs it and discovers its scope.

### What it cost, and what it moved (S7, closed 2026-09-21)

Re-running the fixed builder over every banked score tensor changes 0.23% of
all active blocks — **itself a pooled figure, and understating the same way
the 5.6% below does**: it is 0.0% wherever `seq_len` divides evenly, 0.92% at
`n_blocks=17` and 1.67% at 65, and the evenly-dividing configurations dominate
the denominator. Both numbers in this section's opening sentence are pools;
only one of them was marked as such until 2026-09-21, which is how a
correction gets applied to a figure and not to the figure beside it. The
fraction of *masks* containing such a change is
strongly length-dependent -- 1.7% at 2048, 5.8% at 4096, 35.3% at 8192,
**73.8% at 16384**, 99.2% at 32768 -- because longer contexts pool thinner
probabilities into more candidates per row and tie more often in fp16. Untied
rows are bit-identical, since a different 1e-9 perturbation cannot reorder
distinct scores.

**The pooled figure across all bands is 5.6%, and quoting it was itself an
instance of #13.** 81% of the banked tensors are 2048-band, so the pool
describes the short bands. It was the first number computed and it reached
both `limitations.md` and this file's own commit message before the per-band
split existed. A blast radius averaged over a population the reader will not
inspect is not a bound.

**Whether any reported accuracy number moved is measured — S7, closed
2026-09-21.** The perturbation is confined to one query-block row out of
16–129 — but it is the row the answer is generated from, and at 16384, the
headline band, 76 of 100 examples were in the vulnerable configuration. The
one GPU band it needed has been run: dense canary 200/200 identical, score
canary 200/200 bit-identical, and `niah_multikey` moved 66/59/20 → 65/53/17
(−6.0 at 0.75, 95% CI [−12.0, −1.0], excludes zero) while `niah_single` held
at ceiling despite 14/2/2 of its texts changing underneath it. The honest
statement is now that the ladder was not a ladder in the tied rows, the
affected fraction is small and measured, and the effect on scores is a real,
measured −6.0-point drop at one cell.

---

## 46. The repository's prose has no equivalent of a test suite

Found 2026-09-20, during the commit of instances 45 and the documentation
corrections that came with it. It is the only entry in this file produced by
*fixing* the others, and it is here because the artifact it damaged is this
project's actual deliverable.

Four workstreams had to land as four independently revertable commits. The
edits were interleaved inside shared files — `claims.md` carried hunks from
three of them — so each commit's tree was reconstructed by splicing the
relevant diff hunks onto the file as `git show HEAD:<file>` returned it.

`HEAD` moves as you commit. The first commit was built against the right base.
The second was built against the first, **with hunk offsets still computed
against the original**, so it spliced text at coordinates that no longer meant
what they had. `claims.md` came out mangled: duplicated paragraphs, a table
cut mid-row, roughly eighty lines wrong.

**The test suite was green. 985 passed.** It was green because the corruption
was entirely in prose, and no test in this project reads `claims.md` for
sense. The commit was made and stood for two more build steps.

What caught it was not a test. It was a `--stat` line:

```
docs/claims.md | 79 ++++++--------------------------------
```

in a commit whose workstream does not touch `claims.md`. A file that should
not have appeared, and a deletion count that had no business existing in an
additive change.

**The rule.** *A repository's executable artifacts have a gate and its prose
artifacts do not, so any mechanical transformation of prose needs its own
end-state check.* The fix was to pin the base to a fixed commit rather than
`HEAD`, and — the part that actually made it safe — to verify the final tree
**byte-for-byte against the state that had been validated** before any of the
commits were made, rather than trusting that four correct-looking steps
compose. Every intermediate state also had its test count recorded (976 → 985
→ 986 → 999), which is what makes a silently-skipped or silently-added step
visible.

**Why this matters more here than it usually would.** In most repositories the
prose is documentation *about* the deliverable. In this one `claims.md` **is**
the deliverable — it is the file the write-up is drafted from, and the whole
premise of keeping it is that a claim and its boundary travel as one sentence.
A corrupted `claims.md` that passes CI is a corrupted paper that passes review.

**The shape is the file's own.** A green suite over a wrong artifact, caught by
a number that did not fit — #10 (a correct stamp nobody read), #23 (a measured
value a default overwrote), #45 (a fixture that could not express the failure).
Here the missing gate was not a fixture or a reader; it was that prose has no
executable form to assert against at all. That is not fixable in general, and
the countermeasure is correspondingly blunt: **diff the end state against a
known-good copy, and do not trust a sequence of individually plausible
mechanical edits to compose into it.**

## 47. A guard whose own tests are vacuous on the platform that runs them

`scripts/procmatch.sh` exists because `pkill -f` killed an invoking SSH
command and took a build with it (2026-09-03), and because `pgrep -f` reported
a dead probe as RUNNING for six minutes (2026-09-04). It walks the ancestor
chain and the process group so that neither can happen again. Its test file
says so, at length, and the suite was green.

On 2026-09-20 the suite was run **on the instance** for the first time with
its output kept. Three of that script's own tests fail on Linux:

    test_a_pattern_matching_only_the_caller_reports_not_running
    test_the_reported_pids_exclude_the_matcher_itself
    test_kill_refuses_when_only_the_caller_matches

Reproduced by hand on the box: `status` against a pattern matching nothing
answered `RUNNING pids=10710`, and `kill` against the same pattern printed
`killing -TERM: 3697`. **The guard reproduces both incidents it was written to
prevent.**

One cause. The pid `pgrep` returns is a subshell the command forked for a
pipeline — it carries the argv, so it matches, and it exits in milliseconds.
Both filters then ask `ps` about a pid that is gone, `ps` prints nothing, and
nothing satisfied either test: an empty pgid is not equal to ours, and an
empty command line does not contain `procmatch.sh`. The phantom fell through
and was reported as genuine.

**Two rules, and the second is the one that generalises.**

**A filter must fail closed.** Both of these failed open, in the direction the
script's own docstring rules out — it must err toward NOT_RUNNING, "because a
false 'not running' is investigated while a false 'running' is believed". An
exclusion rule written as "skip it if `ps` says X" silently becomes "keep it"
the moment `ps` says nothing, and `ps` saying nothing is the *normal* outcome
for exactly the pids this filter is aimed at.

**A test can be vacuous by platform, and the docstring saying so is not a
substitute for a test that is not.** This file's own author had written it
down: macOS `pgrep -f` cannot see an ancestor shell's command line, so "an
integration test for it here is vacuous BY PLATFORM, which is exactly the
'green tests can test nothing' trap." The trap was named, in the file, and
then walked into — because naming it did not produce a test that runs
everywhere. The fix adds three that do, by handing the matcher a pid that has
already exited; they fail on macOS too. **A known-vacuous test is a gap with a
comment on it, and a comment does not fail.**

This is #21 and #22's shape ("tested on CPU" that tests the arithmetic and not
the interfaces) applied to the operator tooling rather than the measurement
code, and it is why the suite is now a per-session phase with its output
synced (`scripts/gpu_suite_record.sh`) rather than a green tick remembered
from the workstation.

## 48. A rule that was already derived, already demonstrated, and dropped at the call site

The 2026-09-20 S7 session finished at 17:51Z, on estimate, and was collected
by its own 130-minute cap at 18:40Z. **Forty-nine idle minutes, ~Rs 64, on a
session whose useful work cost Rs 108.**

The first write-up of this called it "a spend ceiling used as a teardown" and
drew the lesson that a bound on the worst case is not a plan for the expected
case. That is true and it is not the finding, because **the project had
already derived that rule, and demonstrated it**:

> A long run must arm its own teardown on completion, so the idle window is
> bounded by the machine rather than by whether anyone is awake.
> — `spend_ledger.md`, after the 2026-09-06 row burned ~7 idle hours

It was demonstrated on 2026-09-07, when a chained `; sudo shutdown -h +5`
fired with nobody watching (341 min, Rs 455, against a Rs 581–641 bracket).
It was re-derived after the 2026-09-17 A100 session idled 90.6 minutes for
**Rs 429**. And the standing rule — *never end a turn with an instance
running* — was written down as well.

**It was in the usage line of the script that was run.**

    # Usage (on the instance, from the repo root):
    #   bash scripts/s7_run_jitter_band.sh; sudo shutdown -h +5

The launch command was `nohup bash scripts/s7_run_jitter_band.sh > /tmp/band.log 2>&1 &`.
The chain was dropped. Not overruled, not traded away — dropped, by an
operator who had written that usage line an hour earlier.

**This is instance 47 in a different costume.** There the lesson was that "a
rule that must be remembered at every call site will be forgotten at one of
them", and the `[n]vcc` bracket trick had been written down twice before it
was forgotten twice. Here the rule had been written down three times, priced
at Rs 575, Rs 429 and a standing turn-level instruction, and was forgotten at
the one call site that mattered. **Documentation is not a mechanism. The
number of times a rule is written down is not correlated with whether it is
followed; it is evidence that it keeps not being followed.**

So it is now armed inside `s7_run_jitter_band.sh`, `s1a_run_band.sh` and
`sink_control_run.sh`, in the same `finish()` that already writes the rc —
+5 minutes on success, +20 on failure so a broken run can still be inspected,
`ATTNBENCH_NO_HALT=1` to opt out when chaining bands. The thing an operator
now has to remember is the *exception*, not the rule.

**The general form.** When an incident's write-up ends in a rule for a human
to follow, the write-up is not finished. Ask what would have to be true for
the rule to be unforgettable, and build that instead. A rule stated for the
third time is a design task that has been deferred twice.

## 49. A correction's replacement value goes stale inside its own disclaimer

**Found 2026-09-21, by the guard written for instance 46 — but not by the
rule that guard was built on.** `docs/claims.md` carries a table titled "What
survives the correction", whose entire function is recording which of the
study's claims have gone stale. One row read:

    | *16 of 31 matched sparse points are dominated by dense* |
      **DOES NOT SURVIVE.** Normalized: **12 of 31**. |

`16 of 31` is correctly marked. `12 of 31` is the answer the row offers in its
place, and it had itself been superseded by `15 of 34` on 2026-09-20 by the
S1a rebuild. The row was wrong in the half a reader would actually use.

**Why nothing found it.** `tests/test_no_stale_figures.py` permits an
occurrence inside a block carrying a withdrawal marker. The marker in that row
is about the left-hand figure; the checker has no way to know that, so it
covers the whole row — including the replacement. **A withdrawal note is a
permanently marked context, so every later supersession hides inside the
previous one's disclaimer.** The more diligently a document records its own
corrections, the more marked ground it creates for the next stale figure to
stand on.

Two adversarial audits read that table and missed it, and so did the first
version of the guard built specifically to catch stale figures in prose. It
was found only when the guard was tightened for an unrelated reason and the
tightening was break-tested against the historical text — the mutation passed,
which was the signal that the diagnosis was wrong.

**The rule that was measured and rejected.** The obvious generalisation is
"inside a marked unit, only the first registry match is covered". Written and
run against the live tree, it produced three violations, all of them on blocks
quoting a withdrawn paragraph verbatim. That construction is the one the
registry exists to encourage — it is how the provenance of a number survives
its correction — so the rule would have bought a narrow catch by taxing the
mechanism's main benefit. **A guard that makes the right practice expensive
will be switched off, and it will be switched off by someone who is right to
do it.**

**What shipped instead.** The rule is restricted to markdown table rows, where
it flags nothing in the live tree and still catches the row above. The
justification is structural rather than empirical: a quotation is one act,
governed by one disclaimer, while a table row is a record whose verdict cell
is asserted in the reader's present tense. Supersession tables are therefore
exactly where a replacement value is stated as current, and exactly where this
fails. `docs/withdrawn_figures.md` gained a second check at the same time —
no entry may name a replacement that is itself a withdrawn figure — because
the moment a figure is withdrawn is the only moment anyone is looking at the
entries that pointed to it.

**The general form.** A disclaimer covers the claim it was written about, not
the text it happens to share a container with. Any mechanism that grants
immunity by proximity — a marker covering a block, an allowlist entry covering
a directory, a `# noqa` covering a line with two problems on it — grants it to
whatever moves in next. Ask what the exemption was *about*, and whether the
thing now sitting inside it is that.


## 50. A false claim copied into new code, one day after the guard against copying was built

**Found 2026-09-21, by running `git log` for the first time.** Both audits ran
in a tree with no `.git`, so every commit-dependent sentence was accepted as
written. When the real checkout arrived, the first check confirmed the era-1
boundary exactly as documented. The second one did not.

`limitations.md` said:

> Era 1 and 2 split on `git_commit`: any commit that is an ancestor of
> `37675a0` is era 1. Era 2 and 3 split on date only — the jitter fix carries
> no schema change, so a row cannot be assigned between them from its own
> contents. That is a deliberate limitation being recorded rather than a gap.

The first sentence is true. The second is false, and false twice over. The
jitter fix **is** a commit, `5cc3a40`, so the mechanism named in the sentence
before it settles the question: all seven era-2 commits predate it, the single
era-3 commit descends from it, and the partition is exact. And the substitute
rule does not work — `179c894`, `44ab65c` and `33598b4` are era 2 and share
2026-09-20 with `39e1d6d`, which is era 3, so a date rule at day granularity
misfiles three of four.

**The part that makes it an instance rather than an erratum.** On 2026-09-21 a
new module, `attnbench/analysis/eras.py`, was written to close the S12 gap —
the discovery that three documents credited `analysis/composition.py` with
refusing cross-era comparisons when it has no concept of a commit. That module
is the project's era authority. Its docstring opened by restating the
paragraph above, verbatim in substance, including the claim that a row "cannot
be assigned between them from its own contents".

So the fix for one propagated false claim propagated another, in the same
session, **one day after `tests/test_no_stale_figures.py` was built to make
exactly this a test failure** — and it was invisible to that guard, because
the guard watches *figures* and this was a *mechanism*. Nobody re-derived the
sentence from the commit graph at any point: not when it was first written,
not when the era table was added, not when the new module was reviewed, and
not when the tier containing it was approved.

**What was actually wrong with the reasoning.** The paragraph reached a
correct conclusion — do not add a `mask_rule` column — through an incorrect
premise. A half-populated provenance field really is worse than none, and that
argument survives untouched. But "the information is unavailable" was doing
the persuading, and it was the part that was false. **An argument that reaches
the right answer through a wrong premise is harder to catch than one that
reaches the wrong answer**, because the conclusion looks reviewed.

**Fixed** by giving `eras.py` the two boundary commits as constants and an
`era_from_git()` that derives the era with `git merge-base --is-ancestor`.
`COMMIT_ERA` is now documented as a cache of that function — it has to exist,
because the comparison scripts must run in a checkout with banked results and
no history, which is the condition that let this survive two audits — and
`tests/test_eras.py` requires the two to agree wherever git is present, with
break-tests for a mislabelled entry and for a wrong boundary commit.

**The general form.** A guard written against a defect you have just found
will be shaped like that defect. This project's prose guard watches withdrawn
*numbers*, so a withdrawn *mechanism* walked straight past it into new code.
When you build a check in response to an incident, ask what the incident's
sibling looks like — same failure, different type of thing — because that is
what will arrive next, and your new check will not be looking at it.
