# attnbench - Copilot Review Plan: Numerical & Logical Integrity

## Why this plan is shaped the way it is

This isn't a generic "look for bugs" checklist. `docs/silent_failure_patterns.md`
records 27 confirmed incidents, and every one shares the same shape: **no
crash, no failed test, no error - a plausible number, produced by machinery
that looked like it was working.** Nothing in this project's history was
caught by review-as-reading; everything was caught by someone distrusting a
number that was too convenient. That's the standard this plan asks Copilot to
apply: for every check below, the question isn't "does this code look wrong"
but "if this were silently wrong, would anything here notice?"

Because `docs/claims.md` is the ledger every write-up is drafted from, and
every claim in it is only as good as the code producing the numbers behind
it, the review target is: **would a bug in this diff change a number in
claims.md without anything downstream objecting?**

---

## Ground rules for every review

- Any new derived number needs a `claims.md` row before it's reportable - if
a diff changes what a number means, ask whether an existing row's qualifier
(`haystack_mode`, `score_source`, `gate_source`, `check_kind`,
`decode_backend`/`prefill_backend`, `clocks_locked`) still holds.
- A recorded field is not a safeguard unless something *reads* it and
*refuses* on a bad value. If a diff adds a field to a row, ask what consumes
it. If nothing does yet, say so explicitly rather than assuming future
coverage.
- Known, deliberately-documented limitations (oracle scores are an upper
bound, decode is dense-only, RULER haystack is substituted, GLA has no
accuracy arm) are **not bugs**. Don't re-flag them. Only flag if the code
fails to enforce or record the boundary the docs claim exists.

---

## Category checklist

Each category names the confirmed incident(s) it comes from, what to check,
where in the codebase it tends to live, and the concrete action for Copilot.

### 1. Command exit status read from stdout, not the return code

**Precedent:** `git rev-parse HEAD` echoing the literal string `"HEAD"` on
failure (#3); `nvidia-smi -lgc` printing a permission error to stdout while
exiting nonzero, read as success (#16).
**Check:** every subprocess call whose result feeds a boolean or a stamped
field must check `.returncode`, never infer success from non-empty stdout.
**Where:** `provenance.py` (`capture`, `lock_clocks`), any GPU
exclusivity/permission probe.
**Action:** grep all `subprocess`/`_sh(` call sites; flag any whose result is
used as a verdict via stdout truthiness rather than `_sh_result`/exit code.

### 2. Division across mismatched regimes or near-zero denominators

**Precedent:** kernel TFLOPS divided by whole-model TFLOPS (42% phantom
speedup); GLA's issued-FLOPS ratio read as a speed ratio; prefill throughput
applied to a bandwidth-bound decode step (177x off) (#15); `max_rel_err`
computed against a masked-to-zero expected value (#14).
**Check:** for every division - (a) is the denominator floored, and is that
floor a genuine resolution floor rather than an anti-crash epsilon that lets
garbage through silently; (b) are numerator and denominator drawn from the
same regime (compute-bound vs. bandwidth-bound, kernel vs. whole-model, same
host/session)?
**Where:** `gates.check_correctness`, `diagnostic_agreement._rel`,
`cross_arch.Speedup.speedup`, any TFLOPS/throughput/cost-model code.
**Action:** for each `/` or `.div(`, trace both operands' provenance; flag
any ratio spanning two measurement regimes or two sessions/hosts.

### 3. Tolerance checks that certify bias as long as it's small enough

**Precedent:** a reconciliation identity off by one decode step passed
"CLOSES" in every cell for the identity's whole life because the bias sat
inside a 10% band; 24/24 same-sign residuals were described in prose and not
tested (#27).
**Check:** anywhere a check reports pass/fail from "within X%", is there also
a test on the *sign* of residuals across cells (a one-line binomial test)?
Same-sign residuals across many cells are bias, not noise, regardless of
magnitude.
**Where:** `gates.TOL` comparisons, `canary.CanaryDrift`, `cross_arch.Speedup`,
`RECONCILE_TOLERANCE`, any Stage 5 reconciliation.
**Action:** flag any tolerance-based pass/fail with no accompanying sign
check; request one be added.

### 4. Aggregation across a composition that isn't held constant

**Precedent:** a marginal median that fell 45% between bands purely because
OOM attrition changed which batch sizes survived at each length - Simpson's
paradox, no error raised (#13); Stage 4 grouped on exact per-example token
count instead of the intended grid band, silently shrinking most cells to
n=1-7 and letting a zero-variance bootstrap CI report confidently (#20).
**Check:** before any `.groupby(...).mean()/median()`, is group cardinality
verified (`.size().min()`)? Is the facet composition matched across the
groups being compared? Is the group key a designed grid level, or does it
vary per example?
**Where:** `analysis/composition.py`, `analysis/matched.py`, cross-architecture
analysis, anything computing a median/mean over `seq_len` or
`context_length`.
**Action:** flag any groupby not routed through `composition.aggregate()` /
`matched_subset()`; flag any group key that isn't from the fixed grid.

### 5. Warmup, JIT, or cache cost bleeding into a timed measurement

**Precedent:** GLA's uncontrolled Triton JIT compile reported as a 4.28x
*batching* benefit, the exact opposite of the truth, from otherwise-correct
totals (#2); a machine image carrying a stale score cache reported a 12.1s
phase as 0.007s / 9041 TFLOPS (#7); FlexAttention rebuilding its block mask
on every call, pinning a 2ms floor and inflating peak memory (limitations.md,
"the mask is built once per config").
**Check:** does every backend have an excluded warmup call before timing? Are
derived caches cleared at session start rather than inherited from a snapshot?
Does the timed region contain only the operation under test - no `seq_len`-sized
factory op or mask construction inside a per-call loop?
**Where:** timing probes, `backends/*.py` `forward()`, instance startup
scripts, `tests/test_timed_region_setup.py`.
**Action:** for any new/changed backend or timing script, confirm a warmup pass
precedes measurement and is excluded from the recorded latency.

### 6. Oracle and kernel-under-test computing different things by construction

**Precedent:** a block-grid mask expansion that didn't apply causality inside
the diagonal block, so the correctness oracle and the real kernel computed
different functions - a gate that would have failed and looked like the
kernel's fault (#1).
**Check:** whenever a reference/oracle represents a mask, dtype, or causality
convention differently from the backend under test, verify both are being
asked the *identical* masked question before treating disagreement as a kernel
bug.
**Where:** `masks.to_dense_bool` / `to_flex_block_mask` /
`to_block_sparse_attn_mask`, `gates.check_for_family`.
**Action:** for any new mask representation or oracle, require a test that
feeds the oracle and the kernel the same causality/mask semantics explicitly.

### 7. A compiler/runtime fallback that silently swaps what's measured

**Precedent:** `torch.compile` hit its recompile limit mid-sweep and ran
`flex_attention` eagerly from then on - numerically correct, but a different
implementation than the one Stage 1 certified 72/72, and the warning fired
once while covering 72 contaminated rows (Python's `once` filter dedups by
message/module/line, not by occurrence) (#8).
**Check:** does any path relying on a compiled/fused kernel *assert* (not just
log) that the compiled path ran, per measured cell, not once per process?
**Where:** `compile_guard.py`, `gates.check_for_family`,
`FlexAttentionBackend`.
**Action:** confirm `compile_guard.guard()` wraps every new
correctness/timing call touching `torch.compile`; confirm detection is sticky
at process scope and voids `passed` rather than leaving it `True`.

### 8. Tests that stay green when the thing they guard is broken

**Precedent:** at least five confirmed vacuous tests - an OOM guard tested
against a `dict` instead of a `Tensor` (#9); a process-matcher test that only
ever exercised self-match (#9); a groupby test whose only fixture used the
exact value production varies (#20); a coverage test whose arm list was
dense-only while the script it covered ran four arms (#22); a guard verified
against a fixture the guard itself never reads (#25).
**Check:** for any test guarding a gate, oracle, refusal, or coverage property
- can you deliberately break the guarded behavior and watch the test go red?
If it stays green, it asserts nothing about that property.
**Where:** the whole test suite, especially tests with a single fixture value
that happens to match production only by coincidence.
**Action:** for any new/modified test guarding a critical property, require the
"break it, confirm red" check be performed and stated in the PR - don't accept
the assertion on inspection alone. Flag any list of cases (arms, backends,
tasks) maintained separately in a test vs. in production code; require they be
cross-checked against each other or share one source.

### 9. Defaults or stamps silently overwriting measured values

**Precedent:** `clocks_locked=True`, correctly measured on every row, was
overwritten to `False` by a wholesale `for k, v in prov.items(): df[k] = v`
merge, because `capture()`'s default is `False` and nothing passed the measured
value (#23) - this exact bug had already been fixed once, locally, in a
different script, and reappeared because the fix lived at a call site instead
of in the shared merge function (#23, "a fix in a call site is not a fix").
**Check:** any field merged onto a row from a `capture()`/default source must
not silently overwrite a value the row itself measured. Any field marked
`GATED` must have a live downstream consumer - if none exists, it's decorative
and should be labeled as such, not left implying protection it doesn't provide.
**Where:** `provenance.py` (`capture`, `stamp_onto`, `GATED_FIELDS` /
`RECORDED_FIELDS`), any `for k, v in dict.items(): df[k] = v` pattern.
**Action:** flag any wholesale dict-to-DataFrame column assignment; require
`stamp_onto`-style behavior (fill only missing fields, raise on disagreement)
instead of unconditional overwrite.

### 10. A rule defined in more than one place, one of them ungreppable

**Precedent:** the GLA accuracy-exclusion rule had four separate definitions
across scripts, two with no justification comment and one as a bare inline
`df.backend != "gla"` that doesn't match a grep for the constant's name - it
survived three prior audits for exactly that reason (#23).
**Check:** any exclusion/inclusion rule (excluded backends, arm lists, task
lists) appearing as a literal in more than one file.
**Action:** grep for repeated tuples/literals of backend or task names; flag
duplicates; require one canonical definition plus a test asserting there's
exactly one.

### 11. Confounds invisible in an end-to-end number, present only where the finding lives

**Precedent:** an end-to-end speedup number was measured with two stacked
confounds - sparse decode silently falling back to a slower kernel than
dense's own decode kernel, and the two arms generating different numbers of
tokens on the one task where all the "wins" lived, while the clean task
(fixed-length output) showed nothing wrong, making the confound invisible to
exactly the sanity-check a reader would run (#24).
**Check:** when comparing two arms on a per-unit derived quantity (ms/token,
ms/example), is the divisor (`n_generated`, decode kernel identity, batch
composition) identical across arms **at the row level**, not just similar in
aggregate?
**Where:** Stage 5/6 analysis, `matched.py`, `decode_confound.py`,
`pareto.py`.
**Action:** for any new "per-X" latency comparison, require an explicit
row-level check that X and the kernel producing each phase match between arms
before they're compared.

### 12. Statistical validity of cell size before trusting a CI

**Precedent:** `paired_bootstrap_diff_ci` accepted n=1 silently and returned
a confident, zero-variance interval for cells that should have had n~300 (#20).
**Check:** does the bootstrap/CI helper reject or flag cells below a stated
minimum n, rather than returning a plausible-looking interval regardless?
**Action:** confirm a minimum-n guard exists at the CI helper; flag call sites
that consume its output without checking `n` first.

### 13. Difference estimators that cancel a value but not its noise

**Precedent:** a two-point decode-step slope, `(T(8) - T(1)) / 7`, cancels
prefill's value but not its variance - each endpoint re-executes prefill
independently, so prefill noise lands in the slope amplified by up to
`prefill / (decode x delta-k)`, worst exactly at the longest context where the
number matters most (#26).
**Check:** any estimator differencing two measurements to cancel a nuisance
term should have its noise-amplification factor stated, and ideally more
points or an independently-measured component instead of two.
**Action:** flag any two-point finite-difference feeding a reported number; ask
whether amplification has been computed and whether cited precision accounts
for it.

### 14. Reconciliation identities: check the algebra, not just the residual

**Precedent:** `end_to_end ~= prefill + n_generated x decode_step` was wrong
by one step for the identity's entire life - the prefill forward already emits
the first generated token's logits, so it's `prefill + (n-1) x decode_step`.
The bias was ~1.5-9% in the same direction every time and hid inside a 10%
tolerance until an independent intercept check (which needs 3+ points, not 2)
caught it (#27).
**Check:** does the reconciliation formula match what the measured functions
actually return (off-by-one counts of steps/calls are the concrete precedent)?
Does a sign-bias test run alongside the tolerance check (see category 3)?
**Action:** for any new reconciliation/identity check, trace each term back to
its exact definition before trusting that "closes" means "correct."

### 15. Unmatched arms in a headline comparison

**Precedent:** the study's central end-to-end dominance number was measured
with sparse decoding through a different kernel than dense's, undisclosed
until Stage 5 decomposed the total (claims.md, "the dominance result was
measured under two confounds").
**Check:** any cross-backend end-to-end comparison - are decode kernel, batch
composition, and generation length matched, or explicitly flagged as unmatched
with the caveat carried into the output data (not just prose)?
**Action:** for any new end-to-end comparison, require the matched fields to be
named and check whether the code actually enforces the match (same object, not
same name).

### 16. Cache/session-state hygiene across reused environments

**Precedent:** a score cache baked into a machine image survived teardown of
the instance that created it and silently zeroed a phase's cost in a later
session (#7).
**Check:** does the session/instance startup path clear recomputable derived
caches on *every* boot, including from a snapshot image? Is the clearing logic
itself tested against a fake layout?
**Action:** for any new cache, confirm it's covered by the startup-clearing step
or is explicitly exempted with a stated reason (for example, it's a
measurement output, not a derived artifact).

### 17. Does this diff change what `claims.md` is allowed to say?

**Check:** for changes to `attnbench/_vendor/ruler/`, `masks.py`,
`backends/*.py`, `gates.py`, `provenance.py`, or `analysis/*.py` - does an
existing "Supported" row's qualifier still hold? If the change alters a
measured behavior (a new `score_source`, a changed decode fallback, a new
mask representation), does `claims.md` or `limitations.md` need a
corresponding update in the same PR?
**Action:** request the PR description name which `claims.md` rows the change
could affect, even if the answer is "none."

---

## Review protocol for a single PR

1. Identify which categories above the diff touches - usually obvious from
the module (masks/backends -> 5, 6, 7; analysis/gates -> 2, 3, 4, 12, 13,
14; provenance/scripts -> 1, 9, 10, 16; anything cross-arm -> 11, 15).
2. Apply each relevant category's specific check.
3. For any new or modified test: perform the "break it, watch it go red"
check (category 8) as a review gate, not an optional nicety.
4. For any new or modified numeric computation: trace units and regimes on
both sides of every ratio (category 2).
5. For any new or modified aggregation: confirm group cardinality and
composition matching before it computes a mean/median (category 4).
6. Confirm every emitted result row carries the provenance fields the write-up
depends on, and that they're populated by measurement, not silently defaulted
(category 9).
7. Cross-check against `claims.md`/`limitations.md` (category 17).

## One-time full-repo audit (in addition to per-PR review)

- [ ] Grep every subprocess call; confirm none infer success from stdout
content alone.
- [ ] Grep every `/` in `analysis/` and `gates.py`; classify each by
regime-match risk.
- [ ] Grep every tolerance/`TOL`/`atol`/`rtol` comparison; confirm a sign/bias
test exists nearby, or note the gap.
- [ ] Grep every `.groupby(` in `analysis/`; confirm cardinality/composition
checks precede any mean/median.
- [ ] Walk every backend's `forward()`; confirm no `seq_len`-sized factory op
or mask construction sits inside the timed path.
- [ ] List every `Provenance` field; confirm the GATED/RECORDED classification
is current and every GATED field has a live consumer.
- [ ] Grep for duplicated exclusion-list literals (backend/task names) across
scripts; consolidate to one source with a cross-check test.
- [ ] For every `claims.md` "Supported" row, locate the code producing the
underlying numbers and confirm the stated qualifiers are still enforced by
current code, not only documented.
- [ ] Spot-check a sample of existing tests via mutation ("break it, watch it
fail"), prioritizing tests that guard gates, oracles, or refusals.

## Out of scope for this review

Style, naming, or structure issues unrelated to result integrity. This plan
is scoped to whether a diff could change a number that reaches `claims.md`
without anything downstream catching it - not general code quality.

## Suggested delivery

Save this as `.github/copilot-instructions.md` so it applies automatically to
Copilot's repo-wide review, or attach it explicitly to PR review requests:

> Review this diff against `docs/copilot_review_plan.md` categories 1-17;
> report findings per category with line references.
