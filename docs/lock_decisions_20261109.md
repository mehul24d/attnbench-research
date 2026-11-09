# Lock decisions: the 11 open items, each with a default (9 Nov 2026)

Adviser's ruling 5 of the ninth 9 Nov 2026 paste, updated by rulings 1 to 8
of the tenth. It replaces `docs/lock_decisions_20261108.md`, whose items 12
and 13 are settled. These are the T4.1 values whose status in
`docs/lock_package_draft.json` was "owed: decision". **User** marks an item
that needs the researcher, because it sets scientific direction or spend.
The adviser rules on the rest. Target: all 11 settled by 12 Nov 2026.
Nothing here is a lock value until the tagged lock commit (T4.4): the draft
keeps every `value` null.

**Authority for items 1, 2 and 5.** The researcher's confirmation, 9 Nov
2026, IST: "I agree with the adviser's rulings on lock items 1, 2 and 5".

| # | Item (T4.1) | Status | Wording |
|---|---|---|---|
| 1 | H0 to H3 tolerances | **Decided** for H1, H2, H3. H0 waits on a ruling (the unit, below) | H1 and H2: 10% absolute relative error, the median over the pairing's scored cells, per pairing. H3: no tolerance beyond the bootstrap widening of the bracket endpoints it already names. None may be below the 6.77% noise floor. On P1 (T3.2d) the recursion's error was 5.05% median and 59.7% max |
| 2 | Ground-truth density set | **Decided** | P2 and P4 run P1's set: dense 1.0; block-sparse 0.25 and 0.5; XAttention calibrated, its density being the mask density it produces. `oracle_d0.5_L32768` is not measured on any pairing, with its limitation (`docs/limitations.md`, 8 Nov 2026). The compiled arms are in only if their ground truth is resolved by 14 Nov 2026. If they are in, each session's card states the host time they add to its cap |
| 3 | H4 equivalence bound | For the next relay (below) | Proposal: 10 points, from the admissibility margin |
| 4 | H4 confirmatory family | For the next relay (below) | Count m from the declared family; the count depends on two choices the plan leaves open |
| 5 | Floors quantile | **Decided** | 0.95 for both floors. Label: with two sessions (T3.2c, T3.2d) the between-session component is close to the largest gap |
| 6 | HDBI boundedness threshold | Default, confirmed | One threshold, two classes, fitted on P1 by sign agreement, ties to the lower value, rows inside their cell's floor dropped (audit D3; `attnbench/lock/hdbi_fit.py`). Matches Section 5.4: a diagnostic only, fitted on the training pairings; P1 is the only training pairing before the lock |
| 7 | K2 correctness tolerance | Default, confirmed | The gate's `TOL` (`attnbench/gates.py`: bf16 2e-2, fp16 5e-3, fp32 1e-4, atol and rtol) |
| 8 | Payload class edges | Default, confirmed | 8 B, 4 KiB, 1 MiB |
| 9 | Realised-length key rule | Default, **corrected** | Each prediction is keyed by the prompt's realised token count; the budget is a label. The prompt-set hash and tokenizer version are their own T4.1 item (owed: data) |
| 10 | Budget cut order | Default, extended | P7 first, then the chunked extension, then the compiled group, then the rehearsal session. The cut never removes ground-truth timing, H1 or H4 |
| 11 | Recompile gate | Default, confirmed | 0 compile events after warm-up |

Also confirmed as defaults (9 Nov 2026, Appendix A): a group short of 10
clean rounds leaves the within-session spread as well as the
between-session one, and the "marginal" label applies at 16384 only.

## Item 1: H0's unit (flag)

The ruling covers H0 only if its tolerance is per family. **It is not.**
Section 4, H0: "Per-operation differences within **[lock]**, after
provider-level variance from the dense workload, compared only between
pairings that share the arm's operation sequence (Section 5.1)". The 9 Nov
table's "per operation family" was this document's wording, not the plan's.

The 0.25 per-family bound is a different quantity. It is Gate A leg 2's
replay-versus-trace bound (T4.1, "replay-versus-trace bound for Gate A"),
which compares one pairing's replay with its own trace. H0 compares the same
component measured on two pairings (host side P1 against P2; device side
P1 against P3, P4 against P6).

Plan-default proposal, for the next relay: the ruling's form on the plan's
unit. Per pairing pair and component side (host, device), the median absolute
relative difference over the compared operations, within 10%. Per-operation
differences are reported. Operations whose cell time is inside the floor are
counted apart. The draft keeps H0 at "owed: decision".

## Item 9: the correction

The ruling's rule is what the plan and the code already do. Only the 8 and
9 Nov tables were wrong.

- **No amendment keys by the budget.** `docs/lock_value_table.md` row 27 already proposes "Key by the prompt's realised token count, budget kept as a label". Appendix A, 25 Oct 2026 keys each prediction by "model, code path, arm, density, context length and batch".
- **The code keys by realised tokens.** Step 5 records `context_length` as the prompt's own token count and `budget` as the cell (`scripts/t31_steps.py`, `device_components_real`). Step 7 copies `context_length` into each committed cell.
- **Checked on the banked prediction files.** T3.2a to T3.2d are keyed 1951, 4068, 8075, 16370 and 32700. CPU rehearsals use random ids sized to the budget, so there the two are equal. No code change.

## Items 3 and 4: for the next relay

**H4, verbatim (Section 4).** "Recall sufficiency. At matched per-layer
recall, accuracy does not depend on which estimator chose the blocks, at
block sizes 32, 64 and 128. A random-selection control with no inductive bias
is held to the same recall: if it matches the estimators, recall is
sufficient; if it falls short, which blocks are kept matters." Test: "Paired
comparison at matched per-layer recall, or the estimator coefficient in a
pre-registered model, against an equivalence bound **[lock]**. … Confirmatory
family and multiplicity as in Section 5.7 (Holm) … The sample size comes from
a power rule fixed before lock (Section 5.7); a result below that power is
reported as inconclusive."

**What the points measure.** Percentage points of task accuracy: the share
of examples scored correct, times 100, on the primary cell (`qa_1`/16384,
n = max(200, the power rule), Section 5.7). For H4 it is the paired
difference between two estimators held to the same per-layer recall, or
between an estimator and the random control. It is not a difference from
dense.

**T1.1's values, and what they are.** −4.3 (`qa_1`/16384, 69 vs 68 of 100,
density 0.59) and −6.0 (`qa_1`/32768, 50 vs 46 of 100, density 0.51). Each
is the one-sided 97.5% lower confidence bound on calibrated XAttention's
paired accuracy minus dense's (`docs/t4_xattention_calibrated.md`), from the
admissibility test (exact paired non-inferiority, Section 5.7).

**T1.1 is not the data H4 tests.** It differs on every axis H4 fixes:

| | T1.1 | H4 |
|---|---|---|
| Comparison | one estimator against dense | estimator against estimator, and against the random control |
| Recall | not matched | matched per layer, through each estimator's own budget settings |
| Block size | 128 | 32, 64 and 128 |
| n | 100 | at least 200 |

T1.1 is banked pilot data, so it is development data. Its rows can feed only
T1.5's disagreement rate p for the power rule, which is exploratory and not
started. H4's rows do not exist yet. Also, the −6.0 cell is at 32768, where
about a fifth of `qa_1` rows decoded past the position limit (Section 5.7),
so that cell is confounded.

**The plan's own rationale.** Section 5.7: the bound is "justified at lock
from the tasks (how large an accuracy difference would change which
estimator a practitioner picks). The default is 10 points; T1.5 also prices
a 5-point bound, which needs about four times the examples." The power rule
uses δ = 10 points: about 1,300 × p at α = 0.025 and 90% power. The plan
gives no rationale for 10 beyond calling it the default. The lock table's
basis column cites T1.1, which is the dependence the ruling removes.

**Proposed bound: 10 points, from the admissibility margin.** H4's use in
the plan is to let recall stand in for accuracy when admissibility is
labelled ("from recall if H4 holds", Section 5.1). The dispatcher reads that
label at a 10-point margin, with 5 points as the sensitivity (Section 5.7).
A difference between estimators smaller than the label's own margin cannot
change which estimator a practitioner picks through the label. That is the
plan's test, and it does not use −4.3 or −6.0.

- **The 5-point alternative** follows from the same argument read at the sensitivity margin. It needs about 5,200 × p before Holm: 520 at 10% disagreement, 1,040 at 20%.
- **The test can resolve either bound.** The exact test's floor at n = 200 is 2.2 points (Section 5.7).

**Item 4, audit D4's text** (`docs/plan_audit.md`). "With k estimators, the
family has 3 × k(k−1)/2 pair tests plus at least k control tests. Three is
too few even for two estimators (3 pairs + 2 controls = 5). The 1,630 × p
figure understates n." Proposed edit: "about (z(1−α/m) + z(0.95))² / δ² × p
for a family of m tests, m fixed at lock".

Two choices set m, and the plan states neither:

1. **Which arms are estimators.** The removed-assert and compiled variants have the same masks as their eager arm (bitwise, or reported as a different arm), so they are not separate estimators.
2. **Whether the random control is tested at each block size** (3k tests) or once per estimator (k tests).

| Estimators | k | Pair tests | Controls once | Controls per block size | m |
|---|---|---|---|---|---|
| oracle, window, mean-pool, XAttention | 4 | 18 | 4 | 12 | 22 or 30 |
| the same, plus vertical-slash (T1.7) | 5 | 30 | 5 | 15 | 35 or 45 |

The window arm is not yet built (T1.7).
