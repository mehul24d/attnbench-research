# Withdrawn and superseded figures

**What this is.** One row per figure this study has withdrawn, superseded or
corrected, paired with what replaced it. `tests/test_no_stale_figures.py`
reads the JSON block below and fails if any of them appears in `docs/`,
`README.md` or the source of `attnbench/` and `scripts/` as a **live**
statement.

**Why it exists.** The 2026-09-21 external audit found five separate defects
that were one defect: a figure was withdrawn, the withdrawal was applied in
`claims.md`, and nothing propagated it to the other places the figure lived —
the End-to-end section of the same file, the conclusion paragraph,
`limitations.md`, a comment in `accuracy/grid_configs.py`, a status block.
Every correction was right where it was made. None reached the next copy.

That is the shape this project keeps meeting in code (`#22`, `#23`: a rule
fixed at the place it was noticed, invisible to the next call site) arriving
in prose. The difference until now was that code had a suite and prose had an
audit — so a stale number sat in the write-up until someone went looking,
which is exactly the position `silent_failure_patterns.md` #46 describes:
*"The repository's prose has no equivalent of a test suite."*

**This file is not a changelog.** It records only figures a reader could still
mistake for current. A figure that has been fully expunged stays here with
`"retired": true` rather than being deleted, so the pattern is not
re-introduced by someone who never knew it was wrong.

**Measurements stay live; statuses retire.** `16 of 31` is *intrinsically*
superseded — whenever it is written it is wrong, so the entry keeps watching
even after the last copy is corrected, and the correction quotes the withdrawn
sentence verbatim so the entry has something to watch. "The re-run is approved
as audit item S1a" was wrong only *at a moment*; it was true when written and
a similar sentence could be true again. Those retire once corrected. The
distinction is why `stage6-count-of-faster-points` is still live (the
`claims.md` withdrawal quotes "two distinct operating points" in full) and
`s1a-still-pending` is not.

---

## The rule the test applies

An occurrence is permitted when any of three things holds:

1. the **unit** containing it carries one of the markers below — a unit being
   one run of consecutive non-blank lines (a paragraph, a blockquote, a
   contiguous comment block), **except in a markdown table, where each row is
   its own unit**;
2. the unit carries the figure's **replacement** — or, in a table, another
   row states the replacement **in the same column**; or
3. in markdown only, the **nearest preceding heading** carries a marker, and
   the scope ends at the next heading of any level.

**Tables split, because a marker in one row says nothing about the next.**
Treating a table as one unit means a table with `DOES NOT SURVIVE` anywhere in
it is permitted everywhere in it — and supersession tables are precisely the
place where some rows are marked and others are live. The column clause in
(2) is what keeps the legitimate case working: a before/after table states its
correction down a column, and

    | | `dominated_measured` | `dominated_normalized` |
    | pre-fix | 16 / 31 (52%)     | 12 / 31 (39%)      |
    | rebuilt | **28 / 34 (82%)** | **15 / 34 (44%)**  |

is the right way to write one. Demanding that each row carry its own
replacement would push authors to delete the `pre-fix` row. A replacement
sitting in some unrelated cell of a wide table rescues nothing, because
column alignment is what a reader uses to pair the two.

**Splitting tables is a tightening, not the fix for a defect that shipped.**
On the tree as it stands it changes no verdict: it flagged one row,
`claims.md`'s pre-fix/rebuilt dominance table, which the column clause then
correctly permitted. It is here for what it will catch, and the fixture tests
in `test_no_stale_figures.py` are what demonstrate the leak is real.

(2) is not a loophole. *"12 of 31 → 15 of 34"* is the correct way to state a
correction; a rule that forbade it would push authors toward deleting the
history instead of marking it, and the provenance of a number is the thing
this project most often needs and least often has.

(3) exists because a reader arrives at a paragraph through its heading.
`claims.md`'s *"## The oracle can put sparse ABOVE dense — mostly withdrawn
2026-09-20"* covers the paragraphs under it; `limitations.md`'s *"### The
oracle is not only a ceiling"* covers nothing, which is why the same figures
were caught in one file and not the other. There is no equivalent in Python —
a comment block is read on its own — which is why the
`accuracy/grid_configs.py` and `analysis/matched.py` occurrences get no cover.

**Markers are a closed list**, and adding to it is deliberate:

`WITHDRAWN` · `DOES NOT SURVIVE` · `superseded` · `until 2026-09-…` ·
`the sentence read` · `this paragraph read` · `the paragraph that stood here` ·
`this subsection originally read` · `this line previously said` ·
`this paragraph named` · `had said`

**A marker's only function must be to disclaim currency.** The first draft of
this list also carried `pre-fix`, `pre-sink-fix`, `era 2` and `era-2`. Those
are not markers — they are this project's ordinary vocabulary for naming a
measurement regime, and they appear inside live statements on nearly every
page. Admitting them silently rescued **five** of the blocks this registry
exists to catch, including `limitations.md`'s S1a status block, which says
"Regenerated post-fix:" in the middle of being wrong about what had been
regenerated. `test_regime_vocabulary_is_not_a_marker` asserts they stay out.

A loose marker (`corrected`, `no longer`, `see above`) would let any nearby
hedge silence the check. That is how an allowlist becomes a permanent
exemption, which `tests/test_timed_region_setup.py` already refuses for
backends and this file refuses for prose.

## What is NOT in here, and why

Numbers that are **derivable** do not belong in a registry — they belong in a
check that recomputes them. A registry entry for a number that changes every
session is a second thing to keep in step, and it would go stale the same way
the prose did.

`tests/test_doc_derived_numbers.py` covers four, all stale at the 2026-09-21
audit and all arithmetic over files in this repository:

| restated in | recomputed from | was wrong by |
|---|---|---|
| README's ₹ total and session count | the ledger's own table | ₹400 / 4 sessions |
| `spend_ledger.md`'s 2026-09-17 day total | its own four rows | ₹1 |
| README's line count for `limitations.md` | `wc -l` | 950 → 2333 |
| README's incident count | numbered headings in `silent_failure_patterns.md` | 46 → 48 |

**The README's pytest counts are covered differently, and the difference is
the point.** No test can run the suite from inside the suite, so the value
cannot be recomputed. What *can* be checked without running anything is that
the document states one number rather than three — and three is what it stated
at the audit: `999 passed, 10 skipped` in the quick-start block, `The 11 skips`
in the paragraph below it, and `2 need CUDA, and 9 read banked result files`
as the breakdown, against a real 23.
`test_the_readme_suite_counts_are_internally_consistent` asserts the block, the
prose and the three-way breakdown agree. It cannot tell you the figure is
right; it can tell you the document does not believe two things at once, which
is the form the defect actually took.

Register a figure here when it is a **measurement** that was withdrawn or
superseded. Check it there when it is a **count** that can be recomputed.

---

## Scope of the two guards

| | registry (`test_no_stale_figures.py`) | derived (`test_doc_derived_numbers.py`) |
|---|---|---|
| catches | a withdrawn measurement stated as live | a restated count that has drifted from its source |
| fails when | prose changes are not propagated | the underlying file changes and the prose does not |
| cannot catch | a stale figure nobody has registered | anything not recomputable from the repo |

Neither reaches `results/*/README.md`, which legitimately quote superseded
numbers and are currently out of scope for both.

### The hole in the registry that markers cannot close

**A withdrawal note is a permanently marked context, so the next supersession
hides inside the previous one's disclaimer.** The row that demonstrated it,
from the table whose entire job is tracking which claims have gone stale:

    | *16 of 31 matched sparse points are dominated by dense* |
      **DOES NOT SURVIVE.** Normalized: **12 of 31**. |

`16 of 31` is correctly marked. `12 of 31` is the answer offered in its place,
and the S1a rebuild superseded it with `15 of 34` on 2026-09-20. Every scoping
rule above permits that row — per-row scanning included, because the row
carries its own marker. The row was wrong in the half a reader would use.

Two checks were added rather than one, because the mechanism has a prose half
and a registry half:

- `test_no_table_row_states_a_superseded_figure_as_its_verdict` — in a marked
  **table row**, a second registry figure whose own replacement is absent is
  flagged. Restricted to table rows deliberately: the same rule applied to
  prose was written, run against this tree, and rejected, because it flagged
  three blocks quoting a withdrawn paragraph verbatim — the one construction
  this whole scheme exists to encourage. A table row is a record whose verdict
  cell is asserted in the present tense; a quotation is one act governed by
  one disclaimer.
- `test_no_entry_is_replaced_by_a_figure_that_is_itself_withdrawn` — no entry
  may name a replacement that another live entry lists as withdrawn. The
  moment a figure is withdrawn is the only moment anyone is looking at the
  entries that pointed to it, so that is where this fires.

Recorded as instance 49 in `silent_failure_patterns.md`. It is **not** fully
closed: outside table rows, a marked block can still carry a superseded
replacement, and nothing here will say so.

---

## The register

```json
{
  "entries": [
    {
      "id": "stage6-dominated-measured",
      "pattern": "16 of 31|16 / 31",
      "was": "dominated_measured over the 31 accuracy-matched Stage 6 points, computed from pre-sink-fix accuracy (results/_superseded/stage6_prefix_sink/)",
      "replacement": "28 of 34|28 / 34",
      "note": "Superseded 2026-09-20 by the S1a rebuild. The measured column carries the cross-arm decode confound and is inadmissible as a reported quantity either way -- see claims.md, 'What survives the correction'."
    },
    {
      "id": "stage6-dominated-normalized",
      "pattern": "12 of 31|12 / 31",
      "was": "dominated_normalized over the 31 accuracy-matched Stage 6 points, pre-sink-fix",
      "replacement": "15 of 34|15 / 34",
      "note": "Superseded 2026-09-20. This is the reportable column."
    },
    {
      "id": "stage6-niah-single-dominated",
      "pattern": "9 of 15|All 15 `?niah_single",
      "was": "niah_single operating points dominated by dense, out of the 15 that existed pre-sink-fix",
      "replacement": "9 of 19|19 of 19|of 19 `?niah_single",
      "note": "The rebuild has 19 niah_single points: 19 dominated on measured, 9 on normalized."
    },
    {
      "id": "stage6-count-of-faster-points",
      "pattern": "two distinct operating points",
      "was": "the count of distinct operating points genuinely faster than dense, pre-sink-fix",
      "replacement": "three distinct operating points",
      "note": "Three clear their resolution floor in the rebuild, all at 8192: niah_single/0.90 (1.059x), niah_single/0.75 (1.039x), vt/0.90 (1.033x) -- the three the Stage 7 decision map recommends."
    },
    {
      "id": "vt-sparse-above-dense-margins",
      "pattern": "\\+14\\.6",
      "was": "vt block_sparse at 0.75 scoring above dense by +10.8 / +5.9 / +14.6 at 2048 / 4096 / 8192",
      "replacement": "\\+3\\.2",
      "note": "WITHDRAWN 2026-09-20 by audit S1a. Re-measured with the sink forced: +3.2 / +0.8 / -2.4. The margin does not survive at any band at 0.75."
    },
    {
      "id": "vt-nine-standard-errors",
      "pattern": "nine standard errors",
      "was": "the claim that the 8192 vt sparse-above-dense gap was roughly nine standard errors, same direction, three independent bands",
      "replacement": "\\+3\\.2",
      "note": "WITHDRAWN 2026-09-20. The gap was real and the statistics were right; the quantity was not what it was taken for. claims.md prints this sentence as removed text -- limitations.md still carried it live until the audit."
    },
    {
      "id": "cheap-estimator-not-implemented",
      "pattern": "no cheap estimator was implemented",
      "was": "the statement that this study implemented and evaluated no cheap importance estimator",
      "replacement": "",
      "note": "False since 2026-09-16. MInference mean-pool is implemented (accuracy/model.py:224, schema.py ScoreSource, run_accuracy.py --score-source) and evaluated in results/accuracy_forced_sink_cheap/ and results/s9_7b_cheap_16384/. Its ORDERING was measured and it fails; its COST was never measured. State the split, not the absence."
    },
    {
      "id": "cheap-estimator-not-a-score-source",
      "pattern": "this study does not yet\\s+have",
      "was": "the statement that a genuinely cheap estimator is a score_source this study does not yet have",
      "replacement": "minference_meanpool",
      "retired": true,
      "note": "Same figure as cheap-estimator-not-implemented, in limitations.md. RETIRED 2026-09-21: the sentence was removed outright when that section was rewritten, so the pattern now matches nothing. Kept so it is not written again by someone who never knew it was wrong -- which is the whole reason retired entries stay."
    },
    {
      "id": "multikey-16384-sink-delta",
      "pattern": "\\+0\\.0 against dense",
      "was": "niah_multikey at 16384/0.5 scoring +0.0 against dense post-sink-fix",
      "replacement": "−1\\.0|-1\\.0",
      "note": "+0.0 is the era-2 cell (results/accuracy_forced_sink/). Re-measured under the fixed tie-break, audit S7, it is -1.0 (era 3, results/s7_jitter/). Each is inside the other's CI."
    },
    {
      "id": "s1a-still-pending",
      "pattern": "re-run is approved as audit item S1a",
      "was": "the status that the S1a re-run was approved but not yet run",
      "replacement": "",
      "retired": true,
      "note": "S1a ran 2026-09-19/20: results/s1a/accuracy_all_bands.parquet, all three bands, n=100. Stages 4/6/7 were rebuilt at 9b336c8 and promoted at 28fa00a."
    },
    {
      "id": "derived-stages-not-regenerated",
      "pattern": "Not regenerated: every 1\\.5B accuracy number",
      "was": "the status that Stage 4 matched budgets, Stage 6 and the Stage 7 decision map had not been regenerated on forced-sink accuracy",
      "replacement": "",
      "retired": true,
      "note": "They were, on 2026-09-20. See claims.md, 'The derived stages rebuilt on forced-sink accuracy'. The 1.5B accuracy bands themselves WERE re-measured by S1a at n=100; stage3_flashdecode was not."
    },
    {
      "id": "stage7-vt-pre-adjustment-speedup",
      "pattern": "1\\.016 → ",
      "was": "vt's Stage 7 recommended speedup before the post-fix block-count re-evaluation",
      "replacement": "1\\.033|1\\.0327",
      "note": "results/stage7/decision_map.parquet gives vt/8192/0.9 at 1.0327 at every epsilon, and claims.md states the range as 1.033x-1.058x eighteen lines earlier. The corrected pair is 1.033 -> 1.031."
    },
    {
      "id": "s11-tax-against-kernel-denominator",
      "pattern": "0\\.8-1\\.7%|0\\.8–1\\.7%",
      "was": "the per-call mask-conversion tax as a fraction of the A100 16384 cells carrying the 1.090x/1.201x/1.282x claim",
      "replacement": "0\\.1[0-9]",
      "note": "Computed against single-call kernel p50s (3.373/2.266/1.795 ms) but attributed to 28-layer end-to-end prefill cells (400.4/363.4/340.4 ms). Correct range for those cells is 0.11-0.25%. Sign and the floor conclusion are unaffected; keep the kernel-cell figures and label them as kernel cells."
    },
    {
      "id": "composition-refuses-cross-era",
      "pattern": "composition\\.py` (?:exists to )?refuses?",
      "was": "the claim that `analysis/composition.py` refuses a cross-era comparison, stated in three documents and relied on by the S12 disposition",
      "replacement": "eras\\.py",
      "note": "The only entry here that withdraws a claim about CODE rather than a measurement, and it belongs in the same register for the same reason: it was corrected in one place and left standing in two others for as long as it went unread. composition.py refuses on facet composition -- whether the groups being averaged contain the same batch levels -- and has no concept of a commit, a date or a mask rule. Nothing refused a cross-era comparison at all until 2026-09-21, when scripts/run_scale_comparison.py and scripts/run_scorer_comparison.py were given the check, from git_commit via attnbench/analysis/eras.py."
    },
    {
      "id": "coverage-test-expands-the-same-way",
      "pattern": "expands the same way",
      "was": "the S14 claim that test_every_registered_backend_is_considered could catch an unobserved kernel variant, which rested on it expanding its expected set through INSTANCES exactly as the observation pass did",
      "replacement": "KERNELS",
      "note": "The second claim about code in this register, and the same shape as the first: a guard described in prose that the code did not implement. Here the description was accurate and the conclusion drawn from it was not -- expanding the same way is precisely what makes a coverage check unable to fail. Mutating SDPABackend to accept a fifth kernel left the suite green at 4/10. Fixed by deriving the expected set from SDPABackend.KERNELS; coverage reads 4/11."
    },
    {
      "id": "readme-2048-no-accuracy-cost",
      "pattern": "0\\.993× \\(sparsity loses\\)",
      "was": "the README's 2048 row in the 'best speedup at no accuracy cost' table",
      "replacement": "0\\.984×",
      "note": "0.993x is the 0.75-sparsity cell at 2048, which scores 99.0. The column requires 100.0, and the only 2048 cell that reaches it is 0.5 sparsity at 0.984x. Registered with the parenthetical in the pattern because 0.993 on its own is a correct measurement in the claims.md grid and a registry entry that fires on a correct statement is a bad entry."
    },
    {
      "id": "era-2-3-split-on-date-only",
      "pattern": "split on date\\s+only",
      "was": "the claim that era 2 and era 3 can only be told apart by date, so a banked row cannot be assigned between them from its own contents",
      "replacement": "5cc3a40|ancestry|ancestor",
      "note": "False, and it justified not adding a mask_rule column on grounds that did not hold. The jitter fix is the commit 5cc3a40 and ancestry partitions the register exactly: seven era-2 commits predate it, the one era-3 commit descends from it. The date rule does not even work -- three era-2 commits share 2026-09-20 with the era-3 one. Verified against the real history 2026-09-21; attnbench/analysis/eras.py now derives the boundary with git merge-base --is-ancestor. Recorded as instance 50, because the sentence was copied into that module one day after the guard against copying stale statements was built."
    },
    {
      "id": "zeros-bug-attributed-to-its-fix",
      "pattern": "bug of `562374a`",
      "was": "562374a described as the commit carrying the cheap-scorer zeros defect",
      "replacement": "562374a` REMOVED|fixed in `562374a`",
      "note": "562374a deletes `out = torch.zeros_like(q)`; it is the fix. silent_failure_patterns.md already used the unambiguous form, 'fixed in 562374a'. A commit cited as a defect when it is the repair sends the next reader to check out the wrong tree."
    }
  ]
}
```

---

## Adding an entry

1. Write the **pattern as a phrase**, not a number. `16` occurs on nearly
   every page; `16 of 31` occurs once per stale copy.
   `test_registry_patterns_are_specific` bounds this.
2. Give the **replacement** as a pattern too, or leave it empty and explain in
   `note` why the figure has no successor.
3. Run the suite. If `test_no_withdrawn_figure_appears_unmarked` fails, it has
   found the copies the correction did not reach — that is the entry working,
   not the entry being wrong.
4. When a figure is gone from the repository entirely,
   `test_no_dead_registry_entry` will say so. Set `"retired": true` rather
   than deleting the row.
