# Claim status (28 Nov 2026)

The paper-quality wave of 28 Nov 2026, item 1. This file restates results that already exist; it changes, rescores or adds no pre-registered result. Section A holds the pre-registered hypotheses as scored. Section B holds the H100 diagnostic's questions, which were declared before their session but are EXPLORATORY. Section C lists EXPLORATORY / POST HOC numbers, kept apart from A.

**Scoring context.** H0 to H3 were scored at lock tag `lock-2026-11-14-r3`, analysis commit `83b2edf`, CPU only, on 16 Nov 2026. The scored pairings are H0 P2, H1 P2 and P4, H2 P2, H3 P4 (`docs/lock_values.json`, `scored_pairings`, at `d640e69`). P1 is the development pairing and is never counted as a test (plan Section 4, H1). The timed-path pin `0e0fb012` covers every scored row.

**Abbreviations used for sources.**
- **S** = `docs/scoring/hypotheses_scored_20261116.md` (`4143ce2`), with its numbers in `.json` (`5d81f4c`).
- **A(date)** = an Appendix A line of `docs/RESEARCH_PLAN.md` on `main` `d11a466`.

## A. Pre-registered hypotheses

| # | Pairing | Pre-registered status | Result | Number type | Source |
|---|---|---|---|---|---|
| H0 (host separability) | P2 (from P1) | Scored hypothesis | **Not scored.** No P2 cell has P1's strict operation sequence: 0 same, 25 different (dense at 1951 runs 1225 ops on P1, 1085 on P2), 20 XAttention cells unmatched because calibrated densities differ. The table's reason text ("no operation is above its cell's floor") is the code's fallback for an empty comparison | Pre-registered (verdict) | S line 7, "H0's reason" note |
| H1 (composition) | P2 | Scored hypothesis | **Fails.** Recursion median abs. rel. error 0.1183 over 35 cells, against tolerance ≤ 0.1; beaten by fitted_sum 0.1005 and step_form 0.0842; below sum 0.6935 and learned 1.4465; sign 25 agree, 0 disagree; XAttention pair 0 agree, 1 disagree | Pre-registered | S line 8 |
| H1 (composition) | P4 | Scored hypothesis | **Fails.** 0.2495 over 35 cells; beaten by fitted_sum 0.1216 and step_form 0.2454; below sum 0.8527 and learned 0.3169; sign 25 agree, 2 disagree; XAttention pair 0 agree, 1 disagree. P4's dense prediction priced `sdpa_flash` while ground truth ran FA3: a fault in the committed prediction, reported and not corrected | Pre-registered | S line 9; A(17 Nov 2026) |
| H2 (transfer) | P2 (host from P1) | Scored hypothesis | **Not scored.** No transferred cell is measured: all 45 left out, for H0's reason | Pre-registered (verdict) | S line 10, "H0's reason" note |
| H3 (forecast) | P4 | Scored hypothesis | **Not scored.** 6 estimators: mean-pool 0.25 and 0.5 and both XAttention arms "unresolved"; oracle 0.25 and 0.5 "both none in range" | Pre-registered (verdict) | S lines 11, 21–30 |
| H4 (recall sufficiency) | — | Hypothesis, withdrawn before scoring | **Not scored.** T1.5's pilot, T1.8's recall matching and audit B1's cross-card control never ran | — | A(12 Nov 2026); plan Section 4 |
| H5 (crossover direction) | — | Secondary hypothesis, withdrawn before scoring | **Not scored**, for H4's reason | — | A(12 Nov 2026) |
| H6 (dispatch) | — | Hypothesis, withdrawn before scoring | **Not scored.** No quality-certified admissible set: H4 is unscored; XAttention's Triton path fails the mask gate on P2 (10 of 10 records beyond ±0.01); an all-dense set cannot fail. The dispatcher is exploratory only (Section C) | — | A(14 Nov 2026), reversing A(12 Nov 2026) |

**Other pre-registered results.** These come from earlier pre-registered stages, not from the lock.

| Result | Pairing | Number type | Source |
|---|---|---|---|
| T1.1 Session B: calibrated XAttention (`authors` table) is non-inferior on `qa_1`/16384: 69 vs 68 of 100, bound −4.3, density 0.59 | P1 (L4, PyTorch path) | Pre-registered certification | `docs/t4_xattention_calibrated.md` (`f8a77df`); plan Section 2 |
| The same at `qa_1`/32768: 50 vs 46, bound −6.0, density 0.51, over all 100 rows (19 to 21 rows decoded past the position limit) | P1 | Pre-registered certification | same; `docs/paper_claims_audit.md` (b) (`d640e69`) |
| P-T4 predicted 0 of 2 primary cells certified; 2 of 2 were. **Failed** | P1 | Pre-registered prediction | plan Section 2 |
| H100 oracle-mask mechanism predictions: 2 of 3 passed (7 of 9 cells) | H100 (earlier study) | Pre-registered (earlier study) | plan Section 2; `docs/limitations.md` "The H100 test (pre-registered)" |

## B. H100 diagnostic questions Q1 to Q4 (declared before the session, EXPLORATORY)

Read in `docs/exploratory/h100_diag_card.md`, and in its rerun card for Q2. None scores a hypothesis.

| Q | Question | Result | Number type | Source |
|---|---|---|---|---|
| Q1 | Did P4 price the FA3-off kernel? | **Confirmed**: `sdpa_flash` attention / P4's priced attention = 1.016 (band 0.9–1.1) | Exploratory | `h100_diag_verdict_20261117.json` (`f6fa3e7`) |
| Q2 | Does the graph replay price FA3 faithfully? | **Faithful**: graph / eager FA3 attention = 1.019. Read with Q1: P4's misprice is the backend mismatch alone | Exploratory | `h100_diag_rerun_verdict_20261117.json` (`7b98834`); A(17 Nov 2026) |
| Q3 | Is this P4's device? | **Comparable at 8192, 16384 and 32768**: all 9 pairs within ±10% (ratios 1.017–1.055) | Exploratory | `h100_diag_verdict_20261117.json` |
| Q4 | Is the residual host cost? | 8192: **not comparable** (traced wall 1.110x P4's, above 1.10). 16384: **host overstated**, φ 0.97, measured / priced host 0.65. 32768: **host overstated**, φ 0.91, ratio 0.60 | Exploratory | same |

The research questions RQ1 to RQ4 map onto hypotheses: RQ1 to H1–H3, RQ2 to H1's rival comparison and the XAttention pair, RQ3 to H4, RQ4 to H6 (plan Section 1). Their status is that of those hypotheses. **Plan default, flagged:** "Q1–Q4" in the request is read as the diagnostic's questions, and the RQs are mapped here.

## C. EXPLORATORY / POST HOC numbers (none replaces a result above)

| Number | Pairing | Type | Source |
|---|---|---|---|
| Error with all device time traced: 0.116 (P2), 0.208 (P4) | P2, P4 | Exploratory | `docs/exploratory/exploratory_wave_20261116.md` (`edaea38`) |
| P4's residual is host pricing: queue nil, s_0 under 5%, priced host above the whole measured time in 23 of 35 cells | P4 | Exploratory | `docs/exploratory/residual_20261122.md` (`6d139f7`) |
| Safe dispatch at margin 0.02264 (a fallback fixed from P1 after P2/P4 cell times were seen): P2 zero regret; P4 worst 13.9% slower than dense | P2, P4 | Exploratory, post hoc margin | same two files; `docs/paper_claims_audit.md` (c) |
| Paired sign: all 58 sparse cells exclude zero | P2, P4 | Exploratory | `exploratory_wave_20261116.md` line 13 |
| `qa_1`/32768 within the position limit: bound −4.2, n = 80 | P1 | Post hoc sensitivity, reported beside −6.0 | `docs/paper_claims_audit.md` (b); A(26 Nov 2026) |
| P3 full transfer from P1: recursion 0.0811, step_form 0.0408 (35 cells) | P3 | Exploratory (run after the lock; not a scored pairing) | `docs/p3_run/p3_exploratory_scored.md` at `584516a` on `p3-g6-2026-11-26` (pushed, **not merged to `main`**) |
| Miss decomposition, knockouts, per-cell intervals, FA3 repricing, host scaling | P2, P4 | Exploratory / post hoc | `docs/paper/exploratory/paper_tables.md` (this wave) |
| Committed host price exceeds the whole measured prefill in 17 (P2), 2 (P3) and 23 (P4) of 35 cells; the bare part alone never does, the idle-gap residual alone does in 14 (P2) and 18 (P4); null pass ÷ committed 1.31 to 1.45 (non-oracle); P4 committed ÷ nsys host 1.53 to 1.68 | P2, P3, P4 (S4: null pass ÷ recording 1.78) | Exploratory / post hoc, lower bounds | `docs/paper/host_bias_and_pay_bound.md`, `exploratory/null_kernel_bias.md` (this wave, item 10) |
| Parameter-free pay condition: no measured win ruled out; rules out 14 of 25 (P2) and 26 of 28 (P4) measured losses before timing | P2, P4 | Exploratory / post hoc | `docs/paper/host_bias_and_pay_bound.md`, `exploratory/pay_bound.md` (this wave, item 11) |
