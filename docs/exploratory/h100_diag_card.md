# EXPLORATORY: the H100 diagnostic session (adviser's rulings, 17 Nov 2026, item 2)

**EXPLORATORY. Nothing here scores a hypothesis. H0–H3, 83b2edf and the timed-path pin `0e0fb012` are unchanged.** This session explains P4's miss and nothing else. No ground-truth row is written.

## Cells (approved by the researcher, 17 Nov 2026)

These run in this order: `configs/exploratory/h100_diag_cells.json`, sha256 `52c576b2959285c8f321b24c473c9676381ce2ec42d0077a62075f87fe4c0a79`. The driver refuses any other list. All cells are batch 1, on P4's prompts (8075 / 16370 / 32700 tokens).

| # | id | arm | dense backend | budget | mode | reads |
|---|---|---|---|---|---|---|
| 1 | D16_fa3 | dense | FA3 (K2's control) | 16384 | eager | Q2, Q3 |
| 2 | D16_off | dense | sdpa_flash (harness) | 16384 | eager | Q1 |
| 3 | D16_fa3_graph | dense | FA3 | 16384 | CUDA graph, as step 6 prices | Q2 |
| 4–6 | X8, X16, X32 | XAttention (Triton) | — | 8192 / 16384 / 32768 | eager | Q3, Q4 |
| 7–8 | D8_fa3, D32_fa3 | dense | FA3 | 8192 / 32768 | eager | Q3 |

Eager cells run 30 untraced prefills, one torch.profiler prefill (the P4 tracer) and 5 prefills under Nsight Systems. On image v6, Nsight Systems is 2025.1.3, at `/usr/local/cuda/bin/nsys`. A failed cell is recorded and the next cell runs; no cell reruns. Cells not started within 55 minutes are skipped.

## What each outcome means (declared before launch)

"Same" means within ±10%. P4's numbers come from `results_20261116.json`.

- **Q1. Did P4 price the FA3-off kernel?** D16_off's attention against P4's priced 0.07415 s. A ratio in 0.9–1.1 **confirms** that step 5 recorded, and step 6 priced, `sdpa_flash` while ground truth ran FA3. Otherwise it is **not confirmed**.
- **Q2. Does the graph replay price FA3 faithfully?** D16_fa3_graph's attention against its eager twin:
  - 0.9–1.1: **faithful**.
  - ≥ 2.0: the graph **alone explains most** of P4's 2.52× (0.07415 / 0.02943).
  - 1.1–2.0: it **inflates** FA3, but by less than P4's misprice.
  - < 0.9: it **deflates** FA3.
- **Reading Q1 and Q2 together:**
  - Q1 confirmed with Q2 faithful: the misprice is the backend mismatch alone.
  - Q1 not confirmed with Q2 ≥ 2.0: the misprice is the replay path.
  - Any other combination is reported without an inference.
- **Q3. Is this P4's device?** Nine pairs: dense FA3 attention and wall at each length, and XAttention wall at each length, each against P4's. Q3 is read **per length** (ruling of 17 Nov). The pairs at a length gate Q1 and Q4 at that length only. If any pair at a length is not the same, or that length has no pair because its cells did not pass, Q1 (read at 16370) or Q4 at that length is reported as not comparable. Q2 is within this session and is never gated.
- **Q4. Is the residual host cost?** P4's median |rel err| with all device time traced is **0.208**. Its median cell is XAttention at 8075 (+0.208); 16370 gives +0.080 and 32700 gives +0.156. Every residual is an overprediction. The recursion is re-run with each priced host cost scaled by measured / priced host (measured with Nsight; torch.profiler only as a flagged fallback).
  - **Tracing overhead** (ruling of 17 Nov): at each length, overhead = the traced median wall per prefill minus P4's untraced wall at that length, taken as zero if negative. It is subtracted from the traced median host before the thresholds below. If the traced wall is above 1.10× P4's, Q4 is **not comparable** at that length.
  - **φ, in full:** φ = (R_all − R_host) / (R_all − M). R_all is the recursion with every priced device time replaced by P4's traced one, R_host is the same with each priced host cost multiplied by h / H, and M is P4's measured wall. Here h is the measured host per prefill after the overhead is taken off, and H is the cell's summed priced host. R_all − M is the residual above.
  - φ ≥ 0.5: the host costs are **overstated**.
  - φ ≤ 0.2: the residual is **not host**; it is overlap, queue or sync.
  - In between: **partly** host.

  The thresholds in measured host seconds per prefill (h, after the overhead is taken off), from P4's data. Each is φ above, solved for 0.5 and 0.2 by bisection on P4's recursion. Re-derived on 17 Nov, they agree to four decimals, and φ falls monotonically as h rises:

  | tokens | priced host | φ ≥ 0.5 if host ≤ | φ ≤ 0.2 if host ≥ |
  |---|---|---|---|
  | 8075 | 0.1259 | 0.0985 | 0.1164 |
  | 16370 | 0.1267 | 0.1045 | 0.1179 |
  | 32700 | 0.2603 | 0.2029 | 0.2374 |

  *Known before launch:* at 8075 the priced host (0.126 s) is longer than P4's measured wall (0.108 s). Host time cannot exceed the wall, so φ ≥ 0.33 at X8: it reads either partly host or overstated, never "not host".

## Driver and money

- **Terms:** Spot, us-central1-a only, image v6, one attempt.
- **Request window:** requests are sent only inside 06:00–14:00Z; one is refused outside the window and within 5 minutes of its end.
- **Limits:** halt at 110 billed minutes, delete at 120 minutes.
- **On reclaim:** stop at the first reclaim; nothing reruns.
- **Projected billed minutes:** about 14.5 to boot, plus about 20 for the 8 cells traced, plus about 3 to pull and delete, so **about 38**. With a 50% margin on the cells it is **about 48**, inside the ~80 the ruling allows, so no cell is dropped and the sha is unchanged.
  - The per-cell estimate comes from P4's b4 ground-truth step, which traced 21 arm-length cells in 400 s.
  - An eager cell is taken at about 2.5 min: 41 in-process prefills, each under 0.5 s; one torch.profiler trace; and a fresh Nsight child that reloads the model and exports.
  - The graph cell is taken at about 2 min.
  - The 55-minute cell budget still skips any cell not started in time.
- **Cost:** ₹629.94/h, so about ₹400 to ₹504 at the projection, ₹1,155 at the halt and ₹1,260 at the delete, against a cap of ₹1,300.
- **Money gate:** spent ≈ ₹9,580 + ₹1,300 ≤ ₹15,500. Spend is booked on billed minutes, from create to delete or reclaim.

## Plan defaults (flagged)

1. FA3 on/off applies to dense only; XAttention runs its own block-sparse kernel.
2. The graph cell is FA3, the backend a fix would record, rather than `sdpa_flash`, which would reproduce P4's price directly. Q1 reads `sdpa_flash` eager instead.
3. Q4 is read on XAttention cells only, because the dense residuals are already within 0.03 of zero after the FA3 substitution. Nsight's tracing overhead is taken off the host as in Q4's overhead rule. Any overhead not visible in the wall still counts as host time and biases φ down.
4. A no-capacity answer is retried inside the window, but there is at most one running session. A request made near 14:00Z may run past it.
5. The CPU pass ran the dense, graph and Nsight paths, using a stand-in Nsight. XAttention has no local checkout, so its cells fail behind the CPU boundary; their logic is covered by unit tests.
6. The chain is P4's, unchanged since 7a24cba, so there is no e2 rehearsal.

## Approval

The researcher approved this cell list (sha above) on 17 Nov 2026, and launched one session on these terms, after the card's 17 Nov rule edits were committed, the load reruns of `tests/test_session_exit.py` passed and the projected minutes fit.
