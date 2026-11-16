# EXPLORATORY: the H100 diagnostic session (adviser's rulings, 17 Nov 2026, item 2)

**EXPLORATORY. Nothing here scores a hypothesis. H0–H3, 83b2edf and the timed-path pin `0e0fb012` are unchanged.** This session explains P4's miss and nothing else. No ground-truth row is written.

## Cells (proposed; approval wanted)

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
- **Q3. Is this P4's device?** Nine pairs: dense FA3 attention and wall at each length, and XAttention wall at each length, each against P4's. If any pair is not the same, Q1 and Q4 are reported as not comparable and only Q2 is read.
- **Q4. Is the residual host cost?** P4's median |rel err| with all device time traced is **0.208**. Its median cell is XAttention at 8075 (+0.208); 16370 gives +0.080 and 32700 gives +0.156. Every residual is an overprediction. The recursion is re-run with each priced host cost scaled by measured / priced host (measured with Nsight; torch.profiler only as a flagged fallback). φ is the share of the residual this removes:
  - φ ≥ 0.5: the host costs are **overstated**.
  - φ ≤ 0.2: the residual is **not host**; it is overlap, queue or sync.
  - In between: **partly** host.

  The thresholds in measured host seconds per prefill, from P4's data:

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
- **Cost:** ₹629.94/h, so ₹1,155 at the halt and ₹1,260 at the delete, against a cap of ₹1,300.
- **Money gate:** spent ≈ ₹9,580 + ₹1,300 ≤ ₹15,500. Spend is booked on billed minutes, from create to delete or reclaim.

## Plan defaults (flagged)

1. FA3 on/off applies to dense only; XAttention runs its own block-sparse kernel.
2. The graph cell is FA3, the backend a fix would record, rather than `sdpa_flash`, which would reproduce P4's price directly. Q1 reads `sdpa_flash` eager instead.
3. Q4 is read on XAttention cells only, because the dense residuals are already within 0.03 of zero after the FA3 substitution. Nsight's tracing overhead counts as host time, which biases φ down.
4. A no-capacity answer is retried inside the window, but there is at most one running session. A request made near 14:00Z may run past it.
5. The CPU pass ran the dense, graph and Nsight paths, using a stand-in Nsight. XAttention has no local checkout, so its cells fail behind the CPU boundary; their logic is covered by unit tests.
6. The chain is P4's, unchanged since 7a24cba, so there is no e2 rehearsal.

## Decision needed

Approve this cell list (sha above), or name changes to it.
