# EXPLORATORY: the H100 diagnostic's rerun (adviser's rulings of 17 Nov 2026, research-79 wave, item 4; rerun wave, item 1)

**EXPLORATORY. Nothing here scores a hypothesis. H0–H3, `83b2edf` and the timed-path pin `0e0fb012` are unchanged** (`scripts/timed_path.py --pin` against `6c6ed82`: identical). This session reads Q2 and nothing else; Q1, Q3 and Q4 stand as read from the first session.

## Why

The first session (17 Nov 2026, `attnbench-h100-diag-20261117-072424-a1`) failed D16_fa3_graph in its recording, before any capture: `launch_check` refused 28 of 4175 calls, every one `flash_attn_3::fwd: predicted nothing, recorded 1 call(s)`. Each FA3 forward allocates three outputs, clears one with `zero_`, then launches its kernel itself. The stand-in's table of GPU custom kernels named only the block-sparse kernel, so it took the `zero_` for the forward's device work and expected nothing of the forward's own. **The expectation was wrong; the graph was never reached.** No earlier GPU recording ran FA3 (P4's step 5 recorded `sdpa_flash`; Q1). On CPU, `launch_check` over that cell's own recorded trace, with FA3's forward added, passes: 0 of 4035 calls (the 140 calls below the 28 forwards are no longer checked).

## The fix

`scripts/h100_diag.py` widens the table with `flash_attn_3::fwd` around the graph cell's recording only (`fa3_launch_table`). `attnbench/cpu_boundary.py` is not changed: it is in the timed-path pin, and widening it there moves the pin (`0e0fb012` → `fff13860`).

## Cells

`configs/exploratory/h100_diag_rerun_cells.json`, sha256 `669b317c03f98f787076a3fa654e6f2ad9acd44cbbc78e250e67c85fb692a397`, in this order: D16_fa3 (the eager twin, so Q2 is read within one session) and D16_fa3_graph. The rules are the first card's; Q2 is never gated by Q3.

## Driver and money

- **Terms:** Spot, `us-central1-a` only, image v6, **one attempt**, inside 06:00–14:00Z (no override). No Flex-start. A reclaim stops; nothing reruns.
- **Limits:** cap ₹525, halt at 40 billed minutes, delete at 50. At ₹629.94/h the halt bills ₹420 and the delete ₹525 (the drafted ₹400 cap was below both; the researcher set ₹525).
- **Projected billed minutes:** about 14.5 to boot, about 4.5 for the two cells (2.5 eager, 2 graph), 7 with a 50% margin, and about 3 to pull and delete: **about 25**, about ₹262 at ₹629.94/h.
- **Money gate:** spent ₹9,737 (ledger, 17 Nov 2026) + the cap ≤ ₹15,500, and the delete bound ≤ the cap.
- **Driver:** `tools/h100_diag_spot/diag_spot.sh --rerun`: this list, this card and these limits; its own approval file, which must name the sha and the terms line; the single-instance lock and the diag-VM listing as for the first session; `--window-override` is refused with `--rerun`.

## Risks known before launch

- FA3 has never been captured in a CUDA graph on a GPU in this study; the CPU pass ran a stand-in. A capture or replay failure is recorded, and the cell is not rerun.

## Approval

The researcher approved this cell list (sha above) on 17 Nov 2026 and one session on these terms: "Yes: one Spot H100 rerun of cells D16_fa3 and D16_fa3_graph, cell list sha 669b317c…, image v6, us-central1-a, requests only inside 06:00–14:00Z, cap ₹525, halt at 40 billed minutes, delete at 50, one attempt, stop on first reclaim, exploratory only, no Flex-start."
