# Are XAttention's Triton kernels priced? (13 Nov 2026)

The adviser's rulings dated 13 Nov 2026, item 1, ask for two things before the tag `lock-2026-11-13` is pushed:

- the two Triton kernels that step 7's null pass now nulls are still priced in the prediction;
- step 6's pricing-mismatch share for the XAttention arm is within 0.05.

"Step 6" is T3.1's name for `device_components`, which is step 8 in P2's session.

**Input:** P2's relaunch (`attnbench-a100-p2r-20261112-2309`), step 7's recording of `xattention_dcal_L2048_b1` (A100, `xattn:triton`), read under the yes of 13 Nov 2026. The script is `scripts/p2_triton_priced_check.py`, and it runs on the worktree's code at `2f88340`.

**Method:** P2 stopped in step 7, so no graph replay or isolated original exists on the A100. The recording's own eager trace stands in for the graph replay: its nodes are taken in the region `attnbench::prefill`, then `replay.match`, `replay.price_positions` and `op_sequence.to_operations` run on them.

## Priced: yes

| | |
|---|---|
| Triton positions | 56: 2 per layer × 28 layers, all device positions, none without a device record |
| Matched and priced | 56 of 56; no position unpriced |
| Their device time | 1.179 ms of 59.494 ms priced (1.98%) |
| Per layer | median 42.1 µs (41.8 to 43.1) |
| Operations' device total | 59.494 ms, equal to the priced total |

**Break tests**, both killed:

1. The Triton positions' device prices are removed. `to_operations` refuses with missing positions 38, 39, 193, 194 and onwards; it does not price them lower.
2. The Triton nodes are removed from both replays. All 56 positions become unpriced, and the cell has no operations, so the arm's gate reads it.

The host null pass and the device pricing are separate. `NullEntry` changes only the host prices. Device prices are keyed by the recording's sequence, which keeps every device position.

## Mismatch share: not shown

P2's data holds no pricing-mismatch share for XAttention on the Triton path: step 8 never ran. Every banked `device_components` file is from the L4, which has no Triton path. The eager stand-in's share is 0 by construction, so it is not evidence.

**Bound for this cell:** if both Triton kernels were priced from the isolated original, they would add 1.98% of the cell's priced time. That bound is for this cell at 2048 only; the arm's share over P2's cells is still unknown.
