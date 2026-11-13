# P2 on Flex-start: the relaunch at `lock-2026-11-13-r4` (13 Nov 2026)

**Not launched.** P2's relaunch needs its own yes. It is asked only after the pre-launch CPU pass
prints PASSED at the commit the tag names (`scripts/prelaunch_cpu_pass.sh p2`, with T3.2d's step 5
XAttention masks staged as P1's). The step list and minutes are in
`docs/launch_cards/t61_p2_a100.md`.

**Why a relaunch.** The three earlier relaunches stopped with no T6.1 data:

- 12 Nov 2026, `3aa0449`: on `xattention_dcal_L2048_b1`, 34 min, ₹169. XAttention's Triton kernels
  were not nulled.
- 13 Nov 2026, `5347b73`, the tag `lock-2026-11-13`: on `xattention_noassert_dcal_L2048_b1`,
  34 min, ₹169. The fix nulled the shipped package's kernels, and the asserts-removed arm launches
  its own copy's.
- 13 Nov 2026, `886fa28`, the tag `lock-2026-11-13-r2`: in preflight, by the smoke check, 14 min, ₹69.
  Steps 7 and 8 on all five arms, the pricing check and K2's screen passed; the ground truth's
  harness then ran `sdpa_efficient` as a control, which K2's screen records as incorrect
  (unsupported on GQA shapes) and step 12 never runs.

Each XAttention arm's kernels are now nulled in its own kernels module. The smoke check runs every
arm before anything long (below), and smokes the ground truth's harness only with the dense
candidates K2's screen found correct at 2048. `-r3` was cut and tagged locally, never pushed or
launched. `-r4` adds two checks to the preflight, after the adviser's rulings dated 13 Nov 2026,
second set: the block-sparse kernel's contract, and the XAttention arms at 32768 in the smoke check.
The lock is re-cut (Appendix A, 13 Nov 2026, `-r2` to `-r4`).

## What is asked

| | |
|---|---|
| Machine | one `a2-ultragpu-1g` (A100 80GB), `asia-southeast1-c` |
| Provisioning | Flex-start; no Spot fallback (the scratchpad's `gcloud` wrapper refuses Spot, since `gcp_launch_a100.sh` falls back to Spot by itself) |
| Code | a fresh clone at the tag `lock-2026-11-13-r4`, deployed by `gcp_deploy_source.sh` |
| Bootstrap | `gcp_bootstrap_instance.sh "$NAME" "$ZONE"`, without `--suite-record` (the researcher's ruling of 12 Nov 2026) |
| Staged by scp | P1's masks (20 files) as `results/<session>/p1_masks/`, the P1 training file (`de7f0f44…`) |
| Start | `nohup bash scripts/session_exit.sh --log ~/p2.log -- python3 scripts/t61_session.py run --session results/t61_p2_<UTC>`; no `--banked-at`, which a real run refuses |
| Cap | ₹1,934 (the halt, 6.50 h at ₹297.47/h) |
| In-guest halt | 390 min (`GCP_HALT_MINUTES=390`) |
| Hard delete | 400 min (`GCP_MAX_RUN=400m`) |
| Watcher | `t61_rehearsal_watch.sh` in the background, looping until the session exits; no blocking question while it bills |

## New in preflight (step 2)

**The block-sparse kernel's contract** (`t61_session.run_kernel_contract`, the adviser's rulings
dated 13 Nov 2026, second set, item 3) runs first, in its own process. It is P1's preflight check:
- the pinned library;
- the kernel against a reference;
- the padded and the library's own buffer, bitwise equal over the 28 layers of a real XAttention
  prompt;
- memcheck with and without the fix.

A refusal, or a result that is not clean, stops the session in preflight before the smoke check.
That includes a run where the library's own buffer faulted, so no bitwise comparison was made.

The smoke check (`t31_steps.preflight_smoke`, the adviser's rulings dated 13 Nov 2026, item 3)
runs before anything long. It is untimed and its outputs are discarded
(`docs/smoke_untimed_note_2026-11-13.md`). It runs all five arms at 2048 batch 1:

1. step 7's whole cell: the recording, the isolated replay, the null pass and its structure check,
   the bare pass and the residual. An XAttention cell off the Triton path is refused;
2. step 8's whole cell, then the pricing check with each arm's mismatch share, reported as step 8
   reports it;
3. the prediction's input from that cell;
4. K2's untimed screen (each dense candidate's capability and correctness checks) at every cell K2
   screens. At 16384 batch 1 the float64 oracle runs here, where the L4 took the cross-backend
   check;
5. the ground truth's harness: one untraced and one traced forward per dense control K2's screen
   found correct at 2048 and per arm,
   under the probe and the NVML throttle sampler, then the density twin, then step 14's trace
   checks on each trace;
6. each XAttention arm at 32768 batch 1 (the second set, item 2): step 7's full-length recording,
   isolated replay and null pass with the structure check, then one untraced ground-truth forward
   under the probe. The prompt (`qa_1_32768_0`, 32700 tokens on the L4) is four query chunks of
   8192 with a partial last one; a whole last chunk is refused on the GPU.

The estimate is 10.5 to 14 minutes, entered as 12:
- steps 1–5, 7 to 9 min. P2 at `lock-2026-11-13-r2` reached K2's screen 243 s into the preflight;
- the contract, about 0.5 min (31 s in P1's preflight on the L4);
- the 32768 cells, about 3 to 4 min. P1's whole step 5 cells there took 142 s and 116 s on the L4.

At 12 minutes no Spot block has room for `host_components`, so neither card plans Spot. P2 is
Flex-start only. A failure stops the session in preflight. Every path that differs from the
L4, and what covers it, is in `docs/gpu_path_audit.md`, with what is not covered listed for the
adviser's acceptance.

## Headroom (`scripts/budget_interlock.py`)

Spent ₹5,359. P4 on Spot is ruled out (the adviser's rulings dated 13 Nov 2026, item 6), and
neither card now plans Spot blocks.

| P2 | P4 | Spent + caps ₹ | Headroom ₹ |
|---|---|---|---|
| Flex-start ₹1,992 | Flex-start ₹3,237 | 10,588 | 3,412 |

The card's P2 cap (₹1,992) includes the suite record's 20 minutes, which P2 does not run. The
launch keeps the ruled ₹1,934, halt 390 and delete 400 minutes. The steps total 250 minutes;
without the suite record the planned time is 289 minutes, and 1.3 × 289 = 375, under the halt.

The ruled rate is $2.40/h (₹230.38/h), the GPU alone. The cap takes the planning rate, ₹297.47/h,
host included (Appendix A, 12 Nov 2026).

## Machine

**Why 80GB:** the reasons are quota and comparability, not device memory.

- T0.17 recorded quota only for the A100 80GB line (`PREEMPTIBLE_NVIDIA_A100_80GB_GPUS`, 1,
  `asia-southeast1`).
- Every earlier A100 measurement ran on this machine in this zone, and so did the image-family and
  host-CPU checks.
- P1 ran the same grid in the L4's 24 GB.
