# GPU path audit: what differs between the L4 and the A100/H100 (13 Nov 2026)

The adviser's rulings dated 13 Nov 2026, item 4. P2 stopped in step 7 twice, on code the L4 never
runs (12 and 13 Nov 2026). This lists every path of P2's and P4's sessions that depends on the
card, and names what covers it before anything long runs. "Smoke" is the preflight smoke check
(`t31_steps.preflight_smoke`, step 2, untimed; `docs/smoke_untimed_note_2026-11-13.md`). It runs
on the session's own card, P2 on the A100 and P4 on the H100.

How the list was made: a search of `attnbench/` and the session scripts for every read of the
device (`get_device_properties`, `get_device_name`, `get_device_capability`, `total_memory`,
`is_available`), for card names (`"100"`, `A100`, `H100`, `L4`), and for hard-coded module paths
and arm names. Then each hit was traced to the steps that reach it.

## Covered

| # | Path | Why it differs | Covered by |
|---|---|---|---|
| 1 | XAttention's estimator takes Triton when the device name contains `"100"` (`backends/xattention.py:75`, the authors' `Xattention.py:58`) | the L4 takes the torch path; the A100 and H100 both take Triton | smoke steps 7 and 8 on every XAttention arm, refused unless the implementation is `xattn:triton`; `tests/test_arm_module_parity.py` on the CPU, the device named as an A100 |
| 2 | The Triton kernels each XAttention arm launches live in that arm's own package (`xattn.src.kernels`, `attnbench_xattn_noassert.src.kernels`) | only reached on the Triton path; the 13 Nov stop | `null_entries` takes the backend's `kernels_module()`; the parity test checks each arm's launch, null entries and null pass against the module it really calls (mutation: alias left unpatched, killed) |
| 3 | `block_sparse_attn` (the extension and its interface), shared by every sparse arm | one build per card; the A100's read past the row mask (1 Nov 2026) | smoke steps 7 and 8 and the ground-truth forward on every sparse arm, through the adapter's row-mask fix; the parity test's one forward per arm through the mask capture and the recorder |
| 4 | FA3, H100 only (`dense_selection.planned_candidates`; `fa3_wheel`, sm_90a) | not built for the A100 or the L4 | P4's `fa3_import` now runs before the preflight; the smoke refuses a planned FA3 that is not available, runs it through K2's screen at every cell and, if the screen finds it correct at 2048, as a ground-truth control (untraced and traced) |
| 5 | K2's correctness check is chosen by device memory (`gates.exact_oracle_fits`, a fraction of `total_memory`) | the float64 oracle runs at 16384 batch 1 on 80 GB (about 48 GiB peak: the score matrix and two copies); the L4 took cross-backend there. K2 (step 9) has never run on an A100 or H100 | the smoke runs K2's untimed part (`dense_selection.screen`, shared with K2) at every cell K2 screens, on the session's card, and reports the peak memory per cell; refused if a cell has no correct candidate (test with a mutation, killed) |
| 6 | cuDNN SDPA at 8192 and below, declined above 8192 on every card (`impls.SDPABackend`, the L4's Xid 31) | never launched on the A100 or H100; an illegal access kills the process | K2's screen in the smoke: a fault stops the session in preflight, not in step 9 |
| 7 | The isolated replay's L2 flush is twice the device's `L2_cache_size` (`replay.l2_flush`) | 48 MB (L4), 40 MB (A100), 50 MB (H100) | smoke step 7, every arm |
| 8 | CUDA-graph replay and its live masks against the bank (step 8) | per-card graph capture | smoke step 8, every arm, with the pricing check reported |
| 9 | The throttle sampler reads NVML (`clock_sampling.nvml_reader_factory`) | throttle-reason bits and clocks are per card | the smoke's ground-truth harness runs with the session's sampler and reports its timing stats |
| 10 | Kineto traces and step 14's checks (`check_trace.py kineto`) | kernel names and streams are per card | the smoke exports one traced forward per control and per arm and runs the trace checks on each; a flagged trace is refused |
| 11 | The ground-truth probe's score targets (`gt.probe_for`) | module objects per arm | the smoke refuses a deployable arm whose untraced or traced forward scored nothing, or that scored in setup; the parity test checks a score event per deployable arm |
| 12 | Hard-coded arm names and module paths | none found that depend on the card | the parity test checks that P2 and P4 run the same five arms and every patch site sees each arm's calls |

## Closed after the first reading (the adviser's rulings dated 13 Nov 2026, second set)

| # | Path | Ruling | Covered by |
|---|---|---|---|
| B | Steps 5–8 above 2048 on the A100/H100: more query chunks, the same two Triton kernels | item 2: close it in the smoke | the smoke's `long_smoke`: each XAttention arm at 32768, batch 1, untimed, through step 7's full-length recording, isolated replay and null pass with the structure check, then one untimed forward of the ground truth's harness under the probe; refused off the Triton path, when the prompt's last query chunk is not partial, or when a deployable forward scored nothing. The session's 32768 prompt (`qa_1_32768_0`) is 32700 tokens on the L4: four chunks of 8192, the last 8124 tokens and padded |
| E | The block-sparse kernel's contract under memcheck (`bsa_contract`, P1's preflight) | item 3: add it to P2's and P4's preflight | `t61_session.run_kernel_contract`, in its own process before the smoke check: P1's check (the pinned library, the kernel against a reference, the padded and the library's own buffer bitwise equal over every layer of a real XAttention prompt, memcheck with and without the fix). A refusal, a missing result or a result that is not clean is a preflight problem, and the smoke does not run. Unlike P1's preflight, a run where the library's own buffer faulted, so no bitwise comparison was made, is refused. `tests/test_preflight_kernel_contract.py`; mutations K1–K6 in `docs/p2_r2_mutations.md` |

## Accepted as not covered (the adviser's rulings dated 13 Nov 2026, second set, item 1)

| # | Path | Why not | Exposure | Ruling |
|---|---|---|---|---|
| A | The ground truth's timed rounds on the A100 and H100: the timer, the drift rule against the floors, the throttle flag on a real block | timed; a timed block before `predict` breaks the stamp rule | step 12. The harness, sampler and traces are smoked; the clocked part ran on the L4 only (T3.2a–d) | accepted |
| C | The H100 boot, image and build beyond FA3 | not examined in this audit | the FA3 gate passed on 6 Nov 2026 (`1af8763`); P4 is asked only after P2 is banked | accepted; revisited before P4's card |
| D | Device memory at 32768 in steps 7 and 8 | not smoked at first | both cards have more device memory than the L4, where every length ran (T3.2a–d); the host RSS ceiling does not depend on the card | accepted; step 7's part is now smoked by B's `long_smoke` |

What remains uncovered is A, the clocked part of step 12, and C, P4's H100 beyond FA3, which is
looked at again before P4's card.

## C, revisited before P4's card (the adviser's rulings dated 14 Nov 2026, item 5)

| What | Status |
|---|---|
| The boot image | **Fixed.** `gcp_launch_h100.sh` defaulted to v5, the H100's earlier image; P2 ran v6. It now boots only `attnbench-env-v6-20260917`, and P4's preflight refuses libraries other than P2's (`configs/t64/p2_libraries.json`), FA3 alone excepted |
| block_sparse_attn and flash_attn on sm_90 | v6's build logs (`bsa_build.log`, `fa_build.log`, pulled with P2's logs) show `archs=80;90` for both, rc 0. Running on the H100 is covered by P4's preflight: the kernel contract (E) and the smoke's steps 7 and 8 and `long_smoke` (B) on every sparse arm |
| XAttention's Triton kernels on sm_90 | covered by the same preflight; Triton JIT-compiles per card |
| FA3 on v6 | v6's Python is 3.10, the wheel's cp310 tag; `fa3_import` checks hash, tags, install and import before the preflight |
| v6 has never booted an `a3-highgpu-1g` | **listed for acceptance.** The driver is read into the preflight's provenance; a boot or driver failure stops in prereqs or preflight, before anything is timed, at the cost of the boot |
| The H100's timed rounds | A, accepted 13 Nov 2026 |
| Flex-start capacity for `a3-highgpu-1g` | availability, not correctness: none in three zones on 6 Nov 2026 |
