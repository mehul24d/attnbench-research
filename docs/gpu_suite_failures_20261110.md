# The suite record's failures on the GPU host (10 Nov 2026)

Source: `results/t61r_l4_20261109T2038Z_b1/home_logs/bootstrap_suite.log`, pulled from attempt 1 of the L4 rehearsal (`attnbench-l4-reh-a-20261110-0144`, commit `2053377`). No new GPU time. Adviser's ruling 2 of 10 Nov 2026: separate the tests that assume a CPU-only host from real defects in the GPU path.

The record ran `python3 -m pytest -q --deselect tests/test_timed_region_setup.py` on the L4 host (Python 3.10, kernels installed): **27 failed, 6,123 passed, 215 skipped**, 9 min 41 s. `tests/test_timed_region_setup.py` ran by itself afterwards (the S8 step): 1 failed, 12 passed.

## The split

| | Count |
|---|---|
| Tests that assume a CPU-only host | 27 of 27 |
| Real defects in the GPU path | 0 of 27 |
| A finding from the separate S8 run (not in the 27) | 1: block-sparse converts its mask each call |

The 27 were classified from the failure text in the log, not by rerunning. The two cases that the log cannot settle alone are named below.

## The 27, by cause

| Cause | Tests | Why it is the test's assumption |
|---|---|---|
| A CPU dry run reaches a GPU kernel because the kernel is installed, and gives it float32 | 15: `test_t227_shakedown` (2), `test_t228_shakedown` (4), `test_t31_session` (2), `test_session_exit` (7) | `UnsupportedConfig: block_sparse: BlockSparseAttention only support fp16 and bf16` and `sdpa/flash: No available kernel`. The dry runs are float32 on CPU and assume the kernels are absent. Two of the four T2.28 failures are `KeyError`s on results the failed step never wrote. The seven `session_exit` tests wrap a dry-run session and exit 1 with no stderr, the same dry run failing; the log does not show its output, so for these seven the cause is inferred from the others. |
| A fake capture is soaked because a CUDA device is present | 7: `test_t31_step6_cells` (4), `test_t31_wiring` (3) | `AttributeError: 'str' object has no attribute 'replay'` in `soak(cap)`. `scripts/t31_steps.py` skips the soak "for a dry run's fake capture" by testing `torch.cuda.is_available()`, so on a GPU host the tests' fake capture (a string) is soaked. Real sessions pass a real capture, and the rehearsal's step 6 passed on this host. |
| The CPU boundary or an absent library is assumed | 2: `test_bsa_contract` | `a CUDA device is present; the CPU boundary is for a CPU`; and the stand-in test expects "not installed" where the library is installed. |
| A profile read on a CUDA host | 2: `test_components_trace_recording::test_the_parser_refuses_a_cpu_only_trace`, `test_isolated_segments::test_a_replay_in_blocks_has_the_operators_of_the_whole_replay_under_one_region` | The parser still refuses a trace with no device work; only the message differs from the one the test expects. The second compares event lists of a CPU toy (`device="cpu"`, no-op sync) and the profiler adds `cudaDeviceSynchronize` and `detach` events when it sees a CUDA device. Not reproduced without a GPU. |
| A count that depends on what is visible | 1: `test_doc_derived_numbers::test_the_suite_counts_match_the_collected_suite` | 6,184 + 162 documented, 6,378 collected. The documented counts are for a host where RULER and the tokenizer are hidden. |

## The one finding from S8

`test_no_backend_rebuilds_setup_inside_the_timed_region` fails for `block_sparse` on the L4: `setup ops per call [1, 1, 1]`. The adapter calls `mask.to_block_sparse_attn_mask(b, h, device)` in every `forward`. The call is a no-op `.to`, two `unsqueeze`s and an `expand`: views, no allocation, no device work. On the Mac's CPU it takes about 3 µs; at 32 layers that is about 0.1 to 0.3 ms a prefill on the instance's host, not measured there.

The test is built to flag exactly this ("work that depends only on cfg is being redone every call") and was never exercised for `block_sparse` before, because on a CPU host the library is absent and the test reads the backend as not exercised. It is not a defect that produces a wrong number: ground truth, the components and the replay all run the same call, so every banked row carries it alike. Hoisting it changes the timed code of every block-sparse arm, and with it the code identity the banked P1 rows and the training file stand on. **No change made.** The options are to leave it and add `("block_sparse", "block_sparse")` to `ALLOWLIST` with this reason, or to hoist it under a new identity. That is the researcher's decision.

## Consequence

`scripts/gpu_suite_record.sh` as written cannot pass on a GPU host. P2's and P4's cards do not run it (the adviser's ruling of 10 Nov). No CPU-assumption test is fixed now. When the record is wanted, the work is a GPU-safe selection (skip these 27, or mark them `cpu_only`) and the soak guard keyed on the capture, not the device.
