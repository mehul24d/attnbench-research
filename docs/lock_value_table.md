# T4.1 lock values still blank (1 Nov 2026)

One row per T4.1 item (32). "Plan default" is what the plan or code already states; "pilot value" is what banked data or a shakedown gives; "proposal" is Claude's, for the researcher to accept or change. A dash means nothing is stated yet. Nothing here is a decision.

| # | Value | Plan default | Pilot value | Proposal |
|---|---|---|---|---|
| 1 | H0 to H3 tolerances | "within [lock]" per hypothesis | none | Set from the T3.2 repeat spread, once the floors exist |
| 2 | H6 regret bound | mean and worst case per pairing | none | After T7.6 shows the regret distribution; no number yet |
| 3 | Ground-truth density set | every admissible density | T4 pilots ran a few densities per arm | Working, pre-lock (T3.1): block-sparse arms at 0.25 and 0.5; the XAttention arms at their calibrated table (density measured, not set). The lock set is still open (audit A4) |
| 4 | H4 equivalence bound | delta = 10 points in the power rule; T1.5 also prices 5 points | T1.1 calibrated XAttention: -4.3 and -6.0 points on the two primary `qa_1` cells | 10 points |
| 5 | H4 confirmatory family | estimator pairs at block 32, 64, 128 plus the random control | none | Count m from the declared family; "three comparisons" is too few (audit D4, open) |
| 6 | Power target and n | alpha 0.025, power 90%, n about 1,300 x p | p (pilot disagreement, upper 95% bound) not yet computed (T1.5) | Keep alpha 0.025 and power 0.90 |
| 7 | Primary cell n and admissibility n | at least 200 | none | 200, raised if 1,300 x p is larger |
| 8 | Admissibility Holm family | declared set, must equal the measured set | none | The (arm, density) set T1.8 produces |
| 9 | Context grid vs position limit | 2048 to 32768; 32,768 positions for Qwen2.5 | Shakedown 1: `qa_1` at 8192 lands at 8075 tokens | As the plan, plus the 32768 pairing rule |
| 10 | Delay-injection slope bound | within a bound of 1.0 | none | From the P1 repeats |
| 11 | Mask-gate bound | agreement and density difference vs P1's masks | none | After the P2 session's first masks |
| 12 | Cross-card dense task-score bound | fixed at lock | none | After the A100 dense control (audit B1, open) |
| 13 | Replay-versus-trace bound (Gate A) | set from the L4 shakedown's agreement distribution | Shakedown 2 gives it | Take the chosen quantile of that distribution |
| 14 | Null-kernel floor multiple | within a fixed multiple of published figures | TaxBreak 4.7 us, SKIP 2.37 us (other hosts); ours from T2.2 | After Shakedown 2's T2.2 smoke |
| 15 | Probe bounds | CV bound; source-vs-target median bound | none | From the P1 repeats |
| 16 | Resolution and noise floors | from measured run-to-run spread | dry runs use a provisional 1.0 ms | From T3.2 |
| 17 | Floors quantile | 0.95, provisional | 0.95 | 0.95 |
| 18 | HDBI boundedness threshold | fitted on training pairings (host-bound / balanced / device-bound in code) | none | Fit on P1; two vs three classes is audit D3 (open) |
| 19 | Learned-predictor features | frozen in code (T2.10) | LightGBM 4.6.0 | Keep the frozen set |
| 20 | Learned-predictor hyperparameters | at lock | none | Library defaults, pinned |
| 21 | K2 correctness tolerance | the gate's `TOL` | bf16 2e-2, fp16 5e-3, fp32 1e-4 (atol = rtol), "loosely calibrated" | Keep until Shakedown 2's float64 gate shows the real error |
| 22 | Head-type criterion | T1.3 | undecided | After T1.3 |
| 23 | Rerun budget and minimum clean rounds | "a stated budget" | dry runs: 1 extra round, 2 clean rounds | One extra round per cell and 3 clean rounds, to be checked against Shakedown 2's flag rate |
| 24 | Throttle set | decided 27 Oct 2026 | hw slowdown, both thermal slowdowns, hw power brake | As decided |
| 25 | Payload class edges | proposed 8 B, 4 KiB, 1 MiB | none | As proposed |
| 26 | Prompt-set hash and tokenizer version | recorded | transformers 4.46.0 on image v6 | Record both from the T3.1 prompts |
| 27 | Realised-length key rule | not stated | prompts land below the budget | Key by the prompt's realised token count, budget kept as a label |
| 28 | Budget cut order | if forced: full grid for the XAttention pair, compiled XAttention pair, mean-pool, dense; others at 8192, 16384, 32768 | none | As the plan |
| 29 | Compiled-versus-eager density flag tolerance | ±0.01 absolute, working (31 Oct 2026 ruling) | Shakedown 2: XAttention 0.66582 compiled, 0.66688 eager (−0.00106) | Keep ±0.01 unless Shakedown 3's compiled rows show a wider spread |
| 30 | Probe-length rule (T2.18) | three lengths at or below 32768 with distinct chunk counts of at least two, else two plus a full recording at 32768 (31 Oct 2026 ruling) | XAttention: 20480, 28672, full at 32768 | As ruled |
| 31 | Flagged-memset sensitivity bound (T2.3) | each memset with no device record priced at the memset zoo's duration for its size (31 Oct 2026 ruling) | Shakedown 2: 84 flagged memsets per XAttention prefill | As ruled; the paper reports the count and the shift |
| 32 | Chain-check tolerance (T2.2) | per operator met at a composite or by its device argument, \|sum of the chain-priced costs - sum of the direct host times in the isolated replay\| within rel x direct + abs x calls (1 Nov 2026 ruling) | none on a GPU; on the CPU boundary the direct time holds the compute (not gated) | Working: rel 0.25, abs 5 us per call; set from Shakedown 4's `chain_check.json` |

Also blank in `configs/t31/p1_session.json`: densities (row 3), ground-truth groups, repeats, dense rounds and the sampler interval, all filled for T3.1 from the block below; the resolution floor (row 16) and the rerun budget (row 23), which stay null as inputs.

## Working values for T3.1 (researcher's decision, 31 Oct 2026)

T3.1 is pre-lock P1 data. It takes the values below, marked "working, pre-lock", and `scripts/t31_steps.py` reads them from this block only (one source). They must never be chosen from P2 or P4 data, and the runner refuses them for any pairing but P1. The floors stay null as inputs: T3.1 produces a session floor from its own dense pre-pass (at the provisional quantile, with the series saved so the lock can recompute it at its own). The rerun budget stays null: it is decided after the T3.2 repeats, and until then T3.1 reruns nothing and says `complete: false` where a cell lacks clean rounds. A Shakedown 2 rehearsal may use any values; its results are flagged "rehearsal" and discarded.

| Value | Working value | Why |
|---|---|---|
| Densities | dense 1.0; oracle and mean-pool 0.25 and 0.5; XAttention arms "calibrated" | The pilots' 0.25 plus one denser point; XAttention's density is its estimator's |
| Ground-truth groups | eager: all five arms; compiled default and compiled capture: mean-pool and XAttention, each with the eager dense control | One compile setting per process (T2.21); the oracle builds on the host |
| Repeats | 10 clean rounds per cell | Enough for a median and p90 at P1 |
| Dense rounds (K2) | 5 | Interleaved candidate rounds per cell |
| Sampler interval | 0.1 s | The sampler process's 10 Hz |
| Chain-check tolerance (row 32) | 0.25 of the direct time plus 5 us per call, per operator | A guess before any GPU figure: the null pass skips a composite's C++ body (an allocation, the inner dispatch, a few us), and the isolated replay's profiler adds about 1 us per inner call. Shakedown 4's `chain_check.json` gives the distribution; a rehearsal records the check and does not stop on it |

```json t31-working-values
{
 "status": "working, pre-lock",
 "pairing": "P1",
 "densities": {"dense": [1.0], "oracle": [0.25, 0.5], "meanpool": [0.25, 0.5],
               "xattention": ["calibrated"], "xattention_noassert": ["calibrated"]},
 "gt_groups": {"eager": ["dense", "oracle", "meanpool", "xattention", "xattention_noassert"],
               "compiled_default": ["meanpool", "xattention"],
               "compiled_capture": ["meanpool", "xattention"]},
 "reps": 10,
 "dense_rounds": 5,
 "sampler_interval_s": 0.1,
 "chain_check_tolerance": {"rel": 0.25, "abs_us": 5.0}
}
```
