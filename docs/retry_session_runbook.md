# Retry session runbook — flash-attn + BSA + deferred measurements

**Budget: 3 hours. One g2-standard-8 + L4, on-demand, asia-south1.**

Supersedes `compile_session_runbook.md` for execution order. That document
keeps the standing rules (agreement is not verification; save the serial log;
re-verify ruled-out causes; kill and start are separate invocations) — read
them once, then work from this file.

This session is **execution, not improvisation**. Every command below is
written out. If something is not in this document, it is not in scope; add it
to a follow-up rather than inventing it on billed hardware.

---

## What went wrong last time, in one paragraph

The 2026-09-02 compile session spent ₹233 and produced no kernels. Three
compounding causes: ninja was not on PATH so torch compiled serially with
`MAX_JOBS` inert; the build compiled four GPU architectures because
`FLASH_ATTN_CUDA_ARCHS` was never looked for; and `MAX_JOBS=5 ×
NVCC_THREADS=2` demanded ~50 GB on a 31 GB box and was killed by the OOM
killer after ~90 minutes. SSH became unreachable ~5 minutes into the build and
never returned, so every measurement queued behind the build was lost. All
three causes are now guarded in `scripts/build_flash_attn.sh`. **The ordering
lesson is what this runbook encodes: the build can take the machine away from
you, so nothing that must be recorded may be scheduled after it.**

---

## Prerequisites (free, do before launching)

```bash
cd ~/Desktop/research/attnbench_scaffold
python3 -m pytest tests/ -q                 # expect all green
bash scripts/gcp_cleanup_check.sh           # expect "Clean"
gcloud compute machine-images list          # v2 must be READY
```

**The suite includes a static call-site check** (`tests/test_script_call_sites.py`)
that verifies every `scripts/*.py` call into `attnbench` matches the current
signature. This is not optional politeness -- scripts are never imported by
the tests, so a signature change that propagates to the library and its tests
but misses a script produces a `TypeError` on the first line of real work:
after launching an instance, after the model downloads, on billed hardware.
That happened twice in one session, both times caught by eye. Running the
suite before launch is what now catches it instead, so **do not skip the
suite because "only scripts changed"** -- that is precisely the case it
covers.

`attnbench-l4-validation-image-v2-20260902` is the base. It carries torch
2.9.1+cu129, transformers pinned to 4.46.0 (the torchaudio ABI break),
flash-linear-attention, the repo, and the Qwen2.5-1.5B weights.

---

## Phase 0 — launch (~5 min)

```bash
GCP_ZONE=asia-south1-c bash scripts/gcp_launch_compile_session.sh
```

`asia-south1-b` stocked out on 2026-09-02 and GCP named `-c` as having
capacity; both draw on the same regional quota. If `-c` is out, try `-b`, then
`-a`. A stockout creates nothing and costs nothing.

**Record the boot time.** Every spend figure is measured from it.

Then, before anything else:

```bash
gcloud compute ssh NAME --zone=ZONE --command="sudo systemctl edit --force --full ssh --no-pager" # or:
gcloud compute ssh NAME --zone=ZONE --command="sudo mkdir -p /etc/systemd/system/ssh.service.d && printf '[Service]\nRestart=always\nRestartSec=5\n' | sudo tee /etc/systemd/system/ssh.service.d/restart.conf && sudo systemctl daemon-reload"
```

Whether sshd was among last session's OOM kills is **unresolved** — the serial
log was destroyed with the instance. The mitigation is cheap and applies
either way.

Sync current code (the image predates every fix from 2026-09-02):

```bash
tar czf /tmp/src.tgz --exclude='__pycache__' --exclude='.git' --exclude='results' \
  --exclude='*.egg-info' attnbench scripts configs tests
gcloud compute scp /tmp/src.tgz NAME:~/src.tgz --zone=ZONE
gcloud compute ssh NAME --zone=ZONE --command="cd ~/attnbench_scaffold && tar xzf ~/src.tgz && python3 -c 'from attnbench.build_guards import check_build_memory; print(\"guards present\")'"
```

The repo is at `~/attnbench_scaffold` (NOT `~/research/...`).

---

## Phase 1 — MEASUREMENTS FIRST, on the idle machine (~30 min)

**Nothing may start a build until this phase is complete and its outputs are
copied off.** This is the whole lesson of the last session: the build
saturates the box, SSH dies, and anything scheduled behind it is lost. These
numbers currently exist only in conversation transcripts.

### 1a — batch scaling probe

```bash
gcloud compute ssh NAME --zone=ZONE --command="cd ~/attnbench_scaffold && python3 scripts/probe_batch_scaling.py --seq-len 16384 --batches 1,2,4,8,12"
```

Writes `results/batch_scaling/batch_scaling.parquet` with a provenance stamp.
Expected from the 2026-09-02 measurements (unrecorded, which is why this runs
first): batch 8 fits at ~12.7–13.6 GiB, batch 12 OOMs, and **per-example wall
time is flat** (sdpa_flash 2.38 → 2.01 → 2.04 → 2.07 s/example across batch
1→8). That flatness is the finding: the workload is compute-bound at 16K, so
batching wins nothing regardless of memory. An OOM row is a result, not a
failure.

#### HARD GATE: batch=8 memory at 16384

**Stop the session and report if batch=8 does not fit, or if peak memory
lands outside the range below.** This is the check that says the machine and
the sizing are both behaving; everything after it assumes they are.

Expected peak memory, **recomputed for token-exact sizing** — do not use the
2026-09-02 numbers directly, they were measured at 19821 tokens because
nominal 16384 then produced a 21% longer context:

| batch | sdpa_flash | GLA | verdict |
|---|---|---|---|
| 1 | ~3.9 GiB | ~4.0 GiB | must fit |
| 2 | ~4.9 | ~5.1 | must fit |
| 4 | ~6.9 | ~7.3 | must fit |
| **8** | **~11.0** | **~11.7** | **must fit — the gate** |
| 12 | ~15.0 | ~16.1 | *expected to fit now* |
| 16 | ~19.0 | ~20.5 | marginal against ~22 GiB usable |

Accept anything within **±15%** of the tabulated figure. Derivation: weights
are a fixed 2.88 GiB; activations were measured at ~1.10 GiB per example per
19821 tokens for GLA (~1.01 for sdpa_flash) and both backends are O(seq_len)
in memory, so activations scale by 16384/19821 = 0.827.

**Do NOT gate on batch=12 OOMing.** It OOM'd on 2026-09-02 and that is
exactly the expectation that must not be carried forward: at 16384 real
tokens it now projects to ~15–16 GiB and should *fit*. A gate asserting the
old behaviour would fire on a correct measurement and abort a healthy
session.

Failure readings and what they mean:

- **batch=8 OOMs** → sizing is not doing what it claims (contexts longer than
  the budget), or the model loaded in the wrong dtype. Check
  `seq_len_real` against `seq_len_nominal` in the parquet before anything
  else; they should agree to ~1%.
- **memory much lower than the table** → contexts are shorter than requested;
  check for a `BudgetTooSmallError` fallback or a truncating tokenizer.
- **batch=1 above ~5 GiB** → weights are not bf16.

### 1b — anchors at both lengths

```bash
gcloud compute ssh NAME --zone=ZONE --command="cd ~/attnbench_scaffold && python3 scripts/time_one_accuracy_example.py --seq-len 16384 --dense-backend sdpa_flash"
```

Do **not** use `sdpa_math` — it materialises the full O(seq²) score matrix and
OOM'd at 32K (69.66 GiB requested).

Reference from 2026-09-02 at 16384 (19821 real tokens): scoring 4.890 TFLOPS,
sdpa_flash 41.399, gla 13.197. The like-for-like comparison against the 8192
anchor is **scoring only** (5.185 → 4.890, a 5.7% degradation with length);
the `measured` blend is not comparable across the two because the dense
backend differed.

### 1c — copy results off IMMEDIATELY

```bash
mkdir -p results/gpu_session_$(date +%Y%m%d)
gcloud compute scp --recurse NAME:~/attnbench_scaffold/results/batch_scaling \
  results/gpu_session_$(date +%Y%m%d)/ --zone=ZONE
```

Do not defer this to teardown. An instance whose SSH dies takes its disk
contents with it.

---

## Phase 2 — flash-attn (~45–60 min)

```bash
gcloud compute ssh NAME --zone=ZONE --command="pip install --user -q ninja psutil packaging wheel"
gcloud compute ssh NAME --zone=ZONE --command="cd ~/attnbench_scaffold && nohup bash scripts/build_flash_attn.sh > ~/fa_build.log 2>&1 & echo started"
```

The script refuses to start if `MAX_JOBS × NVCC_THREADS × 5 GB` exceeds free
memory, asserts ninja is visible to torch, and pins
`FLASH_ATTN_CUDA_ARCHS=80;90`. Defaults are `MAX_JOBS=4`, `NVCC_THREADS=1` —
**do not raise either without re-reading section 4 of that script; they
multiply.**

Expected: 72 CUDA targets, 2 architectures, ~45 min. Confirm within the first
2 minutes that it is genuinely parallel:

```bash
gcloud compute ssh NAME --zone=ZONE --command="ls ~/fa/flash_attn-*/build/temp*/build.ninja >/dev/null 2>&1 && echo NINJA_IN_USE || echo SERIAL_FALLBACK; ps -eo cmd | grep -c '[n]vcc'"
```

`SERIAL_FALLBACK` means stop and fix PATH — serial is 3.6 hours and will not
finish.

**Poll sparingly.** SSH competes with the build for CPU. Once every 10–15
minutes, and prefer the serial console (an API call, no shell needed):

```bash
gcloud compute instances get-serial-port-output NAME --zone=ZONE | grep -iE "oom-kill|Killed process" | tail -5
```

Re-check that OOM grep **every time you report status** — a clean result is
true only as of when it was measured.

---

## Phase 3 — verify, then capture v3 (~15 min)

```bash
gcloud compute ssh NAME --zone=ZONE --command="python3 -c 'import flash_attn; print(flash_attn.__version__)'"
gcloud compute ssh NAME --zone=ZONE --command="cd ~/attnbench_scaffold && python3 -m pytest tests/test_gqa_agreement.py -q"
```

The GQA gate becomes a genuine three-way check (naive-with-explicit-KV-
expansion vs SDPA vs FA2) once flash-attn imports — it adds `FlashAttention2`
automatically. **A failure here invalidates every GQA result in the study**;
stop and report rather than continuing.

Then capture, from the **running** instance:

```bash
gcloud compute ssh NAME --zone=ZONE --command="sync"
gcloud compute machine-images create attnbench-l4-image-v3-$(date +%Y%m%d) \
  --source-instance=NAME --source-instance-zone=ZONE \
  --description="v2 + flash-attn 2.8.3.post1 (sm80,sm90), verified import + GQA gate"
```

Capture from running, not stopped: **the hard cap re-arms on every boot**, so
a stop/start silently grants a fresh 6 hours. If you must stop, immediately
re-arm the original deadline with `sudo shutdown -h HH:MM`.

If the import could not be verified, put `unverified` in the description
naming what was not confirmed, and **keep v2** until a future session boots v3
and confirms the import.

---

## Phase 4 — Block-Sparse-Attention (~40 min, time permitting)

```bash
gcloud compute ssh NAME --zone=ZONE --command="cd ~/attnbench_scaffold && nohup pip install --user --no-build-isolation 'block-sparse-attn @ git+https://github.com/mit-han-lab/Block-Sparse-Attention.git@49d6c39e4dc0303442cda3bb758b3925d4399c49' > ~/bsa_build.log 2>&1 & echo started"
```

`--no-build-isolation` is required (pip's isolated env cannot see torch).
Check whether BSA has its own arch hook before assuming it does not — that
assumption cost 50 minutes last session. If it builds, **re-capture v3** with
both kernels.

---

## Phase 5 — block_sparse anchor (only if BSA built)

```bash
gcloud compute ssh NAME --zone=ZONE --command="cd ~/attnbench_scaffold && python3 scripts/time_one_accuracy_example.py --seq-len 16384 --dense-backend sdpa_flash"
```

This is the one measurement that narrows the Stage 3 estimate from its
current **16–25 h bracket**, which is bounded almost entirely by
block_sparse's unmeasured throughput. With it, the two-sided restore rule in
`configs/accuracy/stage3_grid.yaml` can finally be applied:

- anchor **worse** than extrapolated → fire reserve cut C (`32768: 100 → 50`)
- anchor **better** by enough to open ~5 h → restore `16384: 200 → 300`

Without it, the committed grid stands unchanged. That is an acceptable
outcome, not a failure.

---

## Phase 6 — teardown (~10 min)

```bash
S=results/gpu_session_$(date +%Y%m%d); mkdir -p $S
gcloud compute instances get-serial-port-output NAME --zone=ZONE > $S/serial_console.log   # FIRST
gcloud compute scp --recurse NAME:~/attnbench_scaffold/results $S/ --zone=ZONE
gcloud compute instances delete NAME --zone=ZONE --quiet
bash scripts/gcp_cleanup_check.sh
gcloud compute machine-images list
```

The serial log goes first because it is destroyed with the instance and is the
only diagnostic channel that survives losing SSH. Delete, never stop — a
stopped instance still bills for its disk. Keep v2 until v3 is verified.

Report elapsed instance time and estimated spend.

---

## Hard cutoff

**At cap minus 90 minutes**, stop whatever is building, capture with what is
installed, and run any remaining measurements. An image plus real measurements
beats a complete build that cannot be captured because the cap fired
mid-capture — `machine-images create` is not interruptible.

Decided here, in advance, so it is not decided under time pressure when "ten
more minutes" is most persuasive and most expensive.

## Abort conditions

Stop and report rather than continuing:

- the GQA gate fails (invalidates the study's GQA results)
- any step fails twice
- SSH unreachable for >20 min with no build running to explain it
- an OOM kill appears in the serial log

## Budget note

₹28,663 of credit expires **2026-12-01**; ₹474 spent to date. Money is not the
constraint — calendar time before expiry is. Do not trade a session's success
for instance-minutes: take the extra measurement, keep the redundant image,
buy the certainty.

---

# OUTCOME — 2026-09-03

**Succeeded.** ₹209, 2h35m instance time (boot 04:50:52Z, deleted ~07:25:30Z).
Zero OOM kills across the whole session.

| phase | result |
|---|---|
| 0 launch | `asia-south1-c` after all three zones stocked out once; succeeded on retry |
| 1a batch probe | **hard gate PASSED** — batch=8 at 10.96 GiB (sdpa_flash) / 11.71 (GLA), every prediction within 0.5% |
| 1b anchor | scoring 5.362 TFLOPS, sdpa_flash 41.714 |
| 2 flash-attn | built in 2h12m, 73 ninja edges, sm80;sm90, `MAX_JOBS=4` |
| 3 verify + capture | import + real FA2 forward OK; **v3 captured** |
| 3b GQA gate | **PASSED, genuinely three-way** (FA2 confirmed present) |
| 4 BSA | NOT DONE — no time |

`attnbench-l4-image-v3-20260903` carries flash-attn 2.8.3.post1. Its
description says the GQA gate was not run, because it was captured before the
gate; the gate then PASSED, so the description is pessimistic, not wrong.
**v2 can now be deleted** — v3 supersedes it and the gate has passed.

## What the three guards were worth

All three 2026-09-02 failure modes were prevented, and each guard was
observed working rather than assumed to:

- ninja on PATH — `NINJA_IN_USE`, load average pinned at 4.00 for two hours
- `FLASH_ATTN_CUDA_ARCHS=80;90` — 73 edges, not the ~146 four architectures
  would have produced
- memory guard — `4 concurrent cicc, ~20 GB needed, ~27 GB usable`; peak
  observed `cicc` RSS was 3.83 GB against the 5 GB the guard budgets

## The finding nobody was looking for

The batch probe reported `gla 4.28x -- batching helps`, contradicting the
compute-bound conclusion. It was **JIT warmup**, not batching: two identical
batch=1 GLA calls in one process took 5.394 s and 1.384 s. The tell was that
GLA's batch=2 run took *less total wall time* than batch=1 while doing twice
the work — impossible for any batching effect. The same defect understated
the anchor's GLA figure as 12.018 TFLOPS against a true ~47.7.

Fixed in `batch_scaling.py`, `probe_batch_scaling.py` and
`time_one_accuracy_example.py`, with regression tests built from the real
measurements. Warmup is deliberately NOT applied to the scoring pass, which
writes to `score_cache` — a warmup there would time a cache load.

## Next session — boot from v3, budget 2h

### Precondition: a real git commit (do this BEFORE launching)

```bash
cd ~/Desktop/research/attnbench_scaffold
git status --short                                     # expect empty
python3 -c "from attnbench import provenance; print(provenance.capture().git_commit)"
# must print a 40-hex SHA -- not None, and NOT the string "HEAD"
```

Session 4 produces the 32K baseline and the corrected GLA anchor. Those are
results that later sessions will want to join, so they must carry a verifiable
commit -- five minutes now against re-running measurements later. Everything
measured before commit `eee2c6b32cf4` records `git_commit="HEAD"` and is not
joinable without `allow_unverified=True`. See `docs/silent_failure_patterns.md`
instance 3 for why that string exists.

flash-attn is already in the image, so the whole window goes to BSA. Launch
with `GCP_ZONE_FALLBACKS="asia-south1-c asia-south1-b asia-south1-a"`; the
script now stops at the first success and refuses to launch if an instance
already exists.

1. **BSA build** (Phase 4) — the only unmeasured input to the Stage 3
   bracket. Same 73-edge shape as flash-attn, which took 2h12m with the
   guards in place, so it should fit. **Check whether BSA has its own arch
   hook before assuming it does not** — that assumption cost 50 minutes on
   2026-09-02.
2. **32K dense baseline** — answerable now without any new build, since FA2
   is in the image. Open since the first validation run: `sdpa_math` OOM'd at
   32K requesting 69.66 GiB because it materialises the full O(seq^2) score
   matrix, which left the longest grid length without a dense reference.
   FA2 never materialises that matrix, so it should simply run. Quick, and
   it closes a hole in the study's longest cell.
3. **Corrected anchor re-run** on an idle machine — the warmup fix has not
   yet been exercised on hardware, so the warm GLA figure (~47.7 TFLOPS
   corrected, vs 12.018 measured cold) is still inferred rather than
   measured.
4. block_sparse anchor, re-estimate Stage 3, apply the two-sided rule in
   `configs/accuracy/stage3_grid.yaml`.

**BSA gets ONE attempt.** If the build fails again, do not retry and do not
debug it on billed hardware -- start Stage 2 instead (`docs/stage2_plan.md`).
Since FlexAttention's block-sparse path landed, BSA contributes 288 of 1404
cells, all at block_size=128 where flex already runs the same pattern, and
resume means those cells can be added later at the cost of only themselves.

Ordering note: 2 and 3 are measurements and belong on an IDLE machine, so
run them before starting the BSA build, exactly as this session ran the
batch probe first. The build can take the machine away from you.

### Stale text inside the v3 image description

`attnbench-l4-image-v3-20260903`'s description ends with "Keep v2 until GQA
passes." Both halves are now stale: the GQA gate passed on 2026-09-03 (and
was confirmed genuinely three-way, with `FlashAttention2.is_available()`
True), and v2 was deleted the same day. Machine image descriptions cannot be
edited after creation, so this note is the correction. The description's
"NOT VERIFIED: GQA" is pessimistic, not wrong -- the capture simply happened
before the gate ran.
