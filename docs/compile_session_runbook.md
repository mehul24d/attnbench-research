# Compile session runbook

One instance, 6-hour hard cap, whose only job is to build `flash-attn` and
`Block-Sparse-Attention` from the v2 machine image and capture a v3 image with
both — so no session after this one ever compiles them again.

Two measurements ride along, because the instance is rented anyway and they
cost minutes against a 6-hour cap. They are taken **before** the builds start,
not underneath them — see Phase 2 for why measuring under `nvcc` contention
would produce a number that is worse than useless:

- a second timing anchor at `seq_len=16384`, to check whether the Stage 3
  hour estimate extrapolated from the single 8192 point holds
- a peak-memory check at batch=2 at 16384, to confirm or overturn the
  analytical "batch=1 only at long context" finding

## Rule: agreement is not verification

When a reviewer confirms a parameter, that is a second opinion, not a second
check. Two people can share the same wrong model, and consensus makes a wrong
number feel settled precisely when it most needs testing.

On 2026-09-02, `MAX_JOBS=4` was proposed and independently endorsed. Both
parties missed the same thing -- that `MAX_JOBS` and `NVCC_THREADS`
multiply -- so the endorsement added confidence without adding information,
and the agreed value would have OOM'd exactly as the original did. Nobody had
done the arithmetic; agreeing felt like someone had.

Route resource parameters to a check that can fail (see the pre-flight memory
guard in `scripts/build_flash_attn.sh`), not to a second reader. Approval is
for decisions about intent -- what to build, what to spend. It cannot
validate an arithmetic claim.

## Rule: save the serial log BEFORE deleting an instance

The serial console is the only diagnostic channel that survives losing SSH,
and it is destroyed with the instance. `get-serial-port-output` on a deleted
instance returns "resource not found"; GCE's buffer also rolls, so even a
live instance may have lost its earliest entries.

On 2026-09-02 the serial log was read repeatedly but only ever through
`grep`/`tail` in transient commands -- never saved. After teardown, the
question "was sshd among the OOM kills, which would explain why SSH never
recovered?" became permanently unanswerable, and it is the one question
blocking a confident relaunch.

Before `instances delete`, always:

    gcloud compute instances get-serial-port-output NAME --zone=ZONE \
      > results/<session>/serial_console.log

It costs one API call and nothing in storage. Do it as the first step of
teardown, before copying results off, since an instance that dies unexpectedly
takes the explanation with it.

## Diagnostic rule: a ruled-out cause must be re-verified before it is relied on

A negative finding is true **as of when it was measured**, not thereafter.
Cumulative resource problems -- memory, disk, file descriptors, log growth --
are precisely the ones that are absent early and fatal later, so an early
"ruled that out" is the least trustworthy kind of conclusion to carry
forward.

What happened on 2026-09-02: SSH became unreachable, and the serial console
was checked for OOM messages. It was clean, so the diagnosis was recorded as
"CPU starvation, not memory". That was **correct at 16:51**. It was then
treated as settled and reported twice more as established fact while memory
pressure kept building; the OOM killer fired at 18:17:56, having killed 8
`cicc` processes by the time anyone looked again. Ninety minutes of billed
time were spent reasoning from a stale negative.

The check cost one API call and could have been repeated at every status
poll. Re-verify the ruled-out cause each time you report on it, especially
when the symptom persists and the explanation predicted it would resolve.

## Interpretation rule: a library's magic constant is probably load-bearing

When a dependency's heuristic produces a number that seems too conservative,
the default assumption should be that it encodes a measurement someone took,
not that it is arbitrary padding. Override it only after finding out what it
represents.

flash-attn's setup.py computes `MAX_JOBS = min(cpu_count//2, free_gb/9)`.
The `/9` reads like timidity. It is not: nvcc's frontend (`cicc`) uses
~5 GB resident per instance, and with the default `NVCC_THREADS=2` spawning
one frontend per gencode target, ~9-10 GB per job is very nearly exact. The
constant was a measurement. Overriding it to 5 demanded ~50 GB on a 31 GB
box and cost the session its build.

The same reasoning error produced the follow-up mistake, which was
independently endorsed and still wrong: `MAX_JOBS=4` with `NVCC_THREADS`
left at 2 is 8 concurrent frontends, ~40 GB, and would have OOM'd just the
same. The missing insight both times was that the two settings **multiply** --
`concurrent cicc ~= MAX_JOBS x NVCC_THREADS` -- so raising job count while
leaving thread count alone is not a partial fix, it is the same bug. Being
agreed with is not verification; the arithmetic was never checked by either
party.

`scripts/build_flash_attn.sh` now fails closed on this: it computes
`jobs x threads x 5 GB` against real free memory and refuses to start rather
than letting the OOM killer discover the ceiling 90 minutes in.

## Procedural rule: kill and start are ALWAYS separate SSH invocations

Not a habit to remember -- a rule, because the alternative failed four times
in one session.

**Never issue a kill and a reference to the same target in one remote
command.**

```
# WRONG -- kills this SSH session, exit 255, no error message
gcloud compute ssh NAME --command="pkill -f 'pip install'; ..."
gcloud compute ssh NAME --command="pkill -f build_fa; chmod +x ~/build_fa.sh; ..."

# ALSO WRONG -- the [b]racket trick does not save you here, because the
# same command line contains the literal 'build_fa.sh' further along
gcloud compute ssh NAME --command="pkill -f '[b]uild_fa'; nohup ~/build_fa.sh &"

# RIGHT -- two invocations, the killing one naming nothing it starts
gcloud compute ssh NAME --command="P=\$(pgrep -f '[n]vcc'); [ -n \"\$P\" ] && kill -9 \$P; ps -eo cmd | grep -c '[n]vcc'"
gcloud compute ssh NAME --command="nohup ~/build_fa.sh > ~/log 2>&1 &"
```

`pkill -f` matches against full command lines, and the SSH wrapper's own
command line contains everything you typed. The `[x]yz` regex trick only
protects the pattern from matching *itself* -- it does nothing about the
target's name appearing elsewhere in the same command, which is exactly what
happens when you kill and restart together.

The fix is separation, not a cleverer regex. Verify the kill worked with a
count in the killing invocation, then start in the next one.

## Spending rules (unchanged, in force for this session)

- Confirm machine type, zone, hourly rate, and estimated session cost before
  creating any billable resource.
- Never leave an instance running at the end of a turn. If waiting on
  something long, say so and stop.
- Report elapsed instance time and estimated spend at every report-back.
- The billing account is paid with no auto-suspend. A forgotten instance is
  the worst failure mode available.
- If any step fails twice, stop and report. Do not debug in a loop on billed
  hardware.
- On-demand, not spot. Preemption 70 minutes into a compile wastes far more
  than spot pricing saves.

## Pricing (Cloud Billing Catalog, asia-south1/Mumbai, on-demand)

| component | rate |
|---|---|
| G2 instance core × 8 | $0.026012 /core/h → $0.2081/h |
| G2 instance RAM × 32 GiB | $0.003047 /GiB/h → $0.0975/h |
| Nvidia L4 GPU × 1 | $0.582976/h |
| 200 GB pd-balanced boot disk | $0.120 /GiB/month → $0.0329/h |
| **total** | **$0.9215/h ≈ ₹81/h** |

6-hour cap = **$5.53 ≈ ₹487**. Expected actual is lower; the cap is the
worst case, not the plan.

**Budget context -- the binding constraint is time, not money.** ₹28,663 of
credit is available and **expires 2026-12-01**. Spend to date is ₹474 across
both 2026-09-02 sessions (₹241 validation + ₹233 compile), i.e. under 2% of
the credit. At that rate the credit cannot be exhausted before it expires.

So do **not** optimise for cost at the expense of sessions. A session that
fails and has to be repeated costs far more -- in elapsed calendar time
against a hard expiry -- than the instance-hours saved by cutting it short,
running a smaller machine, or skipping a measurement to save minutes. Where
the two conflict, buy the certainty: take the larger machine, run the extra
probe, keep the redundant image. The guardrails in this runbook exist to
prevent *forgotten* spend, which is waste, not to minimise *deliberate*
spend, which is the point of the credit.

Machine images bill separately from instances and are **not** covered by
`scripts/gcp_cleanup_check.sh`. The v2 image is 23.0 GB. A v3 image roughly
doubles that ongoing storage cost until v2 is deleted.

## Phase 0 — before launch (free)

1. `bash scripts/gcp_cleanup_check.sh` — must report clean.
2. `gcloud compute machine-images list` — v2 must be READY. The launch script
   fails closed if it isn't, rather than silently falling back to a bare DLVM
   image and spending the cap on a reinstall.
3. Confirm the estimate above with the billing-account owner. Stop here for
   approval.

## Phase 1 — launch

```
bash scripts/gcp_launch_compile_session.sh
```

`asia-south1-b` by default (asia-south1-c stocked out last session; both draw
on the same regional quota pool). Override with `GCP_ZONE=` if b is also out.
The script prints the full billable configuration and requires typing
`launch`.

Record the boot time. Every spend figure reported afterward is measured from
it, not from when the runbook started.

## Retry session plan (after the failed 2026-09-02 compile session)

Budget **3 hours**, not 6 -- the guards in `scripts/build_flash_attn.sh` turn
the three failures that consumed the last session into fast, loud errors.

**Order, and it is not the intuitive one:**

1. **Batch probe** (`scripts/probe_batch_scaling.py --seq-len 16384`) on the
   idle machine, first, before any build starts.
2. **Any available anchors** (`time_one_accuracy_example.py`) while still idle.
3. **flash-attn** via `scripts/build_flash_attn.sh`.
4. **Block-Sparse-Attention.**
5. **block_sparse anchor**, if time remains -- the one measurement that would
   narrow the Stage 3 estimate from its current 16-25h bracket.

The measurements come first for a reason that is now demonstrated rather than
theorised: **the build will lock you out.** Last session it saturated the box
badly enough that SSH stopped completing its banner exchange for the
remaining three hours, and every measurement still queued behind it was lost.
The batch-scaling numbers currently exist only in conversation -- they were
measured but never written to a parquet, because the probe was scheduled
after the build.

This is the same principle as the last session's "measure on an idle machine"
ordering, applied harder: not just *before the build for accuracy*, but
*before the build because access itself is at risk*. Anything that must be
recorded gets recorded while the machine is still reachable.

**Before launching, add sshd resilience.** Whether sshd was among the OOM
kills is unresolved (the serial log was destroyed with the instance), but the
cheap mitigation applies either way:

    sudo systemctl edit ssh    # [Service] Restart=always / RestartSec=5

## Phase 2 — measurements first, on an idle machine

Take both measurements **before** starting any build, and do not start a build
until they are recorded.

The instinct is the opposite — the builds are the long pole, so start them
first and measure underneath. That is wrong here. `flash-attn` and
`Block-Sparse-Attention` both saturate all 8 vCPUs with `nvcc`/`ptxas`, and a
throughput measurement taken under that contention reads low: the Python
driver loop, kernel launch overhead, tokenization, and host-side data movement
all compete for cores even though the attention work itself is on the GPU.

That matters because of what the anchor is *for*. It decides whether reserve
cut C fires. A number depressed by CPU contention looks exactly like "the
extrapolation was optimistic", which would fire C on an artifact and
permanently underpower the 32K point for nothing. **A contaminated anchor is
worse than no anchor**, because it would be acted on.

The measurements take minutes. The builds have hours. Order them accordingly.

### 2a — the 16384 timing anchor

```
cd ~/attnbench_scaffold
python scripts/time_one_accuracy_example.py --seq-len 16384 --dense-backend sdpa_flash
```

`sdpa_math` is not viable here: it materializes the full O(seq²) score matrix
and OOM'd at 32K last session (69.66 GiB requested). Use `sdpa_flash`.

Compare the reported `measured` and `scoring` effective TFLOPS against the
8192 anchor (5.185 scoring / 6.647 measured). The extrapolation predicts these
hold roughly constant; the expectation on record is that they come in
**worse**, since attention's share of compute grows quadratically with length.

The decision rule this feeds is two-sided and already recorded in
`configs/accuracy/stage3_grid.yaml`:

- **anchor worse than extrapolated** → fire the reserve cut, `32768: 100 → 50`
  (~6.7h saved, 34.1h → ~27.4h)
- **anchor better, by enough to open ~5h** → restore `16384: 200 → 300`
  (costs 5.05h, lands at 39.13h — so a smaller improvement is banked, not
  spent)

Note the grid's estimate is FLOPs-derived and omits per-example overhead
entirely (tokenization, mask construction, score-cache I/O). At 300 examples
per cell that omission is the largest single source of error in the estimate.
If the probe reports wall-clock per example, capture it — it is the only
direct evidence available about that gap.

### 2b — batch=2 peak memory at 16384

The analytical model says batch=2 does not fit at 16384 (GLA ~26.8 GiB against
~22 GB usable). It is extrapolated from one real data point via a rough
token-ratio proxy and has never been measured. Confirm or overturn it.

Run the same probe at batch=2 and record `torch.cuda.max_memory_allocated()`.
An OOM here is a **successful measurement**, not a failure — it confirms the
model. Record it and move on; do not retry with tuning.

This does not change the committed grid either way. Even if batch=2 fits,
`compute_importance_scores` is hard-blocked at batch=1 by an explicit
`ValueError` in `_scoring_forward_chunked`, and `state.scores` has no batch
dimension at all — so scoring and block_sparse, which are the majority of the
grid's cost, cannot batch regardless of available memory. The check is worth
running because it is nearly free and it closes out an open question, not
because a favourable result unlocks anything.

Record both results before proceeding. Once Phase 3 starts, this machine
cannot produce a trustworthy timing number again for the rest of the session.

## Phase 3 — start the builds

```
cd ~/attnbench_scaffold
nohup pip install --user --no-build-isolation flash-attn > ~/flash_attn_build.log 2>&1 &
```

`--no-build-isolation` is required: pip's isolated build environment cannot
see the already-installed torch, and without it the build dies immediately in
`get_requires_for_build_wheel` with `ModuleNotFoundError: No module named
'torch'`. This was hit and diagnosed last session.

Do **not** bother setting `TORCH_CUDA_ARCH_LIST=8.9`. Both packages compute
their own gencode list internally from the CUDA toolkit version and ignore it
— verified last session by observing `ptxas -arch sm_90` running during a
supposedly sm_89-scoped build. Budget for the unscoped multi-architecture
build; that is what the 6-hour cap is sized for.

Watch progress with `tail -f`, never by re-running the install. When killing a
build, do not use `pkill -f 'pip install'` — that pattern matches the invoking
SSH command's own command line and kills the session (exit 255, no error
message). Match on a narrower pattern and confirm with `ps aux`.

Build Block-Sparse-Attention at its pinned commit the same way once flash-attn
finishes, or in parallel if the box has headroom.

## Phase 3a — priority order once flash-attn lands

If flash-attn finishes with time in hand, do these **in this order**, not in
the order they appear elsewhere in this runbook:

1. **Capture v3.** The image is the only artifact that cannot be recreated
   cheaply -- everything else is a rerunnable measurement. Capture as soon as
   flash-attn is installed and imports, before starting anything else.
2. **Batch probe** (`scripts/probe_batch_scaling.py`) with whatever backends
   are installed.
3. **Block-Sparse-Attention build**, with whatever time remains.

flash-attn alone is a real outcome, not a partial one: it is the most
important missing backend, it unblocks the full three-way GQA gate, and it
may solve the 32K dense-baseline problem that no SDPA variant could
(`sdpa_math` OOMs on its O(seq^2) score matrix, `sdpa_efficient` has no
kernel for the 12:2 GQA shape).

**Capture hazard -- the shutdown timer resets on reboot.** The hard cap is a
`shutdown -h +N` scheduled by the startup script, which runs on *every* boot.
Stopping the instance to capture and then starting it again re-arms the cap
for N minutes from the *new* boot, silently extending the session past the
intended deadline. Two ways to handle it:

- Capture from the **running** instance (`sync` first). GCP supports this and
  it avoids the stop/start entirely. Preferred when what changed is an
  installed package tree whose writes have completed.
- Or stop/capture/start, and then **immediately re-arm a corrected
  shutdown** on the restarted instance so the original wall-clock deadline
  still holds:
  `sudo shutdown -h HH:MM` at the original cap time.

Never leave a restarted instance running on a freshly re-armed 6-hour timer.

## Phase 3b — hard cutoff at cap minus 90 minutes

**Decided in advance, on purpose.** At 90 minutes before the hard cap, stop
whatever is building and move to capture regardless of build state.

For the 2026-09-02 session: cap fires 21:50:33Z, so the cutoff is
**20:20:33Z**.

If Block-Sparse-Attention (or flash-attn) is still compiling at the cutoff:

1. Kill the build. Match on a narrow pattern and confirm with `ps aux` --
   never `pkill -f 'pip install'`, which matches the invoking SSH command's
   own command line and kills the session.
2. Capture v3 with whatever is installed.
3. Spend the remaining time on the batch probe and, if block_sparse built,
   the block_sparse anchor.

The reasoning: an image plus real measurements beats a complete build that
cannot be captured because the cap fired mid-compile. A build killed at the
cutoff costs one session's compile time; a cap that fires during
`machine-images create` costs the entire session, because an instance that
self-terminates mid-capture leaves nothing behind. Capture is not
interruptible and must not be started near the cap.

This decision is made here rather than at hour five specifically so it is not
made under time pressure, when "it's nearly done, give it ten more minutes"
is most persuasive and most expensive.

**If block_sparse does not build this session**, run the batch probe anyway
with the backends that are available. Getting sdpa_flash and GLA into
`results/batch_scaling/batch_scaling.parquet` with proper provenance is worth
the window on its own -- the block_sparse batch=1 row can be appended by a
later session, since `append_checkpoint` appends rather than overwrites.

## Capturing an image you could not verify

If SSH is unreachable at capture time (it starved during this session's
build -- see the MAX_JOBS note in `scripts/build_flash_attn.sh`), **capture
anyway** via the API, which needs no shell. But then:

1. Put **`unverified`** in the machine image `--description`, naming what was
   not confirmed (e.g. "flash-attn built but import never verified -- SSH
   unreachable at capture").
2. **Do not delete the previous image.** It stays until a future session
   boots the new one and confirms the thing it exists to provide actually
   imports:
   ```
   python3 -c 'import flash_attn; print(flash_attn.__version__)'
   ```
3. Only after that check passes does the old image become deletable, and the
   new one's description should be corrected to drop `unverified`.

An image that silently lacks the backend it was built to carry costs a whole
session to discover -- and it would be discovered at the *start* of the next
session, after paying to launch. The ~Rs 100/month to keep a known-good
predecessor is not a close call against that.

## Phase 4 — capture the v3 image

Capture as soon as both builds finish, **before** the cap and before any
further experimentation. An image captured with both kernels compiled is the
entire deliverable of this session; measurements are secondary and
reproducible, a lost 5-hour compile is not.

```
gcloud compute instances stop  NAME --zone=ZONE
gcloud compute machine-images create attnbench-l4-compile-image-v3-YYYYMMDD \
  --source-instance=NAME --source-instance-zone=ZONE
```

If only one of the two builds completes, capture anyway. A v3 with flash-attn
alone still retires half the problem permanently and costs nothing extra to
take.

## Phase 5 — teardown

1. **Save the serial console log first of all** — it is destroyed with the
   instance and is the only diagnostic channel that survives losing SSH:
   ```
   gcloud compute instances get-serial-port-output NAME --zone=ZONE \
     > results/<session>/serial_console.log
   ```
   Do this even when the session went fine and especially when it did not.
   One API call; on 2026-09-02 skipping it made the cause of a three-hour
   SSH outage permanently unknowable.
2. Copy any results off the instance — an instance is deleted, its disk goes
   with it.
3. `gcloud compute instances delete NAME --zone=ZONE` — delete, not stop. A
   stopped instance still bills for its disk.
4. `bash scripts/gcp_cleanup_check.sh` — must report clean.
5. `gcloud compute machine-images list` — confirm the new image exists and
   decide whether to delete the previous one. Keep the predecessor until the
   new image is verified to boot and import both kernels; deleting it early
   trades ~₹100/month for the risk of a broken image.
6. Report elapsed instance time and estimated spend.

Billing data lags by hours to ~24h, so the console's Reports page showing
₹0.00 immediately after teardown is expected and is not evidence of anything.
The cleanup check is the authoritative answer to "is anything still billing".

## Estimate from the cumulative average, not from the last window

When projecting how long a long build has left, divide total work done by
total elapsed. Do NOT extrapolate from the gap between the last two
readings.

On 2026-09-03 the flash-attn build's completion estimate swung 07:24 ->
07:13 -> 06:59 -> 07:15 -> 07:19 across five reports of the same healthy
build. Nothing was wrong: object files in that tree cost very different
amounts (the `flash_bwd_hdim128` kernels are several times heavier than the
forward ones), and ninja completes them in bursts of MAX_JOBS. So a
short-window rate measures which kernels happen to be in flight, not
progress. The cumulative average was stable to within a few minutes the
whole time and was right.

This matters beyond tidiness: the estimate drove real decisions -- whether
the GQA gate would fit, whether to capture a partial image, whether to
abandon the build. An estimator that swings 25 minutes in either direction
makes those calls arbitrary, and each swing spends credibility with whoever
is reading the reports.

Corollary: say which estimator produced a number. "07:16 by cumulative
average" invites a correction that "about 07:16" does not.
