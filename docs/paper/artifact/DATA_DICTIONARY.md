# Data dictionary: the attnbench paper artifact (prepared 28 Nov 2026, not published)

This artifact holds the banked data behind `docs/paper/`. `scripts/build_artifact.py` builds it, and `scripts/artifact_scan.py` must report it clean before anyone receives it.

Every time is in seconds unless a field name says otherwise. Every file sits under `data/` at its banked relative path, so the repository's scripts read it unchanged. `regenerate.sh` runs them and compares the outputs with the committed tables.

## Top level

| File | What |
|---|---|
| `MANIFEST.tsv` | One row per data file: path, bytes, sha256, role |
| `SHA256SUMS` | The same hashes, in `shasum -c` form |
| `regenerate.sh` | Regenerates and compares the tables. Run it from a clone of the repository |
| `SCAN.md` | The identifier scan's report for this copy |

## Data

| Path under `data/` | Role | Format |
|---|---|---|
| `scoring_20261116/run/inputs.json` | The scoring wave's inputs, anonymised | JSON. `bucket` is a placeholder (`gs://attnbench-store`). Local paths read `<BANKED>`, which `paper_reproduce.py` rewrites to `data/`. Per pairing it holds `remote` (the prediction's path in the store), `sessions` and `ground_truth_dir` |
| `scoring_20261116/mirror/` | What the scorer read from the store on 16 Nov 2026 | `index.json` maps each remote path (store name replaced) to `{exists, sha256, created, ...}`. `objects/<store>/<path>` holds the bytes, which are unchanged |
| `scoring_20261116/bucket/.../t64_p4_*_b3`, `..._b4` | P4's two sessions as scored | `manifest.json`, `step_layout.json` and the `13_ground_truth/` JSON files: what the loaders read |
| `bucket_20261117/.../t61_p2_20261114T0022Z/` | P2's session as scored | `manifest.json`, `step_layout.json`, the `12_ground_truth/` JSON files and `07_host_components/` (see below) |
| `bucket_tierA_20261121/.../traces/` | P2's ground-truth traces | Kineto (Chrome trace) JSON, one file per traced repetition: `trace_L<budget>_b<batch>_<rep>_<arm>[_d<density>].json` |
| `p4_traces_20261116/` | P4's ground-truth traces | As P2's |
| `.../t64_p4_20261116T113724Z_b3/08_host_components/<cell>/` | P4's committed host prices | See "Host components" below |
| `.../t64_p4_20261116T113724Z_b3/10_dense_select/` | P4's dense-kernel microbenchmark | `dense_choice_H100_<budget>_<batch>.json` |
| `aws_p3_g6_20261126T0906Z/session/05_host_components/<cell>/` | P3's host prices | See "Host components" below |
| `aws_p3_g6_20261126T0906Z/session/09_ground_truth/` | P3's untraced ground truth | `eager/gt_L<budget>_b<batch>.json`: per arm and density, the rounds' prefill seconds. `stamps.json` holds the code stamps |
| `aws_p3_g6_20261126T0906Z/session/12_delay_injection/` | P3's delay injection: complete, **not analysed** | `plan.json` (64 variants, hashed and stored before timing). `delay_injection.json` holds `samples_s` (10 per variant) and the per-mechanism slopes |
| `p1_training/`, `p2_training/` | The training rows the fitted sum and the learned predictor read (the inputs name P1 and P2 only) | gzip JSON (`attnbench.training` schema 2) |
| `s4/t31_rehearsal_20261101T0723Z/05_host_components/dense_d1_L2048_b1/` | Shakedown 4 (L4, 1 Nov 2026), one cell | `host_profile.json` (null pass), `recording/sequence.json`, `recording/trace.json` (Kineto) |

## Host components (one directory per cell, `<arm>_d<density>_L<budget>_b<batch>`)

- **`host_profile.json`.** The committed host price (`pricing: bare_timer`):
  - `host_s` maps each position in the operation sequence to its seconds.
  - `residual_s` is each position's share of the per-layer idle-gap residual.
  - `price_route` says how each position was priced (`bare_leaf`, `bare_chain`, `isolated_replay`, ...).
  - `launch_floor_s` is the bare-timed null kernel.
- **`host_profile_null_pass.json`.** The null-kernel pass (`pricing: null_pass`). `host_s` uses the same positions, and `reps_s` holds the repetitions.
- **`bare_prices.json`.** The bare timer's calls (`calls`), each with:
  - `position` and `positions`: where it starts and how many host positions it prices;
  - `operator` (null for a Python entry point) and `route`;
  - `bare_s` and `instrumented_s` (the null pass's own time for the call);
  - `launch_floor_s` and the chain gate.

## Not in the artifact

- GPU binaries, model weights and prompts. The prompts are RULER samples, pinned by hash in the repository.
- Any file that names a cloud project, account, store or instance credential.
- P3's 0.9 GB of profiled traces, which this wave's scripts do not read.

## Redactions

- The Shakedown 4 recording trace had a home directory in it. The copy reads `/home/user` (marked in `MANIFEST.tsv`). The scripts read only its layer annotations.
