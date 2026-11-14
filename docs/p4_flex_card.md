# P4 on Flex-start at `lock-2026-11-14-r2` (14 Nov 2026)

**Not launched.** P4's Flex-start request and launch need one yes, asked together. They are asked
only after the pre-launch CPU pass prints PASSED at the commit the tag names
(`scripts/prelaunch_cpu_pass.sh p4` on P2's banked training file, with T3.2d's step 5 XAttention
masks staged), and after P2's banked entry is in the pairing ledger: a real P4 run checks P2's
training file against that entry. The step list and minutes are in
`docs/launch_cards/t64_p4_h100.md`.

**Since `lock-2026-11-13-r4`** (Appendix A, 14 Nov 2026, the adviser's rulings dated 14 Nov 2026):

- P2 ran and is banked: 35 rows, sha256 `96c6670c…`, built at `8542cf0` with `--session`.
- H6 is not scored, so P4's committed file carries no dispatch section. The dispatcher is reported
  as exploratory only.
- P4 boots P2's image, and its preflight refuses libraries other than P2's
  (`configs/t64/p2_libraries.json`); FA3 alone may differ.
- Audit C was revisited (`docs/gpu_path_audit.md`). The one item listed for acceptance is that v6
  has never booted an `a3-highgpu-1g`.

## What is asked

| | |
|---|---|
| Machine | one `a3-highgpu-1g` (H100 80GB), `us-central1-a` |
| Provisioning | Flex-start (`gcp_launch_h100.sh`'s default); no Spot (ruled out for P4, 13 Nov 2026; the scratchpad's `gcloud` wrapper refuses Spot) |
| Image | `attnbench-env-v6-20260917`, P2's, the only one the launcher boots |
| Code | a fresh clone at the tag `lock-2026-11-14-r2`, deployed by `gcp_deploy_source.sh` |
| Bootstrap | `FA3_WHEEL_SHA256=1676ef0b86c377ae911a7de238c2a1f6b58a7666f1a6eef919d4dc7b0973b64a gcp_bootstrap_instance.sh "$NAME" "$ZONE" --suite-record`. **Plan default, flagged:** the suite record runs, as ruling 1h of 10 Nov 2026 and the card's cap have it; the H100 has none. P2 ran without one, by the researcher's ruling of 12 Nov 2026 for P2 |
| Staged by scp | the P1 training file (`de7f0f44…`) and P2's (`96c6670c…`) under `results/`; P2's Triton masks (P2's `06_mask_gate/masks/triton`, read from the bucket for this, under the launch yes) as `results/<session>/p2_masks/` |
| Start | `nohup bash scripts/session_exit.sh --log ~/p4.log -- python3 scripts/t64_session.py run --session results/t64_p4_<UTC> --fa3-sha256 1676ef0b…`; no `--banked-at`, which a real run refuses (it reads P1's and P2's banking times from the ledger) |
| Cap | ₹3,237 (rate ₹462.42/h, `us-central1`, all-in; `GCP_RATE_INR_HR=462.42`) |
| In-guest halt | 419 min (`GCP_HALT_MINUTES=419`) |
| Hard delete | 429 min (`GCP_MAX_RUN=429m`) |
| Watcher | `t61_rehearsal_watch.sh` in the background, looping until the session exits; no blocking question while it bills |

## New in preflight

P4's preflight is P2's: the block-sparse kernel's contract, then the smoke check over all five
arms at 2048, K2's screen with FA3 among the candidates, and the XAttention arms at 32768
(`docs/p2_flex_card.md` describes each). One check is added before the GPU variants. The
instance's libraries, read as every cell manifest records them, must be P2's:

- torch 2.9.1+cu129;
- transformers 4.46.0;
- flash_attn 2.8.3.post1;
- block_sparse_attn 0.0.2, with the row-mask fix `row-mask-tail-1` and the interface `0d28e66dc351`;
- triton 3.5.1;
- xattn `e3798877…`.

A library missing, added or at another version stops the session in preflight. FA3 is the only
library allowed to differ. `fa3_import` checks it before the preflight: the hash, the tags
(cp310, v6's Python), the install and the import.

## Headroom (`scripts/budget_interlock.py`)

Spent ₹6,182. P2 ran, so nothing is reserved for it. P4 on Spot is ruled out.

| P2 | P4 | Spent + caps ₹ | Headroom ₹ |
|---|---|---|---|
| ran (₹823, in the spent figure) | Flex-start ₹3,237 | 9,419 | 4,581 |

The cap is 1.3 × 5.38 planned hours at ₹462.42/h, the suite record's 20 minutes included. It
counts the ground truth's worst case of 5 reruns at every cell.

## What it does not do

- It reads no P2 prediction error, and scores nothing. H0 to H2 are scored only after P4's
  prediction path is frozen and tagged; this tag is that freeze.
- It writes no dispatch section, since H6 is not scored.
