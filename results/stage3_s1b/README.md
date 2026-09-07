# Stage 3, Segment 1b — bands 4096 and 8192

**7200 rows, commit `1126bd2`, clean tree, `clocks_locked=True`, n=300/cell.**
Two backends: `sdpa_flash` (dense reference) and `block_sparse` at
sparsity 0.5 / 0.75 / 0.9. No linear arm — see `gla_arm_verdict.json`.

## Why this is a separate directory from `results/stage3_s1/`

Band 2048 is banked at `d27c650`. The code moved between segments (the GLA
gate fix and the `gate_source` column), so `check_code_continuity` refuses to
append here — correctly. `--allow-mixed-commits` would have worked and would
have been true, but it asks a reader to trust a diff. A separate directory
keeps each file single-commit and makes the join an explicit, dated step with
both commits on their own rows.

Band 2048's rows carry `gate_source = null` because the column did not exist
when they were written. Null means "not recorded", which is the truth.

## Files

- `accuracy.parquet` — both bands, 7200 rows
- `band4096.parquet` — the 4096 band alone, copied off at its boundary
- `gla_arm_verdict.json` — the pre-registered decision, with the gate, the
  control's score, and the commit that produced it
- `bands.log` — the full run log, including `ALL BANDS DONE 09:50:11Z`
- `probe/` — Stage 0/1 gates for this host

## Read the numbers with these two qualifiers

1. The importance ranking is an **oracle** derived from full attention
   scores. It is not merely a ceiling: `block_sparse` at 0.75 scores *above*
   dense on `vt` in all three bands. See `docs/claims.md`, "The oracle can put
   sparse ABOVE dense".
2. **Dense degrades with length** (97.0 → 95.0 → 86.0 on `niah_multikey`), so
   report absolutes beside deltas — some apparent sparse degradation at 8192
   is the baseline falling.
