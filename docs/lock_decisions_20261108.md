# Lock decisions: the 13 items, each with a default (8 Nov 2026)

Adviser's ruling E of 8 Nov 2026 (fourth paste), item 8. These are the T4.1
values whose status in `docs/lock_package_draft.json` is "owed: decision",
plus the two settled earlier the same day. **User** marks the items that
need the researcher, either because they set spend or scientific direction.
Every other item takes its default unless the researcher says otherwise.

| # | Item (T4.1) | Default | Needs |
|---|---|---|---|
| 1 | H0 to H3 tolerances | Set after the floors; none below the 6.77% noise floor | **User**: scientific |
| 2 | Ground-truth density set | P1's set (dense 1.0; block-sparse 0.25, 0.5; XAttention calibrated); oracle 0.5 at 32768 not measured | **User**: scientific, and spend if the missing cell is to be measured |
| 3 | H4 equivalence bound | 10 points (T1.1: −4.3 and −6.0) | **User**: scientific |
| 4 | H4 confirmatory family | m counted from the declared family (audit D4) | **User**: scientific |
| 5 | Floors quantile | 0.95 for both floors | **User**: scientific (with n = 2 sessions it sits near the largest gap) |
| 6 | HDBI boundedness threshold | Fitted on P1, rows inside their cell's floor dropped; two vs three classes (audit D3) | **User**: scientific |
| 7 | K2 correctness tolerance | The gate's `TOL` | default |
| 8 | Payload class edges | 8 B, 4 KiB, 1 MiB | default |
| 9 | Realised-length key rule | Key by the budget; realised tokens kept as a field | default |
| 10 | Budget cut order | As the plan | default |
| 11 | Recompile gate | 0 | default |
| 12 | Block drift bound | 6.77%, a flag, not an exclusion; flagged blocks get a sensitivity run | settled (adviser, 8 Nov 2026) |
| 13 | Resolution floor's form | Per (length, batch), by the declared slope criterion (0.354) | settled (rule `2b08e0d`) |

The learned-predictor hyperparameters left this list. They are now owed
data: a provisional winner from cell-grouped CV exists, and it is to be
rerun on features from the revised identity.
