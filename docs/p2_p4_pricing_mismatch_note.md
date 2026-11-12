# Pricing-mismatch share on P2 and P4 (13 Nov 2026)

The adviser's rulings dated 13 Nov 2026, item 1, option (a), written before any P2 data:

> If an arm's step 8 pricing-mismatch share on P2 or P4 exceeds 0.05, the arm is labelled 'pricing not validated', reported descriptively, and excluded from every hypothesis claim; the session continues.

**Why:** the share of XAttention's Triton path on the A100 has not been measured (`docs/p2_triton_priced_check.md`). Its Triton kernels are priced: 1.98% of the 2048 cell's device time. Appendix A cites this note after P2.

**The code it relies on:**
- `replay.pricing_gate` labels the arm at the lock value `pricing_mismatch_share` (0.05). It doesn't stop the session.
- `predict.commit` writes no prediction for a labelled arm and records it under `not_priced`, so no hypothesis test has rows for that arm.
