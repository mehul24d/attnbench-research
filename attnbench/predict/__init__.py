"""The time model and its baselines (plan Section 5.4).

`recursion` is the critical-path recursion (T2.8). Everything else reads the
components it prices (`recursion.price`), so every predictor uses identical
numbers: `ablations`, `baselines` and the `hdbi` diagnostic (T2.9),
`training` (the committed and retrospective training sets of T2.9 and
T2.10), and `learned` with its analytic `intensity` features (T2.10).
"""
