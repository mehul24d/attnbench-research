"""The time model and its baselines (plan Section 5.4).

`recursion` is the critical-path recursion (T2.8). Everything else reads the
components it prices (`recursion.price`), so every predictor uses identical
numbers: `ablations`, `baselines` and the `hdbi` diagnostic (T2.9),
`training` (the committed and retrospective training sets of T2.9 and
T2.10), `learned` with its analytic `intensity` features (T2.10), and
`naive` (the constant-host-cost and nearest-pairing baselines, T2.19).

`commit` writes, syncs and reads the hashed prediction file, and refuses a
late or altered one (T2.11, K9). `intervals`, `crossover` and `scoring` give
the bootstrap prediction intervals, the crossover bracket rule with the
quadratic fit and crossover error, and the decision error (T2.19).
`sensitivity` and `residuals` read a committed file for the queue
sensitivity and the residual analysis (T2.15). `delay_injection` plans and
scores Gate A's delay-injection slope check (T2.25).
"""
