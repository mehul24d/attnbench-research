# Hardware constraints

See `README.md` for the stage table and measurement-discipline rules. This
file holds the constraints that live in decisions rather than in the code, so
they don't get silently re-inferred wrong. Both entries below exist because
an otherwise-sound inference from the surrounding code produced the wrong
answer at least once.

## Hardware ceiling

**Max 24GB VRAM** (L4- or 4090-class) for any GPU work in this study, **80GB
if the pending H100 quota request lands** (asia-south1, `a3-highgpu-1g` via
Spot/DWS Flex Start — see the GCP quota diagnostics from this project's setup
session).

Any model chosen for Stage 3 (accuracy track) must fit that ceiling **with
activations at the grid's longest `seq_len` (32K, per
`configs/accuracy/stage3_grid.yaml`)**, not just fit in isolation. Don't infer
a model size from unrelated code (e.g. `AttnConfig`'s head geometry) — check
this ceiling explicitly. A large model (e.g. a 400B-parameter one) does not
fit regardless of how plausible it looks from the surrounding config; this
has already happened once during planning and was caught before it shipped.

## Compute capability

GPU work in this study targets **compute capability 8.0+ (Ampere or newer)**.
Do not assume T4 (compute capability 7.5) is a valid target — it has come up
incorrectly before. Check `backends/base.py`'s `Capability.min_compute_capability`
per backend rather than assuming a specific card.
