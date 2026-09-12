"""Empirical timing probe: replaces the FLOPs-based "assumed 15 TFLOPS
effective throughput" estimate (see scripts/run_accuracy.py's docstring)
with a measurement from one real example at the grid's longest configured
length, before committing a rented session to the full Stage 3 grid.

The extrapolation's *structure* doesn't change from the planning-time
estimate -- total wall time is still (total FLOPs) / (effective TFLOPS) --
only the TFLOPS figure does: measured per phase here, instead of guessed.

**A first version of this module got the FLOPs numerator wrong** in a way
that only a real GPU measurement caught: it used `AttnConfig.useful_flops()`
alone, which is exactly one transformer layer's attention core (QK^T + PV).
But the wall-clock actually measured (`accuracy/model.py`'s
`compute_importance_scores`/`run_measured`) is a full `n_layers`-deep
model's forward pass -- every layer's attention core AND its surrounding
Q/K/V/O projections AND its MLP block, plus one lm_head call at the end.
Dividing that whole-model wall-clock by one-layer-attention-only FLOPs
produces a ratio with no physical meaning (confirmed: it read as
"0.05 TFLOPS", ~300x below a real GPU's actual throughput, purely because
the numerator was undercounting by roughly `n_layers`-plus-MLP-share, not
because the GPU is that slow). `whole_model_flops` below is the fix: it
folds in the real model's architecture (fetched from the target model's own
config, e.g. `AutoConfig.from_pretrained`) so the FLOPs figure matches what
the timer actually measured.

Two phase categories, because they have different arithmetic intensity and
must not share one throughput figure:

- "scoring": the dense causal softmax pass (accuracy/model.py's
  `compute_importance_scores`). Runs once per (task, example, seq_len),
  shared/amortized across every block_sparse sparsity level at that cell --
  never once per sparsity level.
- "measured": the actual backend-under-test forward pass -- one per
  (backend, sparsity-if-any, seq_len). A block_sparse measured pass's FLOPs
  figure should be `AttnConfig.useful_flops()` (discounted by its declared
  sparsity), not `issued_flops()`, since that's the work a real kernel that
  actually skips blocks executes; for every other backend the two are equal
  (sparsity is None), so `useful_flops()` is always the right figure to
  pass here, uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import AccuracyGrid
from .ruler import RulerExample
from ..config import AttnConfig

PhaseCategory = str  # "scoring" | "measured", not a closed enum -- see module docstring


@dataclass(frozen=True)
class ModelArchitecture:
    """The real target model's geometry, needed to estimate whole-forward-
    pass FLOPs -- fetched from the model's own config (e.g.
    `AutoConfig.from_pretrained(grid.model_primary)`), never assumed or
    left at `AttnConfig`'s placeholder head counts. `grid_configs.
    build_configs_by_backend` uses `n_heads_q=1, n_heads_kv=1` purely as a
    config-identity placeholder for cell-counting purposes (mask/sparsity/
    seq_len are what distinguish cells, not head count) -- those
    placeholders were never meant to carry real FLOPs weight, and using
    them for that is exactly the kind of silent, plausible-but-wrong
    inference docs/hardware_constraints.md's hardware-ceiling note already exists to prevent
    for model size; this is the same mistake, one level down.
    """

    n_layers: int
    hidden_size: int
    intermediate_size: int
    n_heads_q: int
    n_heads_kv: int
    head_dim: int
    vocab_size: int


def whole_model_flops(cfg: AttnConfig, arch: ModelArchitecture, *,
                       logits_to_keep: int = 1) -> int:
    """Total FLOPs for one real forward pass at cfg's (seq_len, sparsity,
    mask) shape, scaled to `arch`'s real geometry -- see module docstring
    for why `cfg.useful_flops()` alone undercounts what a whole-model
    wall-clock measurement actually pays for.

    `cfg.useful_flops()` (attention core: QK^T + PV, causal- and sparsity-
    aware) is unaffected by this fix and stays the per-layer attention
    figure. This function additionally counts, using the standard
    `2 * batch * seq_len * in_dim * out_dim` FLOPs-per-linear convention:

    - Q/K/V/O projections (`hidden_size` in/out for Q and O; `n_heads_kv *
      head_dim` out/in for K and V -- GQA-aware, matching `_expand_kv`'s
      house rule that a kernel's own cost model reflects its real shape).
    - The MLP block: Qwen2/Llama's gated MLP is 3 matmuls (gate/up/down),
      not 2 -- getting this wrong (assuming a plain 2-matmul MLP) would
      undercount by a third, the dominant FLOPs term at these sequence
      lengths.
    - The lm_head, once (not per layer) -- `logits_to_keep` positions, not
      the full sequence, matching the fix in `accuracy/model.py` for the
      exact OOM this mismatch also caused independently of attention
      backend (11+ GiB computing logits over ~40K positions at a ~150K
      vocab). Defaults to 1 (the real generation-time cost) rather than 0
      (every position), since 0 is never what a real Stage 3 run does.
    """
    b, s = cfg.batch, cfg.seq_len
    qkvo = 2 * b * s * arch.hidden_size * (
        arch.hidden_size                             # q_proj
        + 2 * arch.n_heads_kv * arch.head_dim         # k_proj + v_proj
        + arch.hidden_size)                           # o_proj
    mlp = 6 * b * s * arch.hidden_size * arch.intermediate_size  # gate+up+down
    attention_core = cfg.useful_flops()
    lm_head = 2 * b * logits_to_keep * arch.hidden_size * arch.vocab_size
    return arch.n_layers * (attention_core + qkvo + mlp) + lm_head


def _with_real_heads(cfg: AttnConfig, arch: ModelArchitecture) -> AttnConfig:
    """Swap a config's placeholder head geometry for the real model's,
    leaving everything that actually distinguishes a cell (seq_len, mask,
    sparsity, block_size, mask_source) untouched."""
    return replace(cfg, n_heads_q=arch.n_heads_q, n_heads_kv=arch.n_heads_kv,
                   head_dim=arch.head_dim)


# ---------------------------------------------------------------------------
# The unit of work
# ---------------------------------------------------------------------------
#
# A Stage 3 row is NOT one forward pass. It is one prefill plus `k` greedy
# decode steps, because Stage 3 scores generated text against a RULER answer
# and the answer is several tokens long. The estimate that missed this was
# not out by a rounding factor -- with no KV cache it was out by ~20x, and
# the error was not in any arithmetic. Every input was still valid; the unit
# was wrong. That is why the decode term lives here, in the cost model, with
# `decode_steps_by_task` a REQUIRED argument rather than a default: a caller
# has to say what a row costs, and "no decode" has to be written down as a
# claim rather than arrived at by omission.
#
# See docs/stage3_generation_decision.md for where these come from.

# Expected steps = measured answer length + 1, for the newline the model
# emits to end the line (see the stopping rule -- EOS essentially never
# fires on completion-style prompts). Measured over 200 examples per task
# with the real Qwen tokenizer, and independent of seq_len: the answer is a
# fixed-shape payload, so only BPE splitting varies.
DECODE_STEPS_BY_TASK: dict[str, int] = {
    "niah_single": 8,       # 7 answer tokens, exactly, on all 200
    "niah_multikey": 33,    # 27-36 observed
    "vt": 17,               # 12-20 observed
}

# The other end of the bracket. If the newline stop never fires, every
# example runs to its per-task cap, and these are what a row costs then.
# Pricing both is the point: the expected figure is a point estimate that
# depends on the model behaving, the cap figure is a bound that does not.
DECODE_STEPS_BY_TASK_AT_CAP: dict[str, int] = {
    "niah_single": 14,
    "niah_multikey": 72,
    "vt": 40,
}


def decode_flops(cfg: AttnConfig, arch: ModelArchitecture, *, n_steps: int) -> int:
    """FLOPs for `n_steps` cached greedy decode steps after a prefill of
    `cfg.seq_len` tokens.

    Each step runs the whole model over ONE position -- projections, MLP and
    lm_head all at `seq_len`-independent cost -- and attends that single
    query against a cache that has grown to `seq_len + i` keys. So the
    per-step attention term is linear in context where prefill's is
    quadratic, which is the entire reason the cache makes generation
    affordable: `k` steps cost roughly `k / seq_len` of a prefill.

    Two things this deliberately does NOT do, both conservative:

    1. It charges DENSE attention over the full cache even for a
       block_sparse cfg. That is not an approximation, it is decision C:
       sparsity is applied during prefill only and generation runs full
       attention over the cache (docs/limitations.md). A sparse row's decode
       really does cost this.
    2. It charges the same to GLA, whose decode is a fixed-size recurrent
       state update costing O(head_dim^2) per step with no dependence on
       context at all. GLA's real decode is far cheaper, so its rows are
       overcharged here. An estimate that is too high on one backend is a
       schedule that finishes early; the reverse is a session that runs out
       of hours mid-band.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    b, s, k = cfg.batch, cfg.seq_len, n_steps
    per_pos_qkvo = 2 * b * arch.hidden_size * (
        arch.hidden_size + 2 * arch.n_heads_kv * arch.head_dim + arch.hidden_size)
    per_pos_mlp = 6 * b * arch.hidden_size * arch.intermediate_size
    # sum over i of (s + i) keys attended, x2 matmuls (QK^T, PV) x 2 FLOPs
    cache_keys = k * s + k * (k - 1) // 2
    attention = 4 * b * arch.n_heads_q * arch.head_dim * cache_keys
    lm_head = k * 2 * b * arch.hidden_size * arch.vocab_size
    return arch.n_layers * (k * (per_pos_qkvo + per_pos_mlp) + attention) + lm_head


# ---------------------------------------------------------------------------
# Decode is bandwidth-bound, and dividing its FLOPs by a prefill throughput
# is a category error
# ---------------------------------------------------------------------------
#
# `decode_flops` above counts decode FLOPs correctly. Dividing that count by
# the measured PREFILL TFLOPS does not give a decode time, and the gap is not
# a rounding error: on Qwen2.5-1.5B at batch 1, a decode step is 3.09 GFLOP,
# which at the prefill's measured 42 TFLOPS would take 0.074 ms. The real
# floor is ~13 ms, because the step reads all 3.09 GB of bf16 weights to
# produce one token and an L4 has ~300 GB/s of bandwidth. **177x.**
#
# The two phases sit in different regimes. A prefill at 8192 tokens does
# ~8192 FLOPs of work per byte of weight it reads and is compute-bound; a
# batch-1 decode step does 2 and is bandwidth-bound. One throughput figure
# cannot describe both, and using the compute-bound one for the
# bandwidth-bound phase makes generation look free.
#
# This is the third appearance of the same shape in this project, and the
# most expensive of the three had the same signature -- a ratio taken across
# two regimes that look comparable because they share a unit:
#   * attention-KERNEL TFLOPS read as whole-MODEL TFLOPS (a 42% phantom
#     speedup);
#   * GLA's 1.7 "TFLOPS" against FA2's 61.8, concluding GLA is slow when it
#     is faster in wall clock and simply issues ~30x fewer FLOPs;
#   * and now prefill TFLOPS applied to decode.
# The unit matches every time. The regime does not.

# Fraction of peak memory bandwidth a real kernel achieves. 0.80 is the
# conventional figure for a large streaming read and is an ASSUMPTION, not a
# measurement -- flagged here because the whole point of this module is that
# assumptions must be visible. It sets a FLOOR: real decode also pays Python
# dispatch and kernel-launch latency per layer, which this does not model.
ACHIEVED_BANDWIDTH_FRACTION = 0.80


def model_parameter_bytes(arch: ModelArchitecture, *, dtype_bytes: int = 2) -> int:
    """Weight bytes read to produce one token.

    Counts the lm_head projection (`vocab_size x hidden_size`) once. Where
    embeddings are tied -- as they are on Qwen2.5-1.5B -- that matrix is read
    for the output projection whether or not it is also the input table, so
    it belongs here either way.
    """
    per_layer = (arch.hidden_size * arch.hidden_size                      # q_proj
                 + 2 * arch.hidden_size * arch.n_heads_kv * arch.head_dim  # k, v
                 + arch.hidden_size * arch.hidden_size                     # o_proj
                 + 3 * arch.hidden_size * arch.intermediate_size)          # gated MLP
    return dtype_bytes * (arch.n_layers * per_layer
                          + arch.vocab_size * arch.hidden_size)


def decode_memory_traffic_bytes(cfg: AttnConfig, arch: ModelArchitecture, *,
                                 n_steps: int, dtype_bytes: int = 2) -> int:
    """Bytes read across `n_steps` decode steps: the weights, every step,
    plus the KV cache, which grows.

    The weight term does not depend on context and dominates: at 8192 it is
    3.09 GB against 235 MB of cache. That is why the decode tax is nearly
    FLAT per row rather than proportional to context -- and therefore why it
    is largest, in relative terms, exactly where prefill is cheapest. At the
    2048 band it is a bigger cost than the prefill it follows.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    weights = n_steps * model_parameter_bytes(arch, dtype_bytes=dtype_bytes)
    # K and V, both, un-expanded in GQA layout -- which is how
    # SDPABackend.state_from_prefill really stores them.
    per_token_kv = 2 * arch.n_layers * arch.n_heads_kv * arch.head_dim * dtype_bytes
    cache = per_token_kv * (n_steps * cfg.seq_len + n_steps * (n_steps - 1) // 2)
    return weights + cfg.batch * cache


def decode_seconds(cfg: AttnConfig, arch: ModelArchitecture, *, n_steps: int,
                   peak_bandwidth_bytes_per_s: float,
                   per_step_overhead_s: float = 0.0,
                   achieved_fraction: float = ACHIEVED_BANDWIDTH_FRACTION,
                   dtype_bytes: int = 2) -> float:
    """Wall seconds for `n_steps` decode steps, from bandwidth rather than
    from FLOPs.

    `per_step_overhead_s` is the term that separates the floor from a real
    measurement: this implementation runs one Python-level HF forward per
    step with a custom attention module per layer and no CUDA graphs, so
    dispatch and launch latency are real and are not bandwidth. Pass 0 for
    the hardware floor; pass a measured value once the pre-flight probe has
    one. Do not guess it and then quote the result as an estimate -- quote
    the bracket.
    """
    bytes_read = decode_memory_traffic_bytes(cfg, arch, n_steps=n_steps,
                                             dtype_bytes=dtype_bytes)
    return (bytes_read / (peak_bandwidth_bytes_per_s * achieved_fraction)
            + n_steps * per_step_overhead_s)


# Peak memory bandwidth, for the devices this study actually rents.
PEAK_BANDWIDTH_BYTES_PER_S: dict[str, float] = {
    "L4": 300e9,        # GDDR6, 192-bit
    "A100-80GB": 2039e9,  # HBM2e
}


# NOTE: `MEASURED_TOKEN_INFLATION` and `_at_real_tokens` used to live here.
# They corrected for a generator that sized haystacks by word count, so a
# grid seq_len of 16384 produced ~19821 real tokens and every FLOPs estimate
# built on the nominal number came out ~20% low (worse in the quadratic
# attention term). That correction is retired: `accuracy/sizing.py` now
# binary-searches filler against the real tokenizer, so a grid seq_len IS a
# token count and needs no ratio applied to it.
#
# Fixing it at the source rather than correcting after the fact matters
# because the ratio was an empirical constant that would drift silently with
# a different tokenizer, model, or task template -- and nothing would have
# flagged it.


@dataclass(frozen=True)
class PhaseTiming:
    """One measured phase from the probe: a label for reporting, which
    FLOPs category it belongs to, the FLOPs figure for the probed config
    (caller's responsibility -- see module docstring on issued vs. useful),
    and the wall-clock time it took.
    """

    label: str
    category: PhaseCategory
    flops: int
    wall_seconds: float
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.wall_seconds <= 0:
            raise ValueError(f"{self.label}: wall_seconds must be positive, "
                              f"got {self.wall_seconds}")
        if self.flops <= 0:
            raise ValueError(f"{self.label}: flops must be positive, got {self.flops}")

    @property
    def effective_tflops(self) -> float:
        return self.flops / self.wall_seconds / 1e12


def blended_tflops_by_category(phases: list[PhaseTiming]) -> dict[PhaseCategory, float]:
    """One effective-TFLOPS figure per category, from the total FLOPs and
    total wall time across every probed phase in that category -- not an
    average of per-phase rates, so a slow outlier phase is weighted by how
    much time it actually took, not counted equally with a fast one.
    """
    totals: dict[PhaseCategory, list[float]] = {}
    for p in phases:
        flops_total, seconds_total = totals.setdefault(p.category, [0.0, 0.0])
        totals[p.category][0] = flops_total + p.flops
        totals[p.category][1] = seconds_total + p.wall_seconds
    return {cat: flops / seconds / 1e12 for cat, (flops, seconds) in totals.items()}


def total_grid_flops_by_category(configs_by_backend: dict[str, list[AttnConfig]],
                                  examples_by_task_length: dict[tuple[str, int], list[RulerExample]],
                                  *, arch: ModelArchitecture,
                                  decode_steps_by_task: dict[str, int],
                                  dense_backend: str = "sdpa_math",
                                  logits_to_keep: int = 1,
                                  ) -> dict[PhaseCategory, int]:
    """Total FLOPs the full grid actually requires, split into "scoring"
    and "measured" -- the same split `blended_tflops_by_category` produces
    from the probe, so the two can be divided category-by-category.

    Every config from `configs_by_backend` has two substitutions applied
    before `whole_model_flops` sees it, both of which were originally
    missing and both of which made the estimate wrong in the same
    direction (too low):

    1. Placeholder head geometry is swapped for `arch`'s real one (see
       `_with_real_heads`) -- `configs_by_backend`'s configs carry
       `n_heads_q=1` etc., fine for the cell-counting `build_cells` does
       but silently wrong for a FLOPs estimate.
    A config's `seq_len` is taken at face value as a real token count,
    which it now is: `accuracy/sizing.py` sizes every generated context
    against the real tokenizer, so no inflation ratio is applied here. An
    earlier version multiplied by a measured ~1.21 constant to correct a
    word-count-based generator; that correction is gone along with its
    cause.

    Mirrors the amortization `accuracy/model.py`'s cache already enforces
    at runtime: the scoring pass for a given (task, seq_len) is counted
    once per example, never once per block_sparse sparsity level, even
    though `configs_by_backend["block_sparse"]` lists one config per
    sparsity level.

    `decode_steps_by_task` is REQUIRED and has no default, including no
    empty one. A measured row is a prefill plus that many greedy decode
    steps (see `decode_flops` and the note above it); passing `{}` is the
    legitimate way to price the prefill term alone, but it has to be
    written down. The estimate this parameter exists to prevent was wrong
    by ~20x with every input still valid, purely because nobody stated the
    unit. A task with examples but no entry here raises rather than
    silently costing zero decode.

    Decode is charged to its own "decode" category, NEVER folded into
    "measured". Folding it in was this module's own version of the bug it
    documents: `corrected_grid_hours` divides each category's FLOPs by that
    category's measured TFLOPS, and the measured TFLOPS are compute-bound
    prefill figures. On 2026-09-06 the probe printed "grid decode ~= 5.35 h"
    from the bandwidth model and "+0.03 h (0.18%)" from this path, three
    lines apart, both from the same run. A separate category is what forces
    the caller to cost it with `decode_seconds` instead.
    """
    missing = sorted({task for task, _ in examples_by_task_length}
                     - set(decode_steps_by_task)) if decode_steps_by_task else []
    if missing:
        raise KeyError(
            f"{missing} have examples but no entry in decode_steps_by_task. "
            f"A missing task would be costed at zero decode steps, which is a "
            f"claim about the model's stopping behaviour and must be made "
            f"explicitly -- pass 0 for it, or {{}} to price prefill alone.")
    n_examples = {(task, seq_len): len(examples)
                  for (task, seq_len), examples in examples_by_task_length.items()}

    totals: dict[PhaseCategory, int] = {"scoring": 0, "measured": 0, "decode": 0}
    tasks = sorted({task for task, _ in examples_by_task_length})

    scored_seq_lens = {cfg.seq_len for cfg in configs_by_backend.get("block_sparse", [])}
    dense_cfg_by_seq_len = {cfg.seq_len: cfg for cfg in configs_by_backend.get(dense_backend, [])}
    def _costed(cfg: AttnConfig) -> int:
        return whole_model_flops(_with_real_heads(cfg, arch), arch,
                                 logits_to_keep=logits_to_keep)

    for seq_len in scored_seq_lens:
        scoring_cfg = dense_cfg_by_seq_len.get(seq_len)
        if scoring_cfg is None:
            continue
        scoring_flops = _costed(scoring_cfg)
        for task in tasks:
            n = n_examples.get((task, seq_len), 0)
            totals["scoring"] += n * scoring_flops

    for backend_name, configs in configs_by_backend.items():
        for cfg in configs:
            measured_flops = _costed(cfg)
            real_cfg = _with_real_heads(cfg, arch)
            for task in tasks:
                n = n_examples.get((task, cfg.seq_len), 0)
                if n == 0:
                    continue
                steps = decode_steps_by_task.get(task, 0)
                totals["measured"] += n * measured_flops
                totals["decode"] += n * decode_flops(real_cfg, arch, n_steps=steps)

    return totals


def corrected_grid_hours(total_flops_by_category: dict[PhaseCategory, int],
                          measured_tflops_by_category: dict[PhaseCategory, float],
                          ) -> dict[str, float]:
    """Total grid wall-clock hours per category plus the sum, using the
    probe's measured throughput in place of the planning-time 15 TFLOPS
    assumption. Raises if a category has grid FLOPs but no matching
    measurement -- an incomplete probe should fail loudly, not silently
    under-report the total.
    """
    hours: dict[str, float] = {}
    for category, flops in total_flops_by_category.items():
        if category == "decode" and flops:
            raise ValueError(
                "refusing to convert decode FLOPs to hours through a TFLOPS "
                "figure. A batch-1 decode step is BANDWIDTH-bound and the "
                "measured TFLOPS here are compute-bound prefill figures; the "
                "two are ~177x apart on this model. Use "
                "timing_probe.decode_seconds, or a measured per-step time. "
                "See docs/silent_failure_patterns.md #15.")
        if flops == 0:
            hours[category] = 0.0
            continue
        if category not in measured_tflops_by_category:
            raise KeyError(
                f"grid requires {flops} FLOPs in category {category!r} but "
                f"the probe measured no phase in that category -- run the "
                f"probe against every backend/phase the grid actually uses"
            )
        tflops = measured_tflops_by_category[category]
        seconds = flops / (tflops * 1e12)
        hours[category] = seconds / 3600.0
    hours["total"] = sum(hours.values())
    return hours
