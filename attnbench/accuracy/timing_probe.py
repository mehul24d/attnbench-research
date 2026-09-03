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
    inference CLAUDE.md's hardware-ceiling note already exists to prevent
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
    """
    n_examples = {(task, seq_len): len(examples)
                  for (task, seq_len), examples in examples_by_task_length.items()}

    totals: dict[PhaseCategory, int] = {"scoring": 0, "measured": 0}
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
            for task in tasks:
                n = n_examples.get((task, cfg.seq_len), 0)
                totals["measured"] += n * measured_flops

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
