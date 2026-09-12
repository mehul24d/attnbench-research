"""Stage 3 grid, loaded from data rather than inferred from unrelated code.

The model axis in particular must never be re-derived from Stage 2's
AttnConfig/SweepGrid head geometry -- that inference produced a model that
was ~20x too large for this study's hardware ceiling during planning (see
docs/hardware_constraints.md). `stage3_grid.yaml` is the single source of truth for which
models, context lengths, and block sizes this study actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AccuracyGrid:
    """The Stage 3 grid. Mirrors config.SweepGrid's role for Stage 2, but
    loaded from YAML instead of hardcoded, since the model axis is a fact
    about available hardware and published model architectures -- not
    something that belongs baked into Python.

    `seq_lens` maps seq_len -> n_per_length rather than a flat tuple: the
    bootstrap's discriminating power depends directly on n, and the study
    deliberately runs most lengths at a power-adequate n while accepting a
    smaller, explicitly underpowered n at the longest length rather than
    dropping it -- published work finds tolerable sparsity rises with
    sequence length, so omitting the longest length risks concluding
    "sparsity hurts everywhere" for the wrong reason (no data there, not a
    real effect). `directional_seq_lens` names exactly which lengths that
    applies to, so it's a queryable fact, not something inferred by
    comparing n values across rows.
    """

    model_primary: str
    model_alternate: str
    seq_lens: dict[int, int]           # seq_len -> n_per_length
    directional_seq_lens: frozenset[int]
    block_sizes: tuple[int, ...]
    sparsities: tuple[float, ...]
    tasks: tuple[str, ...]
    score_dtype: str
    score_cache_dir: str
    dense_backend: str

    @property
    def finest_block_size(self) -> int:
        """The block_size importance scores are cached at (score_cache.py
        pools further from this for any coarser block_size request, rather
        than recomputing) -- always the smallest configured block_size,
        since coarser granularities are a further pool of the finer one,
        never the other way around.
        """
        return min(self.block_sizes)

    def is_directional(self, seq_len: int) -> bool:
        """True for a length run at reduced n as a single directional data
        point rather than the study's power-adequate main claim."""
        return seq_len in self.directional_seq_lens


def load_grid(path: str | Path) -> AccuracyGrid:
    """Parse a stage3_grid.yaml into an AccuracyGrid.

    Raises on missing required keys rather than defaulting them silently --
    a grid with an unintended model or an unintended seq_len cap is exactly
    the class of error docs/hardware_constraints.md's hardware-ceiling note exists to prevent,
    and a silent default could reintroduce it.
    """
    p = Path(path)
    with p.open() as f:
        raw = yaml.safe_load(f)

    required = ("model", "seq_lens", "block_sizes", "sparsities", "tasks",
                "score_dtype", "score_cache_dir", "dense_backend")
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"{p}: missing required key(s) {missing}")

    model = raw["model"]
    for k in ("primary", "alternate"):
        if k not in model:
            raise KeyError(f"{p}: model.{k} is required")

    seq_lens = {int(k): int(v) for k, v in raw["seq_lens"].items()}
    directional = frozenset(int(s) for s in raw.get("directional_seq_lens", []))
    unknown_directional = directional - set(seq_lens)
    if unknown_directional:
        raise KeyError(
            f"{p}: directional_seq_lens {sorted(unknown_directional)} not "
            f"in seq_lens {sorted(seq_lens)}"
        )

    return AccuracyGrid(
        model_primary=model["primary"],
        model_alternate=model["alternate"],
        seq_lens=seq_lens,
        directional_seq_lens=directional,
        block_sizes=tuple(raw["block_sizes"]),
        sparsities=tuple(raw["sparsities"]),
        tasks=tuple(raw["tasks"]),
        score_dtype=raw["score_dtype"],
        score_cache_dir=raw["score_cache_dir"],
        dense_backend=raw["dense_backend"],
    )
