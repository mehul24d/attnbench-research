"""The arms the L4 shakedown (T2.27) records, as prefills the components read.

Every GPU step of the shakedown (`scripts/t227_steps.py`) needs the same
thing from an arm: a `run_prefill()` that runs one prefill of one prompt at
one length, the decoder layers to annotate, and the `CodePath` the sequence
is recorded against. `Harness.prepare` builds that, for one arm at a time,
the way the accuracy runs build it (`scripts/run_accuracy.py`,
`accuracy/generation.generate_one`): `SwappableAttentionModel.run_measured`
with the arm's backend and config, `logits_to_keep=1`, no decode.

The arms (`ARMS`):

  - dense: `sdpa_flash`, causal (on a CPU dry run, `sdpa_math`);
  - oracle: the block-sparse kernel with masks from the dense fp32 scoring
    pass, built by the reference builder (`masks.mask_for`) on the host. The
    scores are computed before the prefill and outside it: the oracle is a
    non-deployable reference (plan Appendix A, 26 Oct 2026);
  - meanpool: the block-sparse kernel with the inline mean-pool estimator
    and the device builder, in the forward;
  - xattention: the authors' estimator with the calibrated "authors" table
    (T1.1's arm of record), stride 8;
  - xattention_noassert: the same with its Python asserts removed (T2.17).

The two block-sparse mask arms run at sparsity 0.75, the README's figures.

**Prompts.** On the L4 every prompt is a real RULER example sized by the
model's own tokenizer to exactly the cell's length (`grid_configs.
build_examples_by_task_length`, seed 0, task `qa_1`), so the steps see the
inputs the study runs. A CPU dry run has no tokenizer download and no RULER
data: it uses a two-layer random-weight Qwen2 model built from a config and
seeded random token ids, and only the arms that run on a CPU (dense).

**One wrapper at a time.** `SwappableAttentionModel` replaces each layer's
`self_attn` when it is built and remembers the originals; wrapping a model
that is already wrapped would remember the wrappers. So `prepare` unwraps
the previous arm first.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

from .op_sequence import CodePath

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "configs" / "accuracy" / "stage3_grid.yaml"
XATTN_TABLE = "authors"           # T1.1's calibration of record
SPARSITY = 0.75
TASK = "qa_1"
SEED = 0


@dataclass(frozen=True)
class ArmSpec:
    name: str
    backend: str
    mask: str                      # AttnConfig.mask
    score_source: str              # SwappableAttentionModel score_source
    sparsity: Optional[float] = None
    inline_estimator: Optional[str] = None
    scores_before: bool = False    # oracle: dense scoring pass outside the prefill
    xattention: bool = False
    needs_cuda: bool = True


ARMS: dict[str, ArmSpec] = {s.name: s for s in (
    ArmSpec("dense", "sdpa_flash", "causal", "dense_softmax_fp32", needs_cuda=False),
    ArmSpec("oracle", "block_sparse", "block_sparse", "dense_softmax_fp32",
            sparsity=SPARSITY, scores_before=True),
    ArmSpec("meanpool", "block_sparse", "block_sparse", "minference_meanpool_inline",
            sparsity=SPARSITY, inline_estimator="minference_meanpool"),
    ArmSpec("xattention", "xattention", "causal", "xattention_inline", xattention=True),
    ArmSpec("xattention_noassert", "xattention_noassert", "causal", "xattention_inline",
            xattention=True),
)}


def runnable(name: str, cuda: bool) -> bool:
    return cuda or not ARMS[name].needs_cuda


@dataclass
class PreparedArm:
    spec: ArmSpec
    length: int
    prompt_id: str
    run_prefill: Callable[[], None]
    layers: Sequence[torch.nn.Module]
    backend: object
    code_path: Optional[CodePath] = None
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name


def libraries() -> tuple[tuple[str, str], ...]:
    """(name, version) of everything a sequence can depend on."""
    out = [("torch", torch.__version__)]
    for mod in ("transformers", "flash_attn", "block_sparse_attn", "triton"):
        try:
            m = __import__(mod)
            out.append((mod, str(getattr(m, "__version__", "unknown"))))
        except Exception:
            pass
    try:
        from ..backends.xattention import installed_checkout
        head, dirty = installed_checkout()
        if head is not None:
            out.append(("xattn", head + ("+dirty" if dirty else "")))
    except Exception:
        pass
    return tuple(out)


def tiny_model():
    """A two-layer random Qwen2 for CPU dry runs: no download, head_dim 128."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=512, hidden_size=256, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=8192)
    return Qwen2ForCausalLM(cfg).eval()


class Harness:
    """The model, its tokenizer and the grid, loaded once per process."""

    def __init__(self, *, device: str, dry_run: bool, dtype: str = "bfloat16",
                 score_cache_dir: Optional[str] = None):
        from ..accuracy.config import load_grid
        from ..accuracy.generation import ModelGeometry
        from .. import numerics

        self.device, self.dry_run = device, dry_run
        self.cuda = device.startswith("cuda")
        self.grid = load_grid(GRID)
        self.score_cache_dir = score_cache_dir or str(REPO / "results" / "accuracy" / "score_cache")
        numerics.enforce_fp32_matmul()
        if dry_run:
            self.model_id = "tiny-qwen2-random"
            self.dtype = "float32"
            self.model = tiny_model().to(device)
            self.tokenizer = None
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from ..accuracy.pins import require_pinned_transformers
            self.model_id = self.grid.model_primary
            require_pinned_transformers(self.model_id)
            self.dtype = dtype
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.geometry = ModelGeometry.from_config(self.model.config, self.dtype)
        self._wrapped = None
        self._examples: dict = {}

    # --- prompts -------------------------------------------------------------

    def examples(self, length: int, n: int, *, task: str = TASK):
        """n real RULER examples at `length` tokens (indices 0..n-1)."""
        if self.dry_run:
            raise RuntimeError("a dry run has no RULER prompts")
        key = (task, length)
        have = self._examples.get(key, [])
        if len(have) < n:
            from ..accuracy.grid_configs import build_examples_by_task_length
            from ..accuracy.prompting import prompt_token_counter
            have = build_examples_by_task_length(
                self.grid, SEED, count_tokens=prompt_token_counter(self.tokenizer),
                tasks=(task,), seq_lens={length: n})[(task, length)]
            self._examples[key] = have
        return have[:n]

    def input_ids(self, length: int, *, index: int = 0, task: str = TASK):
        """(input_ids, prompt_id) for one prompt at exactly `length` tokens."""
        if self.dry_run:
            g = torch.Generator().manual_seed(SEED + length + index)
            ids = torch.randint(0, self.model.config.vocab_size, (1, length), generator=g)
            return ids.to(self.device), f"random-{length}-{index}"
        from ..accuracy.prompting import encode_prompt
        ex = self.examples(length, index + 1, task=task)[index]
        ids = encode_prompt(self.tokenizer, ex.context, self.device)
        if int(ids.shape[-1]) != length:
            raise RuntimeError(f"{ex.example_id}: {int(ids.shape[-1])} tokens, not {length}")
        return ids, ex.example_id

    # --- arms ----------------------------------------------------------------

    def _backend(self, spec: ArmSpec):
        from ..accuracy.grid_configs import backend_instance
        if spec.xattention:
            from ..accuracy.t4_pilot import XATTN_CALIBRATION_TABLES, XATTN_STRIDE
            from ..backends import get
            from ..backends.xattention import ThresholdTable
            table = ThresholdTable.load(REPO / XATTN_CALIBRATION_TABLES[XATTN_TABLE])
            return get(spec.backend)(threshold=table, stride=XATTN_STRIDE)
        name = spec.backend
        if name == "sdpa_flash" and not self.cuda:
            name = "sdpa_math"
        return backend_instance(name)

    def _cfg(self, spec: ArmSpec, length: int):
        from ..config import AttnConfig
        sparse = spec.mask == "block_sparse"
        base = AttnConfig(seq_len=length, batch=1, n_heads_q=1, n_heads_kv=1, head_dim=128,
                          mask=spec.mask, sparsity=spec.sparsity if sparse else None,
                          block_size=self.grid.block_sizes[0] if sparse else None,
                          mask_source="importance" if sparse else None)
        return self.geometry.onto(base, seq_len=length)

    def unwrap(self) -> None:
        if self._wrapped is not None:
            self._wrapped.unwrap()
            self._wrapped = None

    def prepare(self, name: str, length: int, *, index: int = 0,
                backend=None) -> PreparedArm:
        """One arm's prefill at one length and prompt. `backend` replaces the
        arm's own (T2.17's gate, T2.21's mask agreement)."""
        from ..accuracy.model import SwappableAttentionModel
        spec = ARMS[name]
        if not runnable(name, self.cuda):
            raise RuntimeError(f"arm {name!r} needs a CUDA device")
        self.unwrap()
        cfg = self._cfg(spec, length)
        template = self.geometry.onto(replace(cfg, mask="causal", sparsity=None,
                                              block_size=None, mask_source=None),
                                      seq_len=length)
        wrapped = SwappableAttentionModel(self.model, template, model_id=self.model_id,
                                          finest_block_size=self.grid.finest_block_size,
                                          score_source=spec.score_source)
        self._wrapped = wrapped
        ids, prompt_id = self.input_ids(length, index=index)
        be = backend if backend is not None else self._backend(spec)
        scores = None
        if spec.scores_before:
            scores = wrapped.compute_importance_scores(
                ids, task=TASK, example_id=prompt_id, cache_dir=self.score_cache_dir)
        selecting = hasattr(be, "reset_density")

        def run_prefill():
            if selecting:
                be.reset_density()
            wrapped.run_measured(ids, be, cfg=cfg, layer_scores=scores, logits_to_keep=1,
                                 inline_estimator=spec.inline_estimator)

        return PreparedArm(spec, length, prompt_id, run_prefill, wrapped._layers, be,
                           extra={"cfg": cfg, "input_ids": ids, "wrapped": wrapped})

    def code_path(self, arm: PreparedArm) -> CodePath:
        """The arm's code path, read after one run of it, so the XAttention
        arms name the path the estimator actually took (Triton or torch)."""
        impl = "eager"
        if arm.spec.xattention:
            eff = getattr(arm.backend, "triton_effective", None)
            impl = "xattn:triton" if eff is True else "xattn:torch"
        gpu = torch.cuda.get_device_name(0) if self.cuda else "cpu"
        return CodePath(arm=arm.name, gpu=gpu, implementation=impl,
                        libraries=libraries(), python_optimize=sys.flags.optimize)

    def ready(self, name: str, length: int, *, index: int = 0) -> PreparedArm:
        """`prepare`, one untraced warm run, and the code path."""
        arm = self.prepare(name, length, index=index)
        arm.run_prefill()
        if self.cuda:
            torch.cuda.synchronize()
        arm.code_path = self.code_path(arm)
        return arm
