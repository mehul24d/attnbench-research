"""SwappableAttentionModel: runs a real HF causal LM with each layer's
self-attention replaced by one of this harness's AttentionBackend instances,
so Stage 3 measures accuracy with the exact same kernel implementations
Stage 2 measures speed with.

Two passes per example, not one:

1. **Scoring pass** -- every layer's attention runs a real, dense,
   causal-masked softmax computation (the harness's own math, not the
   model's built-in attention path) and derives a block-pooled importance
   ranking from it. This is `score_source="dense_softmax_fp32"` (see
   schema.py): a real system would use a cheap approximate estimator here,
   not a full dense pass, so accuracy numbers built on this ranking are an
   **upper bound**, and the estimator's own cost is excluded from every
   latency number in the study. That trade-off is deliberate -- it isolates
   the kernel under test -- and is recorded on every row rather than left in
   this docstring.
2. **Measured pass** -- every layer's attention runs through the actual
   `AttentionBackend` under test, using (for block-sparse configs) the
   importance ranking pass 1 produced.

Both passes reuse the model's own `q_proj/k_proj/v_proj/o_proj` and its own
`apply_rotary_pos_emb` -- never reimplemented, so RoPE can't silently drift
from what the real model does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import torch
import transformers
from torch import nn

from . import score_cache
from .. import masks
from .. import numerics
from ..backends.base import AttentionBackend
from ..backends.impls import SDPABackend
from ..config import AttnConfig

# The decoder layer's unpacking of self_attn's return value changed between
# transformers major versions -- confirmed against two real installs, not
# assumed: 4.46.0's Qwen2DecoderLayer does
# `hidden_states, self_attn_weights, present_key_value = self.self_attn(...)`
# (3-tuple), while 5.16.1's LlamaDecoderLayer does `hidden_states, _ =
# self.self_attn(...)` (2-tuple). Getting this wrong raises ValueError
# ("too many/not enough values to unpack") -- a hard crash, not a silent
# wrong answer, but only on whichever transformers version wasn't tested
# against. There is no tuple length that satisfies both call sites at once.
def _self_attn_returns_3tuple(version: str) -> bool:
    return int(version.split(".")[0]) < 5


_TRANSFORMERS_SELF_ATTN_RETURNS_3TUPLE = _self_attn_returns_3tuple(transformers.__version__)

# The CausalLM.forward kwarg that restricts lm_head computation to the last
# N positions was renamed at the same major-version boundary as the
# self_attn tuple arity above -- confirmed against the same two real
# installs: 4.46.0's Qwen2ForCausalLM.forward takes `num_logits_to_keep`,
# 5.16.1's takes `logits_to_keep`. Both default to 0, and both apply it as
# `hidden_states[:, -N:, :]` -- since `-0 == 0` in Python, the default
# silently means "every position", not "none". Passing 1 explicitly is what
# makes lm_head cost O(1) instead of O(seq_len): a real bug found on the
# first GPU validation session -- computing logits over all ~40K positions
# at a ~150K vocab tried to allocate 11+ GiB and OOM'd, entirely unrelated
# to which attention backend was under test.
def _logits_to_keep_kwarg(version: str) -> str:
    return "num_logits_to_keep" if int(version.split(".")[0]) < 5 else "logits_to_keep"


_LOGITS_TO_KEEP_KWARG = _logits_to_keep_kwarg(transformers.__version__)


# **This gate fails SILENTLY, and its sibling above does not.** That asymmetry
# is the whole reason this function exists.
#
# Get the tuple arity wrong and the decoder layer raises ValueError on the
# first forward -- measured on both real installs 2026-09-21: "not enough
# values to unpack (expected 3, got 2)" on 4.46.0, "too many values to unpack
# (expected 2)" on 5.16/5.17. You cannot ship that.
#
# Get THIS name wrong and nothing happens. `CausalLM.forward` carries
# `**kwargs` on both versions, so the unknown keyword is swallowed, the
# parameter keeps its default of 0, and `hidden_states[:, -0:, :]` is every
# position rather than none. Measured at seq_len=64: the correct name returns
# logits of shape (1, 1, V), the wrong name (1, 64, V) -- byte-identical to
# passing nothing at all. At the real 40K positions and ~150K vocab that is
# the 11+ GiB allocation that OOM'd the first GPU validation session,
# reappearing with no error anywhere.
#
# So the check is made explicit, once, against the model actually being
# wrapped. A silent failure on a paid GPU is worth a signature lookup in a
# constructor.
def _assert_logits_kwarg_accepted(model: nn.Module) -> None:
    import inspect

    try:
        params = inspect.signature(type(model).forward).parameters
    except (AttributeError, TypeError, ValueError):
        # No `forward`, or one that cannot be introspected. Say nothing:
        # the architecture check below is the one that owns "this is not a
        # model we can wrap", and pre-empting it with a message about an
        # lm_head keyword would point the reader at the wrong problem.
        return
    if _LOGITS_TO_KEEP_KWARG in params:
        return
    other = ("logits_to_keep" if _LOGITS_TO_KEEP_KWARG == "num_logits_to_keep"
             else "num_logits_to_keep")
    hint = (f" It does accept {other!r}, so the version test in "
            f"_logits_to_keep_kwarg() is wrong for transformers "
            f"{transformers.__version__}." if other in params else
            " It accepts neither name, so this model cannot restrict lm_head "
            "to the last position and every scoring forward will compute "
            "logits over the whole sequence.")
    raise UnsupportedModelArchitecture(
        f"{type(model).__name__}.forward does not accept "
        f"{_LOGITS_TO_KEEP_KWARG!r}.{hint} Passing it anyway would be "
        f"swallowed by **kwargs and silently compute logits over every "
        f"position -- the allocation that OOM'd the 2026-09-04 session.")

Mode = Literal["score", "measured", "decode"]


class UnsupportedModelArchitecture(Exception):
    """Raised when a model isn't the Llama-family shape this wrapper
    targets. Fails closed -- no silent fallback to a different mechanism."""


def _require_llama_family(model: nn.Module) -> list[nn.Module]:
    """Return the list of decoder layers, or raise.

    Targets the Llama/Mistral/Qwen2 attention module shape: each layer has a
    `.self_attn` with `q_proj/k_proj/v_proj/o_proj` Linear submodules and a
    `.head_dim` attribute. This is a structural check, not a class-name
    check, so it also accepts other architectures that happen to share the
    shape -- but raises rather than guessing on anything that doesn't.
    """
    try:
        layers = list(model.model.layers)
    except AttributeError as e:
        raise UnsupportedModelArchitecture(
            f"expected model.model.layers (Llama-family shape); {type(model).__name__} "
            f"has no such attribute: {e}"
        )
    if not layers:
        raise UnsupportedModelArchitecture("model.model.layers is empty")
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise UnsupportedModelArchitecture(f"layer {i} has no self_attn")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not isinstance(getattr(attn, proj, None), nn.Linear):
                raise UnsupportedModelArchitecture(
                    f"layer {i}.self_attn.{proj} is not an nn.Linear -- "
                    f"not the Llama-family attention shape this wrapper targets"
                )
        if not hasattr(attn, "head_dim"):
            raise UnsupportedModelArchitecture(f"layer {i}.self_attn has no head_dim")
    return layers


@dataclass
class _ModeState:
    """Mutable, shared by reference across every layer's SwappedAttention so
    a single flag flip on SwappableAttentionModel switches every layer's
    behavior for the next forward call, without rebuilding the model.
    """

    mode: Mode = "measured"
    backend: Optional[AttentionBackend] = None
    cfg_template: Optional[AttnConfig] = None
    chunk_blocks: int = 4
    finest_block_size: int = 64
    # Which scorer runs in mode == "score". Defaulted to the dense oracle so
    # every existing caller is unchanged; set to "minference_meanpool" for the
    # cheap arm. It is ALSO hashed into the score-cache key, because the two
    # scorers emit identically-shaped tensors for the same example and would
    # otherwise collide -- see score_cache.cache_key.
    score_source: str = "dense_softmax_fp32"
    # Lazily built dense backend used ONLY to give the cheap scoring pass a
    # faithful layer output. NOT charged to the estimator's cost -- see the
    # minference_meanpool branch in SwappedAttention.forward.
    score_output_backend: Optional[AttentionBackend] = None
    # Populated by the scoring pass, one entry per layer_idx, each
    # (n_heads_kv, n_blocks_finest, n_blocks_finest) fp32.
    scores: dict = field(default_factory=dict)

    # ---- decode ------------------------------------------------------
    # The backend generation runs through, which is NOT always the backend
    # being measured. Sparsity is applied during prefill only (decision C,
    # docs/stage3_generation_decision.md): at 8192 context with 33 generated
    # tokens decode is under 1% of the work, so sparsifying it buys nothing
    # measurable and would add a mask-extrapolation confound to the thing
    # that can be measured -- the scoring pass ranked the prompt's blocks and
    # never scored a token that does not exist yet.
    decode_backend: Optional[AttentionBackend] = None
    # layer_idx -> KVCacheState. Captured during prefill, advanced per step.
    decode_states: dict = field(default_factory=dict)
    # Set during a measured prefill to capture state for the decode that
    # follows. Off by default so correctness runs pay nothing for it.
    capture_decode_state: bool = False


def _scoring_forward_chunked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              *, block_size: int, chunk_blocks: int = 4,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense causal softmax attention AND its block-pooled importance
    ranking, computed together in one chunked pass over query blocks, so
    the full (seq_len, seq_len) matrix never materialises.

    Both outputs come from the same pass deliberately: the scoring pass's
    layer output must be a real, faithful continuation of the model's
    hidden states (later layers depend on it), and recomputing it via a
    second, separate full-precision call would reintroduce exactly the
    O(seq_len^2) materialisation chunking exists to avoid.

    q: (1, n_heads_q, seq_len, head_dim), k/v: (1, n_heads_kv, seq_len, head_dim).
    Returns (attn_output (1, n_heads_q, seq_len, head_dim),
             pooled_scores (n_heads_kv, n_blocks, n_blocks) fp32).
    """
    batch, n_heads_q, seq_len, head_dim = q.shape
    n_heads_kv = k.shape[1]
    if batch != 1:
        raise ValueError("importance scoring assumes batch=1 (one example at a time)")
    group_size = n_heads_q // n_heads_kv
    n_blocks = -(-seq_len // block_size)  # ceil div, same as masks._n_blocks
    scale = 1.0 / math.sqrt(head_dim)

    q32, k32 = q.float(), k.float()
    out = torch.empty(batch, n_heads_q, seq_len, head_dim, dtype=v.dtype, device=q.device)
    # CPU, not q.device: masks.py's mask generation (BlockSparseMask.active,
    # importance_block_mask's jitter) is CPU-only by convention -- see
    # BlockSparseMask's docstring -- so the ranking this feeds is moved to
    # CPU once here (a few hundred floats per layer) rather than forcing
    # every caller of masks.mask_for to special-case a CUDA tensor.
    scores = torch.zeros(n_heads_kv, n_blocks, n_blocks, dtype=torch.float32)

    for qb0 in range(0, n_blocks, chunk_blocks):
        qb1 = min(qb0 + chunk_blocks, n_blocks)
        row0, row1 = qb0 * block_size, min(qb1 * block_size, seq_len)
        key_end = row1  # causal: this chunk never needs keys past its own rows

        q_chunk = q32[:, :, row0:row1, :]
        k_chunk = k32[:, :, :key_end, :].repeat_interleave(group_size, dim=1)
        v_chunk = v[:, :, :key_end, :].repeat_interleave(group_size, dim=1)

        logits = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
        row_idx = torch.arange(row0, row1, device=q.device).unsqueeze(1)
        col_idx = torch.arange(0, key_end, device=q.device).unsqueeze(0)
        logits = logits.masked_fill(col_idx > row_idx, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        out[:, :, row0:row1, :] = torch.matmul(probs.to(v.dtype), v_chunk)

        probs_grouped = probs.view(batch, n_heads_kv, group_size,
                                    row1 - row0, key_end).mean(dim=2)
        key_pad = (-key_end) % block_size
        row_pad = (qb1 - qb0) * block_size - (row1 - row0)
        if key_pad or row_pad:
            probs_grouped = torch.nn.functional.pad(probs_grouped, (0, key_pad, 0, row_pad))
        key_n_blocks = (key_end + key_pad) // block_size
        pooled = probs_grouped.view(batch, n_heads_kv, qb1 - qb0, block_size,
                                     key_n_blocks, block_size).mean(dim=(3, 5))
        scores[:, qb0:qb1, :key_n_blocks] = pooled[0].cpu()

    return out, scores


def minference_meanpool_scores(q: torch.Tensor, k: torch.Tensor, *,
                                n_heads_kv: int, block_size: int) -> torch.Tensor:
    """The CHEAP estimator: MInference's Block-Sparse index estimation.

    Specified in MInference 1.0 (arXiv:2407.02490), Algorithm 3:

        Q_hat <- MeanPooling(Q, block_size)
        K_hat <- MeanPooling(K, block_size)
        A_hat <- softmax(Q_hat K_hat^T / sqrt(d) + m_causal)
        i_b   <- argtopk(A_hat, k_b)

    and cited by Sparse Frontier (arXiv:2504.17768v2, Appendix A.1.1) as the
    source of its "block-wise pooled token representations".

    **How this differs from the oracle, exactly.** The dense scorer above
    computes `mean_block(softmax(Q K^T))`. This computes
    `softmax(mean_block(Q) mean_block(K)^T)`. Same quantity with the pooling
    and the softmax commuted -- which is the approximation MInference
    justifies by the commutativity of mean pooling and MatMul. The two arms
    therefore differ ONLY in how blocks were ranked, which is what makes the
    oracle-versus-estimator comparison interpretable.

    **Why it is cheap.** The oracle materialises an S x S score matrix (and
    is chunked precisely because it cannot be held whole). This never forms
    one: it pools to S/block_size representatives first, so the matmul is
    (S/b) x (S/b). At S=32768, b=128 that is 256 x 256 against 32768 x 32768
    -- about 16000x fewer score entries. Its cost is not excluded from
    latency numbers the way the oracle's is, because it is small enough to
    pay.

    **Head convention** follows the oracle and `masks.py`: scores are meaned
    over the query heads within a KV group, giving one ranking per KV head
    and none per query head. Note that the reference implementation is
    per-head adaptive (uniform budget, independent selection); this study is
    head-uniform, and that divergence is recorded in limitations.md.

    Returns (n_heads_kv, n_blocks, n_blocks) on CPU, matching the oracle.
    """
    batch, n_heads_q, seq_len, head_dim = q.shape
    if batch != 1:
        raise ValueError(f"minference_meanpool_scores expects batch=1, got {batch}")
    group_size = n_heads_q // n_heads_kv
    n_blocks = (seq_len + block_size - 1) // block_size
    pad = n_blocks * block_size - seq_len

    q32 = q.float()
    k32 = k.float()
    if pad:
        q32 = torch.nn.functional.pad(q32, (0, 0, 0, pad))
        k32 = torch.nn.functional.pad(k32, (0, 0, 0, pad))

    # MeanPooling(Q, block_size) and MeanPooling(K, block_size).
    q_hat = q32.view(batch, n_heads_q, n_blocks, block_size, head_dim).mean(dim=3)
    k_hat = k32.view(batch, n_heads_kv, n_blocks, block_size, head_dim).mean(dim=3)
    k_hat = k_hat.repeat_interleave(group_size, dim=1)

    scale = 1.0 / math.sqrt(head_dim)
    logits = torch.matmul(q_hat, k_hat.transpose(-2, -1)) * scale

    # Block-level causal mask: query block qb may not see key block kv > qb.
    qb = torch.arange(n_blocks, device=q.device).unsqueeze(1)
    kv = torch.arange(n_blocks, device=q.device).unsqueeze(0)
    logits = logits.masked_fill(kv > qb, float("-inf"))

    a_hat = torch.softmax(logits, dim=-1)
    # Mean over the query heads within each KV group, as the oracle does.
    a_hat = a_hat.view(batch, n_heads_kv, group_size, n_blocks, n_blocks).mean(dim=2)
    return a_hat[0].cpu()


def pool_scores_to_block_size(scores_finest: torch.Tensor, *,
                               finest_block_size: int,
                               target_block_size: int) -> torch.Tensor:
    """Derive a coarser-block_size score matrix from the cached finest one by
    further average-pooling, never a separate cache entry or recomputation
    -- the same fixed-ranking-then-derive-coarser nesting masks.py already
    uses across sparsity, applied to block_size instead.
    """
    if target_block_size == finest_block_size:
        return scores_finest
    if target_block_size % finest_block_size != 0:
        raise ValueError(
            f"target_block_size={target_block_size} must be a multiple of "
            f"finest_block_size={finest_block_size}"
        )
    factor = target_block_size // finest_block_size
    n_heads_kv, n, _ = scores_finest.shape
    if n % factor != 0:
        raise ValueError(
            f"finest grid of {n} blocks doesn't divide evenly by pooling "
            f"factor {factor} (target/finest block_size ratio)"
        )
    pooled = scores_finest.view(n_heads_kv, n // factor, factor, n // factor, factor)
    return pooled.mean(dim=(2, 4))


class SwappedAttention(nn.Module):
    """Replaces one decoder layer's self_attn. Reuses the original
    q_proj/k_proj/v_proj/o_proj and the caller-supplied apply_rotary_pos_emb
    -- never reimplements either, so a Stage 3 run's RoPE/projection math is
    always identical to the real model's.
    """

    def __init__(self, original: nn.Module, *, layer_idx: int,
                 apply_rotary_pos_emb, state: _ModeState):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        self.head_dim = original.head_dim
        self._apply_rotary_pos_emb = apply_rotary_pos_emb
        self._state = state

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        state = self._state
        cfg = replace(state.cfg_template, seq_len=q.shape[2], batch=q.shape[0])

        if state.mode == "decode":
            # Full attention over the cache, regardless of cfg.mask. The new
            # token legitimately attends to every cached position, and the
            # sparsity decision is made once during prefill -- see
            # _ModeState.decode_backend.
            prior = state.decode_states[self.layer_idx]
            out, new_state = state.decode_backend.decode_step(
                q, k, v, prior, replace(cfg, mask="causal"))
            state.decode_states[self.layer_idx] = new_state
        elif state.mode == "score":
            if state.score_source == "minference_meanpool":
                # The cheap arm. No S x S matrix is ever formed for the
                # SCORES -- that is the point of the estimator, and the
                # mean-pool below is the only thing charged to it.
                layer_scores = minference_meanpool_scores(
                    q, k, n_heads_kv=cfg.n_heads_kv,
                    block_size=cfg.block_size or 64)
                # But this layer must still emit a REAL attention output.
                # An earlier version returned torch.zeros_like(q) on the
                # theory that the scoring pass's output is unused. It is not:
                # layer L's q and k come from the hidden states layers < L
                # produced, so a zeroed output corrupts the activations every
                # later layer is scored against. Measured on a 4-layer toy
                # model, max|zeros - real| on the input hidden states was 0.0
                # at layer 0 and 1.79 / 2.21 / 3.21 at layers 1 / 2 / 3 --
                # and the resulting scores stayed finite, normalised and
                # entirely plausible, so nothing downstream would have
                # flagged them. The cheap arm would simply have looked worse
                # than it is, which in THIS experiment reads as "the oracle
                # matters": a false positive of exactly the shape
                # score_cache.cache_key's collision would have produced in
                # the other direction.
                #
                # Computed with flash SDPA, not the chunked dense-oracle
                # path: O(seq_len) memory, and its cost is NOT charged to the
                # estimator. That accounting is the honest one -- a real
                # deployment gets this forward for free, since it is the
                # attention the layer was going to compute anyway and only
                # the block-ranking is extra work. The dense arm charges its
                # equivalent output to the oracle because there the output
                # and the scores are the same computation (see
                # _scoring_forward_chunked).
                if state.score_output_backend is None:
                    state.score_output_backend = SDPABackend("flash")
                out = state.score_output_backend.forward(
                    q, k, v, replace(cfg, mask="causal"))
            else:
                out, layer_scores = _scoring_forward_chunked(
                    q, k, v, block_size=cfg.block_size or 64,
                    chunk_blocks=state.chunk_blocks)
            state.scores[self.layer_idx] = layer_scores
        else:
            mask = None
            if cfg.mask == "block_sparse":
                finest = state.scores[self.layer_idx]
                pooled = pool_scores_to_block_size(
                    finest, finest_block_size=state.finest_block_size,
                    target_block_size=cfg.block_size)
                # mean over KV heads -> the single (n_blocks, n_blocks)
                # ranking masks.mask_for expects (no per-head masks: this
                # study never varies the sparsity pattern per head, matching
                # masks.py's own stated convention).
                importance = pooled.mean(dim=0)
                mask = masks.mask_for(cfg, importance_scores=importance)
            out = state.backend.forward(q, k, v, cfg, mask=mask)
            if state.capture_decode_state:
                # k/v here are un-expanded (B, n_heads_kv, S, D), which is
                # what state_from_prefill wants and what keeps the cache-size
                # measurement honest -- see SDPABackend.state_from_prefill.
                state.decode_states[self.layer_idx] = (
                    state.decode_backend.state_from_prefill(k, v, cfg))

        out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        out = self.o_proj(out)
        # Tuple arity the calling decoder layer unpacks depends on the
        # installed transformers version -- see
        # _TRANSFORMERS_SELF_ATTN_RETURNS_3TUPLE above. Neither slot beyond
        # `out` is ever real: this wrapper computes no attention weights, and
        # its decode cache lives in _ModeState.decode_states rather than in
        # transformers' own past_key_values (which this deliberately ignores
        # -- two caches, one of them stale, is worse than none).
        if _TRANSFORMERS_SELF_ATTN_RETURNS_3TUPLE:
            return out, None, None
        return out, None


@dataclass(frozen=True)
class GenerationResult:
    """What one greedy decode produced, and under what contract.

    `stop_reason` and `decode_backend` travel with the text for the same
    reason `check_kind` travels with a correctness verdict: the standard a
    result was produced under cannot be reconstructed from the result. A
    prediction that hit its cap and one that terminated on a newline are
    different events, and only one of them is a model failing to answer.
    """

    token_ids: list
    stop_reason: str        # eos | newline | cap
    n_generated: int
    decode_backend: str
    prefill_backend: str

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "cap"


class SwappableAttentionModel:
    """Wraps a Llama-family HF causal LM so every layer's attention can be
    swapped to any AttentionBackend, for one example at a time.
    """

    def __init__(self, model: nn.Module, cfg_template: AttnConfig,
                 *, model_id: str, finest_block_size: int = 64,
                 chunk_blocks: int = 4,
                 score_source: str = "dense_softmax_fp32"):
        _assert_logits_kwarg_accepted(model)
        self.model = model
        self.model_id = model_id
        self.finest_block_size = finest_block_size
        self._state = _ModeState(cfg_template=cfg_template, chunk_blocks=chunk_blocks,
                                  finest_block_size=finest_block_size,
                                  score_source=score_source)
        self._layers = _require_llama_family(model)
        self._originals = [layer.self_attn for layer in self._layers]
        self.n_layers = len(self._layers)
        self.n_heads_kv = model.config.num_key_value_heads
        self._wrapped = False
        self._wrap()

    def _wrap(self) -> None:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        for i, layer in enumerate(self._layers):
            layer.self_attn = SwappedAttention(
                self._originals[i], layer_idx=i,
                apply_rotary_pos_emb=apply_rotary_pos_emb, state=self._state)
        self._wrapped = True

    def unwrap(self) -> None:
        """Restore the original self_attn modules so the model can be
        reused clean after a Stage 3 run."""
        if not self._wrapped:
            return
        for layer, original in zip(self._layers, self._originals):
            layer.self_attn = original
        self._wrapped = False

    def compute_importance_scores(self, input_ids: torch.Tensor, *, task: str,
                                   example_id: str, cache_dir: str,
                                   ) -> dict[int, torch.Tensor]:
        """Per-layer (n_heads_kv, n_blocks, n_blocks) importance rankings for
        this example, at self.finest_block_size. Cache-checked first
        (keyed on model+task+example+seq_len, covering every layer in one
        entry); only runs the dense scoring pass on a miss.
        """
        # Before anything is keyed or computed: a run whose score_source
        # claims fp32 must actually be getting fp32. See numerics.py -- the
        # oracle's QK^T is TF32-eligible on Ampere and later, and the flag
        # that decides it is a global whose default has moved between torch
        # versions this project's pin permits.
        numerics.assert_fp32_matmul(self._state.score_source)

        seq_len = input_ids.shape[-1]
        key = score_cache.cache_key(self.model_id, task, example_id, seq_len,
                                     self._state.score_source)
        cached = score_cache.load(cache_dir, key, expected_n_layers=self.n_layers,
                                   expected_n_heads_kv=self.n_heads_kv)
        if cached is not None:
            return {i: cached[i] for i in range(self.n_layers)}

        self._state.mode = "score"
        self._state.cfg_template = replace(
            self._state.cfg_template, block_size=self.finest_block_size)
        self._state.scores = {}
        with torch.no_grad():
            # use_cache=False: SwappedAttention never returns a real KV
            # cache (no incremental-decode support -- see its forward()),
            # so leaving use_cache at the model's default (True for a causal
            # LM) makes the outer model try to convert a None cache to
            # legacy format and crash with AttributeError.
            #
            # logits_to_keep=1 (see _LOGITS_TO_KEEP_KWARG above): this call's
            # only real purpose is populating self._state.scores as a side
            # effect of the forward pass -- .logits is never read -- so
            # there is no reason to pay for lm_head over every position.
            self.model(input_ids=input_ids, use_cache=False,
                      **{_LOGITS_TO_KEEP_KWARG: 1})
        per_layer = self._state.scores
        stacked = torch.stack([per_layer[i] for i in range(self.n_layers)], dim=0)
        score_cache.save(cache_dir, key, stacked)
        return per_layer

    def run_measured(self, input_ids: torch.Tensor, backend: AttentionBackend,
                      *, cfg: AttnConfig, layer_scores: Optional[dict] = None,
                      logits_to_keep: int = 0,
                      decode_backend: Optional[AttentionBackend] = None):
        """Run the model with every layer's attention going through
        `backend`, using `cfg` (mask/sparsity/block_size/dtype) and, for
        block_sparse configs, the per-layer importance rankings from
        compute_importance_scores.

        `logits_to_keep` is forwarded as-is to transformers' own convention
        (0 == every position, matching its default): full-sequence logits
        are the right choice for correctness testing (test_a/b/c/d compare
        every position at near-machine precision, deliberately -- a bug
        affecting only an interior position's causal masking must still be
        caught, not just the last one). A real generation caller (RULER
        only ever needs the next token from the final position) should pass
        `logits_to_keep=1` explicitly -- see time_one_accuracy_example.py --
        rather than this method silently restricting it, since materializing
        logits over the full context at a ~150K vocab is exactly what OOM'd
        at 32K context on Qwen2.5-1.5B (11+ GiB) when nothing needed it.
        """
        self._state.mode = "measured"
        self._state.backend = backend
        self._state.cfg_template = cfg
        self._state.decode_backend = decode_backend
        self._state.capture_decode_state = decode_backend is not None
        self._state.decode_states = {}
        if layer_scores is not None:
            self._state.scores = layer_scores
        try:
            with torch.no_grad():
                return self.model(input_ids=input_ids, use_cache=False,
                                  **{_LOGITS_TO_KEEP_KWARG: logits_to_keep})
        finally:
            self._state.capture_decode_state = False

    def generate(self, input_ids: torch.Tensor, backend: AttentionBackend, *,
                 cfg: AttnConfig, max_new_tokens: int,
                 eos_token_ids: frozenset = frozenset(),
                 newline_token_ids: frozenset = frozenset(),
                 whitespace_token_ids: frozenset = frozenset(),
                 layer_scores: Optional[dict] = None,
                 decode_backend: Optional[AttentionBackend] = None,
                 ) -> "GenerationResult":
        """Greedy decode from a prefilled prompt.

        Deterministic by construction -- argmax, no sampling, no temperature.
        Sampling noise would swamp the effect Stage 3 measures (whether sparse
        attention changes the answer) and would make a re-run disagree with
        itself, so there is no knob for it.

        `decode_backend` defaults to `backend` where the backend has its own
        decode path and must be named explicitly where it does not. Nothing
        falls back silently: `block_sparse` decodes densely by design, and
        which backend generated a row's text is recorded on the row.

        Stopping: EOS, or any token whose text contains a newline, or
        `max_new_tokens`. The newline stop is load-bearing rather than
        belt-and-braces -- RULER prompts are completion-style and end
        mid-sentence, so an instruct model's EOS (`<|im_end|>`, which closes
        an assistant turn that never opened) is essentially never emitted.
        Without it every example would exit on the cap, and "hit its cap"
        would carry no information about any backend. See
        docs/stage3_generation_decision.md.

        The newline stop does not arm until a non-whitespace token has been
        produced (`whitespace_token_ids`). Without that, a model that opened
        with a newline would stop having generated nothing, and every row of
        every backend would score 0 for a reason unrelated to attention.
        accuracy/stopping.first_stop_index is the pure statement of this rule
        and tests hold the two to the same answer.
        """
        if decode_backend is None:
            if not type(backend).supports_decode():
                raise UnsupportedModelArchitecture(
                    f"{backend.name} has no decode path, so generation needs "
                    f"an explicit decode_backend. This is deliberate: the "
                    f"choice changes what a row means and is recorded on it.")
            decode_backend = backend

        prefill_len = int(input_ids.shape[-1])
        out = self.run_measured(input_ids, backend, cfg=cfg,
                                layer_scores=layer_scores, logits_to_keep=1,
                                decode_backend=decode_backend)
        next_id = int(out.logits[0, -1].argmax())

        generated: list[int] = []
        stop_reason = "cap"
        seen_content = False
        self._state.mode = "decode"
        # q_len=1 per step. The cfg the decode path sees keeps the prompt's
        # head geometry and dtype; only the query length changes.
        decode_cfg = replace(cfg, mask="causal")
        try:
            for step in range(max_new_tokens):
                generated.append(next_id)
                if next_id in eos_token_ids:
                    stop_reason = "eos"
                    break
                if next_id in newline_token_ids and seen_content:
                    stop_reason = "newline"
                    break
                if next_id not in newline_token_ids and next_id not in whitespace_token_ids:
                    seen_content = True
                if len(generated) >= max_new_tokens:
                    stop_reason = "cap"
                    break
                # position_ids is not optional here. With use_cache=False and
                # a one-token input, transformers derives position 0, so RoPE
                # would rotate every generated token as though it were the
                # start of the sequence. No error, plausible text, wrong
                # output -- the same shape as is_causal on a single query.
                pos = prefill_len + len(generated) - 1
                step_ids = torch.tensor([[next_id]], device=input_ids.device)
                position_ids = torch.tensor([[pos]], device=input_ids.device)
                with torch.no_grad():
                    out = self.model(input_ids=step_ids,
                                     position_ids=position_ids, use_cache=False,
                                     **{_LOGITS_TO_KEEP_KWARG: 1})
                next_id = int(out.logits[0, -1].argmax())
        finally:
            self._state.mode = "measured"

        return GenerationResult(token_ids=generated, stop_reason=stop_reason,
                                n_generated=len(generated),
                                decode_backend=decode_backend.name,
                                prefill_backend=backend.name)
