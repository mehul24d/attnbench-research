"""XAttention (Xu et al., ICML 2025; arXiv:2503.16428) as a backend.

The audit's C1 asked for the sparse method's OWN estimator, run as a deployed
system would, on the same kernel family as this study's block_sparse arm.
XAttention is that: its estimator (antidiagonal scoring) selects blocks per
head inside each layer's forward, then calls the same
mit-han-lab/Block-Sparse-Attention `block_sparse_attn_func` this study's
`block_sparse` backend calls.

**Official code, not a reimplementation.** The estimator is
`xattn.src.Xattention.xattn_estimate`, imported from mit-han-lab/x-attention
installed on the instance at a pinned commit (see `XATTN_COMMIT`). The
repository carries no licence, so nothing of it is copied here: it is
installed and called, never redistributed. The only code below that is not a
call is the kernel invocation `Xattention_prefill` makes after the estimate,
reproduced so the selected mask can be kept for density accounting --
`Xattention_prefill` discards it. A CUDA test (Phase A gate) asserts this
backend's output is bitwise identical to `Xattention_prefill`'s on the same
inputs, so the reproduction cannot drift from the original unnoticed.

**Settings follow the method's own evaluation** (`eval/LongBench/pred.py` at
the pinned commit): stride 8, `norm=1`, `keep_sink=True`, `keep_recent=True`,
`use_triton=True` (the official code itself falls back to torch on devices
whose name lacks "100", e.g. the L4, and prints that it did -- on every call,
so once per layer per example; this backend makes the same decision once,
from the same device-name test, and passes the result, which takes the same
code path without the print), KV heads
repeated to the query-head count (`repeat_kv`, which `xattn_estimate`
asserts). One deliberate deviation: that evaluation uses per-layer thresholds
profiled for Llama-3.1-8B (`xattn/threshold/llama_threshold.py`); none exist
for Qwen2.5, so this study uses a scalar threshold tau, which
`Xattention_prefill` accepts, swept over a pre-registered grid. Recorded in
`docs/limitations.md`.

**Sparsity is an outcome, not a setting.** Selection is by cumulative
attention mass >= tau per head, so the fraction of blocks kept varies with
the input. Each forward records the realised density per layer
(`last_layer_density`); budget-matched comparisons against the top-k arms use
it, not tau.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from ..config import AttnConfig
from .base import AttentionBackend, Capability, UnsupportedConfig, register
from .impls import _expand_kv

XATTN_REPO = "https://github.com/mit-han-lab/x-attention"
XATTN_COMMIT = "e37988770b9d1bebd489eba011d615f35587ba08"
XATTN_BLOCK_SIZE = 128          # Xattention_prefill asserts it; also BSA's hardcoded block


def official_chunk_size(k_len: int) -> int:
    """`Xattention_prefill`'s default chunk_size, verbatim in arithmetic."""
    p = 1 << (k_len - 1).bit_length()
    return int(max(min(max(2048, p), 128 * 1024 * 2048 // p), 2048))


def triton_for_device(device_name: str, requested: bool = True) -> bool:
    """Whether `xattn_estimate` would actually use Triton on this device: the
    official test, verbatim -- the device name must contain "100". An L4
    fails it and runs the torch path, so its estimator cost there is not the
    method's official cost (docs/t4_xattention_pilot.md)."""
    return bool(requested) and "100" in device_name


def installed_checkout() -> tuple[Optional[str], Optional[bool]]:
    """(HEAD commit, dirty) of the x-attention checkout `xattn` imports from.

    The package is installed editable from a git checkout (its `xattn/src`
    has no `__init__.py`, so a plain install omits the estimator), so the
    commit is read from that checkout rather than trusted from the install
    command. (None, None) when it is not a git checkout at all.
    """
    import xattn
    root = Path(xattn.__file__).resolve().parent.parent

    def git(*args):
        r = subprocess.run(["git", "-C", str(root), *args],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    if head is None:
        return None, None
    status = git("status", "--porcelain", "--untracked-files=no")
    return head, (status is None or status != "")


def density_tensor(simple_mask: torch.Tensor, q_block_num: int,
                   k_block_num: int) -> torch.Tensor:
    """`realised_density` as a 0-dim tensor on the mask's device, so a
    forward can record it without a host sync inside the timed region."""
    m = simple_mask[0, :, :q_block_num, :k_block_num]
    valid = torch.ones(q_block_num, k_block_num, dtype=torch.bool,
                       device=m.device).tril(diagonal=k_block_num - q_block_num)
    return (m & valid).sum() / (valid.sum() * m.shape[0])


def realised_density(simple_mask: torch.Tensor, q_block_num: int, k_block_num: int) -> float:
    """Fraction of causally valid blocks kept, over heads. The mask's rows
    beyond the causal triangle are excluded: `keep_sink` sets query block 0's
    whole row, and those future blocks are masked again in-kernel by
    `is_causal`, so counting them would overstate the work done."""
    return float(density_tensor(simple_mask, q_block_num, k_block_num))


@dataclass(frozen=True)
class ThresholdTable:
    """Per-(layer, query head) thresholds, as XAttention's own RULER
    evaluation uses them (`llama_fuse_8`, a 32x32 table for Llama-3.1-8B).

    Built by `scripts/calibrate_xattn_thresholds.py` with the method's own
    profiler and committed before any row uses it. `sha256` is over the
    values alone, so a row's `xattn_calibration` names exactly the numbers
    that ran, whatever else the file says.
    """

    name: str
    values: tuple            # n_layers tuples of n_heads floats

    @staticmethod
    def digest(values) -> str:
        canon = json.dumps([[round(float(x), 8) for x in row] for row in values])
        return hashlib.sha256(canon.encode()).hexdigest()

    @property
    def sha256(self) -> str:
        return self.digest(self.values)

    @property
    def label(self) -> str:
        return f"{self.name}:{self.sha256[:12]}"

    @classmethod
    def load(cls, path) -> "ThresholdTable":
        """Read a calibration file, refusing one whose recorded digest does
        not match its values (a hand-edited table is not a calibration)."""
        doc = json.loads(Path(path).read_text())
        values = tuple(tuple(float(x) for x in row) for row in doc["thresholds"])
        if not values or len({len(r) for r in values}) != 1:
            raise ValueError(f"{path}: thresholds must be a non-empty rectangular table")
        if any(not 0.0 <= x <= 1.0 for r in values for x in r):
            raise ValueError(f"{path}: every threshold must be in [0, 1]")
        table = cls(name=str(doc["name"]), values=values)
        if doc.get("sha256") != table.sha256:
            raise ValueError(f"{path}: recorded sha256 {doc.get('sha256')!r} does not "
                             f"match its thresholds ({table.sha256})")
        return table


@register
class XAttentionBackend(AttentionBackend):
    capability = Capability(
        name="xattention",
        family="sparse",
        min_compute_capability=(8, 0),
        supports_backward=False,
        supports_gqa=True,           # via KV repetition, as the method's own eval does
        supports_causal=True,
        supports_block_sparse=False,  # selects its own blocks; takes no external mask
        block_sizes=(XATTN_BLOCK_SIZE,),
        head_dims=(64, 128),
        dtypes=("bfloat16", "float16"),
        notes=(f"Official estimator from {XATTN_REPO} at {XATTN_COMMIT[:7]} "
               "(no licence: installed, not vendored) plus Block-Sparse-"
               "Attention. Per-head, threshold-selected blocks; realised "
               "density recorded per layer. Batch 1, block 128, prefill only."),
    )

    def __init__(self, threshold: Union[float, ThresholdTable] = 0.9, stride: int = 8, *,
                 keep_sink: bool = True, keep_recent: bool = True,
                 use_triton: bool = True):
        if isinstance(threshold, ThresholdTable):
            self.threshold = threshold
        else:
            if not 0.0 < threshold <= 1.0:
                raise ValueError(f"threshold must be in (0, 1], got {threshold}")
            self.threshold = float(threshold)
        # The layer being computed, set by the model wrapper before each
        # prefill layer. A table needs it; a scalar ignores it.
        self.layer_idx: Optional[int] = None
        self._table_cache: dict = {}
        self.stride = int(stride)
        self.keep_sink = keep_sink
        self.keep_recent = keep_recent
        self.use_triton = use_triton
        # Decided once, on the first CUDA call, by the official device test.
        self.triton_effective: Optional[bool] = None
        self.last_layer_density: list[torch.Tensor] = []

    @staticmethod
    def _import_check():
        import block_sparse_attn  # noqa: F401
        import xattn.src.Xattention  # noqa: F401

    def reset_density(self) -> None:
        self.last_layer_density = []

    def set_layer(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def _threshold_for(self, q: torch.Tensor, n_heads: int):
        """The scalar, or this layer's row of the table as a per-head tensor
        on q's device -- the form `Xattention_prefill` receives in the
        authors' RULER path (`threshold[layer_idx]`)."""
        if not isinstance(self.threshold, ThresholdTable):
            return self.threshold
        if self.layer_idx is None:
            raise RuntimeError("a per-layer threshold table needs set_layer() "
                               "before each forward")
        if self.layer_idx >= len(self.threshold.values):
            raise ValueError(f"layer {self.layer_idx} beyond the table's "
                             f"{len(self.threshold.values)} layers")
        row = self.threshold.values[self.layer_idx]
        if len(row) != n_heads:
            raise ValueError(f"table has {len(row)} heads per layer; the model "
                             f"has {n_heads} query heads")
        key = (self.layer_idx, str(q.device))
        if key not in self._table_cache:
            self._table_cache[key] = torch.tensor(row, dtype=torch.float32, device=q.device)
        return self._table_cache[key]

    def estimate(self, q: torch.Tensor, k_rep: torch.Tensor) -> torch.Tensor:
        """The official estimator, with the arguments `Xattention_prefill`
        passes it. Returns the (1, H, q_blocks_padded, k_blocks_padded) mask."""
        from xattn.src.Xattention import xattn_estimate
        use_triton = self.use_triton
        if q.device.type == "cuda":
            if self.triton_effective is None:
                self.triton_effective = triton_for_device(
                    torch.cuda.get_device_properties(q.device).name, self.use_triton)
            use_triton = self.triton_effective
        _, simple_mask = xattn_estimate(
            q, k_rep, block_size=XATTN_BLOCK_SIZE, stride=self.stride, norm=1,
            threshold=self._threshold_for(q, q.shape[1]), select_mode="inverse",
            use_triton=use_triton, causal=True,
            chunk_size=official_chunk_size(k_rep.shape[2]), kdb=1,
            keep_sink=self.keep_sink, keep_recent=self.keep_recent)
        return simple_mask

    def forward(self, q, k, v, cfg: AttnConfig, mask=None):
        if mask is not None:
            raise UnsupportedConfig("xattention selects its own blocks; got an external mask")
        if cfg.mask != "causal":
            raise UnsupportedConfig(f"xattention: mask kind {cfg.mask!r} (expects 'causal')")
        if q.shape[0] != 1:
            raise UnsupportedConfig("xattention: batch 1 only (Xattention_prefill asserts it)")

        k_rep, v_rep = _expand_kv(k, v, cfg)
        simple_mask = self.estimate(q, k_rep)

        from block_sparse_attn import block_sparse_attn_func

        _, h, s, d = q.shape
        nb = -(-s // XATTN_BLOCK_SIZE)
        # Kept on the device: a .item() here would sync the stream once per
        # layer inside the timed prefill. Read after the timer stops
        # (generation.generate_one).
        self.last_layer_density.append(density_tensor(simple_mask, nb, nb))

        # From here to the return: Xattention_prefill's own kernel call, same
        # arguments, same order. Bitwise equality with it is a CUDA test.
        qf = q.transpose(1, 2).reshape(s, h, d)
        kf = k_rep.transpose(1, 2).reshape(s, h, d)
        vf = v_rep.transpose(1, 2).reshape(s, h, d)
        cu = torch.tensor([0, s], dtype=torch.int32, device=q.device)
        head_mask_type = torch.ones(h, dtype=torch.int32, device=q.device)
        out = block_sparse_attn_func(
            qf, kf, vf, cu, cu, head_mask_type, None,
            simple_mask[:, :, :nb, :nb].contiguous(), s, s,
            p_dropout=0.0, deterministic=True, is_causal=True)
        return out.view(1, s, h, d).transpose(1, 2)


# ---- the method's own threshold profiler -----------------------------------

PROFILE_MODULE = "xattn.threshold.profile_threshold.profile_threshold"


def _official_profiler():
    """`xattn_prefill_profile` and `ProfileConfig` from the pinned checkout.

    The module imports `flash_attn` at load for one purpose: the layer
    output its profiler returns, which this study replaces with its own
    (`ThresholdProfiler.forward`). When flash-attn is not installed, a
    stand-in that returns None is registered for the import only, and the
    profiler reports that it did. Nothing it computes passes through it.
    """
    stubbed = False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        stub = types.ModuleType("flash_attn")
        stub.flash_attn_func = lambda *a, **k: None
        sys.modules["flash_attn"] = stub
        stubbed = True
    import importlib
    mod = importlib.import_module(PROFILE_MODULE)
    return mod, stubbed


class ThresholdProfiler:
    """XAttention's per-head threshold calibration, run by the method's own
    code inside each layer of this study's model wrapper.

    Per text and per layer, the official `xattn_prefill_profile`:
      1. computes each head's exact block attention mass;
      2. finds, per query block, the fewest key blocks covering 90% of it;
      3. returns the threshold on XAttention's own estimate that would
         select at least those blocks, one value per head.
    The table is then the maximum over texts, as the official script's
    `final_threshold` is. Nothing of that computation is reimplemented
    here; this class only hands it each layer's q and k and collects what
    it records. The layer output is dense causal SDPA, so later layers see
    the hidden states the real model would.
    """

    def __init__(self, n_layers: int, stride: int = 8):
        self.n_layers = n_layers
        self.stride = stride
        self.layer_idx: Optional[int] = None
        self.per_text: list[torch.Tensor] = []     # each (n_layers, n_heads)
        self._current: Optional[list] = None
        self.flash_attn_stubbed: Optional[bool] = None
        self._dense = None

    def set_layer(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def start_text(self) -> None:
        self._current = [None] * self.n_layers

    def end_text(self) -> None:
        if self._current is None or any(t is None for t in self._current):
            raise RuntimeError("a text ended before every layer was profiled")
        self.per_text.append(torch.stack(self._current))
        self._current = None

    def forward(self, q, k, v, cfg: AttnConfig, mask=None):
        if self._current is None or self.layer_idx is None:
            raise RuntimeError("start_text() and set_layer() come before forward")
        mod, stubbed = _official_profiler()
        # Decided by the first call: after it, a registered stand-in imports
        # cleanly and would read as the real flash-attn.
        if self.flash_attn_stubbed is None:
            self.flash_attn_stubbed = stubbed
        k_rep, v_rep = _expand_kv(k, v, cfg)
        shim = types.SimpleNamespace(layer_idx=0,
                                     profile_config=mod.ProfileConfig(stride=self.stride))
        mod.xattn_prefill_profile(shim, query_states=q, key_states=k_rep,
                                  value_states=v_rep, block_size=XATTN_BLOCK_SIZE,
                                  stride=self.stride)
        (per_head,) = shim.profile_config.history_threshold
        self._current[self.layer_idx] = per_head.reshape(-1).float().cpu()
        if self._dense is None:
            from .impls import SDPABackend
            self._dense = SDPABackend("flash" if q.is_cuda else "math")
        return self._dense.forward(q, k, v, cfg)

    def table(self) -> torch.Tensor:
        """(n_layers, n_heads): the maximum over texts."""
        if not self.per_text:
            raise RuntimeError("no text was profiled")
        return torch.stack(self.per_text).amax(dim=0)

