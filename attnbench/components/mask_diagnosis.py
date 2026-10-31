"""Live/live and live/bank mask comparisons, by arm and layer (researcher's
request, 31 Oct 2026, after Shakedown 3). Findings only: no gate reads them.

Shakedown 3 failed two mask checks:

  - the T2.4 replay (meanpool, 8192): "the replay's live masks differ from
    the bank", with per-call agreement 0.84985 in all 28 layers. In the
    bank each layer keeps 7380 of 49152 entries, and 1 - 7380/49152 is
    0.84985: the live masks read back were all False, in every layer;
  - T2.12's mask replay (meanpool): "the replay's executed density differs
    from its source's", with the logits bitwise equal.

`replay_diagnosis` compares, per layer, the masks the kernel was handed:

  - live/live: two eager runs of the arm;
  - live/bank: an eager run against the recording's bank;
  - tape/bank: the replay's own path without a graph (`EagerBackend`);
  - on a GPU, the graph's live masks read before any replay (as the T2.4
    step and T3.1's device step read them), after one replay, and a copy
    taken inside the graph, after one replay.

`t212_diagnosis` does the same for T2.12: the source arm's handed masks
twice (live/live), against its recorded mask set (live/bank), the
precomputed arm's handed masks against the source's, and whether each
backend keeps the density record the step compares (the step reports each
record's length at the comparison).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from . import mask_bank, replay

KERNEL_ATTRS = ("base_blockmask",)


def guarded(fn: Callable[..., dict], *args, **kwargs) -> dict:
    """The diagnosis, or the error that stopped it: a finding never fails a
    step."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:                  # noqa: BLE001 -- recorded, not raised
        return {"error": f"{type(e).__name__}: {e}"}


def summaries(diag: dict) -> dict:
    """Each comparison's summary (the step's findings; the file keeps the rest)."""
    return {k: (v["summary"] if isinstance(v, dict) and "summary" in v else v)
            for k, v in diag.items()}


def per_layer(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> list[dict]:
    """Per call (layer): bitwise identical, share of equal entries, and each
    side's kept count. Shapes must match after broadcasting `b` to `a`."""
    if len(a) != len(b):
        return [{"error": f"{len(a)} masks against {len(b)}"}]
    rows = []
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = x.detach().cpu().bool(), y.detach().cpu().bool()
        if x.shape != y.shape:
            try:
                y = y.expand_as(x)
            except RuntimeError:
                rows.append({"layer": i, "error": f"shapes {tuple(x.shape)} and {tuple(y.shape)}"})
                continue
        rows.append({"layer": i, "identical": bool(torch.equal(x, y)),
                     "agreement": float((x == y).float().mean()),
                     "kept_a": int(x.sum()), "kept_b": int(y.sum())})
    return rows


def summary(rows: Sequence[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    return {"layers": len(rows), "identical_layers": sum(r["identical"] for r in ok),
            "errors": [r["error"] for r in rows if "error" in r],
            "a_all_false_layers": sum(r["kept_a"] == 0 for r in ok),
            "min_agreement": min((r["agreement"] for r in ok), default=None)}


def handed(run: Callable[[], None], kernel_module=None) -> list[torch.Tensor]:
    """Copies of the mask the kernel was handed at each call, in one run."""
    with mask_bank.MaskCapture(module=kernel_module, copy=True) as cap:
        run()
    return [m.detach().clone() for m in cap.masks()]


def _both(rows: list[dict]) -> dict:
    return {"summary": summary(rows), "by_layer": rows}


def replay_diagnosis(run_prefill: Callable[[], None], rec, *, device,
                     graph_backend=None, kernel_module=None) -> dict:
    """Module docstring. `rec` is the arm's recording (tape and bank)."""
    if rec.masks is None:
        return {"masks": None}
    bank = rec.masks.tensors(device)
    live1, live2 = handed(run_prefill, kernel_module), handed(run_prefill, kernel_module)
    out = {"live_live": _both(per_layer(live1, live2)),
           "live_bank": _both(per_layer(live1, bank))}
    cap = replay.capture(run_prefill, rec.tape, device=device, masks=rec.masks,
                         backend=replay.EagerBackend(), kernel_module=kernel_module)
    out["tape_eager_bank"] = _both(per_layer(cap.live_masks, bank))
    if graph_backend is not None:
        cap = replay.capture(run_prefill, rec.tape, device=device, masks=rec.masks,
                             backend=graph_backend, kernel_module=kernel_module)
        cap.backend.synchronize()
        before = [m.detach().clone() for m in cap.live_masks]
        cap.replay()
        cap.backend.synchronize()
        out["graph_before_replay_bank"] = _both(per_layer(before, bank))
        out["graph_after_replay_bank"] = _both(per_layer(cap.live_masks, bank))
        cap = replay.capture(run_prefill, rec.tape, device=device, masks=rec.masks,
                             backend=graph_backend, kernel_module=kernel_module,
                             copy_live=True)
        cap.replay()
        cap.backend.synchronize()
        out["graph_copy_after_replay_bank"] = _both(per_layer(cap.live_masks, bank))
    return out


def t212_diagnosis(run_source: Callable[[], None], run_precomputed: Callable[[], None],
                   recorded: Sequence[torch.Tensor], *, source_backend,
                   precomputed_backend, kernel_module=None) -> dict:
    """Module docstring. `recorded` is the mask set's masks in layer order."""
    src1, src2 = handed(run_source, kernel_module), handed(run_source, kernel_module)
    pre = handed(run_precomputed, kernel_module)
    rec = [m if m.dim() == src1[0].dim() else m.unsqueeze(0) for m in recorded] if src1 else []
    return {"live_live": _both(per_layer(src1, src2)),
            "live_bank": _both(per_layer(src1, rec)),
            "precomputed_vs_source": _both(per_layer(pre, src1)),
            "source_keeps_density_record": hasattr(source_backend, "last_layer_density"),
            "precomputed_keeps_density_record": hasattr(precomputed_backend,
                                                        "last_layer_density")}
