"""Audit C4: `components/op_sequence.py`, `trace_recording.py` and
`mask_bank.py` against tasks T2.3 and T2.4 (and the Section 5.1 definitions
they implement). Each test fails on main at c93d26d and demonstrates one
finding. The file is not named test_*.py, so the default suite does not
collect it; run it by name.

None of these tests computes with torch. On main `import attnbench` and
`mask_bank` import torch at module level, so in a CPU container without torch
they ran through a scratch loader (the package __init__ skipped, and an
empty `torch` module for `mask_bank`'s import line).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from attnbench.components import mask_bank as mb
from attnbench.components.op_sequence import (CodePath, OpSequence, SeqKind, SeqOp,
                                              to_operations)

PLAN = Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_PLAN.md"
CP = CodePath(arm="xattn", gpu="NVIDIA L4", implementation="xattn:torch")


# ---------------------------------------------------------------------------
# Finding 1 (medium). Section 5.1: "An implicit synchronisation inside an
# allocator call such as `cudaFree` is not costed: it is a trace check, and a
# prefill trace that shows one is flagged (T2.3; Appendix A, 25 Oct 2026)."
# The T2.3 row: "Waits the recursion does not price yet (a blocking copy from
# the host, an event synchronise) are named and refused; an implicit wait in
# `cudaFree` is flagged." op_sequence puts "implicit" in UNPRICED_MECHANISMS
# (op_sequence.py:135-136), so to_operations refuses the whole sequence
# (op_sequence.py:939-942) with "a researcher decides its cost": the decision
# already made (not costed) is not applied, and the prediction for that arm
# cannot be made at all.

def test_an_implicit_cudafree_wait_is_not_costed_and_does_not_block_the_prediction():
    ops = (SeqOp(SeqKind.KERNEL, api="cudaLaunchKernel", kernels=("k1",), stream=7),
           SeqOp(SeqKind.SYNC, api="cudaFree", mechanism="implicit"),
           SeqOp(SeqKind.KERNEL, api="cudaLaunchKernel", kernels=("k2",), stream=7))
    seq = OpSequence(code_path=CP, context_length=2048, batch=1, ops=ops, streams=(7,))
    out = to_operations(seq, host={0: 1.0, 2: 1.0}, device={0: 10.0, 2: 10.0}, link={})
    # Not costed: the recursion sees the two kernels and nothing for the wait.
    assert [o.kind.value for o in out] == ["kernel", "kernel"]


# ---------------------------------------------------------------------------
# Finding 2 (low). T2.4 banks "the estimator's selection (its masks on each
# recorded prompt)" (Section 5.1), and the mask gates (T2.23, T2.24) and the
# density difference of T2.28 read it. mask_bank.load (mask_bank.py:262-286)
# checks the packed masks' sha256s, the npz's and the combined one, but the
# manifest's per-call `density` and `causal` are covered by none of them and
# are not recomputed from the mask: an edited or stale manifest loads with a
# density the mask does not have.

def _bank(tmp_path) -> Path:
    a = np.tril(np.ones((1, 2, 4, 4), dtype=bool))       # causal, every valid block kept
    packed = np.packbits(a.reshape(-1))
    m = mb.BankedMask(tuple(a.shape), True, mb.density(a, True),
                      hashlib.sha256(packed.tobytes()).hexdigest(), packed)
    assert m.density == 1.0
    base = tmp_path / "bank"
    mb.save(mb.MaskRecord(CP, 512, 1, "prompt-0", (m,)), base)
    return base


@pytest.mark.parametrize("field, value", [("density", 0.5), ("causal", False)])
def test_a_manifest_whose_fields_do_not_follow_from_the_masks_is_refused(tmp_path, field, value):
    base = _bank(tmp_path)
    manifest = base.with_suffix(".json")
    meta = json.loads(manifest.read_text())
    meta["calls"][0][field] = value
    manifest.write_text(json.dumps(meta, indent=1))
    with pytest.raises(mb.MaskBankError):
        mb.load(base)


# ---------------------------------------------------------------------------
# Finding 3 (medium; plan ambiguity that changes a result). Section 5.1: "The
# checker classifies each pairing and arm as "same sequence" or "different
# sequence" under a strict definition (primary) and a looser one (secondary,
# reported beside it) (Appendix A, 25 Oct 2026); H0 and H2 compare components
# only within the same sequence under the strict definition." The plan never
# says what the strict definition compares. op_sequence chose (its docstring:
# "Section 5.1 does not define it") to leave kernel names out of `op_key`
# (op_sequence.py:286-303), so an L4 trace and an A100 trace whose GEMMs are
# differently named library kernels are "same sequence". With kernel names in,
# they are "different", and H0 and H2 lose those pairings. The choice decides
# which pairings H0 and H2 are scored on, and it is not the researcher's yet.

def test_the_plan_states_what_the_strict_same_sequence_definition_compares():
    text = PLAN.read_text(encoding="utf-8")
    sentences = [s for s in re.split(r"(?<=[.;])\s+", text)
                 if "strict definition" in s or "strict \"same sequence\"" in s]
    assert sentences, "the plan no longer mentions the strict definition"
    assert any("kernel name" in s for s in sentences), (
        "no sentence that mentions the strict definition says whether kernel "
        "names are compared")
