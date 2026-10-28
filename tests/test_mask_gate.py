"""T2.23 and T2.24: the mask gates, on hand-built fixture banks.

Masks are tiny (one head, 2 x 2 or 3 x 3 blocks) so every agreement and
density below is worked out by hand in its comment. The last test runs real
`mask_bank` records through a save and load (it imports torch, as
`mask_bank` does).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pytest

from attnbench.components import mask_gate as G
from attnbench.components.op_sequence import CodePath
from attnbench.predict.density_sensitivity import _check_delta

L4_TORCH = CodePath(arm="xattention", gpu="NVIDIA L4", implementation="xattn:torch")
A100_TORCH = CodePath(arm="xattention", gpu="NVIDIA A100", implementation="xattn:torch")
A100_TRITON = CodePath(arm="xattention", gpu="NVIDIA A100", implementation="xattn:triton")
H100_TRITON = CodePath(arm="xattention", gpu="NVIDIA H100", implementation="xattn:triton")

FULL = np.array([[[[1, 0], [1, 1]]]], dtype=bool)        # every valid block kept: density 1
DROP = np.array([[[[1, 0], [0, 1]]]], dtype=bool)        # (1, 0) dropped: density 2/3
UPPER = np.array([[[[1, 1], [1, 1]]]], dtype=bool)       # FULL plus an invalid block


@dataclass(frozen=True)
class M:
    a: np.ndarray
    causal: bool = True

    def array(self):
        return self.a


@dataclass(frozen=True)
class R:
    code_path: CodePath
    context_length: int
    batch: int
    prompt_id: str
    masks: tuple


def rec(cp, *layers, prompt="p0", length=256, batch=1):
    return R(cp, length, batch, prompt, tuple(M(x) for x in layers))


LOOSE = G.Bounds(min_agreement=0.5, max_density_difference=0.5)
TIGHT = G.Bounds(min_agreement=1.0, max_density_difference=0.0)


# --- one layer -----------------------------------------------------------------

def test_density_is_kept_over_valid_blocks():
    # Valid blocks under causal: (0,0), (1,0), (1,1).
    assert G.mask_density(FULL, True) == 1.0
    assert G.mask_density(DROP, True) == pytest.approx(2 / 3)
    # The block above the diagonal is not valid, so keeping it changes nothing.
    assert G.mask_density(UPPER, True) == 1.0
    assert G.mask_density(UPPER, False) == 1.0
    assert G.mask_density(FULL, False) == 0.75


def test_agreement_counts_valid_blocks_only():
    # FULL against DROP: they differ at (1, 0), one of 3 valid blocks.
    assert G.valid_agreement(FULL, DROP, True) == pytest.approx(2 / 3)
    # FULL against UPPER: they differ only above the diagonal.
    assert G.valid_agreement(FULL, UPPER, True) == 1.0


def test_a_layer_comparison_by_hand():
    c = G.compare_records(rec(L4_TORCH, FULL, DROP), rec(A100_TRITON, DROP, UPPER))
    first, second = c.layers
    # Layer 0: FULL vs DROP, agreement 2/3, density 1 -> 2/3.
    assert (first.agreement, first.identical) == (pytest.approx(2 / 3), False)
    assert first.density_difference == pytest.approx(-1 / 3)
    # Layer 1: DROP vs UPPER: (1, 0) differs; density 2/3 -> 1.
    assert second.agreement == pytest.approx(2 / 3)
    assert second.density_difference == pytest.approx(1 / 3)
    # Mean density 5/6 on both sides: no shift.
    assert c.mean_density_shift == pytest.approx(0.0)
    assert (c.arm, c.context_length, c.batch, c.prompt_id) == ("xattention", 256, 1, "p0")


def test_identical_is_bitwise_including_the_invalid_blocks():
    c = G.compare_records(rec(L4_TORCH, FULL, FULL), rec(A100_TORCH, FULL, UPPER))
    assert [l.identical for l in c.layers] == [True, False]
    assert [l.agreement for l in c.layers] == [1.0, 1.0]


# --- what is refused ----------------------------------------------------------------

@pytest.mark.parametrize("cand, match", [
    (rec(A100_TRITON, FULL, prompt="p1"), "not the same arm"),
    (rec(A100_TRITON, FULL, length=512), "not the same arm"),
    (rec(A100_TRITON, FULL, FULL), "kernel calls"),
    (rec(L4_TORCH, FULL), "both records are of the code path"),
    (rec(A100_TRITON, np.ones((1, 1, 3, 3), dtype=bool)), "shapes"),
])
def test_records_that_are_not_one_arm_on_one_prompt_are_refused(cand, match):
    with pytest.raises(G.MaskGateError, match=match):
        G.compare_records(rec(L4_TORCH, FULL), cand)


def test_a_causal_flag_that_differs_or_a_non_bool_mask_is_refused():
    other = R(A100_TRITON, 256, 1, "p0", (M(FULL, causal=False),))
    with pytest.raises(G.MaskGateError, match="causal"):
        G.compare_records(rec(L4_TORCH, FULL), other)
    ints = R(A100_TRITON, 256, 1, "p0", (M(FULL.astype(np.uint8)),))
    with pytest.raises(G.MaskGateError, match="bool"):
        G.compare_records(rec(L4_TORCH, FULL), ints)


def test_banks_must_hold_the_same_prompts_once_each_on_one_code_path():
    ref = [rec(L4_TORCH, FULL, prompt="p0"), rec(L4_TORCH, FULL, prompt="p1")]
    with pytest.raises(G.MaskGateError, match="only in the reference"):
        G.compare_banks(ref, [rec(A100_TRITON, FULL, prompt="p0")])
    with pytest.raises(G.MaskGateError, match="twice"):
        G.compare_banks(ref, [rec(A100_TRITON, FULL, prompt="p0")] * 2)
    with pytest.raises(G.MaskGateError, match="mixes code paths"):
        G.compare_banks(ref, [rec(A100_TRITON, FULL, prompt="p0"),
                              rec(A100_TORCH, FULL, prompt="p1")])
    with pytest.raises(G.MaskGateError, match="empty"):
        G.compare_banks([], ref)
    out = G.compare_banks(ref, [rec(A100_TRITON, FULL, prompt=p) for p in ("p1", "p0")])
    assert [c.prompt_id for c in out] == ["p0", "p1"]


# --- the bound ----------------------------------------------------------------------

def test_the_bound_has_no_defaults_and_refuses_nonsense():
    with pytest.raises(TypeError):
        G.Bounds()
    for kw in (dict(min_agreement=0, max_density_difference=0.1),
               dict(min_agreement=1.1, max_density_difference=0.1),
               dict(min_agreement=0.9, max_density_difference=-0.1),
               dict(min_agreement=float("nan"), max_density_difference=0.1),
               dict(min_agreement=True, max_density_difference=0.1)):
        with pytest.raises(G.MaskGateError):
            G.Bounds(**kw)


def test_the_gate_names_every_layer_outside_the_bound():
    cmp = G.compare_banks([rec(L4_TORCH, FULL, FULL)], [rec(A100_TRITON, DROP, FULL)])
    loose = G.gate(cmp, LOOSE)
    assert loose.passed and loose.failures == ()
    assert (loose.n_records, loose.n_layers, loose.identical_layers) == (1, 2, 1)
    assert loose.min_agreement == pytest.approx(2 / 3)
    assert loose.max_abs_density_difference == pytest.approx(1 / 3)
    tight = G.gate(cmp, TIGHT)
    assert not tight.passed
    assert [(f[3], f[4]) for f in tight.failures] == [("p0", 0)]
    assert len(tight.failures[0][5]) == 2        # agreement and density both outside
    # The bound is inclusive: exactly at it passes.
    at = G.Bounds(min_agreement=2 / 3, max_density_difference=1 / 3 + 1e-12)
    assert G.gate(cmp, at).passed


def test_one_gate_is_one_pair_of_code_paths():
    a = G.compare_banks([rec(L4_TORCH, FULL)], [rec(A100_TRITON, FULL)])
    b = G.compare_banks([rec(L4_TORCH, FULL, prompt="p1")], [rec(A100_TORCH, FULL, prompt="p1")])
    with pytest.raises(G.MaskGateError, match="one pair"):
        G.gate(a + b, LOOSE)
    with pytest.raises(G.MaskGateError, match="nothing"):
        G.gate((), LOOSE)


# --- the deltas T2.28 reads ----------------------------------------------------------

def test_density_deltas_by_arm_and_by_length():
    # p0 at 256: shift -1/3 (FULL -> DROP); p1 at 256: shift +1/3 (DROP -> FULL);
    # p2 at 512: shift -1/3.
    ref = [rec(L4_TORCH, FULL, prompt="p0"), rec(L4_TORCH, DROP, prompt="p1"),
           rec(L4_TORCH, FULL, prompt="p2", length=512)]
    cand = [rec(A100_TRITON, DROP, prompt="p0"), rec(A100_TRITON, FULL, prompt="p1"),
            rec(A100_TRITON, DROP, prompt="p2", length=512)]
    cmp = G.compare_banks(ref, cand)
    by_len_mean = G.density_deltas(cmp, statistic="mean", by_length=True)
    assert by_len_mean == {("xattention", 256): pytest.approx(0.0),
                           ("xattention", 512): pytest.approx(1 / 3)}
    by_len_max = G.density_deltas(cmp, statistic="max", by_length=True)
    assert by_len_max == {("xattention", 256): pytest.approx(1 / 3),
                          ("xattention", 512): pytest.approx(1 / 3)}
    # By arm: mean of (-1/3, +1/3, -1/3) is -1/9, reported as 1/9.
    assert G.density_deltas(cmp, statistic="mean", by_length=False) == \
        {"xattention": pytest.approx(1 / 9)}
    for d in by_len_mean.values():
        _check_delta(d, "mask gate")                 # what density_sensitivity accepts
    with pytest.raises(G.MaskGateError, match="statistic"):
        G.density_deltas(cmp, statistic="median", by_length=True)
    with pytest.raises(TypeError):
        G.density_deltas(cmp, by_length=True)


# --- T2.23 and T2.24 -------------------------------------------------------------------

def test_the_p2_gate_needs_both_paths_within_the_bound():
    p1 = [rec(L4_TORCH, FULL)]
    same = G.p2_mask_gate(p1, [rec(A100_TORCH, FULL)], [rec(A100_TRITON, FULL)], TIGHT)
    assert same.passed and same.triton_vs_pytorch_on_p2.passed
    triton_off = G.p2_mask_gate(p1, [rec(A100_TORCH, FULL)], [rec(A100_TRITON, DROP)], TIGHT)
    assert not triton_off.passed
    assert triton_off.pytorch_vs_p1.passed and not triton_off.triton_vs_p1.passed
    assert not triton_off.triton_vs_pytorch_on_p2.passed
    torch_off = G.p2_mask_gate(p1, [rec(A100_TORCH, DROP)], [rec(A100_TRITON, FULL)], TIGHT)
    assert not torch_off.passed
    assert torch_off.triton_vs_p1.arms == ("xattention",)


def test_the_p2_gate_refuses_banks_from_the_wrong_cards_or_paths():
    p1 = [rec(L4_TORCH, FULL)]
    with pytest.raises(G.MaskGateError, match="same card"):
        G.p2_mask_gate(p1, [rec(A100_TORCH, FULL)], [rec(H100_TRITON, FULL)], LOOSE)
    with pytest.raises(G.MaskGateError, match="two implementations"):
        G.p2_mask_gate(p1, [rec(A100_TRITON, FULL)], [rec(A100_TRITON, FULL)], LOOSE)
    l4_triton = CodePath(arm="xattention", gpu="NVIDIA L4", implementation="xattn:triton")
    with pytest.raises(G.MaskGateError, match="different cards"):
        G.p2_mask_gate(p1, [rec(L4_TORCH, FULL)], [rec(l4_triton, FULL)], LOOSE)


def test_p4_against_p2_decides_whether_p6_inherits():
    p2 = [rec(A100_TRITON, FULL)]
    assert G.p4_mask_comparison(p2, [rec(H100_TRITON, FULL)], TIGHT).decision == G.P6_INHERITS
    assert G.p4_mask_comparison(p2, [rec(H100_TRITON, DROP)], TIGHT).decision == G.P6_RECORDS
    with pytest.raises(G.MaskGateError, match="one implementation"):
        G.p4_mask_comparison(p2, [rec(CodePath(arm="xattention", gpu="NVIDIA H100",
                                               implementation="xattn:torch"), FULL)], TIGHT)
    with pytest.raises(G.MaskGateError, match="different cards"):
        G.p4_mask_comparison(p2, [rec(A100_TRITON, FULL)], TIGHT)


# --- real banks ------------------------------------------------------------------------

def test_real_mask_bank_records_go_through_the_gate_after_a_save_and_load(tmp_path):
    pytest.importorskip("torch")
    from attnbench.components import mask_bank as mb

    def banked(a):
        packed = np.packbits(a.reshape(-1))
        return mb.BankedMask(tuple(a.shape), True, mb.density(a, True),
                             hashlib.sha256(packed.tobytes()).hexdigest(), packed)

    for name, cp, mask in (("p1", L4_TORCH, FULL), ("p2", A100_TRITON, DROP)):
        mb.save(mb.MaskRecord(cp, 256, 1, "p0", (banked(mask),)), tmp_path / name)
    p1, p2 = mb.load(tmp_path / "p1"), mb.load(tmp_path / "p2")
    (c,) = G.compare_banks([p1], [p2])
    assert c.layers[0].agreement == pytest.approx(2 / 3)
    assert c.layers[0].reference_density == mb.density(FULL, True)
    assert c.layers[0].candidate_density == mb.density(DROP, True)
    assert not G.gate([c], TIGHT).passed
