# H0 to H3 scored at `lock-2026-11-14-r3` (CPU only)

Analysis commit `83b2edf983a8` (34 imported files byte-identical); lock read from the tag, `main` differing only at ['/plan_sha256'].

| Hypothesis | Pairing | Statistic | Threshold | Verdict |
|---|---|---|---|---|
| H0 (host, P1→P2) | P2 | no operation is above its cell's floor on both pairings (0 cells with the same sequence, 25 with a different one, 20 unmatched) | ≤ 0.1 | **not scored** |
| H1 | P2 | recursion median abs. relative error 0.1183 over 35 cells; rivals fitted_sum 0.1005, sum 0.6935, step_form 0.0842, learned 1.4465; sign 25 agree, 0 disagree; XAttention pair 0 agree, 1 disagree | ≤ 0.1, below every rival, every sign, the pair within 0.1 | **fails** |
| H1 | P4 | recursion median abs. relative error 0.2495 over 35 cells; rivals fitted_sum 0.1216, sum 0.8527, step_form 0.2454, learned 0.3169; sign 25 agree, 2 disagree; XAttention pair 0 agree, 1 disagree | ≤ 0.1, below every rival, every sign, the pair within 0.1 | **fails** |
| H2 (host from P1) | P2 | no transferred cell is measured (45 cells left out) | ≤ — and every sign | **not scored** |
| H3 | P4 | estimators: 6 not scored | measured bracket inside the predicted | **not scored** |
| H4 | — | — | — | **not scored** (Appendix A, 12 Nov 2026) |
| H5 | — | — | — | **not scored** (Appendix A, 12 Nov 2026) |
| H6 | — | — | — | **not scored** (Appendix A, 14 Nov 2026) |

## H1 parts

- P2: {'tolerance': 'fails', 'beats_every_rival': 'fails', 'sign': 'holds', 'xattention_pair': 'fails'}; recursion below: {'fitted_sum': False, 'sum': True, 'step_form': False, 'learned': True}; marginal signs: 3; inside the floor: 4; unmeasured: 0.
- P4: {'tolerance': 'fails', 'beats_every_rival': 'fails', 'sign': 'fails', 'xattention_pair': 'fails'}; recursion below: {'fitted_sum': False, 'sum': True, 'step_form': False, 'learned': True}; marginal signs: 4; inside the floor: 2; unmeasured: 0.

## H3 per estimator (P4)

| Estimator | Status | Verdict |
|---|---|---|
| meanpool@0.25 | unresolved | not scored |
| meanpool@0.5 | unresolved | not scored |
| oracle@0.25 | both none in range | not scored |
| oracle@0.5 | both none in range | not scored |
| xattention@calibrated | unresolved | not scored |
| xattention_noassert@calibrated | unresolved | not scored |

## Inputs

- P1: `predictions/P1-T3.2d/t32d_p1_20261109T0726Z/prediction_P1_20261109T095833754749Z.json` sha256 `280a34238a09…`, trained on nothing, first ground truth 2026-11-09T10:01:23+00:00.
- P2: `predictions/P2/t61_p2_20261114T0022Z/prediction_P2_20261114T022220275757Z.json` sha256 `6cdd10b9460b…`, trained on ['P1'], first ground truth 2026-11-14T02:28:57+00:00.
- P4: `predictions/P4/t64_p4_20261116T113724Z_b3/prediction_P4_20261116T120732377724Z.json` sha256 `f9de9be0afcd…`, trained on ['P1', 'P2'], first ground truth 2026-11-16T12:11:10+00:00.
- P2 measured: 35 cells, blocks ['t61_p2_20261114T0022Z'], pooled across blocks [], groups left out 0, unmeasured 0.
- P4 measured: 35 cells, blocks ['b3', 'b4'], pooled across blocks [['primary', 'NVIDIA H100 80GB HBM3|eager|O0|3761dc918db6', 'dense', 1.0, 8075, 1]], groups left out 0, unmeasured 0.
