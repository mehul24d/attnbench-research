"""Leg 1's cell bootstrap: the recursion's and the step form's median absolute
relative error on one P1 session, overall and per arm, with 95% percentile
intervals (adviser's ruling 2 of the ninth 9 Nov 2026 paste; the limitations
text, `docs/limitations.md`, 9 Nov 2026).

The residuals are Gate A's own (`gate_a.measured`, `gate_a._residuals`), so a
group short of clean rounds is left out as in leg 1. Cells are resampled with
replacement within each scope, the same draw for both predictors (paired),
10,000 draws at seed 0. The difference is the recursion's median minus the
step form's, beside the noise floor it is read against.

The committed prediction is read from a local copy of the bucket file, with
the creation time step 10 read in session.

    python3 scripts/leg1_bootstrap.py --prediction pred/prediction_P1_<t>.json \\
        --created 2026-11-09T09:58:44+00:00 --ground-truth gt/recorded \\
        --noise-floor 0.0677 --out leg1_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

from attnbench.analysis import gate_a as G  # noqa: E402
import _load  # noqa: E402
import gate_a as GA  # noqa: E402

DRAWS, SEED = 10_000, 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prediction", required=True, help="local copy of the committed prediction")
    ap.add_argument("--created", required=True, help="the bucket file's creation time (ISO, UTC)")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--noise-floor", type=float, required=True)
    ap.add_argument("--dense-arm", default="dense")
    ap.add_argument("--out", required=True)
    return ap


class LocalCopy:
    """The store `_load.load` reads: the local file and its .sha256, read only."""

    def __init__(self, path: Path, created: datetime):
        self.path, self.when = path, created

    def get(self, remote):
        return Path(str(self.path) + ".sha256" if remote.endswith(".sha256") else self.path).read_bytes()

    def created(self, remote):
        return self.when


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def scope(errors: dict, keys: list, rng: random.Random) -> dict:
    """{n, recursion, step_form, difference, *_ci95, share_step_form_below}
    over `keys`, cells resampled with replacement, paired."""
    rec = [errors[k]["recursion"] for k in keys]
    st = [errors[k]["step_form"] for k in keys]
    dr, ds, dd = [], [], []
    for _ in range(DRAWS):
        idx = [rng.randrange(len(keys)) for _ in keys]
        a = statistics.median(rec[i] for i in idx)
        c = statistics.median(st[i] for i in idx)
        dr.append(a), ds.append(c), dd.append(a - c)
    return {"n": len(keys), "recursion": statistics.median(rec), "step_form": statistics.median(st),
            "difference": statistics.median(rec) - statistics.median(st),
            "recursion_ci95": [_pct(dr, .025), _pct(dr, .975)],
            "step_form_ci95": [_pct(ds, .025), _pct(ds, .975)],
            "difference_ci95": [_pct(dd, .025), _pct(dd, .975)],
            "share_step_form_below": sum(x > 0 for x in dd) / DRAWS}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    path = Path(args.prediction)
    results, starts = GA.ground_truth(Path(args.ground_truth))
    cp = _load.load("gs://x/" + path.name, str(path), started_at=starts, clock_skew_s=0.0,
                    banked_at=None, store=LocalCopy(path, datetime.fromisoformat(args.created)))
    m = G.measured(cp, results)
    res = G._residuals(cp, m["seconds"], dense_arm=args.dense_arm,
                       relative_floor=args.noise_floor, predictors=G.GATE_PREDICTORS)
    errors: dict = {}
    for r in res:
        errors.setdefault((r.arm, round(r.density, 3), r.context_length, r.batch), {})[
            r.predictor] = abs(r.relative)
    rng = random.Random(SEED)
    scopes = {"all": sorted(errors)}
    scopes.update({a: sorted(k for k in errors if k[0] == a) for a in sorted({k[0] for k in errors})})
    rep = {"prediction": path.name, "prediction_sha256": cp.sha256, "draws": DRAWS, "seed": SEED,
           "unit": "cell, paired", "noise_floor": args.noise_floor,
           "excluded_groups": m["excluded"], "scopes": {}}
    for name, keys in scopes.items():
        s = scope(errors, keys, rng)
        s["difference_exceeds_noise_floor"] = s["difference"] > args.noise_floor
        s["difference_ci95_upper_exceeds_noise_floor"] = s["difference_ci95"][1] > args.noise_floor
        rep["scopes"][name] = s
        print(f"{name}: n {s['n']}, recursion {s['recursion']:.2%} "
              f"[{s['recursion_ci95'][0]:.2%}, {s['recursion_ci95'][1]:.2%}], step form "
              f"{s['step_form']:.2%} [{s['step_form_ci95'][0]:.2%}, {s['step_form_ci95'][1]:.2%}], "
              f"difference {s['difference'] * 100:+.2f} points "
              f"[{s['difference_ci95'][0] * 100:+.2f}, {s['difference_ci95'][1] * 100:+.2f}]")
    Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
