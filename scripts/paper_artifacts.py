"""The paper's LaTeX tables, PDF figures and cited numbers from a committed
prediction and its ground truth (C10).

The committed prediction is read with `load_committed` (never by hash
alone), so a prediction synced after the timed workload, or never synced,
is refused. `--spec` is an `attnbench.paper_spec/1` JSON file naming the
prediction's sha256, every ground-truth file with its sha256 and the cell
each arm's row measures, every arm's result class, admissibility, the
floors and, optionally, the crossover intervals; file paths are relative to
it (`attnbench/figures/paper_inputs.py`). Stale or unhashed inputs are
refused. Writes into --out-dir, which must be new or empty
(`attnbench/figures/paper_artifacts.py`). Run at a clean commit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

from attnbench.figures import paper_artifacts as PA  # noqa: E402
from attnbench.figures import paper_inputs as PI  # noqa: E402
from attnbench.figures import results as RS  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402
from attnbench.predict.recursion import CostModelError  # noqa: E402
import _load  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--remote", required=True, help="the committed prediction in the store")
    ap.add_argument("--local", required=True, help="its local copy, checked against the store")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--clock-skew-s", type=float, required=True)
    ap.add_argument("--banked-at", action="append", metavar="PAIRING=ISO")
    ap.add_argument("--out-dir", required=True)
    return ap


def code_stamp() -> C.CodeStamp:
    return C.current_code_stamp()


def main(argv=None, store=None) -> int:
    args = parser().parse_args(argv)
    try:
        spec_path = Path(args.spec)
        spec = PI.read_spec(spec_path)
        gt = PI.ground_truth(spec, spec_path.parent)
        cp = _load.load(args.remote, args.local,
                        started_at=[r["first_timed_block_started_at"] for _, _, r, _ in gt],
                        clock_skew_s=args.clock_skew_s, banked_at=args.banked_at, store=store)
        inp = PI.build_inputs(spec, spec_path.parent, cp)
        manifest = PA.build(inp, args.out_dir, code_stamp())
    except (PI.PaperInputError, RS.ResultsError, C.PredictionCommitError, CostModelError,
            FileExistsError, OSError, KeyError, TypeError, ValueError) as e:
        print(f"paper_artifacts: {e}", file=sys.stderr)
        return 2
    print(f"wrote {len(manifest['outputs'])} artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
