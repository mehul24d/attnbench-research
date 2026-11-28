#!/usr/bin/env bash
# Regenerate the paper's tables from this artifact and compare them with the
# committed copies in docs/paper/. CPU only, offline; nothing is launched.
#
# Run from a clone of the repository at the artifact's commit:
#     bash <artifact>/regenerate.sh <artifact dir> <scratch dir>
# Exit 0 only if the checksums hold and every regenerated table matches.
set -euo pipefail
A="$(cd "$1" && pwd)"
OUT="$(mkdir -p "$2" && cd "$2" && pwd)"
R="$(git rev-parse --show-toplevel)"
cd "$A" && shasum -a 256 -c --quiet SHA256SUMS
cd "$R"
python3 scripts/paper_reproduce.py --banked "$A/data" --out "$OUT/reproduce"
IN="$OUT/reproduce/inputs_repointed.json"
M="$A/data/scoring_20261116/mirror"
python3 scripts/null_kernel_bias.py --inputs "$IN" --mirror "$M" --banked "$A/data" \
    --s4 "$A/data/s4" --out-dir "$OUT/paper"
python3 scripts/pay_bound.py --inputs "$IN" --mirror "$M" --banked "$A/data" --out-dir "$OUT/paper"
for f in null_kernel_bias.json pay_bound.json; do
    cmp "$OUT/paper/$f" "docs/paper/exploratory/$f"
    echo "$f: identical"
done
