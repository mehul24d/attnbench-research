#!/usr/bin/env bash
# Install mit-han-lab/x-attention on a GPU instance at the commit this study
# pins, and prove the import resolves to that commit, unedited.
#
# Until 2026-10-02 nothing installed it and nothing checked it: the backend
# named a commit (`XATTN_COMMIT`) that no session had ever run. Run from the
# repository root, after the preflight, before the CUDA gate:
#
#   bash scripts/install_xattention.sh
#
# Three things about the package decide how this is done:
#
#   - Editable, from a git checkout. `xattn/src` has no `__init__.py`, so
#     setup.py's find_packages() leaves the estimator out of a normal
#     install; an editable one imports from the checkout, where it resolves.
#     It also keeps a commit to read back (`installed_checkout`).
#   - `--no-deps`. Its requirements.txt pins a whole research environment
#     (vllm, flash-attn, a torch build); the estimator needs only torch,
#     triton and block_sparse_attn, which the image already has. Installing
#     its pins would replace the torch every banked row was measured on.
#   - OUTSIDE the repository. A checkout inside the working tree would make
#     the tree dirty, and every row written afterwards would be stamped
#     git_dirty=True.
#
# The repository carries no licence, so it is cloned and called here, never
# copied into this repository.
set -euo pipefail

COMMIT="$(python3 -c 'from attnbench.backends.xattention import XATTN_COMMIT; print(XATTN_COMMIT)')"
REPO_URL="$(python3 -c 'from attnbench.backends.xattention import XATTN_REPO; print(XATTN_REPO)')"
DEST="${XATTN_DIR:-$HOME/x-attention}"

case "$(cd "$(dirname "$DEST")" && pwd)/" in
  "$(pwd)/"*) echo "REFUSING: $DEST is inside the repository; it would dirty the tree" >&2
              exit 1 ;;
esac

if [[ ! -d "$DEST/.git" ]]; then
  git clone -q "$REPO_URL" "$DEST"
fi
git -C "$DEST" checkout -q --detach "$COMMIT"
python3 -m pip install -q --no-deps -e "$DEST"

python3 - <<'PY'
from attnbench.backends.xattention import XATTN_COMMIT, installed_checkout
import xattn.src.Xattention  # noqa: F401  (imports triton and block_sparse_attn too)
head, dirty = installed_checkout()
assert (head, dirty) == (XATTN_COMMIT, False), (
    f"x-attention resolves to {head} (dirty={dirty}), not {XATTN_COMMIT}")
print(f"x-attention {head[:7]} installed, clean, importable")
PY

if [[ -n "$(git status --porcelain)" ]]; then
  echo "INSTALL LEFT THE REPOSITORY DIRTY:" >&2
  git status --porcelain >&2
  exit 1
fi
