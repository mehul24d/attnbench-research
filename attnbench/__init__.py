"""attnbench: controlled cross-method attention benchmark harness."""

import importlib

from .config import AttnConfig, SweepGrid
from . import provenance

__version__ = "0.1.0"
__all__ = ["AttnConfig", "SweepGrid", "provenance", "timing", "gates"]

# timing and gates import torch, so they load on first use: `import attnbench`
# and the pure modules under it then run without torch.
_LAZY = ("timing", "gates")


def __getattr__(name):
    if name in _LAZY:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
