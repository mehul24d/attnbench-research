"""attnbench: controlled cross-method attention benchmark harness."""

from .config import AttnConfig, SweepGrid
from . import provenance, timing, gates

__version__ = "0.1.0"
__all__ = ["AttnConfig", "SweepGrid", "provenance", "timing", "gates"]
