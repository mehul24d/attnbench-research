"""Backend registry. Importing this module registers every implementation."""

from .base import (AttentionBackend, Capability, KVCacheState, UnsupportedConfig,
                   register, get, all_backends)
from . import impls  # noqa: F401  (import registers the backends)
from . import xformers_backend  # noqa: F401
from . import sage_attention  # noqa: F401
from . import block_sparse  # noqa: F401
from . import linear  # noqa: F401

__all__ = ["AttentionBackend", "Capability", "KVCacheState", "UnsupportedConfig",
           "register", "get", "all_backends"]
