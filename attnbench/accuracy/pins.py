"""Library versions a model's rows must be produced under (estimator-frontier
pre-registration, sec. 4.9, 2026-10-03).

Llama-3.1-8B rows are generated, decoded and scored only on the GPU image's
transformers 4.46.0. Decoding differs between versions: 4.46.0 applies the
tokenizer's `clean_up_tokenization_spaces=True` ("x , y" -> "x, y") and 5.18.0
ignores it for BPE, and substring scoring can be sensitive to spacing before
punctuation. The workstation's tests run on 5.18.0, so a convention would
break silently. This is a hard check instead.

The pin is the image's version: `scripts/gcp_launch_compile_session.sh`
("transformers pinned to 4.46.0") and audit item S9, where the instance ran
4.46.0. It has not been re-read from the v6 image. If the instance reports
anything else, the Llama path refuses, and the pin changes only by a dated
amendment before any Llama row.
"""

from __future__ import annotations

from importlib import metadata
from typing import Optional

PINNED_TRANSFORMERS: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "4.46.0",
}


class TransformersVersionMismatch(RuntimeError):
    """The loaded transformers is not the version this model is pinned to."""


def installed_transformers() -> Optional[str]:
    try:
        return metadata.version("transformers")
    except metadata.PackageNotFoundError:
        return None


def require_pinned_transformers(model_id: Optional[str]) -> None:
    """Refuse to produce or score a pinned model's rows on any other
    transformers version. A no-op for models without a pin."""
    want = PINNED_TRANSFORMERS.get(model_id)
    if want is None:
        return
    import transformers
    have = transformers.__version__
    if have != want:
        raise TransformersVersionMismatch(
            f"{model_id} rows must be produced under transformers {want} "
            f"(the GPU image's pin); loaded {have}. Decoding, and so scoring, "
            f"differs between versions.")
