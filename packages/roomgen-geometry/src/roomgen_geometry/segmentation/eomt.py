"""The EoMT backend: mask queries decoded inside a plain ViT encoder."""

from __future__ import annotations

from typing import Final

import torch
from jaxtyping import Float, Int
from torch import Tensor

# The DINOv2-backed weights are MIT; the otherwise-identical DINOv3 variants
# carry a restrictive licence and are not drop-in replacements here.
DEFAULT_CHECKPOINT: Final = "tue-mps/ade20k_semantic_eomt_large_512"


class EoMT:
    """EoMT as a :class:`~roomgen_geometry.segmentation.base.Segmenter`.

    Args:
        checkpoint: Hugging Face Hub id, resolved against the ambient Hub cache
            (`ROOMGEN_HF_HOME`); weights are fetched on demand, never vendored.
        device: where the weights live and where inputs are expected.
        dtype: parameter dtype.

    The returned ids index
    :data:`~roomgen_geometry.segmentation.ade20k.ADE20K_LABELS`, and the call
    runs under ``torch.no_grad``: the segmentation is evaluated once per photo
    and is a constant for the rest of the run, so nothing here belongs on the
    guidance gradient path.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        raise NotImplementedError

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        raise NotImplementedError
