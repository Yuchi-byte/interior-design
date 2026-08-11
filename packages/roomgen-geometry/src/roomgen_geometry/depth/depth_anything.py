"""The Depth Anything V2 backend: a DPT head on a DINOv2 backbone."""

from __future__ import annotations

from typing import Final

import torch
from jaxtyping import Float
from torch import Tensor

DEFAULT_CHECKPOINT: Final = "depth-anything/Depth-Anything-V2-Large-hf"


class DepthAnythingV2:
    """Depth Anything V2 as a :class:`~roomgen_geometry.depth.base.DepthEstimator`.

    Args:
        checkpoint: Hugging Face Hub id, resolved against the ambient Hub cache
            (`ROOMGEN_HF_HOME`); weights are fetched on demand, never vendored.
        device: where the weights live and where inputs are expected.
        dtype: parameter dtype; a reduced-precision backbone still has to
            produce a usable gradient, so it is the caller's choice.

    Everything between the input image and the returned disparity stays on the
    autograd graph, preprocessing included: the resize and the normalisation are
    tensor ops, never a PIL round-trip, and nothing on the path detaches, calls
    ``.item()``, or runs under ``torch.no_grad``. A detached backend silently
    turns guidance into a no-op instead of failing.
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
    ) -> Float[Tensor, "batch height width"]:
        raise NotImplementedError
