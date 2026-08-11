"""The Depth Anything V2 backend: a DPT head on a DINOv2 backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import interpolate
from transformers import AutoModelForDepthEstimation, DPTImageProcessor

# Small is the only Apache-2.0 checkpoint tier; Base/Large/Giant are CC-BY-NC.
DEFAULT_CHECKPOINT: Final = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2:
    """Depth Anything V2 as a :class:`~roomgen_geometry.depth.base.DepthEstimator`.

    Args:
        checkpoint: Hugging Face Hub id; weights are fetched on demand, never
            vendored.
        device: where the weights live and where inputs are expected.
        dtype: parameter dtype; a reduced-precision backbone still has to
            produce a usable gradient, so it is the caller's choice.
        cache_dir: Hub snapshot cache the weights are read from and downloaded
            into (`RoomgenSettings.hub_cache`); `None` follows the Hub client's
            own environment.

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
        cache_dir: Path | None = None,
    ) -> None:
        # The processor is loaded for its input size and normalisation
        # statistics rather than constants copied out of the checkpoint; the
        # preprocessing itself is the tensor ops below, never a PIL round-trip.
        processor = DPTImageProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
        self._input_size = (int(processor.size["height"]), int(processor.size["width"]))
        self._image_mean = torch.tensor(processor.image_mean, device=device).view(
            1, 3, 1, 1
        )
        self._image_std = torch.tensor(processor.image_std, device=device).view(
            1, 3, 1, 1
        )

        # `torch_dtype` rather than transformers 5's `dtype`: it is the spelling
        # both major versions accept, and the declared floor is 4.54.
        model: Module = AutoModelForDepthEstimation.from_pretrained(
            checkpoint, torch_dtype=dtype, cache_dir=cache_dir
        )
        # Frozen parameters, not a `no_grad` forward: the guidance gradient has
        # to reach the input image, and only the weights are held constant.
        self._model = model.to(device).eval().requires_grad_(False)
        self._dtype = dtype

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch height width"]:
        _, _, height, width = images.shape
        # Square, unlike the checkpoint's processor, which preserves the aspect
        # ratio: reference and sample are always one frame size, so a fixed
        # input costs the guidance loop the same on every step.
        resized = interpolate(
            images,
            size=self._input_size,
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        normalised = (resized - self._image_mean) / self._image_std
        predicted_depth: Tensor = self._model(
            pixel_values=normalised.to(self._dtype)
        ).predicted_depth
        # Back in the caller's precision before the upsample: a half-precision
        # backbone is a memory choice, not a decision about the reward's dtype.
        return interpolate(
            predicted_depth.to(images.dtype).unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
