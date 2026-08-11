"""A weightless stand-in for a monocular depth backend."""

from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


@dataclass(frozen=True)
class FakeDepthEstimator:
    """A `DepthEstimator` that needs no checkpoint: a floor-shaped disparity ramp coupled to image content.

    Args:
        scale: the unknown positive scale a real affine-invariant backend leaves
            undetermined.
        shift: the matching unknown offset. Two instances differing in
            `scale`/`shift` stand for the same backend disagreeing on two
            images, which is the ambiguity `align_scale_shift` exists to remove.
        content_weight: how strongly pixel values move the disparity; non-zero
            keeps every input pixel on the gradient path, which is what makes
            this usable as a depth backend in a guidance test.
    """

    scale: float = 1.0
    shift: float = 0.0
    content_weight: float = 0.5

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch height width"]:
        _, _, height, _ = images.shape
        rows = torch.linspace(
            0.0, 1.0, height, dtype=images.dtype, device=images.device
        )
        # Disparity is inverse depth, so it grows toward the bottom of the
        # frame: the floor there is the nearest thing a room photo contains.
        floor_ramp = rows[None, :, None]
        return (
            self.scale * (floor_ramp + self.content_weight * images.mean(dim=1))
            + self.shift
        )
