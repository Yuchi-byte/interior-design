"""A weightless stand-in for a semantic segmentation backend."""

from dataclasses import dataclass
from typing import Final

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS

_CEILING: Final = ADE20K_LABELS.index("ceiling")
_WALL: Final = ADE20K_LABELS.index("wall")
_FLOOR: Final = ADE20K_LABELS.index("floor")
_WINDOWPANE: Final = ADE20K_LABELS.index("windowpane")
_SOFA: Final = ADE20K_LABELS.index("sofa")


@dataclass(frozen=True)
class FakeSegmenter:
    """A `Segmenter` that needs no checkpoint: a ceiling/wall/floor room with one window and one sofa.

    The layout spans both sides of the ADE20K shell/contents partition the
    architectural mask is built from. Label values ignore pixel content and
    never carry gradient.

    Args:
        ceiling_fraction: share of the frame height labelled ceiling, from the
            top.
        floor_fraction: share of the frame height labelled floor, from the
            bottom.
    """

    ceiling_fraction: float = 0.2
    floor_fraction: float = 0.3

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        batch, _, height, width = images.shape
        wall_start = int(height * self.ceiling_fraction)
        wall_end = height - int(height * self.floor_fraction)
        if wall_start < 1 or wall_end - wall_start < 4 or height - wall_end < 2:
            raise ValueError(
                f"frame {height}x{width} is too small to lay out a ceiling/wall/floor "
                f"room at fractions {self.ceiling_fraction}/{self.floor_fraction}"
            )

        labels = torch.full(
            (height, width), _WALL, dtype=torch.int64, device=images.device
        )
        labels[:wall_start] = _CEILING
        labels[wall_end:] = _FLOOR
        wall_height = wall_end - wall_start
        labels[
            wall_start + wall_height // 4 : wall_end - wall_height // 4,
            width // 8 : width // 2,
        ] = _WINDOWPANE
        # The sofa stands on the floor but stops short of the bottom row, so the
        # frame edge stays shell and a mask test can rely on the extremes.
        labels[wall_end : height - 1, width // 2 :] = _SOFA
        return labels.expand(batch, height, width).clone()
