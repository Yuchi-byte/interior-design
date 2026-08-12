"""A weightless stand-in for a semantic segmentation backend."""

from dataclasses import dataclass
from typing import Final

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS, NUM_CLASSES

_CEILING: Final = ADE20K_LABELS.index("ceiling")
_WALL: Final = ADE20K_LABELS.index("wall")
_FLOOR: Final = ADE20K_LABELS.index("floor")
_WINDOWPANE: Final = ADE20K_LABELS.index("windowpane")
_SOFA: Final = ADE20K_LABELS.index("sofa")
_DOOR: Final = ADE20K_LABELS.index("door")
_BOOKCASE: Final = ADE20K_LABELS.index("bookcase")

# What a pixel of each class looks like to `FakeClassLogitSegmenter`: one
# brightness per class, far enough apart that a test can render a class by
# painting its key and read the label back. Two structural surfaces (wall,
# floor), the two openings the class term exists for (glazing, door), and one
# piece of furniture that may legitimately cover them.
BRIGHTNESS_BY_CLASS: Final[dict[int, float]] = {
    _FLOOR: 0.1,
    _BOOKCASE: 0.3,
    _WALL: 0.5,
    _DOOR: 0.7,
    _WINDOWPANE: 0.9,
}


@dataclass(frozen=True)
class FakeSegmenter:
    """A `Segmenter` that needs no checkpoint: a ceiling/wall/floor room with one window and one sofa.

    The layout spans all three sides of the ADE20K class lists the architectural
    mask is built from: shell (ceiling, wall, floor), contents (the sofa), and
    the glazing the two lists between them leave unclaimed. Label values ignore
    pixel content and never carry gradient.

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


@dataclass(frozen=True)
class FakeClassLogitSegmenter:
    """A `ClassLogitSegmenter` that needs no checkpoint: class by brightness, differentiably.

    Every pixel's class scores are Gaussian bumps around the brightnesses in
    `BRIGHTNESS_BY_CLASS`, so a test paints a class and reads it back, and the
    scores are a smooth function of the pixels — which is what a gradient test of
    the class energy needs and what `FakeSegmenter`, labelling by frame position
    alone, cannot give.

    The id path is the argmax of the very same scores, as EoMT's is: the mask
    deciding which pixels are scored and the scores it gates can never disagree
    about what this segmenter saw.

    Args:
        sharpness: the bumps' standard deviation in brightness units. Small
            enough to keep the classes apart, large enough that a pixel between
            two keys has a gradient towards both.
    """

    sharpness: float = 0.1

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        with torch.no_grad():
            return self.class_logits(images.detach()).argmax(dim=1)

    @jaxtyped(typechecker=beartype)
    def class_logits(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch 150 height width"]:
        keys = torch.tensor(
            [BRIGHTNESS_BY_CLASS.get(label, 0.0) for label in range(NUM_CLASSES)],
            dtype=images.dtype,
            device=images.device,
        ).view(1, -1, 1, 1)
        # The unrendered classes score a flat zero rather than a bump at a
        # sentinel brightness: a NaN key would survive the `where` forwards and
        # come back as a NaN gradient over every pixel.
        rendered = torch.tensor(
            [label in BRIGHTNESS_BY_CLASS for label in range(NUM_CLASSES)],
            device=images.device,
        ).view(1, -1, 1, 1)
        distance = images.mean(dim=1, keepdim=True) - keys
        bumps = torch.exp(-0.5 * (distance / self.sharpness) ** 2)
        return torch.where(rendered, bumps, torch.zeros((), dtype=images.dtype))
