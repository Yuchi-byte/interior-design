"""The architectural shell mask and the boundary erosion that trims it."""

from __future__ import annotations

import torch
from jaxtyping import Bool, Int
from torch import Tensor
from torch.nn.functional import max_pool2d

from roomgen_geometry.segmentation.ade20k import ARCHITECTURAL_CLASSES


def architectural_mask(
    segmentation: Int[Tensor, "batch height width"], *, erosion_radius: int
) -> Bool[Tensor, "batch height width"]:
    """The pixels the geometric reward is allowed to score.

    Args:
        segmentation: ADE20K-150 model output ids, as
            :class:`~roomgen_geometry.segmentation.base.Segmenter` returns them.
        erosion_radius: in pixels, passed on to :func:`erode_boundaries`.

    True on the shell classes
    (:data:`~roomgen_geometry.segmentation.ade20k.ARCHITECTURAL_CLASSES`) and
    False on everything a renovation may change. The result is a constant: it is
    derived from a segmentation that never carries gradient.
    """
    shell = torch.tensor(
        sorted(ARCHITECTURAL_CLASSES),
        dtype=segmentation.dtype,
        device=segmentation.device,
    )
    return erode_boundaries(torch.isin(segmentation, shell), radius=erosion_radius)


def erode_boundaries(
    mask: Bool[Tensor, "batch height width"], *, radius: int
) -> Bool[Tensor, "batch height width"]:
    """Drop the pixels within `radius` of a False neighbour.

    Monocular depth is unreliable in the band either side of a segment edge,
    where a single pixel straddles two surfaces at different distances, so those
    pixels are excluded rather than trusted.
    """
    if radius < 0:
        raise ValueError(f"erosion radius must not be negative, got {radius}")
    if radius == 0:
        return mask
    # Binary erosion as a max-pool over the inverted mask: a pixel survives only
    # if no False pixel falls in its (2*radius+1)² neighbourhood. The pool pads
    # with -inf rather than zeros, so the frame edge erodes against real
    # neighbours only and a mask touching the border keeps its border.
    neighbourhood_has_false = max_pool2d(
        (~mask).to(torch.float32).unsqueeze(1),
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    return neighbourhood_has_false.squeeze(1) == 0.0
