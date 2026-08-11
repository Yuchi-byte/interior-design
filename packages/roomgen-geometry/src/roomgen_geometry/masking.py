"""The architectural shell mask and the boundary erosion that trims it."""

from __future__ import annotations

from jaxtyping import Bool, Int
from torch import Tensor


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
    raise NotImplementedError


def erode_boundaries(
    mask: Bool[Tensor, "batch height width"], *, radius: int
) -> Bool[Tensor, "batch height width"]:
    """Drop the pixels within `radius` of a False neighbour.

    Monocular depth is unreliable in the band either side of a segment edge,
    where a single pixel straddles two surfaces at different distances, so those
    pixels are excluded rather than trusted.
    """
    raise NotImplementedError
