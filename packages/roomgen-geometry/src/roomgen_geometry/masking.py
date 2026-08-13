"""The architectural shell mask and the boundary erosion that trims it."""

from __future__ import annotations

import torch
from jaxtyping import Bool, Int
from torch import Tensor
from torch.nn.functional import max_pool2d

from roomgen_geometry.segmentation.ade20k import (
    ARCHITECTURAL_CLASSES,
    CONTENTS_CLASSES,
    STRUCTURAL_CLASSES,
)


def scorable_mask(
    segmentation: Int[Tensor, "batch height width"], *, erosion_radius: int
) -> Bool[Tensor, "batch height width"]:
    """The pixels one *prediction's* segmentation leaves available to the reward.

    Args:
        segmentation: ADE20K-150 model output ids, as
            :class:`~roomgen_geometry.segmentation.base.Segmenter` returns them.
        erosion_radius: in pixels, passed on to :func:`erode_boundaries`.

    Only what the segmenter affirmatively calls contents
    (:data:`~roomgen_geometry.segmentation.ade20k.CONTENTS_CLASSES`) is excused,
    so an id neither class list names — an ignore index, or the sky, tree or
    person a window frames — stays scored rather than excused by default. The
    opposite polarity, keeping only affirmative shell, was measured and lost: it
    tied at noise 0.6, gave up 1.8pp of depth-error reduction at 0.9, and scored
    less of the room while doing it.

    Erosion is applied to the kept side and never to the exemption: the
    unreliable band either side of a segment edge is dropped from the score
    rather than trusted, so the excused region grows by the radius. The result is
    a constant — it is derived from a segmentation that never carries gradient.
    """
    return erode_boundaries(
        ~_is_any_of(segmentation, CONTENTS_CLASSES), radius=erosion_radius
    )


def architectural_mask(
    segmentation: Int[Tensor, "batch height width"], *, erosion_radius: int
) -> Bool[Tensor, "batch height width"]:
    """The pixels the geometric reward is allowed to score, from the *photo's* segmentation.

    True on the shell classes
    (:data:`~roomgen_geometry.segmentation.ade20k.ARCHITECTURAL_CLASSES`) and
    False on everything a renovation may change. The reference side is read the
    other way round from the prediction side on purpose: what the photo
    affirmatively shows is the room, while what a render declines to commit on
    has not earned an exemption from being scored.
    """
    return erode_boundaries(
        _is_any_of(segmentation, ARCHITECTURAL_CLASSES), radius=erosion_radius
    )


def structural_mask(
    segmentation: Int[Tensor, "batch height width"], *, erosion_radius: int
) -> Bool[Tensor, "batch height width"]:
    """The pixels whose class a renovation may not change: the reference side of the class term.

    True on :data:`~roomgen_geometry.segmentation.ade20k.STRUCTURAL_CLASSES` —
    the shell plus the glazing `architectural_mask` leaves out, since a window's
    depth is the view's but its identity is the room's — and eroded like every
    other mask, because a class label flickers in the band either side of a
    segment edge and a term that chases the flicker is charging for noise.

    Read one way only, from the photo, exactly as `architectural_mask` is: what a
    *prediction's* segmentation is allowed to excuse is a separate question,
    which `scorable_mask` answers.
    """
    return erode_boundaries(
        _is_any_of(segmentation, STRUCTURAL_CLASSES), radius=erosion_radius
    )


def _is_any_of(
    segmentation: Int[Tensor, "batch height width"], classes: frozenset[int]
) -> Bool[Tensor, "batch height width"]:
    ids = torch.tensor(
        sorted(classes), dtype=segmentation.dtype, device=segmentation.device
    )
    return torch.isin(segmentation, ids)


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
