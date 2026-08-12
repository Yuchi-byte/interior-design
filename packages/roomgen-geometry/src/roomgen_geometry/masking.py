"""The architectural shell mask and the boundary erosion that trims it."""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Bool, Int
from torch import Tensor
from torch.nn.functional import max_pool2d

from roomgen_geometry.segmentation.ade20k import (
    ARCHITECTURAL_CLASSES,
    CONTENTS_CLASSES,
    STRUCTURAL_CLASSES,
)

MaskMode = Literal["intersect_shell", "subtract_contents"]
"""Which way round a segmentation is read when it decides what may be scored."""


def scorable_mask(
    segmentation: Int[Tensor, "batch height width"],
    *,
    mode: MaskMode,
    erosion_radius: int,
) -> Bool[Tensor, "batch height width"]:
    """The pixels one segmentation leaves available to the reward, under either polarity.

    Args:
        segmentation: ADE20K-150 model output ids, as
            :class:`~roomgen_geometry.segmentation.base.Segmenter` returns them.
        mode: `intersect_shell` exempts everything that is not affirmatively
            shell (:data:`~roomgen_geometry.segmentation.ade20k.ARCHITECTURAL_CLASSES`);
            `subtract_contents` exempts only what is affirmatively contents
            (:data:`~roomgen_geometry.segmentation.ade20k.CONTENTS_CLASSES`), so
            an id neither list names — an ignore index, or the sky, tree, or
            person a window frames — stays scored rather than excused.
        erosion_radius: in pixels, passed on to :func:`erode_boundaries`.

    Erosion is applied to the kept side in both modes, never to the exemption:
    the unreliable band either side of a segment edge is dropped from the score
    rather than trusted, so the region a mode exempts grows by the radius.

    The two modes agree wherever every id is named by one of the two class
    lists; they part company on every id named by neither — which is most of the
    outdoor and stuff classes — and on the eroded band around them. The result is
    a constant either way: it is derived from a segmentation that never carries
    gradient.
    """
    exempt = (
        torch.isin(segmentation, _class_ids(CONTENTS_CLASSES, segmentation))
        if mode == "subtract_contents"
        else ~torch.isin(segmentation, _class_ids(ARCHITECTURAL_CLASSES, segmentation))
    )
    return erode_boundaries(~exempt, radius=erosion_radius)


def architectural_mask(
    segmentation: Int[Tensor, "batch height width"], *, erosion_radius: int
) -> Bool[Tensor, "batch height width"]:
    """The pixels the geometric reward is allowed to score: `scorable_mask` at `intersect_shell`.

    True on the shell classes
    (:data:`~roomgen_geometry.segmentation.ade20k.ARCHITECTURAL_CLASSES`) and
    False on everything a renovation may change. This is the reference photo's
    side of the reward, which is read one way only — the modes are a statement
    about a *prediction's* segmentation, not about the room as photographed.
    """
    return scorable_mask(
        segmentation, mode="intersect_shell", erosion_radius=erosion_radius
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
        torch.isin(segmentation, _class_ids(STRUCTURAL_CLASSES, segmentation)),
        radius=erosion_radius,
    )


def _class_ids(classes: frozenset[int], like: Tensor) -> Tensor:
    return torch.tensor(sorted(classes), dtype=like.dtype, device=like.device)


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
