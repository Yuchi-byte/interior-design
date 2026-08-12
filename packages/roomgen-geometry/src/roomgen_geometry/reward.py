"""The scale- and shift-invariant masked depth objective."""

from typing import Final

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from roomgen_geometry.alignment import (
    AlignmentMode,
    align_scale_shift,
    apply_scale_shift,
)

# The energy is a sum over every masked pixel of a squared residual, and at
# 512x512 that is a quarter of a million terms: in float16, whose largest finite
# value is 65504, the sum saturates to inf long before the division rescales it
# and the whole backward pass comes back zero — guidance silently switched off
# while sampling looks perfectly healthy. The cast is an autograd op like any
# other, so the gradient still reaches a half-precision prediction; only the
# reduction is lifted.
_ENERGY_DTYPE: Final = torch.float32

# The class scores are non-negative query mass and normalise to a distribution
# by their own sum, which is ~1 wherever any query fires at all. The floor bounds
# the normalisation's derivative at 1/floor where no query does: without it a
# pixel the segmenter has no opinion about would hand back a 1/epsilon spike, and
# an unbounded gradient from the term's least reliable pixels is the whole reason
# this reward is a margin rather than a cross-entropy.
_SCORE_FLOOR: Final = 1e-2


def masked_depth_error(
    prediction: Float[Tensor, "batch height width"],
    target: Float[Tensor, "batch height width"],
    mask: Bool[Tensor, "batch height width"],
    *,
    alignment: AlignmentMode = "scale_shift",
) -> Float[Tensor, "batch"]:
    """How far the sample's geometry has drifted from the reference's, per batch element.

    Args:
        prediction: inverse depth of the current sample; carries gradient.
        target: inverse depth of the reference photo; a detached constant.
        mask: the architectural shell, the only pixels that count.
        alignment: how much of the affine freedom the fit hands back. The
            default is the full scale-and-shift solve an affine-invariant
            backend requires; a *metric* backend, whose disparity is determined
            absolutely, may be scored under `shift` or `none` instead, which is
            the only way absolute scale ever enters the energy at all. See
            :data:`roomgen_geometry.alignment.AlignmentMode`.

    Both maps are affine-invariant inverse depth, so they share no frame until
    :func:`roomgen_geometry.alignment.align_scale_shift` has fitted the
    prediction onto the target over the masked pixels; the error is the masked
    mean squared residual of that fit, and is therefore unchanged by any
    *positive* affine rescaling of `prediction`. A negative one is not a change
    of frame but a change of geometry — it swaps near for far — and scores
    above the flat fit rather than the same as the unreflected map.

    The energy is float32 by contract whatever dtype the two maps arrive in, so
    that a half-precision sampling run scores against a reduction that cannot
    overflow.
    """
    # Excluded pixels are replaced, not weighted to zero, because `0 * NaN` is
    # NaN in both directions: forwards a single non-finite pixel outside the
    # mask would poison the whole score, and backwards it would still reach the
    # fitted scale — whose gradient is a sum over *every* pixel of the map — and
    # come back as a NaN gradient over the entire image. A non-finite pixel
    # *inside* the mask does still propagate, deliberately: garbage in the
    # scored region is a fault to surface, not to hide.
    zero = torch.zeros((), dtype=_ENERGY_DTYPE, device=prediction.device)
    predicted = torch.where(mask, prediction.to(_ENERGY_DTYPE), zero)
    reference = torch.where(mask, target.to(_ENERGY_DTYPE), zero)
    fit = align_scale_shift(predicted, reference, mask, mode=alignment)
    aligned = apply_scale_shift(predicted, fit)
    weight = mask.to(_ENERGY_DTYPE)
    squared_residual = (
        (torch.where(mask, aligned - reference, zero) ** 2).flatten(1).sum(-1)
    )
    # An empty mask has nothing to score, and dividing its zero residual by its
    # own zero weight would propagate a NaN into every pixel's gradient; the
    # clamped denominator scores it zero instead. That agrees with the identity
    # fit `align_scale_shift` falls back to on an empty mask, which is the one
    # case it leaves the maps unaligned — every non-empty mask, however
    # degenerate, is scored against a fit that has at least matched their means.
    mean_squared_residual = squared_residual / weight.flatten(1).sum(-1).clamp(min=1.0)
    # A prediction whose depth ordering is reversed is fitted at scale zero, so
    # the residual above is the flat fit's — the reference's own masked variance
    # — and it is flat in the prediction: no gradient, and nothing telling a
    # sampler which way out of an inverted render. Adding back the variance only
    # a reflection could have explained restores both, at the cost of an energy
    # that exceeds the flat fit exactly where the geometry is inside out. It is
    # identically zero, and so bit-for-bit inert, on any fit the constraint did
    # not bind on.
    return mean_squared_residual + fit.refused_variance


def masked_class_error(
    logits: Float[Tensor, "batch classes height width"],
    target: Int[Tensor, "batch height width"],
    mask: Bool[Tensor, "batch height width"],
    *,
    margin: float,
) -> Float[Tensor, "batch"]:
    """How far the sample's structure has drifted from the reference's *identity*, per batch element.

    Args:
        logits: the prediction's per-class scores, non-negative and normalised
            here into a per-pixel distribution; carries gradient.
        target: the reference photo's class ids, one per pixel; a constant.
        mask: the structural pixels the prediction has not affirmatively covered
            with contents — the only pixels that count.
        margin: the share of a pixel's class mass the reference's class must
            hold before the pixel is considered settled. Above it the pixel
            contributes nothing, forwards or backwards.

    A squared hinge on the reference class's probability, not a cross-entropy.
    `-log p` is unbounded exactly where this term fires — a render that painted a
    wall over the window has `p ≈ 0` — so it would make the mixture of this term
    against a bounded depth energy a function of the frame's worst pixel rather
    than of the weight set for it, and its `1/p` gradient is largest where the
    segmenter is least confident and so least worth believing. The hinge is
    bounded by `margin²`, its gradient by `2·margin`, and its dead zone says the
    thing that is actually meant: keep the window a window, and stop pulling once
    it is one. That dead zone is also what makes a wrong label in the reference
    map survivable — the render pays until it is plausibly that class, not
    forever — and what leaves the term bit-for-bit inert on a prediction that
    already agrees.

    The energy is float32 by contract, like `masked_depth_error`, and an empty
    mask scores zero rather than dividing by its own zero weight.
    """
    if not 0.0 < margin <= 1.0:
        raise ValueError(f"margin must fall in (0, 1], got {margin}")
    zero = torch.zeros((), dtype=_ENERGY_DTYPE, device=logits.device)
    # Excluded pixels are replaced rather than weighted to zero, for the reason
    # `masked_depth_error` spells out: `0 * NaN` is NaN backwards as well as
    # forwards, so a single non-finite score outside the mask would come back as
    # a NaN gradient over the whole image. Substituting zeros leaves those pixels
    # at probability zero and a finite constant deficit, which the final `where`
    # then drops from the sum.
    scores = torch.where(mask.unsqueeze(1), logits.to(_ENERGY_DTYPE), zero)
    probability = scores.gather(1, target.unsqueeze(1)).squeeze(1) / scores.sum(
        dim=1
    ).clamp(min=_SCORE_FLOOR)
    deficit = torch.where(mask, (margin - probability).clamp(min=0.0), zero)
    weight = mask.to(_ENERGY_DTYPE)
    return (deficit**2).flatten(1).sum(-1) / weight.flatten(1).sum(-1).clamp(min=1.0)
