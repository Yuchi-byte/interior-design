"""The scale- and shift-invariant masked depth objective."""

from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.alignment import align_scale_shift, apply_scale_shift

# The energy is a sum over every masked pixel of a squared residual, and at
# 512x512 that is a quarter of a million terms: in float16, whose largest finite
# value is 65504, the sum saturates to inf long before the division rescales it
# and the whole backward pass comes back zero — guidance silently switched off
# while sampling looks perfectly healthy. The cast is an autograd op like any
# other, so the gradient still reaches a half-precision prediction; only the
# reduction is lifted.
_ENERGY_DTYPE: Final = torch.float32


def masked_depth_error(
    prediction: Float[Tensor, "batch height width"],
    target: Float[Tensor, "batch height width"],
    mask: Bool[Tensor, "batch height width"],
) -> Float[Tensor, "batch"]:
    """How far the sample's geometry has drifted from the reference's, per batch element.

    Args:
        prediction: inverse depth of the current sample; carries gradient.
        target: inverse depth of the reference photo; a detached constant.
        mask: the architectural shell, the only pixels that count.

    Both maps are affine-invariant inverse depth, so they share no frame until
    :func:`roomgen_geometry.alignment.align_scale_shift` has fitted the
    prediction onto the target over the masked pixels; the error is the masked
    mean squared residual of that fit, and is therefore unchanged by any affine
    rescaling of `prediction`.

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
    aligned = apply_scale_shift(
        predicted, align_scale_shift(predicted, reference, mask)
    )
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
    return squared_residual / weight.flatten(1).sum(-1).clamp(min=1.0)
