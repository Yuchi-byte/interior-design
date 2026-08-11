"""The scale- and shift-invariant masked depth objective."""

from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.alignment import align_scale_shift, apply_scale_shift


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
    """
    aligned = apply_scale_shift(prediction, align_scale_shift(prediction, target, mask))
    weight = mask.to(aligned.dtype)
    squared_residual = (weight * (aligned - target) ** 2).flatten(1).sum(-1)
    # An empty mask has nothing to score, and dividing its zero residual by its
    # own zero weight would propagate a NaN into every pixel's gradient; the
    # clamped denominator scores it zero instead, which is the same
    # nothing-to-say answer as the identity fit `align_scale_shift` falls back
    # to on that input.
    return squared_residual / weight.flatten(1).sum(-1).clamp(min=1.0)


def gradient_matching_error(
    prediction: Float[Tensor, "batch height width"],
    target: Float[Tensor, "batch height width"],
    mask: Bool[Tensor, "batch height width"],
    *,
    levels: int,
) -> Float[Tensor, "batch"]:
    """Masked disagreement in depth *gradients* across `levels` image-pyramid levels.

    Shares the alignment convention of :func:`masked_depth_error`, and sharpens
    the planarity and edge placement that a pure value error tolerates.
    """
    raise NotImplementedError
