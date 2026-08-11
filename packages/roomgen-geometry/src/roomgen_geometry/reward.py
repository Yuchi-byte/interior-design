"""The scale- and shift-invariant masked depth objective."""

from jaxtyping import Bool, Float
from torch import Tensor


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
    raise NotImplementedError


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
