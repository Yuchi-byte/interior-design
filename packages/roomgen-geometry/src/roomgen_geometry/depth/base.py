"""The DepthEstimator contract: images in, affine-invariant inverse depth out."""

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class DepthEstimator(Protocol):
    """A monocular depth backend, differentiable with respect to its input.

    Args:
        images: NCHW, floating point, values in ``[0, 1]``. This is the image
            convention throughout the package; a backend converts to whatever
            its network wants internally, without leaving the autograd graph.

    Returns per-pixel *inverse* depth (disparity): larger means nearer, and the
    values are determined only up to an unknown positive scale and an unknown
    shift that differ between any two images, so two predictions are never
    comparable until :func:`roomgen_geometry.alignment.align_scale_shift` has
    removed that freedom.
    """

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch height width"]: ...
