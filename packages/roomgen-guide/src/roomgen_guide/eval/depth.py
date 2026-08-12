"""Gate (a): how far a finished render's geometry has drifted from the photo's."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_geometry.alignment import align_scale_shift, apply_scale_shift
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.reward import masked_depth_error


def depth_error(
    image: Float[Tensor, "batch 3 height width"],
    *,
    reference: GeometricReference,
    depth_estimator: DepthEstimator,
) -> float:
    """Masked, scale-and-shift-invariant depth error of `image` against `reference`.

    Args:
        image: NCHW, floating point, values in ``[0, 1]``.
        depth_estimator: the backend `reference` was built with — a reference is
            only meaningful to the backend that produced it.

    The score is a plain number, taken under ``no_grad``: nothing downstream of
    a finished render is differentiated.
    """
    with torch.no_grad():
        disparity = depth_estimator(image)
        error = masked_depth_error(disparity, reference.disparity, reference.mask)
    return float(error.mean())


def signed_depth_residual_map(
    image: Float[Tensor, "batch 3 height width"],
    *,
    reference: GeometricReference,
    depth_estimator: DepthEstimator,
) -> Float[Tensor, "batch height width"]:
    """Which way `image` bent the room, pixel by pixel: aligned disparity minus reference.

    Args:
        image: NCHW, floating point, values in ``[0, 1]``.
        depth_estimator: the backend `reference` was built with.

    Positive where the render's inverse depth sits above the photo's — the
    surface reads closer than it was — negative where it reads further away,
    and zero outside the shell mask. In the aligned inverse-depth units of
    `depth_error`, whose score is the mean of this map's square over the mask.
    Taken under ``no_grad``, like every finished-render measurement.
    """
    with torch.no_grad():
        disparity = depth_estimator(image)
        # Affine-invariant inverse depth shares no frame with the reference
        # until it is fitted onto it, exactly as `masked_depth_error` does.
        aligned = apply_scale_shift(
            disparity,
            align_scale_shift(disparity, reference.disparity, reference.mask),
        )
        weight = reference.mask.to(aligned.dtype)
        return weight * (aligned - reference.disparity)
