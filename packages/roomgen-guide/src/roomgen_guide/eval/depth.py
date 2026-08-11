"""Gate (a): how far a finished render's geometry has drifted from the photo's."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

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
