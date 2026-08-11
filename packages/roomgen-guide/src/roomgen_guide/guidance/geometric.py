"""The energy that holds the room still: depth of the sample against the photo."""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.reference import GeometricReference


@dataclass(frozen=True)
class GeometricGuidance:
    """Masked depth error of the current sample against the reference photo.

    The same depth backend that built the reference is re-run on the predicted
    clean image, so the two predictions are comparable once aligned; the whole
    evaluation stays differentiable back to the latent.
    """

    depth_estimator: DepthEstimator
    reference: GeometricReference

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        raise NotImplementedError
