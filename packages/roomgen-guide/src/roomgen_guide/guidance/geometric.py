"""The energy that holds the room still: depth of the sample against the photo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.masking import architectural_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.reward import masked_depth_error
from roomgen_geometry.segmentation.base import Segmenter

MaskObserver = Callable[[float, Bool[Tensor, "batch height width"]], None]
"""Handed each guided step's trajectory position and the mask that step scored."""


@dataclass(frozen=True)
class GeometricGuidance:
    """Masked depth error of the current sample against the reference photo.

    The same depth backend that built the reference is re-run on the predicted
    clean image, so the two predictions are comparable once aligned; the whole
    evaluation stays differentiable back to the latent.

    Args:
        segmenter: re-segments the predicted clean image once the trajectory is
            far enough along to be worth segmenting, so the score falls on the
            reference's shell *intersected* with the prediction's own — furniture
            the renovation puts in front of a wall stops being scored, while a
            wall that has moved still is. `None` scores the reference's shell
            alone, occlusion included, for the whole trajectory.
        erosion_radius: for the prediction's mask, as
            :func:`roomgen_geometry.masking.architectural_mask` takes it;
            required with `segmenter` and meaningless without it.
        resegment_after: fraction of the walked trajectory before which the
            reference's shell is scored alone. Early x̂₀ is blur the segmenter
            reads as furniture-free, so its shell covers nearly everything and
            intersecting with it is a no-op that still spends the step's budget
            pulling nascent furniture towards the reference's wall depth; layout
            forms early and there is nothing there to occlude yet. 0 intersects
            from the first step, 1 only on the last.
        on_mask: called with `(progress, mask)` after each evaluation decides
            what it scores, for a caller that wants to record it. Observation
            only — the energy is what it would be without one.
    """

    depth_estimator: DepthEstimator
    reference: GeometricReference
    segmenter: Segmenter | None = None
    erosion_radius: int | None = None
    resegment_after: float = 0.5
    on_mask: MaskObserver | None = None

    def __post_init__(self) -> None:
        if (self.segmenter is None) != (self.erosion_radius is None):
            raise ValueError(
                "re-segmenting the prediction needs both a segmenter and an "
                "erosion_radius; pass neither to score the reference's shell alone"
            )
        if not 0.0 <= self.resegment_after <= 1.0:
            raise ValueError(
                "resegment_after is a fraction of the walked trajectory and must "
                f"lie in [0, 1], got {self.resegment_after}"
            )

    def __call__(
        self,
        predicted_image: Float[Tensor, "batch 3 height width"],
        *,
        progress: float,
    ) -> Float[Tensor, ""]:
        scored = self._scored_pixels(predicted_image, progress=progress)
        if self.on_mask is not None:
            self.on_mask(progress, scored)
        disparity = self.depth_estimator(predicted_image)
        error = masked_depth_error(disparity, self.reference.disparity, scored)
        return error.mean()

    def _scored_pixels(
        self,
        predicted_image: Float[Tensor, "batch 3 height width"],
        *,
        progress: float,
    ) -> Bool[Tensor, "batch height width"]:
        segmenter, erosion_radius = self.segmenter, self.erosion_radius
        if segmenter is None or erosion_radius is None:
            return self.reference.mask
        if progress < self.resegment_after:
            return self.reference.mask
        # Detached *and* under `no_grad`: the segmentation is a constant of the
        # step, so not even a graph is built for it. The undetached
        # `predicted_image` is what the depth backend sees, and is what carries
        # the gradient back to the latent.
        with torch.no_grad():
            segmentation = segmenter(predicted_image.detach())
        prediction_shell = architectural_mask(
            segmentation, erosion_radius=erosion_radius
        )
        return self.reference.mask & prediction_shell
