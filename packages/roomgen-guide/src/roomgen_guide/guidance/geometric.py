"""The energy that holds the room still: depth of the sample against the photo."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.masking import MaskMode, scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.reward import masked_depth_error
from roomgen_geometry.segmentation.base import Segmenter

MaskObserver = Callable[[Bool[Tensor, "batch height width"]], None]
"""Handed the mask each guided step scored."""


@dataclass(frozen=True)
class GeometricGuidance:
    """Masked depth error of the current sample against the reference photo.

    The same depth backend that built the reference is re-run on the predicted
    clean image, so the two predictions are comparable once aligned; the whole
    evaluation stays differentiable back to the latent.

    Args:
        segmenter: re-segments the predicted clean image at every step, so the
            score falls on the reference's shell *intersected* with the
            prediction's own — furniture the renovation puts in front of a wall
            stops being scored, while a wall that has moved still is. `None`
            scores the reference's shell alone, occlusion included.
        erosion_radius: for the prediction's mask, as
            :func:`roomgen_geometry.masking.architectural_mask` takes it;
            required with `segmenter` and meaningless without it.
        mask_mode: which way the prediction's segmentation is read, as
            :func:`roomgen_geometry.masking.scorable_mask` defines it —
            `intersect_shell` scores only what the prediction also calls shell,
            `subtract_contents` scores everything it does not affirmatively call
            contents. The eval that judges the run must read it the same way.
        on_mask: called with the mask after each evaluation decides what it
            scores, for a caller that wants to record it. Observation only —
            the energy is what it would be without one.
    """

    depth_estimator: DepthEstimator
    reference: GeometricReference
    segmenter: Segmenter | None = None
    erosion_radius: int | None = None
    mask_mode: MaskMode = "intersect_shell"
    on_mask: MaskObserver | None = None

    def __post_init__(self) -> None:
        if (self.segmenter is None) != (self.erosion_radius is None):
            raise ValueError(
                "re-segmenting the prediction needs both a segmenter and an "
                "erosion_radius; pass neither to score the reference's shell alone"
            )

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        scored = self._scored_pixels(predicted_image)
        if self.on_mask is not None:
            self.on_mask(scored)
        disparity = self.depth_estimator(predicted_image)
        return masked_depth_error(disparity, self.reference.disparity, scored)

    def _scored_pixels(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Bool[Tensor, "batch height width"]:
        segmenter, erosion_radius = self.segmenter, self.erosion_radius
        if segmenter is None or erosion_radius is None:
            return self.reference.mask
        # Detached *and* under `no_grad`: the segmentation is a constant of the
        # step, so not even a graph is built for it. The undetached
        # `predicted_image` is what the depth backend sees, and is what carries
        # the gradient back to the latent.
        with torch.no_grad():
            segmentation = segmenter(predicted_image.detach())
        prediction_shell = scorable_mask(
            segmentation, mode=self.mask_mode, erosion_radius=erosion_radius
        )
        return self.reference.mask & prediction_shell
