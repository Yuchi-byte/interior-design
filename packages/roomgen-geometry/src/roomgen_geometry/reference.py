"""The frozen geometry of the input photo that sampling is steered back towards."""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.segmentation.base import Segmenter


@dataclass(frozen=True)
class GeometricReference:
    """The room as photographed: its inverse depth and its architectural shell.

    Computed once per run and constant thereafter — construction rejects a
    tensor that still carries gradient, because a reference on the graph makes
    the reward differentiate the target it is being compared against.
    """

    disparity: Float[Tensor, "batch height width"]
    mask: Bool[Tensor, "batch height width"]

    def __post_init__(self) -> None:
        if self.disparity.shape != self.mask.shape:
            raise ValueError(
                f"mask shape {tuple(self.mask.shape)} does not match disparity shape "
                f"{tuple(self.disparity.shape)}"
            )
        if self.disparity.requires_grad or self.mask.requires_grad:
            raise ValueError(
                "GeometricReference must be a constant: detach its tensors before "
                "freezing them into a reference"
            )

    @classmethod
    def from_photo(
        cls,
        photo: Float[Tensor, "batch 3 height width"],
        *,
        depth_estimator: DepthEstimator,
        segmenter: Segmenter,
        erosion_radius: int,
    ) -> GeometricReference:
        """Run both backends on the input photo and freeze the result.

        Args:
            photo: NCHW, floating point, values in ``[0, 1]``.
            erosion_radius: passed to
                :func:`roomgen_geometry.masking.architectural_mask`.
        """
        raise NotImplementedError
