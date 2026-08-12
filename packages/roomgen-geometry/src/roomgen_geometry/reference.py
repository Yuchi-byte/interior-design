"""The frozen geometry of the input photo that sampling is steered back towards."""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Bool, Float, Int
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.masking import architectural_mask, structural_mask
from roomgen_geometry.segmentation.ade20k import NUM_CLASSES
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
        mask = architectural_mask(segmenter(photo), erosion_radius=erosion_radius)
        # The depth backend is differentiable by design, so its output is on the
        # graph whenever the photo is; the reference is a target and must not be.
        return cls(disparity=depth_estimator(photo).detach(), mask=mask)


@dataclass(frozen=True)
class StructuralReference:
    """The room's structural *identity* as photographed: what each pixel is, and where that claim holds.

    The class term's target, and the counterpart to `GeometricReference`:
    together they say a renovation may not move the shell (depth) and may not
    change what its surfaces and openings *are* (class). Windowpane belongs to
    this half only — depth behind glass reads the view, while the fact that
    there is a window there is the room's.
    """

    classes: Int[Tensor, "batch height width"]
    mask: Bool[Tensor, "batch height width"]

    def __post_init__(self) -> None:
        if self.classes.shape != self.mask.shape:
            raise ValueError(
                f"mask shape {tuple(self.mask.shape)} does not match class-map shape "
                f"{tuple(self.classes.shape)}"
            )
        # Checked once here rather than per step in the reward, where the ids
        # index a `gather` over the class axis: an id off the end of the table is
        # a device-side assert mid-sample on GPU, and reading `.max()` every step
        # to catch it would cost a synchronisation on the guidance path.
        low, high = int(self.classes.min()), int(self.classes.max())
        if low < 0 or high >= NUM_CLASSES:
            raise ValueError(
                f"class ids outside the ADE20K-150 table: [{low}, {high}] — a "
                f"Segmenter returns ids indexing ADE20K_LABELS"
            )

    @classmethod
    def from_photo(
        cls,
        photo: Float[Tensor, "batch 3 height width"],
        *,
        segmenter: Segmenter,
        erosion_radius: int,
    ) -> StructuralReference:
        """Segment the input photo once and freeze what it says the room is made of.

        Args:
            photo: NCHW, floating point, values in ``[0, 1]``.
            erosion_radius: passed to
                :func:`roomgen_geometry.masking.structural_mask`.

        A `Segmenter`'s ids never carry gradient, so there is nothing to detach:
        the constant is a constant by construction.
        """
        segmentation = segmenter(photo)
        return cls(
            classes=segmentation,
            mask=structural_mask(segmentation, erosion_radius=erosion_radius),
        )
