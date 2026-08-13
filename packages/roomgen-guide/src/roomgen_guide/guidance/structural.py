"""The energy that keeps a window a window: class of the sample against the photo."""

from dataclasses import dataclass
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import StructuralReference
from roomgen_geometry.reward import masked_class_error
from roomgen_geometry.segmentation.base import ClassLogitSegmenter
from roomgen_guide.guidance.geometric import MaskObserver

DEFAULT_MARGIN: Final = 0.5
"""Half a pixel's class mass: what the reference's class must hold to be settled."""


@dataclass(frozen=True)
class StructuralGuidance:
    """Masked class error of the current sample against the reference photo.

    The term depth cannot express. A depth energy pins where the wall plane is,
    and a render that paints flat wall over the window or over the doorway
    satisfies it exactly — the geometry is unchanged, the room is not. This
    energy asks the second question about the same pixels: whatever was
    structure in the photo must still read as *that* structure, which is why
    windowpane, excluded from the depth mask because depth behind glass is the
    view's, is scored here.

    Args:
        segmenter: the same segmentation backend, used twice and differently.
            `class_logits` supplies what is scored and carries the gradient back
            to the image; the id path decides what is scored and is a detached
            constant, as it has always been.
        reference: the photo's frozen class map and structural mask.
        erosion_radius: for the prediction's mask, as
            :func:`roomgen_geometry.masking.scorable_mask` takes it.
        margin: handed to :func:`roomgen_geometry.reward.masked_class_error`.
        on_mask: called with the mask after each evaluation decides what it
            scores, for a caller that wants to record it. Observation only.

    The prediction's segmentation excuses only affirmative contents, as
    :func:`roomgen_geometry.masking.scorable_mask` always does: a renovation may
    legitimately stand a bookcase in front of a window, but a pixel the segmenter
    calls sky — or declines to commit on — stays scored. Keeping only affirmative
    shell instead would let a render that replaced the glazing with sky excuse
    itself, inverting the term into one that rewards erasing the window.
    """

    segmenter: ClassLogitSegmenter
    reference: StructuralReference
    erosion_radius: int
    margin: float = DEFAULT_MARGIN
    on_mask: MaskObserver | None = None

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        scored = self._scored_pixels(predicted_image)
        if self.on_mask is not None:
            self.on_mask(scored)
        # Undetached on purpose, and the one place in this repo where a gradient
        # enters the segmentation network: the energy is differentiated through
        # these scores back to the latent, at the cost of a forward *and* a
        # backward through the encoder per guided step.
        logits = self.segmenter.class_logits(predicted_image)
        return masked_class_error(
            logits, self.reference.classes, scored, margin=self.margin
        )

    def _scored_pixels(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Bool[Tensor, "batch height width"]:
        # Detached *and* under `no_grad`, exactly as `GeometricGuidance` reads
        # its mask: which pixels to score is a discrete decision — an argmax, a
        # set membership, an erosion — and there is no gradient in it to keep.
        with torch.no_grad():
            segmentation = self.segmenter(predicted_image.detach())
        uncovered = scorable_mask(
            segmentation, erosion_radius=self.erosion_radius
        )
        return self.reference.mask & uncovered
