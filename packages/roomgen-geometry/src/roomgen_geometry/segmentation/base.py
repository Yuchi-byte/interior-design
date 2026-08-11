"""The Segmenter contract: images in, per-pixel ADE20K-150 class ids out."""

from typing import Protocol, runtime_checkable

from jaxtyping import Float, Int
from torch import Tensor


@runtime_checkable
class Segmenter(Protocol):
    """A semantic segmentation backend labelling every pixel with a class id.

    Ids index :data:`roomgen_geometry.segmentation.ade20k.ADE20K_LABELS`. The
    result never carries gradient: segmentation is evaluated once on the input
    photo and its output is a constant for the rest of the run.
    """

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]: ...
