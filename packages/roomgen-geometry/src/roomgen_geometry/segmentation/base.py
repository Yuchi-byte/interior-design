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


@runtime_checkable
class ClassLogitSegmenter(Segmenter, Protocol):
    """A `Segmenter` that also exposes the per-pixel class scores its ids are read off.

    Two paths into one network, deliberately different:

    - `__call__` decides *which pixels* a reward may score. That is a discrete
      decision — an argmax, a set membership, an erosion — with no derivative
      worth keeping, so it stays a detached constant of the step.
    - `class_logits` is *what* a reward scores. It is a smooth function of the
      image, it carries gradient back to it, and it is the only thing on the
      segmentation side that can.

    The scores are non-negative per-class mass rather than log-odds (EoMT sums
    each query's class distribution weighted by that query's mask probability),
    so a caller comparing classes normalises them itself; what the contract
    fixes is that `__call__` is this tensor's own argmax, so the mask and the
    quantity it gates can never disagree about what the network saw.
    """

    def class_logits(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch classes height width"]: ...
