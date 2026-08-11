"""Gate (b): the boundary every prompt-adherence metric sits behind."""

from __future__ import annotations

from typing import Protocol

from jaxtyping import Float
from torch import Tensor

from roomgen_geometry.reference import GeometricReference


class PromptAdherence(Protocol):
    """Higher means the render is more of a renovation and less of a copy.

    Args:
        image: the render, NCHW in ``[0, 1]``.
        prompt: what the render was asked for.
        photo: the input photo the render started from, same shape.
        reference: the frozen geometry, whose mask separates the shell that must
            not move from the contents that were free to change.
    """

    def __call__(
        self,
        image: Float[Tensor, "batch 3 height width"],
        *,
        prompt: str,
        photo: Float[Tensor, "batch 3 height width"],
        reference: GeometricReference,
    ) -> float: ...
