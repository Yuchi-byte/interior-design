"""The Guidance contract: a predicted clean image in, a scalar energy out."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class Guidance(Protocol):
    """A constraint on the sample, expressed as an energy the sampler descends.

    Args:
        predicted_image: the decoded x̂₀ of the current step, NCHW,
            floating point, values in ``[0, 1]``, still attached to the latent
            it was formed from.

    Low energy means the constraint holds. The sampler differentiates the
    returned scalar with respect to that latent, so an implementation that
    detaches anywhere returns a valid-looking energy with no gradient behind it.
    """

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]: ...
