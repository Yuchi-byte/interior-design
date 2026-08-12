"""The Guidance contract: a predicted clean image in, one energy per batch element out."""

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

    Low energy means the constraint holds. The energy is **per batch element**
    and is never reduced here: the caller reduces it, because what the reduction
    should be is the caller's property and not the reward's. A sampler taking one
    gradient over a batch reduces with `.mean()`; a sampler that weights members
    of the batch against each other by their own scores — MFM-G's IWAE estimator
    over posterior draws is the worked example — needs them apart, and a reward
    that collapses the batch hands every draw an identical weight while looking
    perfectly healthy.

    The caller differentiates that reduction with respect to the latent, so an
    implementation that detaches anywhere returns a valid-looking energy with no
    gradient behind it.
    """

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]: ...
