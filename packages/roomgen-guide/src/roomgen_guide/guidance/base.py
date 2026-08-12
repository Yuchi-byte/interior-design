"""The Guidance contract: a predicted clean image in, one energy per batch element out."""

from typing import Final, Literal, Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor

# Which way a batch-callable's numbers point. A `Guidance` is an energy a
# sampler *descends*; Meta Flow Maps' rewards are scores it *ascends*. Both are
# `__call__(images) -> Float[Tensor, "batch"]`, so no structural check — the
# Protocol below, `isinstance`, or mypy — can tell one from the other, and
# handing a score to a driver that descends steers the sample away from
# everything the reward was for, silently and at full strength.
type RewardDirection = Literal["energy", "score"]

DESCENDED_ENERGY: Final[RewardDirection] = "energy"
ASCENDED_SCORE: Final[RewardDirection] = "score"


class WrongDirectionError(TypeError):
    """A reward that is ascended was handed to a caller that descends its energy."""


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

    An implementation may declare `direction: ClassVar[RewardDirection]`; absent,
    it is read as `"energy"`, so a plain callable is still a `Guidance`. Only the
    sign-inverted adapters need it — see :func:`require_descended_energy`.
    """

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]: ...


def require_descended_energy(reward: object) -> None:
    """Reject a reward that declares itself a score where an energy is descended.

    Structural typing cannot separate the two conventions, so the convention is
    declared rather than inferred: an object carrying `direction = "score"` is
    ascended by whoever owns it (MFM's sampler, MFM's exp-tilted fine-tuning
    target) and must not be walked by a driver that steps *down* the gradient.
    """
    direction = getattr(reward, "direction", DESCENDED_ENERGY)
    if direction != DESCENDED_ENERGY:
        raise WrongDirectionError(
            f"{type(reward).__name__} declares direction={direction!r}: it is a "
            f"reward to ascend, and this caller descends the energy it is given, "
            f"so steering by it would optimise away from the reference at full "
            f"strength with every gradient finite and every step in range. Hand "
            f"this driver a Guidance — "
            f"roomgen_guide.guidance.geometric.GeometricGuidance is the same "
            f"depth energy without the sign flip — and keep the MFM adapter for "
            f"MFM's own sampler."
        )


def require_per_element(energy: Tensor, *, reward: object) -> Float[Tensor, "batch"]:
    """Reject an energy that collapsed the batch it scored (cbe914e).

    A `Guidance` returns one number per batch element and the caller reduces. A
    reward that reduces for the caller still broadcasts, still differentiates and
    still produces a healthy-looking walk — but a sampler that weights members of
    a batch against each other by their own energies (MFM-G's IWAE `logsumexp`
    over `mc_samples` posterior draws) then hands every draw an identical weight
    and degenerates into a plain average, with nothing to show for it.
    """
    if energy.ndim == 0:
        raise ValueError(
            f"{type(reward).__name__} returned a 0-dim energy: a Guidance is per "
            f"batch element and the caller reduces (cbe914e). A collapsed energy "
            f"hands every posterior draw an identical weight to a sampler that "
            f"weights draws by their own energies, while the loop runs and the "
            f"gradient stays finite."
        )
    return energy
