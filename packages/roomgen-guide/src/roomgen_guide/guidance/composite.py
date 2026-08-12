"""One energy out of several: the weighted sum a sampler descends."""

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import (
    Guidance,
    require_descended_energy,
    require_per_element,
)


@dataclass(frozen=True)
class WeightedTerm:
    """One term of a composite energy, in the units its own `guidance` returns."""

    guidance: Guidance
    weight: float = 1.0


@dataclass(frozen=True)
class WeightedGuidance:
    """Several constraints on one sample, summed into the single energy a sampler steers by.

    Every term scores the same predicted image and stays on its graph, so the
    gradient the sampler takes is the weighted sum of the terms' own gradients.
    Weights carry the terms' unit conversion as well as their relative pull: two
    energies in different units are not comparable at weight 1.

    The sum is taken per batch element, as `Guidance` requires: summing the
    terms' batch means instead would leave a composite energy that no caller
    could take apart again. A term that hands back a reduced energy is rejected
    rather than broadcast into the sum, where it would contribute its own batch
    mean to every element and read as a term that simply pulls weakly.

    Every term must descend, too: an MFM reward summed in here would subtract
    its constraint from the composite while the sum still looked like an energy.
    """

    terms: tuple[WeightedTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError(
                "a composite energy needs at least one term; an empty sum has no "
                "gradient to steer by"
            )
        for term in self.terms:
            require_descended_energy(term.guidance)

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        energy = self.terms[0].weight * self._term_energy(
            self.terms[0], predicted_image
        )
        for term in self.terms[1:]:
            energy = energy + term.weight * self._term_energy(term, predicted_image)
        return energy

    @staticmethod
    def _term_energy(
        term: WeightedTerm, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        return require_per_element(term.guidance(predicted_image), reward=term.guidance)
