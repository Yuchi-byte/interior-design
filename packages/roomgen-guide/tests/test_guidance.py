import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.composite import WeightedGuidance, WeightedTerm


class MeanBrightnessEnergy:
    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        return predicted_image.mean()


class TopLeftEnergy:
    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        return predicted_image[:, :, :4, :4].square().mean()


def test_a_fake_energy_stands_in_for_the_geometric_one() -> None:
    images = torch.rand(1, 3, 8, 8, requires_grad=True)
    energy = MeanBrightnessEnergy()

    assert isinstance(energy, Guidance)
    (gradient,) = torch.autograd.grad(energy(images), images)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_a_weighted_sum_steers_by_each_terms_own_share() -> None:
    images = torch.rand(1, 3, 8, 8, requires_grad=True)
    brightness, corner = MeanBrightnessEnergy(), TopLeftEnergy()
    composite = WeightedGuidance(
        (WeightedTerm(brightness, weight=2.0), WeightedTerm(corner, weight=0.5))
    )

    (gradient,) = torch.autograd.grad(composite(images), images)

    (brightness_gradient,) = torch.autograd.grad(brightness(images), images)
    (corner_gradient,) = torch.autograd.grad(corner(images), images)
    torch.testing.assert_close(
        gradient, 2.0 * brightness_gradient + 0.5 * corner_gradient
    )
    # A term at weight 0 is not merely small: it must leave the composite exactly
    # where the remaining terms alone put it.
    silenced = WeightedGuidance(
        (WeightedTerm(brightness, weight=2.0), WeightedTerm(corner, weight=0.0))
    )
    torch.testing.assert_close(
        silenced(images), 2.0 * brightness(images), rtol=0.0, atol=0.0
    )


def test_a_composite_with_no_terms_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one term"):
        WeightedGuidance(())
