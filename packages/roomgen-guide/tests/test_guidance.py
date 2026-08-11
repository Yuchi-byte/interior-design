import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.schedule import ConstantStrength, GuidanceSchedule


class MeanBrightnessEnergy:
    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        return predicted_image.mean()


def test_a_fake_energy_stands_in_for_the_geometric_one() -> None:
    images = torch.rand(1, 3, 8, 8, requires_grad=True)
    energy = MeanBrightnessEnergy()

    assert isinstance(energy, Guidance)
    (gradient,) = torch.autograd.grad(energy(images), images)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_constant_strength_ignores_where_in_the_trajectory_it_is_asked() -> None:
    schedule = ConstantStrength(strength=2.5)

    assert isinstance(schedule, GuidanceSchedule)
    assert schedule(0, 50) == schedule(49, 50) == 2.5


def test_a_negative_strength_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ConstantStrength(strength=-1.0)
