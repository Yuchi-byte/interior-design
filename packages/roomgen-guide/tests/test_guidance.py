from itertools import pairwise

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.schedule import (
    ConstantStrength,
    DecayingStrength,
    GuidanceSchedule,
)


class MeanBrightnessEnergy:
    def __call__(
        self,
        predicted_image: Float[Tensor, "batch 3 height width"],
        *,
        progress: float,
    ) -> Float[Tensor, ""]:
        return predicted_image.mean()


def test_a_fake_energy_stands_in_for_the_geometric_one() -> None:
    images = torch.rand(1, 3, 8, 8, requires_grad=True)
    energy = MeanBrightnessEnergy()

    assert isinstance(energy, Guidance)
    (gradient,) = torch.autograd.grad(energy(images, progress=0.0), images)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_constant_strength_ignores_where_in_the_trajectory_it_is_asked() -> None:
    schedule = ConstantStrength(strength=2.5)

    assert isinstance(schedule, GuidanceSchedule)
    assert schedule(0, 50) == schedule(49, 50) == 2.5


def test_a_negative_strength_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ConstantStrength(strength=-1.0)


def test_decaying_strength_falls_from_initial_to_final_through_their_midpoint() -> None:
    schedule = DecayingStrength(initial=10.0, final=2.0)

    assert isinstance(schedule, GuidanceSchedule)
    assert schedule(0, 5) == 10.0
    assert schedule(2, 5) == pytest.approx(6.0)
    assert schedule(4, 5) == 2.0


def test_decaying_strength_never_rises_and_never_dips_below_its_final_value() -> None:
    schedule = DecayingStrength(initial=8.0, final=1.5)
    strengths = [schedule(step, 32) for step in range(32)]

    assert all(later <= earlier for earlier, later in pairwise(strengths))
    assert max(strengths) == 8.0
    assert min(strengths) == 1.5


def test_a_decaying_strength_that_rises_over_the_trajectory_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least"):
        DecayingStrength(initial=1.0, final=2.0)


def test_a_negative_final_strength_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DecayingStrength(initial=1.0, final=-1.0)
