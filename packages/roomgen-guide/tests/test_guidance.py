import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import Guidance


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
