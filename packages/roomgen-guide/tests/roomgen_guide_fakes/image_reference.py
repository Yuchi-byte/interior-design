"""Weightless stand-ins for the image-reference energies: no checkpoint, same contracts."""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Identity
from torch.nn.functional import avg_pool2d

from roomgen_guide.guidance.clip import ClipPreprocessing
from roomgen_guide.guidance.image_reference import (
    ImageReferenceEnergies,
    PerceptualReferenceEnergies,
)
from roomgen_guide_fakes.clip import INPUT_SIZE, FakeClipImageEncoder


@dataclass(frozen=True)
class FakePerceptualExtractor:
    """A two-stage pyramid with no weights: the pixels themselves, then their half-scale pool.

    Smooth in every pixel and deterministic, which is what a gradient test of
    the perceptual energy needs and what a checkpoint-free run of a framework
    can be steered by.
    """

    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> tuple[Float[Tensor, "batch channels height width"], ...]:
        return (pixel_values, avg_pool2d(pixel_values, kernel_size=2))


def fake_image_reference_energies(
    *,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> ImageReferenceEnergies:
    """The embedding-similarity factory with no checkpoint behind it.

    Distinct seeds give distinct fixed projections, so two fake "models" pull a
    render in different directions the way two real backbones would.
    """
    return ImageReferenceEnergies(
        encoder=FakeClipImageEncoder(seed=seed),
        preprocessing=_preprocessing(device=device, dtype=dtype),
        module=Identity(),
    )


def fake_perceptual_energies(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> PerceptualReferenceEnergies:
    """The perceptual-distance factory with no checkpoint behind it."""
    return PerceptualReferenceEnergies(
        extractor=FakePerceptualExtractor(),
        preprocessing=_preprocessing(device=device, dtype=dtype),
        module=Identity(),
    )


def _preprocessing(
    *, device: torch.device | None, dtype: torch.dtype
) -> ClipPreprocessing:
    return ClipPreprocessing(
        size=INPUT_SIZE,
        image_mean=torch.full((1, 3, 1, 1), 0.5, device=device, dtype=dtype),
        image_std=torch.full((1, 3, 1, 1), 0.25, device=device, dtype=dtype),
    )
