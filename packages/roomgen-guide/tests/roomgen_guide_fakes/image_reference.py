"""Weightless stand-ins for the image-reference energies: no checkpoint, same contracts."""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Identity
from torch.nn.functional import adaptive_avg_pool2d, avg_pool2d

from roomgen_guide.guidance.clip import ClipPreprocessing
from roomgen_guide.guidance.image_reference import (
    ImageReferenceEnergies,
    MaskedPatchReferenceEnergies,
    PerceptualReferenceEnergies,
)
from roomgen_guide_fakes.clip import INPUT_SIZE, FakeClipImageEncoder

# Coarse enough that a test can name a patch by painting a region, fine enough
# that a mask covering half the frame masks some patches and not others.
PATCH_GRID_SIDE = 4


@dataclass(frozen=True)
class FakeSpatialPatchEncoder:
    """A `SpatialPatchEncoder` with no weights: each patch token is its cell's mean colour.

    Deterministic, smooth in every pixel, and spatial by construction — a change
    inside one cell moves that cell's token and no other — which is what the
    masked patch energy's tests need to name patches by painting regions.
    """

    grid: int = PATCH_GRID_SIDE

    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> Float[Tensor, "batch tokens embedding"]:
        pooled = adaptive_avg_pool2d(pixel_values, self.grid)
        return pooled.flatten(2).transpose(1, 2)


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


def fake_masked_patch_energies(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> MaskedPatchReferenceEnergies:
    """The mask-restricted patch-cosine factory with no checkpoint behind it."""
    return MaskedPatchReferenceEnergies(
        encoder=FakeSpatialPatchEncoder(),
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
