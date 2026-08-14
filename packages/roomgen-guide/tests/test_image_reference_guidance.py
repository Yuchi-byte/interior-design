from dataclasses import replace

import pytest
import torch
from torch import Generator, Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.image_reference import (
    ImageReferenceEnergyFactory,
    ImageReferenceGuidance,
    PerceptualReferenceGuidance,
)
from roomgen_guide.sampling.guided import decode_latent
from roomgen_guide_fakes import (
    EMBEDDING_WIDTH,
    LATENT_CHANNELS,
    FakePipeline,
    fake_image_reference_energies,
    fake_perceptual_energies,
)


@pytest.fixture
def image_guidance(room_photo: Tensor) -> ImageReferenceGuidance:
    return fake_image_reference_energies()(room_photo)


@pytest.fixture
def perceptual_guidance(room_photo: Tensor) -> PerceptualReferenceGuidance:
    return fake_perceptual_energies()(room_photo)


def test_both_factories_freeze_the_reference_and_satisfy_the_contracts(
    image_guidance: ImageReferenceGuidance,
    perceptual_guidance: PerceptualReferenceGuidance,
) -> None:
    assert isinstance(image_guidance, Guidance)
    assert isinstance(perceptual_guidance, Guidance)
    assert isinstance(fake_image_reference_energies(), ImageReferenceEnergyFactory)
    assert isinstance(fake_perceptual_energies(), ImageReferenceEnergyFactory)
    assert not image_guidance.reference_features.requires_grad
    assert all(
        not stage.requires_grad for stage in perceptual_guidance.reference_features
    )


def test_the_reference_photo_itself_is_each_energys_optimum(
    image_guidance: ImageReferenceGuidance,
    perceptual_guidance: PerceptualReferenceGuidance,
    room_photo: Tensor,
    generator: Generator,
) -> None:
    other = torch.rand(room_photo.shape, generator=generator)

    # The cosine energy bottoms out at exactly -1 on the photo it froze; the
    # perceptual distance at exactly 0. Any other image scores above both.
    assert float(image_guidance(room_photo)) == pytest.approx(-1.0)
    assert float(perceptual_guidance(room_photo)) == pytest.approx(0.0, abs=1e-10)
    assert float(image_guidance(other)) > -1.0
    assert float(perceptual_guidance(other)) > 0.0


def test_the_energy_falls_as_the_render_moves_towards_the_photo(
    image_guidance: ImageReferenceGuidance, room_photo: Tensor, generator: Generator
) -> None:
    render = torch.rand(room_photo.shape, generator=generator).requires_grad_(True)
    (gradient,) = torch.autograd.grad(image_guidance(render).mean(), render)
    towards = -gradient / gradient.abs().max()

    near = float(image_guidance((render.detach() + 0.05 * towards).clamp(0.0, 1.0)))
    far = float(image_guidance((render.detach() - 0.05 * towards).clamp(0.0, 1.0)))

    assert near < float(image_guidance(render.detach())) < far


def test_each_energy_scores_every_batch_element_on_its_own(
    image_guidance: ImageReferenceGuidance,
    perceptual_guidance: PerceptualReferenceGuidance,
    room_photo: Tensor,
) -> None:
    batch = torch.cat([room_photo, torch.zeros_like(room_photo)])

    for energy in (image_guidance, perceptual_guidance):
        scores = energy(batch)
        assert scores.shape == (2,)
        assert float(scores[0]) != float(scores[1])


def test_each_energys_gradient_reaches_the_latent_through_the_decoder(
    image_guidance: ImageReferenceGuidance,
    perceptual_guidance: PerceptualReferenceGuidance,
    fake_pipeline: FakePipeline,
    generator: Generator,
) -> None:
    for energy in (image_guidance, perceptual_guidance):
        latent = torch.randn(
            (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
        )

        scores = energy(decode_latent(fake_pipeline.vae, latent))

        gradient = assert_gradient_is_finite_and_nonzero(scores.mean(), latent)
        assert gradient.shape == latent.shape


def test_a_reference_still_on_the_graph_is_refused(
    image_guidance: ImageReferenceGuidance,
    perceptual_guidance: PerceptualReferenceGuidance,
) -> None:
    # A target that is itself differentiated is a target the walk can move
    # instead of moving the render towards it.
    with pytest.raises(ValueError, match="constant"):
        replace(
            image_guidance,
            reference_features=torch.zeros((1, EMBEDDING_WIDTH), requires_grad=True),
        )
    with pytest.raises(ValueError, match="constant"):
        replace(
            perceptual_guidance,
            reference_features=tuple(
                stage.clone().requires_grad_(True)
                for stage in perceptual_guidance.reference_features
            ),
        )


def test_a_mismatched_feature_pyramid_is_refused(
    perceptual_guidance: PerceptualReferenceGuidance, room_photo: Tensor
) -> None:
    truncated = replace(
        perceptual_guidance,
        reference_features=perceptual_guidance.reference_features[:1],
    )

    with pytest.raises(ValueError, match="same extractor"):
        truncated(room_photo)


def test_an_empty_feature_pyramid_is_refused(
    perceptual_guidance: PerceptualReferenceGuidance,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        replace(perceptual_guidance, reference_features=())
