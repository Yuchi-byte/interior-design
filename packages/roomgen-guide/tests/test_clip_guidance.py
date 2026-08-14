from dataclasses import replace

import pytest
import torch
from torch import Generator, Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.clip import ClipTextGuidance
from roomgen_guide.sampling.guided import decode_latent
from roomgen_guide_fakes import (
    EMBEDDING_WIDTH,
    LATENT_CHANNELS,
    FakePipeline,
    fake_clip_text_guidance,
)

_PROMPT = "a warm mid-century living room"


@pytest.fixture
def clip_guidance() -> ClipTextGuidance:
    return fake_clip_text_guidance(_PROMPT)


def test_the_energy_falls_as_the_render_moves_towards_the_prompt(
    clip_guidance: ClipTextGuidance, room_photo: Tensor
) -> None:
    # The prompt's own direction in embedding space, walked towards in pixel
    # space: the encoder is a fixed linear map of the pooled pixels, so the
    # steepest-descent direction on the energy is a real image perturbation.
    (gradient,) = torch.autograd.grad(
        clip_guidance(room_photo.requires_grad_(True)).mean(), room_photo
    )
    towards = -gradient / gradient.abs().max()

    near = float(clip_guidance((room_photo.detach() + 0.05 * towards).clamp(0.0, 1.0)))
    far = float(clip_guidance((room_photo.detach() - 0.05 * towards).clamp(0.0, 1.0)))

    assert isinstance(clip_guidance, Guidance)
    assert near < float(clip_guidance(room_photo.detach())) < far


def test_the_energy_is_the_negated_cosine_and_so_bounded(
    clip_guidance: ClipTextGuidance, room_photo: Tensor
) -> None:
    aligned = replace(
        clip_guidance,
        text_features=clip_guidance.encoder(clip_guidance.preprocessing(room_photo)),
    )

    assert float(aligned(room_photo)) == pytest.approx(-1.0)
    assert float(
        replace(aligned, text_features=-aligned.text_features)(room_photo)
    ) == (pytest.approx(1.0))


def test_the_energy_scores_every_batch_element_on_its_own(
    clip_guidance: ClipTextGuidance, room_photo: Tensor
) -> None:
    batch = torch.cat([room_photo, torch.zeros_like(room_photo)])

    energy = clip_guidance(batch)

    assert energy.shape == (2,)
    assert float(energy[0]) != float(energy[1])


def test_the_energy_gradient_reaches_the_latent_through_the_decoder(
    clip_guidance: ClipTextGuidance, fake_pipeline: FakePipeline, generator: Generator
) -> None:
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    energy = clip_guidance(decode_latent(fake_pipeline.vae, latent))

    gradient = assert_gradient_is_finite_and_nonzero(energy.mean(), latent)
    assert gradient.shape == latent.shape


def test_a_prompt_embedding_still_on_the_graph_is_refused(
    clip_guidance: ClipTextGuidance,
) -> None:
    # A target that is itself differentiated is a target the walk can move
    # instead of moving the render towards it.
    with pytest.raises(ValueError, match="constant"):
        replace(
            clip_guidance,
            text_features=torch.zeros((1, EMBEDDING_WIDTH), requires_grad=True),
        )
