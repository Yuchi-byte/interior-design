from dataclasses import dataclass, field

import pytest
import torch
from jaxtyping import Float
from torch import Generator, Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.guidance.schedule import ConstantStrength
from roomgen_guide.sampling.guided import (
    _normalized,
    decode_latent,
    guidance_gradient,
    sample_guided,
)
from roomgen_guide.sampling.pipeline import Img2ImgPipeline
from roomgen_guide_fakes import LATENT_CHANNELS, FakePipeline, FakeScheduler

_PROMPT = "a mid-century living room"
_NORMALIZATION_EPSILON = 1e-12


class MeanBrightnessEnergy:
    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        return predicted_image.mean()


class FlatEnergy:
    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        # Attached to the image but flat in it: a literal constant would leave
        # the energy off the graph entirely, which autograd refuses rather than
        # differentiates to zero.
        return predicted_image.sum() * 0.0


@dataclass(frozen=True)
class AmplifiedEnergy:
    energy: Guidance
    factor: float

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        return self.factor * self.energy(predicted_image)


@dataclass
class RecordingGuidance:
    energy: Guidance
    scored_on_the_graph: list[bool] = field(default_factory=list)

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        self.scored_on_the_graph.append(
            predicted_image.requires_grad and predicted_image.grad_fn is not None
        )
        return self.energy(predicted_image)


def _sample(
    pipeline: FakePipeline,
    photo: Tensor,
    guidance: Guidance,
    *,
    strength: float,
    steps: int = 6,
    noise_strength: float = 0.6,
    text_guidance_scale: float = 7.5,
    seed: int = 0,
    normalize_gradient: bool = False,
) -> Tensor:
    return sample_guided(
        pipeline,
        photo,
        _PROMPT,
        guidance=guidance,
        schedule=ConstantStrength(strength=strength),
        steps=steps,
        noise_strength=noise_strength,
        text_guidance_scale=text_guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        normalize_gradient=normalize_gradient,
    )


def test_a_fake_pipeline_stands_in_for_a_diffusers_one(
    fake_pipeline: FakePipeline,
) -> None:
    assert isinstance(fake_pipeline, Img2ImgPipeline)


def test_the_same_seed_reproduces_the_same_image(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    image = _sample(fake_pipeline, room_photo, geometric_guidance, strength=500.0)
    repeated = _sample(fake_pipeline, room_photo, geometric_guidance, strength=500.0)
    reseeded = _sample(
        fake_pipeline, room_photo, geometric_guidance, strength=500.0, seed=1
    )

    assert image.shape == room_photo.shape
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert torch.equal(image, repeated)
    assert not torch.equal(image, reseeded)


def test_zero_strength_leaves_the_trajectory_unguided(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    unguided = _sample(fake_pipeline, room_photo, geometric_guidance, strength=0.0)
    other_energy = _sample(
        fake_pipeline, room_photo, MeanBrightnessEnergy(), strength=0.0
    )
    guided = _sample(fake_pipeline, room_photo, geometric_guidance, strength=500.0)

    assert torch.equal(unguided, other_energy)
    assert not torch.equal(unguided, guided)


def test_guidance_lowers_the_depth_energy_of_the_sampled_image(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    unguided = _sample(fake_pipeline, room_photo, geometric_guidance, strength=0.0)
    guided = _sample(fake_pipeline, room_photo, geometric_guidance, strength=500.0)

    assert float(geometric_guidance(guided)) < 0.9 * float(geometric_guidance(unguided))


def test_the_raw_energy_gradient_is_the_step_unless_normalization_is_asked_for(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    default = _sample(fake_pipeline, room_photo, geometric_guidance, strength=500.0)
    unnormalized = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=500.0,
        normalize_gradient=False,
    )
    normalized = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=500.0,
        normalize_gradient=True,
    )

    assert torch.equal(default, unnormalized)
    assert not torch.equal(default, normalized)


def test_normalization_leaves_an_unguided_trajectory_alone(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    unguided = _sample(fake_pipeline, room_photo, geometric_guidance, strength=0.0)
    normalized = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=0.0,
        normalize_gradient=True,
    )

    assert torch.equal(unguided, normalized)


def test_a_flat_energy_normalizes_to_no_step_at_all(
    fake_pipeline: FakePipeline, room_photo: Tensor
) -> None:
    unguided = _sample(fake_pipeline, room_photo, FlatEnergy(), strength=0.0)
    normalized = _sample(
        fake_pipeline,
        room_photo,
        FlatEnergy(),
        strength=500.0,
        normalize_gradient=True,
    )

    assert bool(torch.isfinite(normalized).all())
    assert torch.equal(normalized, unguided)


def test_the_normalized_trajectory_is_invariant_to_the_scale_of_the_energy(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    amplified = AmplifiedEnergy(geometric_guidance, factor=1000.0)
    step_size = 0.1

    normalized = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=step_size,
        normalize_gradient=True,
    )
    normalized_amplified = _sample(
        fake_pipeline,
        room_photo,
        amplified,
        strength=step_size,
        normalize_gradient=True,
    )
    raw = _sample(fake_pipeline, room_photo, geometric_guidance, strength=step_size)
    raw_amplified = _sample(fake_pipeline, room_photo, amplified, strength=step_size)

    torch.testing.assert_close(normalized_amplified, normalized, rtol=1e-4, atol=1e-6)
    assert not torch.allclose(raw_amplified, raw, rtol=1e-4, atol=1e-6)


def test_the_normalized_gradient_keeps_its_direction_at_unit_rms(
    generator: Generator,
) -> None:
    gradient = torch.randn((1, LATENT_CHANNELS, 8, 8), generator=generator)

    normalized = _normalized(gradient, _NORMALIZATION_EPSILON)
    amplified = _normalized(gradient * 1e6, _NORMALIZATION_EPSILON)

    root_mean_square = normalized.square().mean().sqrt()
    torch.testing.assert_close(root_mean_square, torch.ones(()))
    torch.testing.assert_close(
        normalized * gradient.square().mean().sqrt(), gradient, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(amplified, normalized)


def test_a_zero_gradient_normalizes_to_zero() -> None:
    zero = torch.zeros((1, LATENT_CHANNELS, 8, 8))

    assert torch.equal(_normalized(zero, _NORMALIZATION_EPSILON), zero)


def test_the_guided_step_differentiates_the_energy_through_the_predicted_original(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    generator: Generator,
) -> None:
    latent = torch.randn((1, LATENT_CHANNELS, 8, 8), generator=generator)
    noise_prediction = torch.randn((1, LATENT_CHANNELS, 8, 8), generator=generator)
    alpha_cumprod = torch.tensor(0.5)

    steering = guidance_gradient(
        fake_pipeline.vae,
        geometric_guidance,
        latent=latent,
        noise_prediction=noise_prediction,
        alpha_cumprod=alpha_cumprod,
    )

    # Tweedie's x̂₀ spelled out, so the step is pinned to that estimate rather
    # than to whatever the loop happens to feed the decoder.
    differentiable_latent = latent.clone().requires_grad_(True)
    predicted_original = (
        differentiable_latent - (1.0 - alpha_cumprod).sqrt() * noise_prediction
    ) / alpha_cumprod.sqrt()
    expected = assert_gradient_is_finite_and_nonzero(
        geometric_guidance(decode_latent(fake_pipeline.vae, predicted_original)),
        differentiable_latent,
    )

    torch.testing.assert_close(steering, expected)


def test_the_image_the_guidance_scores_is_still_attached_to_the_latent(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    recording = RecordingGuidance(geometric_guidance)

    _sample(fake_pipeline, room_photo, recording, strength=1.0)

    assert recording.scored_on_the_graph == [True] * 3


def test_noise_strength_decides_how_much_of_the_trajectory_is_redone(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    partial = RecordingGuidance(geometric_guidance)
    complete = RecordingGuidance(geometric_guidance)

    _sample(
        fake_pipeline, room_photo, partial, strength=1.0, steps=4, noise_strength=0.5
    )
    _sample(
        fake_pipeline, room_photo, complete, strength=1.0, steps=4, noise_strength=1.0
    )

    assert len(partial.scored_on_the_graph) == 2
    assert len(complete.scored_on_the_graph) == 4


def test_classifier_free_guidance_on_the_prompt_changes_the_image(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    conditioned = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=0.0,
        text_guidance_scale=7.5,
    )
    unamplified = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=0.0,
        text_guidance_scale=1.0,
    )

    assert not torch.equal(conditioned, unamplified)


def test_a_scheduler_that_is_not_epsilon_parameterised_is_rejected(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    pipeline = FakePipeline(scheduler=FakeScheduler(prediction_type="v_prediction"))

    with pytest.raises(ValueError, match="prediction_type"):
        _sample(pipeline, room_photo, geometric_guidance, strength=1.0)


@pytest.mark.parametrize("noise_strength", [0.0, 1.5])
def test_a_noise_strength_outside_the_unit_interval_is_rejected(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
    noise_strength: float,
) -> None:
    with pytest.raises(ValueError, match="noise_strength"):
        _sample(
            fake_pipeline,
            room_photo,
            geometric_guidance,
            strength=1.0,
            noise_strength=noise_strength,
        )
