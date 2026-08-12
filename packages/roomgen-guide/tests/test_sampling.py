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
    _guided_update,
    _prompt_embeddings,
    decode_latent,
    guidance_gradient,
    guidance_gradient_through_unet,
    sample_guided,
)
from roomgen_guide.sampling.pipeline import Img2ImgPipeline
from roomgen_guide_fakes import LATENT_CHANNELS, FakePipeline, FakeScheduler

_PROMPT = "a mid-century living room"
# The raw step at strength 500 on this stack has an RMS of a few hundredths, so
# one cap sits well clear of it and the other binds on every step.
_SLACK_CAP = 1.0
_BINDING_CAP = 0.01


def _normalized(gradient: Tensor) -> Tensor:
    return _guided_update(
        gradient, strength=1.0, normalize_gradient=True, clip_step_rms=None
    )


def _clipped(update: Tensor, cap: float) -> Tensor:
    return _guided_update(
        update, strength=1.0, normalize_gradient=False, clip_step_rms=cap
    )


class MeanBrightnessEnergy:
    def __call__(
        self,
        predicted_image: Float[Tensor, "batch 3 height width"],
        *,
        progress: float,
    ) -> Float[Tensor, ""]:
        return predicted_image.mean()


class FlatEnergy:
    def __call__(
        self,
        predicted_image: Float[Tensor, "batch 3 height width"],
        *,
        progress: float,
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
        self,
        predicted_image: Float[Tensor, "batch 3 height width"],
        *,
        progress: float,
    ) -> Float[Tensor, ""]:
        return self.factor * self.energy(predicted_image, progress=progress)


@dataclass
class RecordingGuidance:
    energy: Guidance
    scored_on_the_graph: list[bool] = field(default_factory=list)
    walked: list[float] = field(default_factory=list)

    def __call__(
        self,
        predicted_image: Float[Tensor, "batch 3 height width"],
        *,
        progress: float,
    ) -> Float[Tensor, ""]:
        self.scored_on_the_graph.append(
            predicted_image.requires_grad and predicted_image.grad_fn is not None
        )
        self.walked.append(progress)
        return self.energy(predicted_image, progress=progress)


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
    clip_step_rms: float | None = None,
    differentiate_through_unet: bool = False,
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
        clip_step_rms=clip_step_rms,
        differentiate_through_unet=differentiate_through_unet,
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

    assert float(geometric_guidance(guided, progress=1.0)) < 0.9 * float(
        geometric_guidance(unguided, progress=1.0)
    )


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

    normalized = _normalized(gradient)
    amplified = _normalized(gradient * 1e6)

    root_mean_square = normalized.square().mean().sqrt()
    torch.testing.assert_close(root_mean_square, torch.ones(()))
    torch.testing.assert_close(
        normalized * gradient.square().mean().sqrt(), gradient, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(amplified, normalized)


def test_a_zero_gradient_normalizes_to_zero() -> None:
    zero = torch.zeros((1, LATENT_CHANNELS, 8, 8))

    assert torch.equal(_normalized(zero), zero)


def test_a_step_under_the_cap_is_the_raw_step_untouched(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    raw = _sample(fake_pipeline, room_photo, geometric_guidance, strength=500.0)
    clipped = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=500.0,
        clip_step_rms=_SLACK_CAP,
    )
    bound = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=500.0,
        clip_step_rms=_BINDING_CAP,
    )

    assert torch.equal(raw, clipped)
    assert not torch.equal(raw, bound)


def test_a_step_over_the_cap_lands_exactly_on_it_without_turning(
    generator: Generator,
) -> None:
    update = torch.randn((1, LATENT_CHANNELS, 8, 8), generator=generator) * 100.0

    clipped = _clipped(update, _BINDING_CAP)

    torch.testing.assert_close(
        clipped.square().mean().sqrt(), torch.tensor(_BINDING_CAP)
    )
    torch.testing.assert_close(
        _normalized(clipped),
        _normalized(update),
    )
    assert torch.equal(_clipped(update, 1e6), update)


def test_the_capped_trajectory_stops_growing_with_the_energy(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    amplified = AmplifiedEnergy(geometric_guidance, factor=1000.0)

    capped = _sample(
        fake_pipeline,
        room_photo,
        amplified,
        strength=500.0,
        clip_step_rms=_BINDING_CAP,
    )
    # A cap that binds on every step *is* a fixed step of that size along the
    # gradient, which is what normalization spells out directly.
    at_the_cap = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=_BINDING_CAP,
        normalize_gradient=True,
    )
    uncapped = _sample(fake_pipeline, room_photo, amplified, strength=500.0)

    torch.testing.assert_close(capped, at_the_cap, rtol=1e-4, atol=1e-6)
    assert not torch.allclose(uncapped, at_the_cap, rtol=1e-4, atol=1e-6)


def test_a_flat_energy_clips_to_no_step_at_all(
    fake_pipeline: FakePipeline, room_photo: Tensor
) -> None:
    unguided = _sample(fake_pipeline, room_photo, FlatEnergy(), strength=0.0)
    clipped = _sample(
        fake_pipeline,
        room_photo,
        FlatEnergy(),
        strength=500.0,
        clip_step_rms=_BINDING_CAP,
    )

    assert bool(torch.isfinite(clipped).all())
    assert torch.equal(clipped, unguided)


def test_capping_a_step_that_is_already_fixed_size_is_rejected(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    with pytest.raises(ValueError, match="clip_step_rms"):
        _sample(
            fake_pipeline,
            room_photo,
            geometric_guidance,
            strength=500.0,
            clip_step_rms=_BINDING_CAP,
            normalize_gradient=True,
        )


def test_a_non_positive_cap_is_rejected(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    with pytest.raises(ValueError, match="clip_step_rms must be positive"):
        _sample(
            fake_pipeline,
            room_photo,
            geometric_guidance,
            strength=500.0,
            clip_step_rms=0.0,
        )


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
        progress=0.0,
    )

    # Tweedie's x̂₀ spelled out, so the step is pinned to that estimate rather
    # than to whatever the loop happens to feed the decoder.
    differentiable_latent = latent.clone().requires_grad_(True)
    predicted_original = (
        differentiable_latent - (1.0 - alpha_cumprod).sqrt() * noise_prediction
    ) / alpha_cumprod.sqrt()
    expected = assert_gradient_is_finite_and_nonzero(
        geometric_guidance(
            decode_latent(fake_pipeline.vae, predicted_original), progress=0.0
        ),
        differentiable_latent,
    )

    torch.testing.assert_close(steering, expected)


def test_differentiating_epsilon_adds_the_denoisers_jacobian_to_the_step(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    generator: Generator,
) -> None:
    latent = torch.randn((1, LATENT_CHANNELS, 8, 8), generator=generator)
    timestep = torch.tensor(500)
    alpha_cumprod = torch.tensor(0.5)
    text_guidance_scale = 7.5
    prompt_embeddings = _prompt_embeddings(
        fake_pipeline,
        _PROMPT,
        device=latent.device,
        images_per_prompt=1,
        apply_text_guidance=True,
    )

    noise_prediction, steering = guidance_gradient_through_unet(
        fake_pipeline,
        geometric_guidance,
        latent=latent,
        timestep=timestep,
        prompt_embeddings=prompt_embeddings,
        text_guidance_scale=text_guidance_scale,
        apply_text_guidance=True,
        alpha_cumprod=alpha_cumprod,
        progress=0.0,
    )

    # The same chain the loop plumbs, composed by hand: classifier-free ε̂ of the
    # latent, Tweedie's x̂₀ over it, decode, energy.
    differentiable_latent = latent.clone().requires_grad_(True)
    model_input = fake_pipeline.scheduler.scale_model_input(
        torch.cat([differentiable_latent, differentiable_latent]), timestep
    )
    unconditional, conditional = fake_pipeline.unet(
        model_input, timestep, encoder_hidden_states=prompt_embeddings
    ).sample.chunk(2)
    attached_noise_prediction = unconditional + text_guidance_scale * (
        conditional - unconditional
    )
    predicted_original = (
        differentiable_latent - (1.0 - alpha_cumprod).sqrt() * attached_noise_prediction
    ) / alpha_cumprod.sqrt()
    expected = assert_gradient_is_finite_and_nonzero(
        geometric_guidance(
            decode_latent(fake_pipeline.vae, predicted_original), progress=0.0
        ),
        differentiable_latent,
    )
    constant_epsilon = guidance_gradient(
        fake_pipeline.vae,
        geometric_guidance,
        latent=latent,
        noise_prediction=noise_prediction,
        alpha_cumprod=alpha_cumprod,
        progress=0.0,
    )

    torch.testing.assert_close(steering, expected)
    assert not torch.allclose(steering, constant_epsilon)
    assert not noise_prediction.requires_grad
    torch.testing.assert_close(noise_prediction, attached_noise_prediction.detach())


def test_differentiating_through_the_unet_moves_the_trajectory(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    through_unet = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=500.0,
        differentiate_through_unet=True,
    )
    repeated = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=500.0,
        differentiate_through_unet=True,
    )
    constant_epsilon = _sample(
        fake_pipeline, room_photo, geometric_guidance, strength=500.0
    )

    assert bool(torch.isfinite(through_unet).all())
    assert torch.equal(through_unet, repeated)
    assert not torch.equal(through_unet, constant_epsilon)


def test_zero_strength_leaves_the_trajectory_unguided_through_the_unet_too(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    unguided = _sample(fake_pipeline, room_photo, geometric_guidance, strength=0.0)
    through_unet = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=0.0,
        differentiate_through_unet=True,
    )

    assert torch.equal(unguided, through_unet)


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


def test_the_energy_is_told_how_far_through_the_walk_each_step_is(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    whole = RecordingGuidance(geometric_guidance)
    through_unet = RecordingGuidance(geometric_guidance)
    # Half the schedule renoised: the walk is the two steps that are redone, and
    # the first of them is its start — the fraction is of the walk, not of the
    # trajectory the photo was never taken back up.
    renoised = RecordingGuidance(geometric_guidance)
    single = RecordingGuidance(geometric_guidance)

    _sample(fake_pipeline, room_photo, whole, strength=1.0, steps=6, noise_strength=1.0)
    _sample(
        fake_pipeline,
        room_photo,
        through_unet,
        strength=1.0,
        steps=3,
        noise_strength=1.0,
        differentiate_through_unet=True,
    )
    _sample(
        fake_pipeline, room_photo, renoised, strength=1.0, steps=4, noise_strength=0.5
    )
    _sample(
        fake_pipeline, room_photo, single, strength=1.0, steps=1, noise_strength=1.0
    )

    assert whole.walked == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert through_unet.walked == pytest.approx([0.0, 0.5, 1.0])
    assert renoised.walked == pytest.approx([0.0, 1.0])
    assert single.walked == pytest.approx([0.0])


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
