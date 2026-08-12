from dataclasses import dataclass, field

import pytest
import torch
from jaxtyping import Float
from torch import Generator, Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.sampling.guided import (
    _encode_photo,
    _guided_update,
    _predict_noise,
    _prompt_embeddings,
    _renoised_timesteps,
    decode_latent,
    guidance_gradient,
    guidance_gradient_through_unet,
    sample_guided,
)
from roomgen_guide.sampling.pipeline import Img2ImgPipeline
from roomgen_guide_fakes import (
    LATENT_CHANNELS,
    FakePipeline,
    FakeScheduler,
    FakeVae,
)

_PROMPT = "a mid-century living room"
# The raw step at strength 500 on this stack has an RMS of a few hundredths, so
# one cap sits well clear of it and the other binds on every step.
_SLACK_CAP = 1.0
_BINDING_CAP = 0.01
# The recorded digest of the reference trajectory; see the test that reads them.
_GOLDEN_SUM = 2902.737061
_GOLDEN_PIXEL = [0.25365448, 0.37026709, 0.13712433]


def _clipped(update: Tensor, cap: float) -> Tensor:
    return _guided_update(update, strength=1.0, clip_step_rms=cap)


def _direction(update: Tensor) -> Tensor:
    """The step with its size divided out, so a test can ask whether it turned."""
    return update / update.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()


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
    images: list[Tensor] = field(default_factory=list)

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        self.scored_on_the_graph.append(
            predicted_image.requires_grad and predicted_image.grad_fn is not None
        )
        self.images.append(predicted_image.detach().clone())
        return self.energy(predicted_image)


class RecordingVae:
    """A `FakeVae` that keeps every latent it was asked to decode.

    The guidance only ever sees the decoded image, and this stack's decoder
    saturates: two very different x̂₀ can land on the same picture. Catching the
    latent on its way in is what makes an assertion about x̂₀ an assertion about
    x̂₀ rather than about the decoder.
    """

    def __init__(self, vae: FakeVae) -> None:
        self.vae = vae
        self.config = vae.config
        self.decoded: list[Tensor] = []

    def encode(self, x: Tensor) -> object:
        return self.vae.encode(x)

    def decode(self, z: Tensor) -> object:
        self.decoded.append(z.detach().clone())
        return self.vae.decode(z)


class RecordingScheduler(FakeScheduler):
    """A `FakeScheduler` that keeps what each step was handed and what it made of it."""

    def __init__(self) -> None:
        super().__init__()
        self.stepped: list[tuple[Tensor, Tensor]] = []

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor,
        sample: Tensor,
        generator: Generator | None = None,
    ) -> object:
        output = super().step(model_output, timestep, sample, generator=generator)
        self.stepped.append((sample.detach().clone(), output.pred_original_sample))
        return output


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
    clip_step_rms: float | None = None,
    differentiate_through_unet: bool = False,
) -> Tensor:
    return sample_guided(
        pipeline,
        photo,
        _PROMPT,
        guidance=guidance,
        strength=strength,
        steps=steps,
        noise_strength=noise_strength,
        text_guidance_scale=text_guidance_scale,
        generator=torch.Generator().manual_seed(seed),
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

    assert float(geometric_guidance(guided)) < 0.9 * float(geometric_guidance(unguided))


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
    torch.testing.assert_close(_direction(clipped), _direction(update))
    assert torch.equal(_clipped(update, 1e6), update)


def test_the_capped_trajectory_stops_growing_with_the_energy(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    amplified = AmplifiedEnergy(geometric_guidance, factor=1000.0)
    louder = AmplifiedEnergy(geometric_guidance, factor=1e6)

    capped = _sample(
        fake_pipeline,
        room_photo,
        amplified,
        strength=500.0,
        clip_step_rms=_BINDING_CAP,
    )
    # A thousand times the energy is a thousand times the gradient in the same
    # direction: while the cap binds, both take the identical step of exactly
    # that size, and the trajectory stops reading the scale of the energy at all.
    capped_louder = _sample(
        fake_pipeline,
        room_photo,
        louder,
        strength=500.0,
        clip_step_rms=_BINDING_CAP,
    )
    uncapped = _sample(fake_pipeline, room_photo, amplified, strength=500.0)
    uncapped_louder = _sample(fake_pipeline, room_photo, louder, strength=500.0)

    torch.testing.assert_close(capped_louder, capped, rtol=1e-4, atol=1e-6)
    assert not torch.allclose(uncapped_louder, uncapped, rtol=1e-4, atol=1e-6)


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
        geometric_guidance(decode_latent(fake_pipeline.vae, predicted_original)),
        differentiable_latent,
    )
    constant_epsilon = guidance_gradient(
        fake_pipeline.vae,
        geometric_guidance,
        latent=latent,
        noise_prediction=noise_prediction,
        alpha_cumprod=alpha_cumprod,
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


def test_each_batch_element_is_capped_on_its_own(
    generator: Generator,
) -> None:
    # The cap is a per-image contract. Reduced over the whole batch it is not a
    # contract at all: one loud element sets the divisor, and every quiet
    # element beside it takes a step orders of magnitude below the one it was
    # promised — silently, since the batch's own RMS still looks right.
    gradient = torch.stack(
        [
            1000.0 * torch.randn((LATENT_CHANNELS, 8, 8), generator=generator),
            0.001 * torch.randn((LATENT_CHANNELS, 8, 8), generator=generator),
            0.001 * torch.randn((LATENT_CHANNELS, 8, 8), generator=generator),
        ]
    )

    clipped = _clipped(gradient, _BINDING_CAP)

    # The cap binds on the loud element and on nothing else, which is the whole
    # point of it: the quiet elements pass through as the raw step.
    torch.testing.assert_close(
        clipped[0].square().mean().sqrt(), torch.tensor(_BINDING_CAP)
    )
    assert torch.equal(clipped[1:], gradient[1:])


def test_one_element_of_a_batch_steps_as_it_would_have_stepped_alone(
    generator: Generator,
) -> None:
    quiet = 0.001 * torch.randn((1, LATENT_CHANNELS, 8, 8), generator=generator)
    loud = 1000.0 * torch.randn((1, LATENT_CHANNELS, 8, 8), generator=generator)
    batched = torch.cat([loud, quiet])

    torch.testing.assert_close(
        _clipped(batched, _BINDING_CAP)[1:], _clipped(quiet, _BINDING_CAP)
    )


def test_a_half_precision_gradient_is_still_sized_by_a_cap_that_binds() -> None:
    # A guidance gradient runs around 1e-5 per element. Squared in float16 that
    # is under the subnormal floor, so its RMS underflows to exactly zero: the
    # cap then divides by the epsilon instead, gets a factor of 1e6, clamps that
    # to 1 and passes the step through untouched — a cap that reports success
    # while never binding.
    gradient = torch.full((1, LATENT_CHANNELS, 8, 8), 1e-5, dtype=torch.float16)
    cap = 1e-6

    clipped = _clipped(gradient, cap)

    assert clipped.dtype == torch.float16
    # Loosely, because a step this small is subnormal in float16 and lands on a
    # grid about a twentieth of itself: the sizing is exact, the dtype it is
    # handed back in is what rounds — and the latent it is subtracted from would
    # round it identically anyway.
    torch.testing.assert_close(
        clipped.float().square().mean().sqrt(),
        torch.tensor(cap),
        rtol=0.1,
        atol=0.0,
    )
    # Which is a tenth of the raw step it was handed: the cap bound.
    assert float(clipped.float().abs().max()) < 0.5 * float(
        gradient.float().abs().max()
    )


@pytest.mark.parametrize("differentiate_through_unet", [False, True])
def test_the_loops_predicted_original_is_the_one_the_scheduler_would_form(
    room_photo: Tensor, differentiate_through_unet: bool
) -> None:
    # A flat energy steps by exactly zero, so the latent the guidance scored is
    # the latent the scheduler is then handed, and the two x̂₀ are comparable.
    # `alphas_cumprod` is indexed by *training* timestep, not by loop step: the
    # two agree only on the first step of a full-strength schedule, and reading
    # it by step index is a whole trajectory denoised against the wrong noise
    # level — which the decoder happily saturates into a plausible image.
    steps = 4
    vae = RecordingVae(FakeVae())
    pipeline = FakePipeline(vae=vae, scheduler=RecordingScheduler())

    _sample(
        pipeline,
        room_photo,
        RecordingGuidance(FlatEnergy()),
        strength=500.0,
        steps=steps,
        noise_strength=1.0,
        differentiate_through_unet=differentiate_through_unet,
    )

    scheduler = pipeline.scheduler
    assert isinstance(scheduler, RecordingScheduler)
    # One decode per guided step, plus the loop's final one.
    assert len(vae.decoded) == steps + 1
    for decoded, (_, predicted_original) in zip(
        vae.decoded, scheduler.stepped, strict=False
    ):
        torch.testing.assert_close(
            decoded * vae.config.scaling_factor, predicted_original
        )


def test_the_photo_is_renoised_to_the_first_walked_timesteps_noise_level(
    room_photo: Tensor,
) -> None:
    steps, noise_strength, seed = 4, 0.5, 0
    pipeline = FakePipeline(scheduler=RecordingScheduler())

    _sample(
        pipeline,
        room_photo,
        RecordingGuidance(FlatEnergy()),
        strength=500.0,
        steps=steps,
        noise_strength=noise_strength,
        seed=seed,
    )

    # The loop's own draw order, replayed: encode the photo, then the noise.
    scheduler = pipeline.scheduler
    assert isinstance(scheduler, RecordingScheduler)
    generator = torch.Generator().manual_seed(seed)
    encoded = _encode_photo(pipeline.vae, room_photo, generator=generator)
    noise = torch.randn(encoded.shape, generator=generator, dtype=encoded.dtype)
    walked = _renoised_timesteps(scheduler.timesteps, noise_strength=noise_strength)

    def renoised_to(timestep: Tensor) -> Tensor:
        alpha = scheduler.alphas_cumprod[timestep]
        return alpha.sqrt() * encoded + (1.0 - alpha).sqrt() * noise

    started_at, _ = scheduler.stepped[0]
    assert torch.equal(started_at, renoised_to(walked[0]))
    # Renoising to the *end* of the walk instead would hand the loop a latent
    # already at the noise level of the step it is meant to finish on.
    assert not torch.allclose(started_at, renoised_to(walked[-1]))


def test_the_walked_timesteps_are_the_least_noisy_tail_of_the_schedule() -> None:
    scheduler = FakeScheduler()
    scheduler.set_timesteps(8)
    timesteps = scheduler.timesteps

    walked = _renoised_timesteps(timesteps, noise_strength=0.25)
    discarded = timesteps[: len(timesteps) - len(walked)]

    assert len(walked) == 2
    assert torch.equal(walked, timesteps[-2:])
    # SDEdit renoises part of the way up and denoises back down, so a partial
    # strength redoes the *low*-noise end of the schedule. Keeping the head
    # instead would restart from near-pure noise and forget the photo, while
    # still running the right number of steps and looking entirely normal.
    assert int(walked.max()) < int(discarded.min())
    assert torch.equal(_renoised_timesteps(timesteps, noise_strength=1.0), timesteps)


def test_the_guided_step_moves_the_latent_before_the_scheduler_advances_it(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    steps, strength, text_guidance_scale, seed = 3, 500.0, 7.5, 0

    sampled = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=strength,
        steps=steps,
        noise_strength=1.0,
        text_guidance_scale=text_guidance_scale,
        seed=seed,
    )

    # The same trajectory composed by hand, in the order it is supposed to run:
    # the guided update lands on x_t and the scheduler then advances the moved
    # latent, so the step's own gain carries the nudge with it.
    scheduler = fake_pipeline.scheduler
    generator = torch.Generator().manual_seed(seed)
    scheduler.set_timesteps(steps)
    timesteps = _renoised_timesteps(scheduler.timesteps, noise_strength=1.0)
    alphas_cumprod = scheduler.alphas_cumprod.to(dtype=room_photo.dtype)
    latent = _encode_photo(fake_pipeline.vae, room_photo, generator=generator)
    noise = torch.randn(latent.shape, generator=generator, dtype=latent.dtype)
    latent = scheduler.add_noise(latent, noise, timesteps[:1])
    prompt_embeddings = _prompt_embeddings(
        fake_pipeline,
        _PROMPT,
        device=room_photo.device,
        images_per_prompt=room_photo.shape[0],
        apply_text_guidance=True,
    )
    first: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
    for timestep in timesteps:
        noise_prediction = _predict_noise(
            fake_pipeline,
            latent,
            timestep,
            prompt_embeddings,
            text_guidance_scale=text_guidance_scale,
            apply_text_guidance=True,
        )
        steering = guidance_gradient(
            fake_pipeline.vae,
            geometric_guidance,
            latent=latent,
            noise_prediction=noise_prediction,
            alpha_cumprod=alphas_cumprod[timestep],
        )
        update = _guided_update(steering, strength=strength, clip_step_rms=None)
        if first is None:
            first = (noise_prediction, timestep, latent, update)
        latent = scheduler.step(noise_prediction, timestep, latent - update).prev_sample

    assert torch.equal(sampled, decode_latent(fake_pipeline.vae, latent))
    # And the order is load-bearing: the scheduler's step is affine with a gain
    # of sqrt(alpha_prev / alpha_t), so the same nudge applied afterwards lands
    # somewhere else. Without this the two orderings are indistinguishable by
    # any test that only checks the loop runs and moves the image.
    assert first is not None
    noise_prediction, timestep, before, update = first
    assert not torch.allclose(
        scheduler.step(noise_prediction, timestep, before - update).prev_sample,
        scheduler.step(noise_prediction, timestep, before).prev_sample - update,
    )


def test_classifier_free_guidance_extrapolates_away_from_the_unconditional(
    fake_pipeline: FakePipeline, generator: Generator
) -> None:
    # `uncond + scale * (cond - uncond)`, not the other way round. Swapping the
    # two halves of the batched embedding leaves a loop that still runs, still
    # responds to the prompt string and still produces a different image with
    # and without conditioning — it just walks away from the prompt.
    latent = torch.randn((1, LATENT_CHANNELS, 8, 8), generator=generator)
    timestep = torch.tensor(500)
    text_guidance_scale = 7.5
    prompt_embeddings = _prompt_embeddings(
        fake_pipeline,
        _PROMPT,
        device=latent.device,
        images_per_prompt=1,
        apply_text_guidance=True,
    )

    prediction = _predict_noise(
        fake_pipeline,
        latent,
        timestep,
        prompt_embeddings,
        text_guidance_scale=text_guidance_scale,
        apply_text_guidance=True,
    )

    conditional, unconditional = fake_pipeline.encode_prompt(
        _PROMPT, latent.device, 1, True
    )
    assert unconditional is not None
    model_input = fake_pipeline.scheduler.scale_model_input(latent, timestep)
    conditional_epsilon = fake_pipeline.unet(
        model_input, timestep, encoder_hidden_states=conditional
    ).sample
    unconditional_epsilon = fake_pipeline.unet(
        model_input, timestep, encoder_hidden_states=unconditional
    ).sample

    torch.testing.assert_close(
        prediction,
        unconditional_epsilon
        + text_guidance_scale * (conditional_epsilon - unconditional_epsilon),
    )
    assert not torch.allclose(
        prediction,
        conditional_epsilon
        + text_guidance_scale * (unconditional_epsilon - conditional_epsilon),
    )


def test_the_encoded_photo_carries_the_autoencoders_scaling_factor(
    fake_pipeline: FakePipeline, room_photo: Tensor
) -> None:
    # Pinned here rather than through a round trip: the decoder undoes the
    # factor, so dropping it from the encode cancels out end to end on the fake
    # stack and changes the latent scale — and so every step-size in the loop —
    # on the real one.
    scaling_factor = fake_pipeline.vae.config.scaling_factor

    latent = _encode_photo(
        fake_pipeline.vae, room_photo, generator=torch.Generator().manual_seed(0)
    )

    distribution = fake_pipeline.vae.encode(room_photo * 2.0 - 1.0).latent_dist
    sampled = distribution.sample(generator=torch.Generator().manual_seed(0))
    assert torch.equal(latent, sampled * scaling_factor)
    assert not torch.allclose(latent, sampled)


def test_the_fake_stacks_trajectory_matches_its_recorded_digest(
    fake_pipeline: FakePipeline,
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
) -> None:
    # A backstop, not a specification: it pins the whole loop — noise draw,
    # renoise level, step order, arithmetic — to numbers that any silent change
    # to any of them moves, including changes that leave every behavioural test
    # in this file passing. Update these only when the change that moved them is
    # a deliberate one, and say so in the commit.
    image = _sample(
        fake_pipeline,
        room_photo,
        geometric_guidance,
        strength=500.0,
        steps=6,
        noise_strength=0.6,
        seed=0,
    )

    assert float(image.sum()) == pytest.approx(_GOLDEN_SUM, abs=1e-3)
    assert image[0, :, 17, 29].tolist() == pytest.approx(_GOLDEN_PIXEL, abs=1e-6)
