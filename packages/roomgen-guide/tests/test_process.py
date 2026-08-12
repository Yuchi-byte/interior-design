from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Generator, Tensor

from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.process import (
    EndpointProcess,
    EnergyGradient,
    StepContext,
    TiltPolicy,
    sample_endpoint_process,
)
from roomgen_guide.sampling.guided import ClippedGradientTilt, LatentDiffusionProcess
from roomgen_guide_fakes import FakePipeline, ToyFlowMapProcess

_CLASS_LABEL = 7
_STEPS = 4
# Raw-gradient units on the toy stack, whose tanh-squashed endpoint hands back
# an energy gradient around 1e-7 per element.
_STRENGTH = 100_000.0


@dataclass
class CountingGuidance:
    energy: Guidance
    calls: int = 0

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, ""]:
        self.calls += 1
        return self.energy(predicted_image)


@dataclass(frozen=True)
class InnerIterationTilt:
    """FMRG's shape: `iterations` gradient steps on the endpoint before the state moves."""

    strength: float
    iterations: int

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, "batch 3 height width"],
        step: StepContext,
    ) -> Float[Tensor, "batch 3 height width"]:
        steering = torch.zeros_like(state)
        for _ in range(self.iterations):
            steering = steering - self.strength * energy_gradient(state + steering)
        return steering


@dataclass(frozen=True)
class WindowedTilt:
    """A high-noise guidance window with an early stop, in step-index units."""

    tilt: ClippedGradientTilt
    guided_fraction: float

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, "batch 3 height width"],
        step: StepContext,
    ) -> Float[Tensor, "batch 3 height width"] | None:
        if step.index >= self.guided_fraction * step.total:
            return None
        return self.tilt(energy_gradient, state=state, step=step)


@dataclass(frozen=True)
class RecordedWalk:
    image: Tensor
    process: ToyFlowMapProcess


def _walk(
    photo: Tensor,
    guidance: Guidance,
    *,
    tilt: TiltPolicy,
    seed: int = 0,
    steps: int = _STEPS,
) -> RecordedWalk:
    process = ToyFlowMapProcess(
        photo,
        class_label=_CLASS_LABEL,
        steps=steps,
        generator=torch.Generator().manual_seed(seed),
    )
    return RecordedWalk(
        sample_endpoint_process(process, reward=guidance, tilt=tilt), process
    )


def test_both_backends_are_the_same_kind_of_process(
    fake_pipeline: FakePipeline, room_photo: Tensor
) -> None:
    latent_diffusion = LatentDiffusionProcess(
        fake_pipeline,
        room_photo,
        "a mid-century living room",
        steps=_STEPS,
        noise_strength=1.0,
        text_guidance_scale=7.5,
        generator=torch.Generator().manual_seed(0),
    )
    flow_map = ToyFlowMapProcess(
        room_photo,
        class_label=_CLASS_LABEL,
        steps=_STEPS,
        generator=torch.Generator().manual_seed(0),
    )

    assert isinstance(latent_diffusion, EndpointProcess)
    assert isinstance(flow_map, EndpointProcess)
    assert latent_diffusion.steps == flow_map.steps == _STEPS


def test_the_driver_steers_a_second_backend_by_the_same_geometric_reward(
    room_photo: Tensor, geometric_guidance: GeometricGuidance
) -> None:
    guided = _walk(
        room_photo, geometric_guidance, tilt=ClippedGradientTilt(strength=_STRENGTH)
    )
    repeated = _walk(
        room_photo, geometric_guidance, tilt=ClippedGradientTilt(strength=_STRENGTH)
    )
    unguided = _walk(
        room_photo, geometric_guidance, tilt=ClippedGradientTilt(strength=0.0)
    )

    assert float(geometric_guidance(guided.image)) < 0.9 * float(
        geometric_guidance(unguided.image)
    )
    assert torch.equal(guided.image, repeated.image)


def test_an_unsteered_walk_never_evaluates_the_reward(
    room_photo: Tensor, geometric_guidance: GeometricGuidance
) -> None:
    counted = CountingGuidance(geometric_guidance)

    unguided = _walk(room_photo, counted, tilt=ClippedGradientTilt(strength=0.0))

    assert counted.calls == 0
    # The endpoint was never formed, so every step's model call was the one its
    # own advance had to make: an unguided arm pays for the walk and nothing else.
    assert unguided.process.model_calls == _STEPS


def test_an_inner_iteration_costs_one_endpoint_and_reuses_the_first(
    room_photo: Tensor, geometric_guidance: GeometricGuidance
) -> None:
    once = _walk(
        room_photo,
        geometric_guidance,
        tilt=InnerIterationTilt(strength=_STRENGTH, iterations=1),
    )
    twice = _walk(
        room_photo,
        geometric_guidance,
        tilt=InnerIterationTilt(strength=_STRENGTH, iterations=2),
    )

    # One model call per evaluated endpoint and none in `advance`: the step
    # reuses what its first evaluation already predicted, however many candidate
    # states the policy went on to try.
    assert once.process.model_calls == _STEPS
    assert twice.process.model_calls == 2 * _STEPS
    assert not torch.equal(once.image, twice.image)


def test_a_guidance_window_leaves_the_steps_outside_it_alone(
    room_photo: Tensor, geometric_guidance: GeometricGuidance
) -> None:
    tilt = ClippedGradientTilt(strength=_STRENGTH)
    counted = CountingGuidance(geometric_guidance)

    everywhere = _walk(room_photo, geometric_guidance, tilt=tilt)
    whole_walk = _walk(
        room_photo, geometric_guidance, tilt=WindowedTilt(tilt, guided_fraction=1.0)
    )
    early = _walk(room_photo, counted, tilt=WindowedTilt(tilt, guided_fraction=0.5))
    unguided = _walk(
        room_photo, geometric_guidance, tilt=ClippedGradientTilt(strength=0.0)
    )

    assert torch.equal(whole_walk.image, everywhere.image)
    assert not torch.equal(early.image, everywhere.image)
    assert not torch.equal(early.image, unguided.image)
    # The steps past the window are not scored at all, which is what an early
    # stop is for: they cost the reward nothing.
    assert counted.calls == _STEPS // 2


def test_the_same_seed_reproduces_the_toy_walk_and_a_new_one_does_not(
    room_photo: Tensor, geometric_guidance: GeometricGuidance, generator: Generator
) -> None:
    tilt = ClippedGradientTilt(strength=_STRENGTH)

    walked = _walk(room_photo, geometric_guidance, tilt=tilt, seed=0)
    reseeded = _walk(room_photo, geometric_guidance, tilt=tilt, seed=1)

    assert float(walked.image.min()) >= 0.0
    assert float(walked.image.max()) <= 1.0
    assert not torch.equal(walked.image, reseeded.image)
