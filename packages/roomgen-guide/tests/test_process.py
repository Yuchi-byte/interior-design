from dataclasses import dataclass, field

import pytest
import torch
from jaxtyping import Float
from torch import Generator, Tensor

from roomgen_fakes.depth import FakeDepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.adapters.mfm_reward import MfmGeometricReward
from roomgen_guide.guidance.base import Guidance, WrongDirectionError
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.process import (
    ClippedGradientTilt,
    EndpointProcess,
    EnergyGradient,
    StepContext,
    TiltPolicy,
    sample_endpoint_process,
)
from roomgen_guide.sampling.guided import LatentDiffusionProcess
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
    ) -> Float[Tensor, "batch"]:
        self.calls += 1
        return self.energy(predicted_image)


@dataclass(frozen=True)
class _PerElementEnergy:
    """One energy per batch element, and a different one for each."""

    weights: Tensor

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        return self.weights * predicted_image.square().mean(dim=(1, 2, 3))


@dataclass
class _RecordingTilt:
    """A pass-through tilt that keeps the gradient the driver handed it."""

    states: list[Tensor] = field(default_factory=list)
    gradients: list[Tensor] = field(default_factory=list)

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, "batch 3 height width"],
        step: StepContext,
    ) -> Float[Tensor, "batch 3 height width"]:
        self.states.append(state.detach().clone())
        gradient = energy_gradient(state)
        self.gradients.append(gradient.detach().clone())
        return torch.zeros_like(state)


def _seeded(seed: int = 0) -> Generator:
    return torch.Generator().manual_seed(seed)


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


def test_the_driver_is_where_the_rewards_batch_is_reduced(room_photo: Tensor) -> None:
    # The reward hands back one energy per batch element and the driver takes a
    # single backward over the batch, so the reduction is the driver's. Spelled
    # out here against the mean, because it is the mean `strength` has always
    # been calibrated against — a `sum` would multiply every step by the batch
    # size and quietly rescale every recorded experiment.
    photo = torch.cat([room_photo, 1.0 - room_photo])
    reward = _PerElementEnergy(torch.tensor([1.0, 4.0]))
    process = ToyFlowMapProcess(
        photo, class_label=_CLASS_LABEL, steps=_STEPS, generator=_seeded()
    )
    recorded = _RecordingTilt()

    sample_endpoint_process(process, reward=reward, tilt=recorded)

    state = recorded.states[0].detach().requires_grad_(True)
    endpoint = ToyFlowMapProcess(
        photo, class_label=_CLASS_LABEL, steps=_STEPS, generator=_seeded()
    ).endpoint(state, StepContext(index=0, total=_STEPS))
    energy = reward(endpoint.image)
    assert energy.shape == (2,)
    (expected,) = torch.autograd.grad(energy.mean(), endpoint.state)
    torch.testing.assert_close(recorded.gradients[0], expected, rtol=0.0, atol=0.0)


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
        tilt=ClippedGradientTilt(strength=_STRENGTH, iterations=1),
    )
    twice = _walk(
        room_photo,
        geometric_guidance,
        tilt=ClippedGradientTilt(strength=_STRENGTH, iterations=2),
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
        room_photo,
        geometric_guidance,
        tilt=ClippedGradientTilt(strength=_STRENGTH, guided_steps=_STEPS),
    )
    early = _walk(
        room_photo,
        counted,
        tilt=ClippedGradientTilt(strength=_STRENGTH, guided_steps=_STEPS // 2),
    )
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


@dataclass
class _ConstantGradient:
    """A fixed ∂energy/∂state, and a record of every state it was asked at."""

    gradient: Tensor
    states: list[Tensor] = field(default_factory=list)

    def __call__(
        self, state: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch 3 height width"]:
        self.states.append(state.detach().clone())
        return self.gradient


def _rms(tensor: Tensor) -> float:
    return float(tensor.square().mean().sqrt())


def test_the_cap_binds_on_every_inner_iteration(room_photo: Tensor) -> None:
    # The failure this exists for: FMRG's inner loop is n_opt unconstrained
    # ascent steps on the endpoint against a frozen judge, which is projected
    # gradient descent — an adversarial attack by construction. Capping the
    # applied update once per *step* would leave the iterations inside it
    # uncapped, so the cap is applied to each one and the whole step is bounded
    # by `iterations * clip_step_rms` rather than by nothing.
    cap = 0.01
    energy_gradient = _ConstantGradient(torch.full_like(room_photo, 1000.0))
    tilt = ClippedGradientTilt(strength=_STRENGTH, clip_step_rms=cap, iterations=3)

    steering = tilt(
        energy_gradient, state=room_photo, step=StepContext(index=0, total=_STEPS)
    )

    assert steering is not None
    assert _rms(steering) == pytest.approx(3 * cap, rel=1e-6)
    # And each iteration re-evaluates at the state the previous ones displaced,
    # which is what makes the loop an optimisation rather than three copies of
    # one step. The first evaluation is at the step's own state, because its
    # model output is the one `advance` reuses.
    assert torch.equal(energy_gradient.states[0], room_photo)
    assert _rms(energy_gradient.states[1] - room_photo) == pytest.approx(cap, rel=1e-6)


def test_an_uncapped_inner_loop_is_what_the_cap_is_holding_back(
    room_photo: Tensor,
) -> None:
    energy_gradient = _ConstantGradient(torch.full_like(room_photo, 1000.0))
    uncapped = ClippedGradientTilt(strength=_STRENGTH, iterations=3)
    capped = ClippedGradientTilt(strength=_STRENGTH, clip_step_rms=0.01, iterations=3)
    step = StepContext(index=0, total=_STEPS)

    loose = uncapped(energy_gradient, state=room_photo, step=step)
    held = capped(energy_gradient, state=room_photo, step=step)

    assert loose is not None
    assert held is not None
    assert _rms(loose) > 1000.0 * _rms(held)


def test_a_reward_that_is_ascended_is_refused_by_the_driver(
    room_photo: Tensor, fake_depth_estimator: FakeDepthEstimator
) -> None:
    # `MfmGeometricReward.__call__(images) -> [batch]` is the `Guidance` shape
    # exactly, so nothing structural — `isinstance`, mypy, the driver's own
    # signature — separates it from an energy. What it is not is a *sign*: MFM
    # ascends it, and a driver that descends would push every sample away from
    # the room at full strength, with a finite gradient and a healthy-looking
    # walk. The declared direction is what turns that into a refusal.
    reference = GeometricReference(
        disparity=fake_depth_estimator(room_photo).detach(),
        mask=torch.ones(room_photo.shape[0], *room_photo.shape[-2:], dtype=torch.bool),
    )
    reward = MfmGeometricReward(
        reference=reference, depth_estimator=fake_depth_estimator
    )

    assert isinstance(reward, Guidance)
    with pytest.raises(WrongDirectionError, match="reward to ascend"):
        _walk(room_photo, reward, tilt=ClippedGradientTilt(strength=_STRENGTH))


def test_the_refusal_lands_before_the_walk_takes_a_step(
    room_photo: Tensor, fake_depth_estimator: FakeDepthEstimator
) -> None:
    reference = GeometricReference(
        disparity=fake_depth_estimator(room_photo).detach(),
        mask=torch.ones(room_photo.shape[0], *room_photo.shape[-2:], dtype=torch.bool),
    )
    process = ToyFlowMapProcess(
        room_photo, class_label=_CLASS_LABEL, steps=_STEPS, generator=_seeded()
    )

    with pytest.raises(WrongDirectionError):
        sample_endpoint_process(
            process,
            reward=MfmGeometricReward(
                reference=reference, depth_estimator=fake_depth_estimator
            ),
            tilt=ClippedGradientTilt(strength=_STRENGTH),
        )

    assert process.model_calls == 0


def test_a_reward_that_collapsed_its_batch_is_refused(room_photo: Tensor) -> None:
    class _ScalarEnergy:
        def __call__(self, predicted_image: Tensor) -> Tensor:
            return predicted_image.square().mean()

    # The pre-cbe914e contract, which still broadcasts and still walks: the
    # driver's `.mean()` of a 0-dim tensor is that tensor, so a legacy reward
    # steers a batch of one perfectly well and hands a batch of eight one
    # gradient it can no longer take apart.
    with pytest.raises(ValueError, match="0-dim energy"):
        _walk(room_photo, _ScalarEnergy(), tilt=ClippedGradientTilt(strength=_STRENGTH))


def test_a_policy_that_probes_a_candidate_before_its_own_state_is_refused(
    room_photo: Tensor,
) -> None:
    @dataclass
    class _CandidateFirstTilt:
        def __call__(
            self,
            energy_gradient: EnergyGradient,
            *,
            state: Float[Tensor, "batch 3 height width"],
            step: StepContext,
        ) -> Float[Tensor, "batch 3 height width"]:
            energy_gradient(state + 1.0)
            energy_gradient(state)
            return torch.zeros_like(state)

    # It returns zero steering, so the walk it produces *should* be the unguided
    # one — and without the guard it is not, because `advance` reuses the model
    # output of whichever state was evaluated first. Silent, finite, and off by
    # a whole prediction.
    energy = _PerElementEnergy(torch.ones(room_photo.shape[0]))
    with pytest.raises(ValueError, match="evaluate the step's own state"):
        _walk(room_photo, energy, tilt=_CandidateFirstTilt())


def test_a_policy_that_starts_at_its_own_state_is_left_alone(
    room_photo: Tensor,
) -> None:
    @dataclass
    class _StateFirstTilt:
        def __call__(
            self,
            energy_gradient: EnergyGradient,
            *,
            state: Float[Tensor, "batch 3 height width"],
            step: StepContext,
        ) -> Float[Tensor, "batch 3 height width"]:
            energy_gradient(state)
            energy_gradient(state + 1.0)
            return torch.zeros_like(state)

    energy = _PerElementEnergy(torch.ones(room_photo.shape[0]))
    probing = _walk(room_photo, energy, tilt=_StateFirstTilt())
    unguided = _walk(room_photo, energy, tilt=ClippedGradientTilt(strength=0.0))

    # The guard is on the *order*, not on probing: a policy that evaluates its
    # own state first may then wander wherever it likes, and a zero steering
    # still lands on the unguided trajectory.
    assert torch.equal(probing.image, unguided.image)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"strength": -1.0}, "strength must be non-negative"),
        ({"strength": 1.0, "clip_step_rms": 0.0}, "clip_step_rms must be positive"),
        ({"strength": 1.0, "iterations": 0}, "iterations must be at least 1"),
        ({"strength": 1.0, "guided_steps": -1}, "guided_steps must be non-negative"),
    ],
)
def test_a_policy_that_cannot_steer_is_refused_at_construction(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ClippedGradientTilt(**kwargs)  # type: ignore[arg-type]
