import pytest
import torch
from jaxtyping import Bool, Float
from torch import Generator, Tensor

from roomgen_fakes.depth import FakeDepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.adapters.flow_map import (
    FlowMapProcess,
    FlowMapStepping,
    GuidanceWindow,
)
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.process import (
    ClippedGradientTilt,
    EndpointProcess,
    StepContext,
    sample_endpoint_process,
)
from roomgen_guide_fakes import ToyFlowMapBackend

_STEPS = 4
# Raw-gradient units on the toy stack, whose tanh-squashed endpoint hands back
# an energy gradient around 1e-7 per element.
_STRENGTH = 100_000.0
# Far enough up the interpolant that the walk has work to do, far enough down
# that the photo it started from is still visible in the sample.
_RENOVATION_SIGMA = 0.6
# A steering step nothing in the codebase rejects and every recorded run stayed
# above: an order of magnitude under bfloat16's ulp at O(1), which is exactly
# the band a state accumulated in the backend's dtype rounds away.
_LEGAL_STEERING_RMS = 1e-3


@pytest.fixture
def photographed_room(
    room_photo: Float[Tensor, "1 3 64 64"],
) -> Float[Tensor, "1 3 64 64"]:
    """A photo with structure a denoiser can keep, which the shared fixture is not.

    `room_photo` is white noise: it survives nothing, so a walk that starts from
    it and one that starts from pure noise arrive at the same flat image and the
    img2img claim has nothing to measure. A pair of ramps plus a patch of that
    noise stands in for a room — smooth where a wall is smooth, textured enough
    that every pixel still moves the energy.
    """
    _, _, height, width = room_photo.shape
    rows = torch.linspace(0.0, 1.0, height)[None, None, :, None]
    columns = torch.linspace(0.0, 1.0, width)[None, None, None, :]
    return (0.15 + 0.6 * rows * (0.4 + 0.6 * columns) + 0.1 * room_photo).clamp(
        0.0, 1.0
    )


@pytest.fixture
def room_guidance(
    fake_depth_estimator: FakeDepthEstimator,
    photographed_room: Float[Tensor, "1 3 64 64"],
    shell_mask: Bool[Tensor, "1 64 64"],
) -> GeometricGuidance:
    reference = GeometricReference(
        disparity=fake_depth_estimator(photographed_room).detach(), mask=shell_mask
    )
    return GeometricGuidance(depth_estimator=fake_depth_estimator, reference=reference)


def _seeded(seed: int = 0) -> Generator:
    return torch.Generator().manual_seed(seed)


def _process(
    photo: Tensor,
    backend: ToyFlowMapBackend,
    *,
    start_sigma: float = _RENOVATION_SIGMA,
    steps: int = _STEPS,
    seed: int = 0,
    window: GuidanceWindow | None = None,
    stepping: FlowMapStepping = "endpoint_reuse",
    differentiate_through_velocity: bool = False,
) -> FlowMapProcess:
    return FlowMapProcess(
        backend,
        photo,
        steps=steps,
        start_sigma=start_sigma,
        generator=_seeded(seed),
        window=window,
        stepping=stepping,
        differentiate_through_velocity=differentiate_through_velocity,
    )


def _walk(
    photo: Tensor,
    guidance: GeometricGuidance,
    *,
    tilt: ClippedGradientTilt,
    start_sigma: float = _RENOVATION_SIGMA,
    steps: int = _STEPS,
    seed: int = 0,
    backend: ToyFlowMapBackend | None = None,
) -> Tensor:
    process = _process(
        photo,
        backend if backend is not None else ToyFlowMapBackend(),
        start_sigma=start_sigma,
        steps=steps,
        seed=seed,
    )
    return sample_endpoint_process(process, reward=guidance, tilt=tilt)


def test_the_flow_map_adapter_is_an_endpoint_process(photographed_room: Tensor) -> None:
    process = _process(photographed_room, ToyFlowMapBackend())

    assert isinstance(process, EndpointProcess)
    assert process.steps == _STEPS


def test_the_reward_moves_the_flow_maps_sample(
    photographed_room: Tensor, room_guidance: GeometricGuidance
) -> None:
    guided = _walk(
        photographed_room, room_guidance, tilt=ClippedGradientTilt(strength=_STRENGTH)
    )
    repeated = _walk(
        photographed_room, room_guidance, tilt=ClippedGradientTilt(strength=_STRENGTH)
    )
    unguided = _walk(
        photographed_room, room_guidance, tilt=ClippedGradientTilt(strength=0.0)
    )

    # The adapter's first test, as `docs/process-adapters.md` states it: below
    # the unguided sample's energy at the same seed, or the gradient is detached
    # somewhere and the walk only looks steered.
    assert float(room_guidance(guided)) < 0.9 * float(room_guidance(unguided))
    assert torch.equal(guided, repeated)


def test_the_photo_the_walk_starts_from_is_the_geometry_it_holds(
    photographed_room: Tensor, room_guidance: GeometricGuidance
) -> None:
    # The fix the adversarial-collapse diagnosis turns on: from noise, matching
    # one specific room's depth is only reachable by inventing texture the depth
    # network reads as that room, and the reward is the only thing pulling. From
    # the photo, the geometry starts correct and the reward's job is to hold it —
    # so the *unguided* img2img arm already scores far below the text-to-image
    # one, before any steering at all.
    unguided = ClippedGradientTilt(strength=0.0)
    renovation = _walk(photographed_room, room_guidance, tilt=unguided)
    from_noise = _walk(photographed_room, room_guidance, tilt=unguided, start_sigma=1.0)

    assert float(room_guidance(renovation)) < 0.5 * float(room_guidance(from_noise))


def test_a_walk_from_pure_noise_is_the_same_walk_whatever_the_photo(
    photographed_room: Tensor,
) -> None:
    # `start_sigma` 1 discards the photo, exactly as v1's `noise_strength` 1
    # does: the arm is text-to-image, and its state at step 0 cannot depend on
    # the room. The generator is drawn from identically either way, so the two
    # states are comparable at all.
    backend = ToyFlowMapBackend()
    from_room = _process(photographed_room, backend, start_sigma=1.0).initial_state()
    from_negative = _process(
        1.0 - photographed_room, backend, start_sigma=1.0
    ).initial_state()
    renoised = _process(photographed_room, backend, start_sigma=0.5).initial_state()
    renoised_negative = _process(
        1.0 - photographed_room, backend, start_sigma=0.5
    ).initial_state()

    assert torch.equal(from_room, from_negative)
    assert not torch.equal(renoised, renoised_negative)


def test_advance_never_pays_for_a_model_call_the_endpoint_already_made(
    photographed_room: Tensor, room_guidance: GeometricGuidance
) -> None:
    once = ToyFlowMapBackend()
    twice = ToyFlowMapBackend()
    unguided = ToyFlowMapBackend()

    _walk(
        photographed_room,
        room_guidance,
        tilt=ClippedGradientTilt(strength=_STRENGTH, iterations=1),
        backend=once,
    )
    _walk(
        photographed_room,
        room_guidance,
        tilt=ClippedGradientTilt(strength=_STRENGTH, iterations=2),
        backend=twice,
    )
    _walk(
        photographed_room,
        room_guidance,
        tilt=ClippedGradientTilt(strength=0.0),
        backend=unguided,
    )

    # One velocity call per evaluated endpoint and none in `advance`: what the
    # reuse saves is the *step's* second call, not the inner loop's. An extra
    # iteration re-evaluates the reward at a displaced state, and a displaced
    # state needs its own velocity — n iterations cost n model calls, exactly as
    # FMRG's `n_opt` does. The cheap part is the pullback, not the loop.
    assert once.velocity_calls == _STEPS
    assert twice.velocity_calls == 2 * _STEPS
    assert unguided.velocity_calls == _STEPS


def test_guidance_stops_where_the_window_does(
    photographed_room: Tensor, room_guidance: GeometricGuidance
) -> None:
    everywhere = _walk(
        photographed_room, room_guidance, tilt=ClippedGradientTilt(strength=_STRENGTH)
    )
    early = _walk(
        photographed_room,
        room_guidance,
        tilt=ClippedGradientTilt(strength=_STRENGTH, guided_steps=_STEPS // 2),
    )
    unguided = _walk(
        photographed_room, room_guidance, tilt=ClippedGradientTilt(strength=0.0)
    )
    never = _walk(
        photographed_room,
        room_guidance,
        tilt=ClippedGradientTilt(strength=_STRENGTH, guided_steps=0),
    )

    assert not torch.equal(early, everywhere)
    assert not torch.equal(early, unguided)
    # An empty window is the unguided walk bit for bit, not merely close to it.
    assert torch.equal(never, unguided)


def test_the_endpoint_is_differentiable_in_the_state(
    photographed_room: Tensor, room_guidance: GeometricGuidance
) -> None:
    process = _process(photographed_room, ToyFlowMapBackend())
    state = process.initial_state()

    endpoint = process.endpoint(state, StepContext(index=0, total=_STEPS))
    (gradient,) = torch.autograd.grad(
        room_guidance(endpoint.image).mean(), endpoint.state
    )

    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().max()) > 0.0


def test_the_schedule_the_backend_returns_has_to_reach_the_data_end(
    photographed_room: Tensor,
) -> None:
    class TruncatedBackend(ToyFlowMapBackend):
        def schedule(
            self, *, steps: int, start_sigma: float, latent: Tensor
        ) -> list[float]:
            walked = super().schedule(
                steps=steps, start_sigma=start_sigma, latent=latent
            )
            return [*walked[:-1], 0.1]

    with pytest.raises(ValueError, match="the schedule ends at sigma"):
        _process(photographed_room, TruncatedBackend())


def test_a_schedule_that_does_not_name_every_step_is_refused(
    photographed_room: Tensor,
) -> None:
    class ShortBackend(ToyFlowMapBackend):
        def schedule(
            self, *, steps: int, start_sigma: float, latent: Tensor
        ) -> list[float]:
            walked = super().schedule(
                steps=steps, start_sigma=start_sigma, latent=latent
            )
            return walked[:-1]

    # steps + 1, never steps: the sigma before every step and the one after the
    # last. A schedule one short would walk the last step to the wrong sigma, or
    # index past the end, depending on which end it dropped.
    with pytest.raises(
        ValueError, match=f"returned {_STEPS} sigmas for {_STEPS} steps"
    ):
        _process(photographed_room, ShortBackend())


def test_the_state_is_float32_whatever_the_backend_runs_in(
    photographed_room: Tensor,
) -> None:
    process = _process(photographed_room, ToyFlowMapBackend(dtype=torch.bfloat16))

    state = process.initial_state()
    endpoint = process.endpoint(state, StepContext(index=0, total=_STEPS))

    assert state.dtype == torch.float32
    assert endpoint.state.dtype == torch.float32
    assert (
        process.advance(
            state, StepContext(index=0, total=_STEPS), prediction=None, steering=None
        ).dtype
        == torch.float32
    )


def test_a_steering_finer_than_the_backends_dtype_still_moves_the_state(
    photographed_room: Tensor,
) -> None:
    # bfloat16's ulp at O(1) is about 8e-3, so a steering at 1e-3 RMS — well
    # inside the caps the recorded runs used — is rounded away entirely by
    # `state + steering` if the state is kept in the backend's dtype. The walk
    # then comes back bit-identical to the unguided one with nothing to report.
    process = _process(photographed_room, ToyFlowMapBackend(dtype=torch.bfloat16))
    state = process.initial_state()
    step = StepContext(index=0, total=_STEPS)
    steering = torch.full_like(state, _LEGAL_STEERING_RMS)

    steered = process.advance(state, step, prediction=None, steering=steering)
    unsteered = process.advance(state, step, prediction=None, steering=None)

    assert not torch.equal(steered, unsteered)


def test_a_bfloat16_backend_walks_a_guided_trajectory_and_not_the_unguided_one(
    photographed_room: Tensor, room_guidance: GeometricGuidance
) -> None:
    capped = ClippedGradientTilt(strength=_STRENGTH, clip_step_rms=_LEGAL_STEERING_RMS)
    guided = _walk(
        photographed_room,
        room_guidance,
        tilt=capped,
        backend=ToyFlowMapBackend(dtype=torch.bfloat16),
    )
    unguided = _walk(
        photographed_room,
        room_guidance,
        tilt=ClippedGradientTilt(strength=0.0),
        backend=ToyFlowMapBackend(dtype=torch.bfloat16),
    )

    # `_STRENGTH` saturates the cap, so every applied step is exactly
    # `_LEGAL_STEERING_RMS`. Nothing enforces a floor on that number anywhere —
    # the config accepts it, the gradient is finite, the cap binds — so the only
    # thing standing between it and a silently unguided run is the state's dtype.
    assert not torch.equal(guided, unguided)


def test_two_time_stepping_buys_a_second_velocity_and_a_different_walk(
    photographed_room: Tensor, room_guidance: GeometricGuidance
) -> None:
    reused, separate, unguided = (
        ToyFlowMapBackend(),
        ToyFlowMapBackend(),
        ToyFlowMapBackend(),
    )
    tilt = ClippedGradientTilt(strength=_STRENGTH)

    endpoint_reuse = _walk(photographed_room, room_guidance, tilt=tilt, backend=reused)
    two_time = sample_endpoint_process(
        _process(photographed_room, separate, stepping="two_time"),
        reward=room_guidance,
        tilt=tilt,
    )
    sample_endpoint_process(
        _process(photographed_room, unguided, stepping="two_time"),
        reward=room_guidance,
        tilt=ClippedGradientTilt(strength=0.0),
    )

    # The endpoint's (sigma, 0) jump and the step's (sigma, sigma_next) velocity
    # are two different network evaluations, so `two_time` pays a second call per
    # guided step — upstream's flow_map2, which is what its high-NFE results and
    # roomgen's own tuned operating point run. Reusing the endpoint is the
    # low-NFE mode, and the shortcut's error accumulates as the grid gets finer.
    assert reused.velocity_calls == _STEPS
    assert separate.velocity_calls == 2 * _STEPS
    assert unguided.velocity_calls == _STEPS
    assert not torch.equal(endpoint_reuse, two_time)


def test_the_jacobian_path_pulls_the_reward_back_through_the_backend(
    photographed_room: Tensor, room_guidance: GeometricGuidance
) -> None:
    step = StepContext(index=0, total=_STEPS)
    cheap = _process(photographed_room, ToyFlowMapBackend())
    jacobian = _process(
        photographed_room, ToyFlowMapBackend(), differentiate_through_velocity=True
    )
    state = cheap.initial_state()

    gradients = []
    for process in (cheap, jacobian):
        endpoint = process.endpoint(state, step)
        (gradient,) = torch.autograd.grad(
            room_guidance(endpoint.image).mean(), endpoint.state
        )
        gradients.append(gradient)

    # Both are finite and both steer; what separates them is that the Jacobian
    # one carries d(velocity)/d(state), which is the projection FMRG-J's
    # on-manifold claim rests on. A backend held constant cannot supply it, and
    # nothing about the walk looks different when it is missing.
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert float(gradients[1].abs().max()) > 0.0
    assert not torch.equal(gradients[0], gradients[1])


def test_the_window_truncates_the_grid_and_covers_the_rest_in_one_jump(
    photographed_room: Tensor,
) -> None:
    steps, window = 8, GuidanceWindow(early_stop=3, unguided_steps=2)
    backend = ToyFlowMapBackend()
    whole = _process(photographed_room, backend, steps=steps)
    truncated = _process(photographed_room, backend, steps=steps, window=window)

    # The walk visits the window's sigmas and then jumps: `steps` is only the
    # resolution the sigma grid is drawn at, and everything below the stopping
    # sigma is covered by `unguided_steps` steps rather than by the grid's own.
    assert whole.steps == steps
    assert truncated.steps == window.guided_steps + window.unguided_steps
    assert truncated.step_span(0) == whole.step_span(0)
    assert truncated.step_span(truncated.steps - 1) == pytest.approx(
        _RENOVATION_SIGMA * (1.0 - 2 / steps) / window.unguided_steps
    )


def test_a_window_the_walk_never_reaches_is_refused(
    photographed_room: Tensor,
) -> None:
    with pytest.raises(ValueError, match="past the end of a schedule"):
        _process(
            photographed_room,
            ToyFlowMapBackend(),
            steps=_STEPS,
            window=GuidanceWindow(early_stop=_STEPS + 1),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"early_stop": 1}, "early_stop must be at least 2"),
        ({"early_stop": 3, "unguided_steps": 0}, "unguided_steps must be at least 1"),
    ],
)
def test_a_window_that_cannot_stop_is_refused_at_construction(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GuidanceWindow(**kwargs)


@pytest.mark.parametrize(
    ("steps", "start_sigma", "message"),
    [
        (0, 0.6, "steps must be at least 1"),
        (4, 0.0, "start_sigma must lie in"),
        (4, 1.5, "start_sigma must lie in"),
    ],
)
def test_a_walk_that_cannot_be_taken_is_refused_at_construction(
    photographed_room: Tensor, steps: int, start_sigma: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _process(
            photographed_room, ToyFlowMapBackend(), steps=steps, start_sigma=start_sigma
        )
