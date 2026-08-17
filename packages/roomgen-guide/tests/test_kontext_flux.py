"""The Kontext backend under the reward driver, walked on the weightless doubles."""

import itertools

import pytest
import torch
from jaxtyping import Bool, Float
from torch import Generator, Tensor

from roomgen_fakes.depth import FakeDepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.adapters.flow_map import FlowMapBackend, FlowMapProcess
from roomgen_guide.adapters.kontext_flux import (
    KontextFluxBackend,
    build_kontext_backend,
)
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.process import (
    ClippedGradientTilt,
    EndpointProcess,
    StepContext,
    sample_endpoint_process,
)
from roomgen_guide_fakes import FAKE_KONTEXT_PROMPT, fake_kontext_stack

_STEPS = 4
# Raw-gradient units on the fake stack, whose tanh-squashed decode hands back
# an energy gradient around 1e-6 per element. The reference tokens already
# anchor the unguided walk near the photo, so an uncapped step this large
# overshoots the tiny remaining energy; the cap is what keeps each update at a
# scale the anchored trajectory can absorb, exactly as the real walks run.
_STRENGTH = 10_000.0
_CLIP_STEP_RMS = 1e-2


@pytest.fixture
def photographed_room(
    room_photo: Float[Tensor, "1 3 64 64"],
) -> Float[Tensor, "1 3 64 64"]:
    """A photo with structure the fake VAE's 8x pooling survives; the shared fixture's noise does not."""
    _, _, height, width = room_photo.shape
    rows = torch.linspace(0.0, 1.0, height)[None, None, :, None]
    columns = torch.linspace(0.0, 1.0, width)[None, None, None, :]
    return (0.15 + 0.6 * rows * (0.4 + 0.6 * columns) + 0.1 * room_photo).clamp(
        0.0, 1.0
    )


@pytest.fixture
def backend(photographed_room: Tensor) -> KontextFluxBackend:
    return build_kontext_backend(
        fake_kontext_stack(),
        prompt=FAKE_KONTEXT_PROMPT,
        reference_image=photographed_room,
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
    photo: Tensor, backend: KontextFluxBackend, *, seed: int = 0
) -> FlowMapProcess:
    # Kontext's own operating point: pure noise, the photo in the context alone.
    return FlowMapProcess(
        backend,
        photo,
        steps=_STEPS,
        start_sigma=1.0,
        generator=_seeded(seed),
    )


def test_the_kontext_backend_is_a_flow_map_backend(
    backend: KontextFluxBackend, photographed_room: Tensor
) -> None:
    assert isinstance(backend, FlowMapBackend)
    assert isinstance(_process(photographed_room, backend), EndpointProcess)


def test_the_schedule_is_kontexts_own_grid(
    backend: KontextFluxBackend, photographed_room: Tensor
) -> None:
    latent = backend.encode(photographed_room, generator=_seeded())

    sigmas = list(backend.schedule(steps=_STEPS, start_sigma=1.0, latent=latent))

    # linspace(1, 1/steps, steps) under the mu shift, then the data end: the
    # shift fixes sigma 1 in place, so a walk from pure noise starts at exactly 1.
    assert len(sigmas) == _STEPS + 1
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[-1] == 0.0
    assert all(later < earlier for earlier, later in itertools.pairwise(sigmas))


def test_every_velocity_call_attends_over_the_reference_tokens(
    backend: KontextFluxBackend, photographed_room: Tensor
) -> None:
    from roomgen_guide_fakes.kontext import FakeKontextTransformer  # noqa: PLC0415

    latent = backend.encode(photographed_room, generator=_seeded())

    velocity = backend.velocity(latent, sigma=0.7, target_sigma=0.0)

    transformer = backend.transformer
    assert isinstance(transformer, FakeKontextTransformer)
    # The model saw the noisy tokens and the reference tokens as one sequence;
    # what came back is the velocity of the noisy head alone.
    assert transformer.sequence_lengths == [
        latent.shape[1] + backend.reference_tokens.shape[1]
    ]
    assert velocity.shape == latent.shape


def test_the_reference_image_conditions_the_velocity(
    photographed_room: Tensor,
) -> None:
    same = build_kontext_backend(
        fake_kontext_stack(),
        prompt=FAKE_KONTEXT_PROMPT,
        reference_image=photographed_room,
    )
    different = build_kontext_backend(
        fake_kontext_stack(),
        prompt=FAKE_KONTEXT_PROMPT,
        reference_image=1.0 - photographed_room,
    )
    latent = same.encode(photographed_room, generator=_seeded())

    assert not torch.equal(
        same.velocity(latent, sigma=0.7, target_sigma=0.0),
        different.velocity(latent, sigma=0.7, target_sigma=0.0),
    )


def test_the_endpoint_is_differentiable_through_the_decoder(
    backend: KontextFluxBackend,
    photographed_room: Tensor,
    room_guidance: GeometricGuidance,
) -> None:
    process = _process(photographed_room, backend)
    state = process.initial_state()

    endpoint = process.endpoint(state, StepContext(index=0, total=_STEPS))
    (gradient,) = torch.autograd.grad(
        room_guidance(endpoint.image).mean(), endpoint.state
    )

    # The project's number-one failure mode is a silently detached graph: the
    # gradient must be non-zero and finite, not merely the loop finishing.
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().max()) > 0.0


def test_the_reward_moves_the_kontext_sample(
    photographed_room: Tensor, room_guidance: GeometricGuidance
) -> None:
    def walk(strength: float) -> Tensor:
        backend = build_kontext_backend(
            fake_kontext_stack(),
            prompt=FAKE_KONTEXT_PROMPT,
            reference_image=photographed_room,
        )
        return sample_endpoint_process(
            _process(photographed_room, backend),
            reward=room_guidance,
            tilt=ClippedGradientTilt(strength=strength, clip_step_rms=_CLIP_STEP_RMS),
        )

    guided, repeated, unguided = walk(_STRENGTH), walk(_STRENGTH), walk(0.0)

    # The adapter's first test, as `docs/process-adapters.md` states it: below
    # the unguided sample's energy at the same seed, or the gradient is
    # detached somewhere and the walk only looks steered.
    assert float(room_guidance(guided)) < 0.9 * float(room_guidance(unguided))
    assert torch.equal(guided, repeated)


def test_decode_speaks_the_reward_convention(
    backend: KontextFluxBackend, photographed_room: Tensor
) -> None:
    latent = backend.encode(photographed_room, generator=_seeded())

    image = backend.decode(latent)

    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert image.shape == photographed_room.shape


def test_an_unencoded_prompt_is_refused_by_name(photographed_room: Tensor) -> None:
    with pytest.raises(KeyError, match="never encoded"):
        build_kontext_backend(
            fake_kontext_stack(),
            prompt="a prompt the stack never met",
            reference_image=photographed_room,
        )


def test_a_misaligned_reference_is_refused(photographed_room: Tensor) -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        build_kontext_backend(
            fake_kontext_stack(),
            prompt=FAKE_KONTEXT_PROMPT,
            reference_image=photographed_room[..., :60, :],
        )
