from dataclasses import dataclass, field, replace
from typing import Final

import torch
from jaxtyping import Float, Int
from torch import Generator, Tensor

from roomgen_fakes import (
    FakeDepthEstimator,
    FakeSegmenter,
    assert_gradient_is_finite_and_nonzero,
)
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.guidance.schedule import ConstantStrength
from roomgen_guide.sampling.guided import decode_latent, sample_guided
from roomgen_guide_fakes import LATENT_CHANNELS, FakePipeline

_EROSION_RADIUS: Final = 2
# The bottom-right block `FakeSegmenter` labels sofa, well inside the reference
# fixture's shell rectangle and well clear of the eroded boundary either side.
_SOFA_REGION: Final = (slice(48, 54), slice(40, 52))
_WALL: Final = ADE20K_LABELS.index("wall")
_SOFA: Final = ADE20K_LABELS.index("sofa")


def _brightened(image: Tensor, amount: float, region: tuple[slice, slice]) -> Tensor:
    changed = image.clone()
    changed[:, :, region[0], region[1]] += amount
    return changed


@dataclass(frozen=True)
class _CornerShellSegmenter:
    # Shell only in the bottom-right corner, which is exactly the block the
    # `shell_mask` fixture excludes: the intersection is empty.
    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        batch, _, height, width = images.shape
        ids = torch.full((batch, height, width), _SOFA, dtype=torch.int64)
        ids[:, -8:, -8:] = _WALL
        return ids


@dataclass
class _RecordingSegmenter:
    inner: FakeSegmenter
    calls: list[tuple[bool, bool]] = field(default_factory=list)

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        self.calls.append((images.requires_grad, torch.is_grad_enabled()))
        return self.inner(images)


def test_the_energy_vanishes_on_the_photo_its_reference_was_frozen_from(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    assert isinstance(geometric_guidance, Guidance)
    assert float(geometric_guidance(room_photo)) < 1e-10


def test_the_energy_grows_with_the_drift_of_the_sample_from_the_photo(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    shell = (slice(0, 32), slice(0, 32))

    small = float(geometric_guidance(_brightened(room_photo, 0.05, shell)))
    large = float(geometric_guidance(_brightened(room_photo, 0.2, shell)))

    assert 0.0 < small < large


def test_the_energy_ignores_the_affine_freedom_the_depth_backend_leaves(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    rescaled_backend = GeometricGuidance(
        depth_estimator=FakeDepthEstimator(scale=3.0, shift=-2.0),
        reference=geometric_guidance.reference,
    )
    drifted = _brightened(room_photo, 0.2, (slice(0, 32), slice(0, 32)))

    torch.testing.assert_close(rescaled_backend(drifted), geometric_guidance(drifted))


def test_the_energy_scores_only_the_masked_shell(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    # The fixture's mask excludes the bottom-right 8x8 corner.
    outside = _brightened(room_photo, 0.5, (slice(-8, None), slice(-8, None)))
    inside = _brightened(room_photo, 0.5, (slice(0, 8), slice(0, 8)))

    assert float(geometric_guidance(outside)) < 1e-10
    assert float(geometric_guidance(inside)) > 0.0


def test_the_energy_gradient_reaches_the_latent_through_the_decoder(
    geometric_guidance: GeometricGuidance,
    fake_pipeline: FakePipeline,
    generator: Generator,
) -> None:
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    energy = geometric_guidance(decode_latent(fake_pipeline.vae, latent))

    gradient = assert_gradient_is_finite_and_nonzero(energy, latent)
    assert gradient.shape == latent.shape


def test_resegmenting_stops_scoring_the_reference_shell_the_prediction_covers(
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
    fake_segmenter: FakeSegmenter,
) -> None:
    resegmenting = replace(
        geometric_guidance, segmenter=fake_segmenter, erosion_radius=_EROSION_RADIUS
    )
    occluded = _brightened(room_photo, 0.5, _SOFA_REGION)

    assert float(geometric_guidance(occluded)) > 0.0
    assert float(resegmenting(occluded)) < 1e-10


def test_a_prediction_shell_disjoint_from_the_reference_scores_nothing(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    disjoint = replace(
        geometric_guidance,
        segmenter=_CornerShellSegmenter(),
        erosion_radius=_EROSION_RADIUS,
    )

    assert float(disjoint(_brightened(room_photo, 0.5, _SOFA_REGION))) == 0.0
    assert float(disjoint(room_photo)) == 0.0


def test_an_empty_intersection_leaves_the_sampler_finite(
    geometric_guidance: GeometricGuidance,
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
) -> None:
    disjoint = replace(
        geometric_guidance,
        segmenter=_CornerShellSegmenter(),
        erosion_radius=_EROSION_RADIUS,
    )

    rendered = sample_guided(
        fake_pipeline,
        room_photo,
        "a mid-century living room",
        guidance=disjoint,
        schedule=ConstantStrength(strength=500.0),
        steps=4,
        noise_strength=0.6,
        text_guidance_scale=7.5,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.isfinite(rendered).all()


def test_the_energy_gradient_survives_resegmenting_the_prediction(
    geometric_guidance: GeometricGuidance,
    fake_segmenter: FakeSegmenter,
    fake_pipeline: FakePipeline,
    generator: Generator,
) -> None:
    resegmenting = replace(
        geometric_guidance, segmenter=fake_segmenter, erosion_radius=_EROSION_RADIUS
    )
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    energy = resegmenting(decode_latent(fake_pipeline.vae, latent))

    gradient = assert_gradient_is_finite_and_nonzero(energy, latent)
    assert gradient.shape == latent.shape


def test_the_segmenter_never_sees_a_tensor_on_the_gradient_graph(
    geometric_guidance: GeometricGuidance,
    fake_segmenter: FakeSegmenter,
    fake_pipeline: FakePipeline,
    generator: Generator,
) -> None:
    recording = _RecordingSegmenter(fake_segmenter)
    resegmenting = replace(
        geometric_guidance, segmenter=recording, erosion_radius=_EROSION_RADIUS
    )
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    energy = resegmenting(decode_latent(fake_pipeline.vae, latent))

    assert recording.calls == [(False, False)]
    assert_gradient_is_finite_and_nonzero(energy, latent)
