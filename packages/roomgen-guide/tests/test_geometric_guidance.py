from dataclasses import dataclass, field, replace
from typing import Final

import pytest
import torch
from jaxtyping import Bool, Float, Int
from torch import Generator, Tensor

from roomgen_fakes import (
    FakeDepthEstimator,
    FakeSegmenter,
    assert_gradient_is_finite_and_nonzero,
)
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.masking import architectural_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.reward import masked_depth_error
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.guidance.schedule import ConstantStrength
from roomgen_guide.sampling.guided import decode_latent, sample_guided
from roomgen_guide_fakes import LATENT_CHANNELS, FakePipeline

_PROMPT: Final = "a mid-century living room"
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


@dataclass(frozen=True)
class _AlwaysIntersecting:
    """The energy as it scored before the threshold existed: both shells, every step.

    Spelled out rather than configured, so `resegment_after=0.0` is pinned
    against an independent statement of what it is supposed to mean.
    """

    depth_estimator: DepthEstimator
    reference: GeometricReference
    segmenter: Segmenter
    erosion_radius: int

    def __call__(
        self,
        predicted_image: Float[Tensor, "batch 3 height width"],
        *,
        progress: float,
    ) -> Float[Tensor, ""]:
        with torch.no_grad():
            shell = architectural_mask(
                self.segmenter(predicted_image.detach()),
                erosion_radius=self.erosion_radius,
            )
        error = masked_depth_error(
            self.depth_estimator(predicted_image),
            self.reference.disparity,
            self.reference.mask & shell,
        )
        return error.mean()


@dataclass
class _MaskLog:
    """The recording hook, kept as the pair of things a guided step decided."""

    entries: list[tuple[float, Tensor]] = field(default_factory=list)

    def __call__(
        self, progress: float, mask: Bool[Tensor, "batch height width"]
    ) -> None:
        self.entries.append((progress, mask))

    @property
    def walked(self) -> list[float]:
        return [progress for progress, _ in self.entries]

    def matches(self, mask: Tensor) -> list[bool]:
        return [torch.equal(recorded, mask) for _, recorded in self.entries]


@dataclass
class _RecordingSegmenter:
    inner: FakeSegmenter
    calls: list[tuple[bool, bool]] = field(default_factory=list)

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        self.calls.append((images.requires_grad, torch.is_grad_enabled()))
        return self.inner(images)


def _walk(
    pipeline: FakePipeline,
    photo: Tensor,
    guidance: Guidance,
    *,
    steps: int = 6,
) -> Tensor:
    return sample_guided(
        pipeline,
        photo,
        _PROMPT,
        guidance=guidance,
        schedule=ConstantStrength(strength=500.0),
        steps=steps,
        noise_strength=1.0,
        text_guidance_scale=7.5,
        generator=torch.Generator().manual_seed(0),
    )


def test_the_energy_vanishes_on_the_photo_its_reference_was_frozen_from(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    assert isinstance(geometric_guidance, Guidance)
    assert float(geometric_guidance(room_photo, progress=1.0)) < 1e-10


def test_the_energy_grows_with_the_drift_of_the_sample_from_the_photo(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    shell = (slice(0, 32), slice(0, 32))

    small = float(
        geometric_guidance(_brightened(room_photo, 0.05, shell), progress=1.0)
    )
    large = float(geometric_guidance(_brightened(room_photo, 0.2, shell), progress=1.0))

    assert 0.0 < small < large


def test_the_energy_ignores_the_affine_freedom_the_depth_backend_leaves(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    rescaled_backend = GeometricGuidance(
        depth_estimator=FakeDepthEstimator(scale=3.0, shift=-2.0),
        reference=geometric_guidance.reference,
    )
    drifted = _brightened(room_photo, 0.2, (slice(0, 32), slice(0, 32)))

    torch.testing.assert_close(
        rescaled_backend(drifted, progress=1.0),
        geometric_guidance(drifted, progress=1.0),
    )


def test_the_energy_scores_only_the_masked_shell(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    # The fixture's mask excludes the bottom-right 8x8 corner.
    outside = _brightened(room_photo, 0.5, (slice(-8, None), slice(-8, None)))
    inside = _brightened(room_photo, 0.5, (slice(0, 8), slice(0, 8)))

    assert float(geometric_guidance(outside, progress=1.0)) < 1e-10
    assert float(geometric_guidance(inside, progress=1.0)) > 0.0


def test_the_energy_gradient_reaches_the_latent_through_the_decoder(
    geometric_guidance: GeometricGuidance,
    fake_pipeline: FakePipeline,
    generator: Generator,
) -> None:
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    energy = geometric_guidance(decode_latent(fake_pipeline.vae, latent), progress=1.0)

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

    assert float(geometric_guidance(occluded, progress=1.0)) > 0.0
    assert float(resegmenting(occluded, progress=1.0)) < 1e-10


def test_a_prediction_shell_disjoint_from_the_reference_scores_nothing(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    disjoint = replace(
        geometric_guidance,
        segmenter=_CornerShellSegmenter(),
        erosion_radius=_EROSION_RADIUS,
    )

    occluded = _brightened(room_photo, 0.5, _SOFA_REGION)

    assert float(disjoint(occluded, progress=1.0)) == 0.0
    assert float(disjoint(room_photo, progress=1.0)) == 0.0


def test_an_empty_intersection_leaves_the_sampler_finite(
    geometric_guidance: GeometricGuidance,
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
) -> None:
    disjoint = replace(
        geometric_guidance,
        segmenter=_CornerShellSegmenter(),
        erosion_radius=_EROSION_RADIUS,
        resegment_after=0.0,
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

    energy = resegmenting(decode_latent(fake_pipeline.vae, latent), progress=1.0)

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

    energy = resegmenting(decode_latent(fake_pipeline.vae, latent), progress=1.0)

    assert recording.calls == [(False, False)]
    assert_gradient_is_finite_and_nonzero(energy, latent)


def test_the_shell_the_energy_scores_switches_over_at_the_threshold(
    geometric_guidance: GeometricGuidance,
    room_photo: Tensor,
    fake_segmenter: FakeSegmenter,
) -> None:
    log = _MaskLog()
    resegmenting = replace(
        geometric_guidance,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
        resegment_after=0.5,
        on_mask=log,
    )
    occluded = _brightened(room_photo, 0.5, _SOFA_REGION)

    early = float(resegmenting(occluded, progress=0.49))
    late = float(resegmenting(occluded, progress=0.5))

    # Before the threshold the sofa's pixels are still the reference's wall and
    # are charged for; after it, the prediction's own shell has excused them.
    assert early > 0.0
    assert late < 1e-10
    assert log.walked == pytest.approx([0.49, 0.5])
    assert log.matches(geometric_guidance.reference.mask) == [True, False]


def test_the_scored_mask_follows_the_threshold_across_the_walked_trajectory(
    geometric_guidance: GeometricGuidance,
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_segmenter: FakeSegmenter,
) -> None:
    log = _MaskLog()
    resegmenting = replace(
        geometric_guidance,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
        resegment_after=0.5,
        on_mask=log,
    )

    _walk(fake_pipeline, room_photo, resegmenting)

    reference_mask = geometric_guidance.reference.mask
    late = log.entries[-1][1]
    assert log.walked == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert log.matches(reference_mask) == [True, True, True, False, False, False]
    # The late mask is the reference's shell with the prediction's furniture
    # taken out of it, so it is a strictly smaller subset of the same region.
    assert torch.equal(late & reference_mask, late)
    assert int(late.sum()) < int(reference_mask.sum())


def test_resegmenting_from_the_first_step_intersects_the_whole_way(
    geometric_guidance: GeometricGuidance,
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_segmenter: FakeSegmenter,
) -> None:
    log = _MaskLog()
    from_the_start = replace(
        geometric_guidance,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
        resegment_after=0.0,
        on_mask=log,
    )
    spelled_out = _AlwaysIntersecting(
        depth_estimator=geometric_guidance.depth_estimator,
        reference=geometric_guidance.reference,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )

    rendered = _walk(fake_pipeline, room_photo, from_the_start)
    equivalent = _walk(fake_pipeline, room_photo, spelled_out)

    assert not any(log.matches(geometric_guidance.reference.mask))
    assert torch.equal(rendered, equivalent)


def test_an_energy_without_a_segmenter_holds_the_reference_shell_the_whole_way(
    geometric_guidance: GeometricGuidance,
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_segmenter: FakeSegmenter,
) -> None:
    log = _MaskLog()
    reference_only = replace(geometric_guidance, on_mask=log)
    resegmenting = replace(
        geometric_guidance,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
        resegment_after=0.0,
    )

    rendered = _walk(fake_pipeline, room_photo, reference_only)
    resegmented = _walk(fake_pipeline, room_photo, resegmenting)

    assert all(log.matches(geometric_guidance.reference.mask))
    assert not torch.equal(rendered, resegmented)


def test_recording_the_mask_leaves_the_energy_and_the_trajectory_alone(
    geometric_guidance: GeometricGuidance,
    fake_pipeline: FakePipeline,
    room_photo: Tensor,
    fake_segmenter: FakeSegmenter,
) -> None:
    resegmenting = replace(
        geometric_guidance, segmenter=fake_segmenter, erosion_radius=_EROSION_RADIUS
    )
    observed = replace(resegmenting, on_mask=_MaskLog())

    assert torch.equal(
        _walk(fake_pipeline, room_photo, resegmenting),
        _walk(fake_pipeline, room_photo, observed),
    )


@pytest.mark.parametrize("resegment_after", [-0.1, 1.5])
def test_a_threshold_outside_the_walked_trajectory_is_rejected(
    geometric_guidance: GeometricGuidance,
    fake_segmenter: FakeSegmenter,
    resegment_after: float,
) -> None:
    with pytest.raises(ValueError, match="resegment_after"):
        replace(
            geometric_guidance,
            segmenter=fake_segmenter,
            erosion_radius=_EROSION_RADIUS,
            resegment_after=resegment_after,
        )
