from dataclasses import dataclass, field, replace
from typing import Final

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
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS, NUM_CLASSES
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.sampling.guided import decode_latent, sample_guided
from roomgen_guide_fakes import LATENT_CHANNELS, FakePipeline

_PROMPT: Final = "a mid-century living room"
_EROSION_RADIUS: Final = 2
# The bottom-right block `FakeSegmenter` labels sofa, well inside the reference
# fixture's shell rectangle and well clear of the eroded boundary either side.
_SOFA_REGION: Final = (slice(48, 54), slice(40, 52))
_WALL: Final = ADE20K_LABELS.index("wall")
_SOFA: Final = ADE20K_LABELS.index("sofa")
# Past the end of the label table: an id the shell/contents partition does not
# name, which is the only place the two mask modes can disagree.
_UNNAMED: Final = NUM_CLASSES


def _brightened(image: Tensor, amount: float, region: tuple[slice, slice]) -> Tensor:
    changed = image.clone()
    changed[:, :, region[0], region[1]] += amount
    return changed


def _flooded(image: Tensor, region: tuple[slice, slice]) -> Tensor:
    changed = image.clone()
    changed[:, :, region[0], region[1]] = 1.0
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
class _UnsureWhereFlooded:
    """Wall, except where the prediction came out saturated: an id neither set names.

    Content-sensitive on purpose. `FakeSegmenter` labels by frame position, so it
    cannot stage a prediction whose own segmentation depends on what was
    rendered into it, which is what a mask-mode difference is made of.
    """

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        unsure = images.amin(dim=1) >= 1.0
        return torch.where(unsure, torch.tensor(_UNNAMED), torch.tensor(_WALL)).expand(
            images.shape[0], *images.shape[-2:]
        )


@dataclass(frozen=True)
class _WallAroundTheSofa:
    """Every pixel affirmatively named, by one list or the other.

    `FakeSegmenter` cannot stage this: its window is glazing, which is in
    neither list precisely so the reward never scores the view behind it.
    """

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        batch, _, height, width = images.shape
        ids = torch.full((batch, height, width), _WALL, dtype=torch.int64)
        ids[:, _SOFA_REGION[0], _SOFA_REGION[1]] = _SOFA
        return ids


@dataclass(frozen=True)
class _AlwaysIntersecting:
    """Both shells, every step, spelled out rather than configured.

    An independent statement of what a configured segmenter is supposed to
    mean, which the guidance is pinned against.
    """

    depth_estimator: DepthEstimator
    reference: GeometricReference
    segmenter: Segmenter
    erosion_radius: int

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
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
    """The recording hook, keeping what each guided step decided to score."""

    entries: list[Bool[Tensor, "batch height width"]] = field(default_factory=list)

    def __call__(self, mask: Bool[Tensor, "batch height width"]) -> None:
        self.entries.append(mask)

    def matches(self, mask: Tensor) -> list[bool]:
        return [torch.equal(recorded, mask) for recorded in self.entries]


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
        strength=500.0,
        steps=steps,
        noise_strength=1.0,
        text_guidance_scale=7.5,
        generator=torch.Generator().manual_seed(0),
    )


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

    torch.testing.assert_close(
        rescaled_backend(drifted),
        geometric_guidance(drifted),
    )


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

    occluded = _brightened(room_photo, 0.5, _SOFA_REGION)

    assert float(disjoint(occluded)) == 0.0
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
        strength=500.0,
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


def test_a_configured_segmenter_intersects_at_every_step_of_the_walk(
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
        on_mask=log,
    )
    spelled_out = _AlwaysIntersecting(
        depth_estimator=geometric_guidance.depth_estimator,
        reference=geometric_guidance.reference,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )

    rendered = _walk(fake_pipeline, room_photo, resegmenting)
    equivalent = _walk(fake_pipeline, room_photo, spelled_out)

    reference_mask = geometric_guidance.reference.mask
    assert not any(log.matches(reference_mask))
    # Every scored mask is the reference's shell with the prediction's furniture
    # taken out of it, so a strictly smaller subset of the same region.
    for scored in log.entries:
        assert torch.equal(scored & reference_mask, scored)
        assert int(scored.sum()) < int(reference_mask.sum())
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


def test_the_mask_modes_score_a_different_shell_of_the_same_prediction(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    intersecting = replace(
        geometric_guidance,
        segmenter=_UnsureWhereFlooded(),
        erosion_radius=_EROSION_RADIUS,
    )
    subtracting = replace(intersecting, mask_mode="subtract_contents")
    flooded = _flooded(room_photo, _SOFA_REGION)

    # Where the segmenter declines to commit, intersecting stops scoring and
    # subtracting keeps going: only an affirmative contents label excuses a pixel.
    assert float(intersecting(flooded)) < 1e-10
    assert float(subtracting(flooded)) > 0.0


def test_the_mask_modes_agree_when_the_segmenter_names_every_pixel(
    geometric_guidance: GeometricGuidance, room_photo: Tensor
) -> None:
    intersecting = replace(
        geometric_guidance,
        segmenter=_WallAroundTheSofa(),
        erosion_radius=_EROSION_RADIUS,
    )
    subtracting = replace(intersecting, mask_mode="subtract_contents")
    drifted = _brightened(room_photo, 0.2, (slice(0, 32), slice(0, 32)))

    energy = intersecting(drifted)

    assert float(energy) > 0.0
    torch.testing.assert_close(subtracting(drifted), energy)
