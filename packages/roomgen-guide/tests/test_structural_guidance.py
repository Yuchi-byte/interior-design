from dataclasses import dataclass, field, replace
from typing import Final

import pytest
import torch
from jaxtyping import Bool, Float, Int
from torch import Generator, Tensor

from roomgen_fakes import (
    FakeClassLogitSegmenter,
    FakeDepthEstimator,
    assert_gradient_is_finite_and_nonzero,
)
from roomgen_fakes.segmentation import BRIGHTNESS_BY_CLASS
from roomgen_geometry.reference import GeometricReference, StructuralReference
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.composite import WeightedGuidance, WeightedTerm
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.guidance.structural import StructuralGuidance
from roomgen_guide.sampling.guided import decode_latent
from roomgen_guide_fakes import LATENT_CHANNELS, FakePipeline
from roomgen_guide_fakes.fixtures import DOOR_REGION, WINDOW_REGION

_WALL: Final = BRIGHTNESS_BY_CLASS[ADE20K_LABELS.index("wall")]
_BOOKCASE: Final = BRIGHTNESS_BY_CLASS[ADE20K_LABELS.index("bookcase")]
_WINDOWPANE: Final = ADE20K_LABELS.index("windowpane")


def _painted(
    image: Tensor, brightness: float, region: tuple[slice, slice]
) -> Float[Tensor, "1 3 64 64"]:
    """The render's answer to one region of the room, in the class it comes out as."""
    changed = image.clone()
    changed[:, :, region[0], region[1]] = brightness
    return changed


@dataclass
class _MaskLog:
    entries: list[Bool[Tensor, "batch height width"]] = field(default_factory=list)

    def __call__(self, mask: Bool[Tensor, "batch height width"]) -> None:
        self.entries.append(mask)


@dataclass
class _RecordingSegmenter:
    """Both paths into one network, recording what each was handed.

    The mask path must never see the graph; the score path must always see it,
    or the energy is a constant and guidance is a no-op that looks healthy.
    """

    inner: FakeClassLogitSegmenter
    ids: list[tuple[bool, bool]] = field(default_factory=list)
    scores: list[tuple[bool, bool]] = field(default_factory=list)

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        self.ids.append((images.requires_grad, torch.is_grad_enabled()))
        return self.inner(images)

    def class_logits(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch 150 height width"]:
        self.scores.append((images.requires_grad, torch.is_grad_enabled()))
        return self.inner.class_logits(images)


def test_the_energy_vanishes_on_the_photo_its_reference_was_frozen_from(
    structural_guidance: StructuralGuidance, structural_room_photo: Tensor
) -> None:
    assert isinstance(structural_guidance, Guidance)
    assert float(structural_guidance(structural_room_photo)) == 0.0


def test_a_window_painted_over_as_wall_is_charged_for(
    structural_guidance: StructuralGuidance, structural_room_photo: Tensor
) -> None:
    # The case depth cannot see: the wall plane is where it always was, and the
    # room has lost its window.
    painted = _painted(structural_room_photo, _WALL, WINDOW_REGION)

    assert float(structural_guidance(painted)) > 0.0


def test_a_door_painted_over_as_wall_is_charged_for(
    structural_guidance: StructuralGuidance, structural_room_photo: Tensor
) -> None:
    painted = _painted(structural_room_photo, _WALL, DOOR_REGION)

    assert float(structural_guidance(painted)) > 0.0


def test_a_render_that_kept_the_glass_pays_nothing_and_one_that_did_not_pays(
    structural_guidance: StructuralGuidance, structural_room_photo: Tensor
) -> None:
    kept = _painted(
        structural_room_photo, BRIGHTNESS_BY_CLASS[_WINDOWPANE], WINDOW_REGION
    )
    painted = _painted(structural_room_photo, _WALL, WINDOW_REGION)

    assert float(structural_guidance(kept)) == 0.0
    assert float(structural_guidance(painted)) > 0.0


def test_furniture_standing_in_front_of_the_window_is_exempt_rather_than_charged(
    structural_guidance: StructuralGuidance, structural_room_photo: Tensor
) -> None:
    # A renovation may legitimately put a bookcase against the window wall. The
    # pixel is excused because the prediction affirmatively calls it contents —
    # the same `subtract_contents` polarity the depth term uses — and not
    # because a bookcase looks anything like glass.
    bookcase = _painted(structural_room_photo, _BOOKCASE, WINDOW_REGION)
    painted = _painted(structural_room_photo, _WALL, WINDOW_REGION)

    assert float(structural_guidance(bookcase)) == 0.0
    assert float(structural_guidance(painted)) > 0.0


def test_the_exemption_is_the_mask_dropping_those_pixels_not_the_score_forgiving_them(
    structural_guidance: StructuralGuidance, structural_room_photo: Tensor
) -> None:
    bookcase = _MaskLog()
    painted = _MaskLog()

    replace(structural_guidance, on_mask=bookcase)(
        _painted(structural_room_photo, _BOOKCASE, WINDOW_REGION)
    )
    replace(structural_guidance, on_mask=painted)(
        _painted(structural_room_photo, _WALL, WINDOW_REGION)
    )

    window = (slice(None), *WINDOW_REGION)
    assert not bool(bookcase.entries[0][window].any())
    assert bool(painted.entries[0][window].all())


def test_the_reference_holds_the_glazing_the_depth_reference_leaves_out(
    structural_guidance: StructuralGuidance,
) -> None:
    reference = structural_guidance.reference

    glazing = reference.classes == _WINDOWPANE
    assert bool(glazing.any())
    assert bool(reference.mask[glazing].any())
    assert not reference.classes.requires_grad
    assert not reference.mask.requires_grad


def test_the_energy_scores_every_batch_element_on_its_own(
    structural_guidance: StructuralGuidance, structural_room_photo: Tensor
) -> None:
    reference = structural_guidance.reference
    batched = replace(
        structural_guidance,
        reference=StructuralReference(
            classes=reference.classes.expand(2, -1, -1),
            mask=reference.mask.expand(2, -1, -1),
        ),
    )
    predictions = torch.cat(
        [
            structural_room_photo,
            _painted(structural_room_photo, _WALL, WINDOW_REGION),
        ]
    )

    energy = batched(predictions)

    assert energy.shape == (2,)
    assert float(energy[0]) == 0.0
    assert float(energy[1]) > 0.0


def test_the_energy_gradient_reaches_the_image_through_the_class_scores(
    structural_guidance: StructuralGuidance, structural_room_photo: Tensor
) -> None:
    painted = _painted(structural_room_photo, _WALL, WINDOW_REGION).requires_grad_(True)

    gradient = assert_gradient_is_finite_and_nonzero(
        structural_guidance(painted).mean(), painted
    )

    # Only the pixels the room lost its window at are being pushed on; the rest
    # of the frame already agrees with the photo and sits in the dead zone.
    window = (slice(None), slice(None), *WINDOW_REGION)
    assert float(gradient[window].abs().max()) > 0.0
    assert float(gradient.abs().sum()) == pytest.approx(
        float(gradient[window].abs().sum())
    )


def test_the_energy_gradient_reaches_the_latent_through_the_decoder(
    structural_guidance: StructuralGuidance,
    fake_pipeline: FakePipeline,
    generator: Generator,
) -> None:
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    energy = structural_guidance(decode_latent(fake_pipeline.vae, latent))

    gradient = assert_gradient_is_finite_and_nonzero(energy.mean(), latent)
    assert gradient.shape == latent.shape


def test_the_mask_path_stays_off_the_graph_while_the_score_path_stays_on_it(
    structural_guidance: StructuralGuidance,
    fake_pipeline: FakePipeline,
    generator: Generator,
) -> None:
    recording = _RecordingSegmenter(FakeClassLogitSegmenter())
    watched = replace(structural_guidance, segmenter=recording)
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    energy = watched(decode_latent(fake_pipeline.vae, latent))

    assert recording.ids == [(False, False)]
    assert recording.scores == [(True, True)]
    assert_gradient_is_finite_and_nonzero(energy.mean(), latent)


def test_the_class_term_rides_beside_the_depth_term_as_one_energy(
    structural_guidance: StructuralGuidance,
    fake_depth_estimator: FakeDepthEstimator,
    structural_room_photo: Tensor,
) -> None:
    depth = GeometricGuidance(
        depth_estimator=fake_depth_estimator,
        reference=GeometricReference(
            disparity=fake_depth_estimator(structural_room_photo).detach(),
            mask=structural_guidance.reference.mask,
        ),
    )
    composite = WeightedGuidance(
        terms=(
            WeightedTerm(depth),
            WeightedTerm(structural_guidance, weight=0.5),
        )
    )
    painted = _painted(structural_room_photo, _WALL, WINDOW_REGION).requires_grad_(True)

    energy = composite(painted)

    torch.testing.assert_close(
        energy, depth(painted) + 0.5 * structural_guidance(painted)
    )
    assert_gradient_is_finite_and_nonzero(energy.mean(), painted)


def test_the_margin_decides_how_settled_a_pixel_has_to_be_to_stop_paying(
    structural_guidance: StructuralGuidance, structural_room_photo: Tensor
) -> None:
    # Halfway between the door key and the glazing key: the segmenter splits the
    # pixel's class mass between the two, so it is settled under a lenient
    # margin and not under a strict one.
    halfway = _painted(
        structural_room_photo,
        0.5
        * (
            BRIGHTNESS_BY_CLASS[ADE20K_LABELS.index("door")]
            + BRIGHTNESS_BY_CLASS[_WINDOWPANE]
        ),
        WINDOW_REGION,
    )

    lenient = float(replace(structural_guidance, margin=0.4)(halfway))
    strict = float(replace(structural_guidance, margin=0.9)(halfway))

    assert lenient == 0.0
    assert strict > 0.0
