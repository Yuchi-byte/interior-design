"""The pytest plugin exposing the pipeline fakes; registered by the root conftest."""

from typing import Final

import pytest
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_fakes.depth import FakeDepthEstimator
from roomgen_fakes.segmentation import BRIGHTNESS_BY_CLASS, FakeClassLogitSegmenter
from roomgen_geometry.reference import GeometricReference, StructuralReference
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.guidance.structural import StructuralGuidance
from roomgen_guide_fakes.pipeline import FakePipeline


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def shell_mask(room_photo: Float[Tensor, "1 3 64 64"]) -> Bool[Tensor, "1 64 64"]:
    mask = torch.zeros((room_photo.shape[0], *room_photo.shape[-2:]), dtype=torch.bool)
    # A shell that covers most of the frame but not all of it, so a test that
    # perturbs the unmasked corner is testing something.
    mask[:, :-8, :-8] = True
    return mask


@pytest.fixture
def geometric_guidance(
    fake_depth_estimator: FakeDepthEstimator,
    room_photo: Float[Tensor, "1 3 64 64"],
    shell_mask: Bool[Tensor, "1 64 64"],
) -> GeometricGuidance:
    reference = GeometricReference(
        disparity=fake_depth_estimator(room_photo).detach(), mask=shell_mask
    )
    return GeometricGuidance(depth_estimator=fake_depth_estimator, reference=reference)


# A room laid out in the brightnesses `FakeClassLogitSegmenter` reads as classes:
# wall behind everything, floor along the bottom, and the two openings the class
# term exists for — a window and a door — which a test can paint over.
WINDOW_REGION: Final = (slice(16, 32), slice(8, 24))
DOOR_REGION: Final = (slice(24, 48), slice(40, 56))
_FLOOR_ROWS: Final = slice(48, None)


@pytest.fixture
def structural_room_photo(
    room_photo: Float[Tensor, "1 3 64 64"],
) -> Float[Tensor, "1 3 64 64"]:
    photo = torch.full_like(
        room_photo, BRIGHTNESS_BY_CLASS[ADE20K_LABELS.index("wall")]
    )
    photo[:, :, _FLOOR_ROWS] = BRIGHTNESS_BY_CLASS[ADE20K_LABELS.index("floor")]
    photo[:, :, WINDOW_REGION[0], WINDOW_REGION[1]] = BRIGHTNESS_BY_CLASS[
        ADE20K_LABELS.index("windowpane")
    ]
    photo[:, :, DOOR_REGION[0], DOOR_REGION[1]] = BRIGHTNESS_BY_CLASS[
        ADE20K_LABELS.index("door")
    ]
    return photo


@pytest.fixture
def structural_guidance(
    fake_class_logit_segmenter: FakeClassLogitSegmenter,
    structural_room_photo: Float[Tensor, "1 3 64 64"],
) -> StructuralGuidance:
    return StructuralGuidance(
        segmenter=fake_class_logit_segmenter,
        reference=StructuralReference.from_photo(
            structural_room_photo,
            segmenter=fake_class_logit_segmenter,
            erosion_radius=2,
        ),
        erosion_radius=2,
    )
