"""The pytest plugin exposing the pipeline fakes; registered by the root conftest."""

import pytest
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_fakes.depth import FakeDepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.guidance.geometric import GeometricGuidance
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
