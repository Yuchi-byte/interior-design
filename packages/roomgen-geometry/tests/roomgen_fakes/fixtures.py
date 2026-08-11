"""The pytest plugin exposing the fakes; registered for both suites by the root conftest."""

import pytest
import torch
from jaxtyping import Float
from torch import Generator, Tensor

from roomgen_fakes.depth import FakeDepthEstimator
from roomgen_fakes.segmentation import FakeSegmenter

_PHOTO_SIZE = (64, 64)


@pytest.fixture
def generator() -> Generator:
    return torch.Generator().manual_seed(0)


@pytest.fixture
def room_photo(generator: Generator) -> Float[Tensor, "1 3 64 64"]:
    return torch.rand((1, 3, *_PHOTO_SIZE), generator=generator)


@pytest.fixture
def fake_depth_estimator() -> FakeDepthEstimator:
    return FakeDepthEstimator()


@pytest.fixture
def fake_segmenter() -> FakeSegmenter:
    return FakeSegmenter()
