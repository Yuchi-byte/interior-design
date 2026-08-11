import pytest
import torch
from torch import Tensor

from roomgen_geometry.segmentation.ade20k import NUM_CLASSES
from roomgen_geometry.segmentation.eomt import EoMT

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def eomt() -> EoMT:
    return EoMT(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


@pytest.fixture
def synthetic_room() -> Tensor:
    # Ceiling/wall/floor bands with a bright window and a dark block standing on
    # the floor: the coarsest layout a room photo can have, and non-square, so
    # the ids have to come back on the input frame rather than the model's.
    photo = torch.full((1, 3, 192, 256), 0.55)
    photo[:, :, :48] = 0.85
    photo[:, :, 140:] = 0.25
    photo[:, :, 60:120, 24:96] = 0.98
    photo[:, :, 120:180, 150:230] = 0.1
    return photo


def test_eomt_labels_every_pixel_of_the_input_frame_with_an_ade20k_id(
    eomt: EoMT, synthetic_room: Tensor
) -> None:
    segmentation = eomt(synthetic_room)

    assert segmentation.shape == (1, 192, 256)
    assert segmentation.dtype == torch.int64
    assert not segmentation.requires_grad
    assert int(segmentation.min()) >= 0
    assert int(segmentation.max()) < NUM_CLASSES


def test_eomt_splits_a_structured_scene_into_more_than_one_class(
    eomt: EoMT, synthetic_room: Tensor
) -> None:
    segmentation = eomt(synthetic_room)

    assert len(segmentation.unique()) > 1
