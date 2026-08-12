import pytest
import torch
from torch import Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS, NUM_CLASSES
from roomgen_geometry.segmentation.base import ClassLogitSegmenter
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


def test_eomt_class_logits_come_back_on_the_input_frame_as_class_mass(
    eomt: EoMT, synthetic_room: Tensor
) -> None:
    logits = eomt.class_logits(synthetic_room)

    assert isinstance(eomt, ClassLogitSegmenter)
    assert logits.shape == (1, NUM_CLASSES, 192, 256)
    # Non-negative query mass rather than log-odds: each query's class
    # distribution weighted by that query's mask probability.
    assert float(logits.min().detach()) >= 0.0


def test_eomt_ids_are_the_argmax_of_the_scores_a_reward_differentiates(
    eomt: EoMT, synthetic_room: Tensor
) -> None:
    # The mask deciding which pixels are scored and the scores it gates must be
    # the same network's opinion, or the term gates one image with another's.
    assert torch.equal(
        eomt(synthetic_room), eomt.class_logits(synthetic_room).argmax(1)
    )


def test_eomt_class_logits_carry_a_gradient_back_to_the_image(
    eomt: EoMT, synthetic_room: Tensor
) -> None:
    # The one path into the segmentation network a reward may differentiate;
    # `__call__` stays under `no_grad` and a detached constant.
    photo = synthetic_room.clone().requires_grad_(True)
    wall = ADE20K_LABELS.index("wall")

    gradient = assert_gradient_is_finite_and_nonzero(
        eomt.class_logits(photo)[:, wall].sum(), photo
    )

    assert gradient.shape == photo.shape


def test_eomt_ids_stay_off_the_graph_even_when_the_image_is_on_it(
    eomt: EoMT, synthetic_room: Tensor
) -> None:
    segmentation = eomt(synthetic_room.clone().requires_grad_(True))

    assert not segmentation.requires_grad
    assert segmentation.grad_fn is None


def test_eomt_weights_stay_frozen_under_the_backward_the_class_path_runs(
    eomt: EoMT, synthetic_room: Tensor
) -> None:
    # A trainable parameter would have every guided step's backward accumulate a
    # `.grad` the size of the weights, for a gradient nothing reads.
    photo = synthetic_room.clone().requires_grad_(True)

    eomt.class_logits(photo).sum().backward()

    assert all(not parameter.requires_grad for parameter in eomt._model.parameters())
    assert all(parameter.grad is None for parameter in eomt._model.parameters())
