import pytest
import torch
from jaxtyping import TypeCheckError
from torch import Tensor

from roomgen_fakes import FakeDepthEstimator, FakeSegmenter
from roomgen_geometry.reference import GeometricReference, StructuralReference
from roomgen_geometry.reward import masked_depth_error
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS, NUM_CLASSES


def test_reference_rejects_tensors_still_on_the_autograd_graph() -> None:
    disparity = torch.rand(1, 4, 4, requires_grad=True)
    with pytest.raises(ValueError, match="constant"):
        GeometricReference(
            disparity=disparity, mask=torch.ones(1, 4, 4, dtype=torch.bool)
        )


def test_reference_rejects_a_mask_of_a_different_shape() -> None:
    # `TypeCheckError` under the suite's jaxtyping hook, `ValueError` from the
    # dataclass's own check unhooked; both name the mask.
    with pytest.raises((TypeCheckError, ValueError), match="mask"):
        GeometricReference(
            disparity=torch.rand(1, 4, 4),
            mask=torch.ones(1, 4, 5, dtype=torch.bool),
        )


def test_from_photo_leaves_the_reference_off_the_graph_a_gradient_could_reach(
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
    room_photo: Tensor,
) -> None:
    photo = room_photo.requires_grad_(True)

    reference = GeometricReference.from_photo(
        photo,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=2,
    )

    assert not reference.disparity.requires_grad
    assert reference.disparity.grad_fn is None
    assert reference.disparity.shape == (1, 64, 64)
    assert reference.mask.shape == reference.disparity.shape
    assert reference.mask.dtype == torch.bool


def test_from_photo_scores_the_shell_and_glazing_and_ignores_the_furniture(
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
    room_photo: Tensor,
) -> None:
    reference = GeometricReference.from_photo(
        room_photo,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=0,
    )

    segmentation = fake_segmenter(room_photo)
    sofa = segmentation == ADE20K_LABELS.index("sofa")
    # The glazing is shell by owner decision: a window's depth is held rather
    # than masked out, since the opening may not move during a renovation.
    glass = segmentation == ADE20K_LABELS.index("windowpane")
    assert not bool(reference.mask[sofa].any())
    assert bool(reference.mask[glass].all())
    assert bool(reference.mask[~sofa].all())


def test_the_reward_scores_an_affine_rescale_of_the_reference_as_no_drift(
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
    room_photo: Tensor,
) -> None:
    reference = GeometricReference.from_photo(
        room_photo,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=2,
    )
    rescaled = FakeDepthEstimator(scale=3.0, shift=-2.0)(room_photo)

    error = masked_depth_error(rescaled, reference.disparity, reference.mask)

    torch.testing.assert_close(error, torch.zeros(1), atol=1e-10, rtol=0.0)


def test_the_structural_reference_holds_the_glazing_beside_the_shell(
    fake_segmenter: FakeSegmenter, room_photo: Tensor
) -> None:
    reference = StructuralReference.from_photo(
        room_photo, segmenter=fake_segmenter, erosion_radius=0
    )

    glazing = reference.classes == ADE20K_LABELS.index("windowpane")
    sofa = reference.classes == ADE20K_LABELS.index("sofa")
    assert reference.classes.shape == (1, 64, 64)
    assert reference.mask.shape == reference.classes.shape
    assert not reference.classes.requires_grad
    assert bool(reference.mask[glazing].all())
    assert not bool(reference.mask[sofa].any())


def test_the_structural_reference_rejects_an_id_off_the_end_of_the_label_table() -> (
    None
):
    # The ids index a `gather` over the class axis every guided step; caught once
    # here rather than as a device-side assert mid-sample, or by a `.max()` that
    # would synchronise on the guidance path.
    classes = torch.zeros((1, 4, 4), dtype=torch.int64)
    classes[0, 0, 0] = NUM_CLASSES

    with pytest.raises(ValueError, match="ADE20K-150"):
        StructuralReference(
            classes=classes, mask=torch.ones((1, 4, 4), dtype=torch.bool)
        )


def test_the_structural_reference_rejects_a_mask_of_a_different_shape() -> None:
    with pytest.raises((TypeCheckError, ValueError), match="mask"):
        StructuralReference(
            classes=torch.zeros((1, 4, 4), dtype=torch.int64),
            mask=torch.ones((1, 4, 5), dtype=torch.bool),
        )
