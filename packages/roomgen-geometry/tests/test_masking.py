import pytest
import torch
from torch import Tensor

from roomgen_fakes import FakeSegmenter
from roomgen_geometry.masking import architectural_mask, erode_boundaries
from roomgen_geometry.segmentation.ade20k import (
    ADE20K_LABELS,
    ARCHITECTURAL_CLASSES,
    CONTENTS_CLASSES,
)


def _square_mask(size: int, side: int) -> Tensor:
    mask = torch.zeros(1, size, size, dtype=torch.bool)
    start = (size - side) // 2
    mask[:, start : start + side, start : start + side] = True
    return mask


def test_erosion_at_radius_zero_leaves_the_mask_untouched() -> None:
    mask = _square_mask(16, 6)

    assert torch.equal(erode_boundaries(mask, radius=0), mask)


def test_erosion_rejects_a_negative_radius() -> None:
    with pytest.raises(ValueError, match="negative"):
        erode_boundaries(_square_mask(8, 4), radius=-1)


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_erosion_shrinks_a_square_by_exactly_the_radius(radius: int) -> None:
    mask = _square_mask(16, 9)

    eroded = erode_boundaries(mask, radius=radius)

    assert torch.equal(eroded, _square_mask(16, 9 - 2 * radius))


def test_erosion_keeps_a_border_touching_mask_at_the_frame_edge() -> None:
    # Outside the frame is unknown, not False: eroding an all-True mask against
    # the image border would silently delete the walls at the edge of a photo.
    mask = torch.ones(1, 8, 8, dtype=torch.bool)

    assert torch.equal(erode_boundaries(mask, radius=2), mask)


def test_the_mask_is_true_on_exactly_the_shell_classes() -> None:
    segmentation = torch.tensor(
        [
            [
                sorted(ARCHITECTURAL_CLASSES),
                sorted(CONTENTS_CLASSES)[: len(ARCHITECTURAL_CLASSES)],
            ]
        ]
    )

    mask = architectural_mask(segmentation, erosion_radius=0)

    assert torch.equal(mask[0, 0], torch.ones_like(mask[0, 0]))
    assert torch.equal(mask[0, 1], torch.zeros_like(mask[0, 1]))


def test_the_mask_keeps_the_room_shell_and_drops_the_furniture(
    fake_segmenter: FakeSegmenter, room_photo: Tensor
) -> None:
    segmentation = fake_segmenter(room_photo)

    mask = architectural_mask(segmentation, erosion_radius=0)

    sofa = segmentation == ADE20K_LABELS.index("sofa")
    assert not bool(mask[sofa].any())
    assert bool(mask[~sofa].all())


def test_eroding_the_room_mask_drops_the_band_around_the_furniture(
    fake_segmenter: FakeSegmenter, room_photo: Tensor
) -> None:
    segmentation = fake_segmenter(room_photo)
    uneroded = architectural_mask(segmentation, erosion_radius=0)

    eroded = architectural_mask(segmentation, erosion_radius=2)

    assert bool((eroded <= uneroded).all())
    # The floor row directly under the sofa is shell, but it borders the sofa.
    sofa_rows, sofa_columns = torch.nonzero(
        segmentation[0] == ADE20K_LABELS.index("sofa"), as_tuple=True
    )
    below_sofa = (int(sofa_rows.max()) + 1, int(sofa_columns.max()))
    assert bool(uneroded[0][below_sofa])
    assert not bool(eroded[0][below_sofa])
