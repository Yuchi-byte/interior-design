import pytest
import torch
from torch import Tensor

from roomgen_fakes import FakeSegmenter
from roomgen_geometry.masking import (
    MaskMode,
    architectural_mask,
    erode_boundaries,
    scorable_mask,
)
from roomgen_geometry.segmentation.ade20k import (
    ADE20K_LABELS,
    ARCHITECTURAL_CLASSES,
    CONTENTS_CLASSES,
    NUM_CLASSES,
)

_WALL = ADE20K_LABELS.index("wall")
_SOFA = ADE20K_LABELS.index("sofa")
# An id past the end of the 150-class table: neither shell nor contents, which
# is what a backend emits where it declines to commit to a class at all.
_UNNAMED = NUM_CLASSES


def _room_with_block(block: int, *, size: int = 16) -> Tensor:
    segmentation = torch.full((1, size, size), _WALL)
    segmentation[:, 6:10, 6:10] = block
    return segmentation


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


def test_the_shell_mask_is_the_default_polarity_of_the_scorable_mask() -> None:
    segmentation = _room_with_block(_SOFA)

    assert torch.equal(
        architectural_mask(segmentation, erosion_radius=2),
        scorable_mask(segmentation, mode="intersect_shell", erosion_radius=2),
    )


def test_the_two_modes_agree_on_every_id_the_partition_covers() -> None:
    # The whole label table, which the partition covers exactly: contents is
    # defined as the complement of shell over it, so keeping what is shell and
    # dropping what is contents are the same statement here.
    segmentation = torch.arange(NUM_CLASSES).reshape(1, 10, 15)

    intersecting = scorable_mask(segmentation, mode="intersect_shell", erosion_radius=0)
    subtracting = scorable_mask(
        segmentation, mode="subtract_contents", erosion_radius=0
    )

    assert torch.equal(intersecting, subtracting)
    assert int(intersecting.sum()) == len(ARCHITECTURAL_CLASSES)


def test_the_modes_part_company_on_an_id_neither_set_names() -> None:
    segmentation = _room_with_block(_UNNAMED)

    intersecting = scorable_mask(segmentation, mode="intersect_shell", erosion_radius=0)
    subtracting = scorable_mask(
        segmentation, mode="subtract_contents", erosion_radius=0
    )

    # Intersecting exempts everything it cannot confirm is shell; subtracting
    # exempts only what it can confirm is contents, so an unnamed id stays scored.
    assert not bool(intersecting[:, 6:10, 6:10].any())
    assert bool(subtracting.all())


@pytest.mark.parametrize("mode", ["intersect_shell", "subtract_contents"])
def test_erosion_grows_the_region_a_mode_exempts_whichever_mode_it_is(
    mode: MaskMode,
) -> None:
    # Both modes exempt an affirmative contents block, and erosion is applied to
    # the kept side in either: the exemption spreads by the radius, never shrinks.
    segmentation = _room_with_block(_SOFA)

    uneroded = scorable_mask(segmentation, mode=mode, erosion_radius=0)
    eroded = scorable_mask(segmentation, mode=mode, erosion_radius=2)

    assert int(uneroded.sum()) == 16 * 16 - 4 * 4
    assert bool((eroded <= uneroded).all())
    assert int(eroded.sum()) == 16 * 16 - 8 * 8


def test_erosion_trims_the_kept_side_against_whatever_the_mode_exempted() -> None:
    segmentation = _room_with_block(_UNNAMED)

    intersecting = scorable_mask(segmentation, mode="intersect_shell", erosion_radius=2)
    subtracting = scorable_mask(
        segmentation, mode="subtract_contents", erosion_radius=2
    )

    # One mode has an edge to erode against here and the other has none, which
    # is the point: erosion follows the exemption rather than the label map.
    assert not bool(intersecting[:, 4:12, 4:12].any())
    assert bool(subtracting.all())
