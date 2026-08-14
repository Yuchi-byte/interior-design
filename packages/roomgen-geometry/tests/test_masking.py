import pytest
import torch
from torch import Tensor

from roomgen_fakes import FakeSegmenter
from roomgen_geometry.masking import (
    architectural_mask,
    contents_mask,
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


def test_the_mask_keeps_the_shell_and_the_glazing_and_drops_the_furniture(
    fake_segmenter: FakeSegmenter, room_photo: Tensor
) -> None:
    segmentation = fake_segmenter(room_photo)

    mask = architectural_mask(segmentation, erosion_radius=0)

    sofa = segmentation == ADE20K_LABELS.index("sofa")
    # The glass stays, by owner decision: a window is an opening a renovation
    # must leave where it is, so its depth is scored like the wall around it.
    glass = segmentation == ADE20K_LABELS.index("windowpane")
    assert not bool(mask[sofa].any())
    assert bool(mask[glass].all())
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


def test_the_two_sides_agree_on_every_id_one_of_the_lists_names() -> None:
    # Restricted to the classes the two lists between them name, keeping what is
    # shell (the photo's side) and dropping what is contents (a render's) are the
    # same statement.
    named = sorted(ARCHITECTURAL_CLASSES | CONTENTS_CLASSES)
    segmentation = torch.tensor(named).reshape(1, 1, len(named))

    shell = architectural_mask(segmentation, erosion_radius=0)
    scorable = scorable_mask(segmentation, erosion_radius=0)

    assert torch.equal(shell, scorable)
    assert int(shell.sum()) == len(ARCHITECTURAL_CLASSES)


def test_an_id_neither_list_names_is_the_photo_s_to_drop_and_a_render_s_to_keep() -> (
    None
):
    unclaimed = sorted(
        set(range(NUM_CLASSES)) - ARCHITECTURAL_CLASSES - CONTENTS_CLASSES
    )
    segmentation = torch.tensor(unclaimed).reshape(1, 1, len(unclaimed))

    shell = architectural_mask(segmentation, erosion_radius=0)
    scorable = scorable_mask(segmentation, erosion_radius=0)

    # The asymmetry is the design: the reference shell is what the photo
    # affirmatively shows, while a render only escapes being scored by
    # affirmatively covering the pixel with contents. An id the segmenter
    # declines to commit on stays scored rather than excusing itself.
    assert not bool(shell.any())
    assert bool(scorable.all())


def test_the_glass_is_shell_while_the_view_through_it_is_not() -> None:
    # A wall with a window onto sky and a tree, a sofa in front of it, and a
    # person crossing the room. The sofa is contents on either side; the view
    # and the person are named by neither list, so a render's mask keeps them
    # and the photo's shell does not claim them. The glazing itself is shell on
    # both sides now — the window is an opening the renovation must keep.
    segmentation = torch.full((1, 16, 16), _WALL)
    segmentation[:, 2:8, 2:8] = ADE20K_LABELS.index("windowpane")
    segmentation[:, 3:5, 3:7] = ADE20K_LABELS.index("sky")
    segmentation[:, 5:7, 3:7] = ADE20K_LABELS.index("tree")
    segmentation[:, 10:14, 2:8] = _SOFA
    segmentation[:, 4:12, 11:14] = ADE20K_LABELS.index("person")

    shell = architectural_mask(segmentation, erosion_radius=0)
    scorable = scorable_mask(segmentation, erosion_radius=0)

    view, person, sofa, glass, wall_above = (
        (slice(None), slice(3, 7), slice(3, 7)),
        (slice(None), slice(4, 12), slice(11, 14)),
        (slice(None), slice(10, 14), slice(2, 8)),
        (slice(None), 2, slice(2, 8)),
        (slice(None), 1, slice(2, 8)),
    )
    assert not bool(shell[view].any())
    assert not bool(shell[person].any())
    assert bool(scorable[view].all())
    assert bool(scorable[person].all())
    assert not bool(shell[sofa].any())
    assert not bool(scorable[sofa].any())
    # The glass sits with the frame: whatever depth a monocular network reads on
    # it (the pane or the scene beyond), a render that kept the window reads the
    # same thing, so the pixel is held rather than excused.
    assert bool(shell[glass].all())
    assert bool(scorable[glass].all())
    # And the wall the opening is cut into is shell on both sides, as ever.
    assert bool(shell[wall_above].all())
    assert bool(scorable[wall_above].all())


def test_erosion_grows_the_region_a_render_is_excused_from() -> None:
    # Erosion is applied to the kept side, so the exemption around an
    # affirmative contents block spreads by the radius and never shrinks.
    segmentation = _room_with_block(_SOFA)

    uneroded = scorable_mask(segmentation, erosion_radius=0)
    eroded = scorable_mask(segmentation, erosion_radius=2)

    assert int(uneroded.sum()) == 16 * 16 - 4 * 4
    assert bool((eroded <= uneroded).all())
    assert int(eroded.sum()) == 16 * 16 - 8 * 8


def test_the_contents_mask_is_true_on_exactly_the_furniture() -> None:
    segmentation = torch.tensor(
        [
            [
                sorted(CONTENTS_CLASSES)[: len(ARCHITECTURAL_CLASSES)],
                sorted(ARCHITECTURAL_CLASSES),
            ]
        ]
    )

    contents = contents_mask(segmentation, dilation_radius=0)

    assert torch.equal(contents[0, 0], torch.ones_like(contents[0, 0]))
    assert torch.equal(contents[0, 1], torch.zeros_like(contents[0, 1]))


def test_an_id_neither_list_names_is_not_contents_to_remove() -> None:
    unclaimed = sorted(
        set(range(NUM_CLASSES)) - ARCHITECTURAL_CLASSES - CONTENTS_CLASSES
    )
    segmentation = torch.tensor(unclaimed).reshape(1, 1, len(unclaimed))

    assert not bool(contents_mask(segmentation, dilation_radius=0).any())


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_dilation_grows_the_furniture_by_exactly_the_radius(radius: int) -> None:
    segmentation = _room_with_block(_SOFA)

    dilated = contents_mask(segmentation, dilation_radius=radius)

    assert int(dilated.sum()) == (4 + 2 * radius) ** 2
    assert bool(dilated[:, 6 - radius : 10 + radius, 6 - radius : 10 + radius].all())


def test_the_strip_removes_exactly_what_the_reward_declines_to_score(
    fake_segmenter: FakeSegmenter, room_photo: Tensor
) -> None:
    # The two masks are one decision read in opposite directions: a pixel the
    # inpainter is handed is a pixel the depth reward would have excused anyway,
    # so a strip can never paint over geometry the reward is still holding.
    segmentation = fake_segmenter(room_photo)

    contents = contents_mask(segmentation, dilation_radius=2)
    scorable = scorable_mask(segmentation, erosion_radius=2)

    assert torch.equal(contents, ~scorable)


def test_erosion_follows_the_exemption_rather_than_the_label_map() -> None:
    segmentation = _room_with_block(_UNNAMED)

    shell = architectural_mask(segmentation, erosion_radius=2)
    scorable = scorable_mask(segmentation, erosion_radius=2)

    # The unnamed block is an edge for the photo's shell to erode against and no
    # edge at all for a render's mask, which excuses nothing here.
    assert not bool(shell[:, 4:12, 4:12].any())
    assert bool(scorable.all())
