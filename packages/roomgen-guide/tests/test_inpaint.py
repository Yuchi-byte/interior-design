import pytest
import torch
from torch import Tensor

from roomgen_guide.sampling.inpaint import (
    InpaintPipeline,
    fill_from_surroundings,
    inpaint_masked_region,
)
from roomgen_guide_fakes import FakeInpaintPipeline

_PROMPT = "empty room, bare floor, bare wall, no furniture"
_HOLE = (slice(16, 40), slice(8, 32))


@pytest.fixture
def contents() -> Tensor:
    mask = torch.zeros((1, 64, 64), dtype=torch.bool)
    mask[:, _HOLE[0], _HOLE[1]] = True
    return mask


def _strip(
    photo: Tensor, mask: Tensor, *, pipeline: InpaintPipeline, seed: int = 0
) -> tuple[Tensor, Tensor]:
    inpainting = inpaint_masked_region(
        photo,
        mask,
        pipeline=pipeline,
        prompt=_PROMPT,
        steps=4,
        guidance_scale=7.5,
        generator=torch.Generator().manual_seed(seed),
    )
    return inpainting.image, inpainting.generated


def test_every_pixel_outside_the_mask_is_the_photograph_itself(
    room_photo: Tensor, contents: Tensor
) -> None:
    # The whole claim of a segmentation-based strip: what was not removed was
    # never regenerated. A pipeline that decodes the frame through a VAE moves
    # those pixels, so this holds only because they are composited back.
    stripped, generated = _strip(room_photo, contents, pipeline=FakeInpaintPipeline())

    outside = ~contents.unsqueeze(1).expand_as(room_photo)
    assert torch.equal(stripped[outside], room_photo[outside])
    assert not torch.equal(generated[outside], room_photo[outside])


def test_the_masked_region_is_the_pipeline_s_render(
    room_photo: Tensor, contents: Tensor
) -> None:
    stripped, generated = _strip(room_photo, contents, pipeline=FakeInpaintPipeline())

    inside = contents.unsqueeze(1).expand_as(room_photo)
    assert torch.equal(stripped[inside], generated[inside])
    assert not torch.equal(stripped[inside], room_photo[inside])


def test_a_seed_strips_the_same_room_twice_and_two_seeds_do_not(
    room_photo: Tensor, contents: Tensor
) -> None:
    pipeline = FakeInpaintPipeline()

    once, _ = _strip(room_photo, contents, pipeline=pipeline, seed=3)
    again, _ = _strip(room_photo, contents, pipeline=pipeline, seed=3)
    other, _ = _strip(room_photo, contents, pipeline=pipeline, seed=4)

    assert torch.equal(once, again)
    assert not torch.equal(once, other)


def test_the_hole_is_filled_with_the_colours_around_it(contents: Tensor) -> None:
    # Two surfaces meeting at a horizontal seam, with the hole cut across both:
    # the fill has to take each side's own colour rather than one average of the
    # frame, or a partial-strength repaint starts from a wall where a floor is.
    photo = torch.zeros((1, 3, 64, 64))
    photo[:, :, :32] = 0.9
    photo[:, :, 32:] = 0.2

    filled = fill_from_surroundings(photo, contents)

    outside = ~contents.unsqueeze(1).expand_as(photo)
    inside = filled[:, :, _HOLE[0], _HOLE[1]]
    above, below = float(inside[:, :, :8].mean()), float(inside[:, :, -4:].mean())
    assert torch.equal(filled[outside], photo[outside])
    # Each end of the hole carries its own side's surface, and nothing in it
    # leaves the range the two surfaces span.
    assert 0.2 <= below < above <= 0.9
    assert float(inside.min()) >= 0.2
    assert float(inside.max()) <= 0.9


def test_an_empty_mask_leaves_the_photograph_alone(room_photo: Tensor) -> None:
    stripped, _ = _strip(
        room_photo,
        torch.zeros((1, 64, 64), dtype=torch.bool),
        pipeline=FakeInpaintPipeline(),
    )

    assert torch.equal(stripped, room_photo)
