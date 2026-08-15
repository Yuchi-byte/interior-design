from dataclasses import dataclass, replace

import pytest
import torch
from jaxtyping import Float, Int
from torch import Generator, Tensor

from roomgen_fakes import assert_gradient_is_finite_and_nonzero
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.image_reference import (
    ImageReferenceEnergyFactory,
    ImageReferenceGuidance,
    MaskedPatchReferenceGuidance,
    patch_positions_from_mask,
)
from roomgen_guide.sampling.guided import decode_latent
from roomgen_guide_fakes import (
    EMBEDDING_WIDTH,
    LATENT_CHANNELS,
    FakePipeline,
    fake_image_reference_energies,
    fake_masked_patch_energies,
)


@pytest.fixture
def image_guidance(room_photo: Tensor) -> ImageReferenceGuidance:
    return fake_image_reference_energies()(room_photo)


def test_the_factory_freezes_the_reference_and_satisfies_the_contracts(
    image_guidance: ImageReferenceGuidance,
) -> None:
    assert isinstance(image_guidance, Guidance)
    assert isinstance(fake_image_reference_energies(), ImageReferenceEnergyFactory)
    assert not image_guidance.reference_features.requires_grad


def test_the_reference_photo_itself_is_the_energys_optimum(
    image_guidance: ImageReferenceGuidance,
    room_photo: Tensor,
    generator: Generator,
) -> None:
    other = torch.rand(room_photo.shape, generator=generator)

    # The cosine energy bottoms out at exactly -1 on the photo it froze; any
    # other image scores above it.
    assert float(image_guidance(room_photo)) == pytest.approx(-1.0)
    assert float(image_guidance(other)) > -1.0


def test_the_energy_falls_as_the_render_moves_towards_the_photo(
    image_guidance: ImageReferenceGuidance, room_photo: Tensor, generator: Generator
) -> None:
    render = torch.rand(room_photo.shape, generator=generator).requires_grad_(True)
    (gradient,) = torch.autograd.grad(image_guidance(render).mean(), render)
    towards = -gradient / gradient.abs().max()

    near = float(image_guidance((render.detach() + 0.05 * towards).clamp(0.0, 1.0)))
    far = float(image_guidance((render.detach() - 0.05 * towards).clamp(0.0, 1.0)))

    assert near < float(image_guidance(render.detach())) < far


def test_the_energy_scores_every_batch_element_on_its_own(
    image_guidance: ImageReferenceGuidance,
    room_photo: Tensor,
) -> None:
    batch = torch.cat([room_photo, torch.zeros_like(room_photo)])

    scores = image_guidance(batch)
    assert scores.shape == (2,)
    assert float(scores[0]) != float(scores[1])


def test_the_energys_gradient_reaches_the_latent_through_the_decoder(
    image_guidance: ImageReferenceGuidance,
    fake_pipeline: FakePipeline,
    generator: Generator,
) -> None:
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    scores = image_guidance(decode_latent(fake_pipeline.vae, latent))

    gradient = assert_gradient_is_finite_and_nonzero(scores.mean(), latent)
    assert gradient.shape == latent.shape


def test_a_reference_still_on_the_graph_is_refused(
    image_guidance: ImageReferenceGuidance,
) -> None:
    # A target that is itself differentiated is a target the walk can move
    # instead of moving the render towards it.
    with pytest.raises(ValueError, match="constant"):
        replace(
            image_guidance,
            reference_features=torch.zeros((1, EMBEDDING_WIDTH), requires_grad=True),
        )


# The fake patch grid is 4x4 over a 64x64 photo, so the left half of the frame
# is exactly patch columns 0 and 1 and the right half columns 2 and 3.
@pytest.fixture
def half_frame_mask(room_photo: Tensor) -> Tensor:
    mask = torch.zeros((room_photo.shape[0], *room_photo.shape[-2:]), dtype=torch.bool)
    mask[:, :, : room_photo.shape[-1] // 2] = True
    return mask


@pytest.fixture
def masked_patch_guidance(
    room_photo: Tensor, half_frame_mask: Tensor
) -> MaskedPatchReferenceGuidance:
    return fake_masked_patch_energies()(room_photo, mask=half_frame_mask)


@dataclass(frozen=True)
class _UniformSegmenter:
    """A `Segmenter` calling every pixel one class, to force the per-step mask."""

    label: int

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        return torch.full(
            (images.shape[0], *images.shape[-2:]), self.label, dtype=torch.int64
        )


def test_the_masked_patch_energy_bottoms_out_on_the_photo(
    masked_patch_guidance: MaskedPatchReferenceGuidance, room_photo: Tensor
) -> None:
    assert isinstance(masked_patch_guidance, Guidance)
    assert float(masked_patch_guidance(room_photo)) == pytest.approx(-1.0)


def test_the_masked_patch_energy_is_blind_outside_its_mask(
    masked_patch_guidance: MaskedPatchReferenceGuidance,
    room_photo: Tensor,
    generator: Generator,
) -> None:
    # Repaint the rightmost quarter — patch column 3, well clear of the masked
    # left half even through the bicubic resize — and the energy must not move:
    # that region is exactly what the renovation is free to change.
    repainted = room_photo.clone()
    quarter = 3 * room_photo.shape[-1] // 4
    repainted[:, :, :, quarter:] = torch.rand(
        repainted[:, :, :, quarter:].shape, generator=generator
    )

    assert float(masked_patch_guidance(repainted)) == pytest.approx(-1.0, abs=1e-4)


def test_a_masked_change_moves_the_masked_patch_energy(
    masked_patch_guidance: MaskedPatchReferenceGuidance,
    room_photo: Tensor,
    generator: Generator,
) -> None:
    repainted = room_photo.clone()
    quarter = room_photo.shape[-1] // 4
    repainted[:, :, :, :quarter] = torch.rand(
        repainted[:, :, :, :quarter].shape, generator=generator
    )

    assert float(masked_patch_guidance(repainted)) > -1.0 + 1e-4


def test_the_prediction_side_resegmentation_mirrors_the_depth_rewards_mask(
    room_photo: Tensor, half_frame_mask: Tensor, generator: Generator
) -> None:
    render = torch.rand(room_photo.shape, generator=generator)
    reference_only = fake_masked_patch_energies()(room_photo, mask=half_frame_mask)

    # A prediction segmented all shell intersects to the reference mask itself;
    # one segmented all contents empties the set and scores zero with nothing
    # to steer by — the same convention as the depth reward's empty mask.
    all_shell = fake_masked_patch_energies()(
        room_photo,
        mask=half_frame_mask,
        segmenter=_UniformSegmenter(ADE20K_LABELS.index("wall")),
        erosion_radius=0,
    )
    all_contents = fake_masked_patch_energies()(
        room_photo,
        mask=half_frame_mask,
        segmenter=_UniformSegmenter(ADE20K_LABELS.index("sofa")),
        erosion_radius=0,
    )

    assert float(all_shell(render)) == pytest.approx(float(reference_only(render)))
    assert float(all_contents(render)) == 0.0


def test_the_masked_patch_energys_gradient_reaches_the_latent(
    masked_patch_guidance: MaskedPatchReferenceGuidance, generator: Generator
) -> None:
    latent = torch.randn(
        (1, LATENT_CHANNELS, 8, 8), generator=generator, requires_grad=True
    )

    scores = masked_patch_guidance(decode_latent(FakePipeline().vae, latent))

    gradient = assert_gradient_is_finite_and_nonzero(scores.mean(), latent)
    assert gradient.shape == latent.shape


def test_patch_positions_take_a_majority_vote_per_patch() -> None:
    # A 4x4 grid over 8x8 pixels: each patch is a 2x2 cell. Full coverage and
    # exactly-half coverage count; a single pixel (a quarter) does not.
    mask = torch.zeros((1, 8, 8), dtype=torch.bool)
    mask[:, 0:2, 0:2] = True  # patch (0, 0): fully covered
    mask[:, 0:1, 2:4] = True  # patch (0, 1): half covered
    mask[:, 0:1, 4:5] = True  # patch (0, 2): a quarter covered

    positions = patch_positions_from_mask(mask, tokens=16)

    assert positions.shape == (1, 16)
    assert positions[0, 0].item() is True
    assert positions[0, 1].item() is True
    assert positions[0, 2].item() is False
    assert int(positions.sum()) == 2


def test_the_masked_patch_energy_refuses_a_degenerate_construction(
    masked_patch_guidance: MaskedPatchReferenceGuidance,
    room_photo: Tensor,
    half_frame_mask: Tensor,
) -> None:
    with pytest.raises(ValueError, match="empty"):
        fake_masked_patch_energies()(room_photo, mask=torch.zeros_like(half_frame_mask))
    with pytest.raises(ValueError, match="constant"):
        replace(
            masked_patch_guidance,
            reference_tokens=masked_patch_guidance.reference_tokens.clone().requires_grad_(
                True
            ),
        )
    with pytest.raises(ValueError, match="segmenter"):
        replace(
            masked_patch_guidance,
            segmenter=_UniformSegmenter(ADE20K_LABELS.index("wall")),
        )
    with pytest.raises(ValueError, match="square"):
        patch_positions_from_mask(half_frame_mask, tokens=15)
