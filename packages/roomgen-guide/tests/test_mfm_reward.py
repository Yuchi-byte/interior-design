from typing import Final

import pytest
import torch
from jaxtyping import Bool, Float
from torch import Generator, Tensor

from roomgen_fakes import (
    FakeDepthEstimator,
    FakeSegmenter,
    assert_gradient_is_finite_and_nonzero,
)
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.reward import masked_depth_error
from roomgen_guide.adapters.mfm_reward import MfmGeometricReward

_EROSION_RADIUS: Final = 2


@pytest.fixture
def reference(
    fake_depth_estimator: FakeDepthEstimator,
    room_photo: Float[Tensor, "1 3 64 64"],
    shell_mask: Bool[Tensor, "1 64 64"],
) -> GeometricReference:
    return GeometricReference(
        disparity=fake_depth_estimator(room_photo).detach(), mask=shell_mask
    )


@pytest.fixture
def reward(
    reference: GeometricReference, fake_depth_estimator: FakeDepthEstimator
) -> MfmGeometricReward:
    return MfmGeometricReward(reference=reference, depth_estimator=fake_depth_estimator)


def test_the_score_is_the_negated_energy_and_never_positive(
    reward: MfmGeometricReward, generator: Generator
) -> None:
    images = torch.rand((3, 3, 64, 64), generator=generator)

    scored = reward.score(images)
    assert torch.allclose(scored, -reward.energy(images))
    assert (scored <= 0.0).all()


def test_the_photo_itself_scores_zero(
    reward: MfmGeometricReward, room_photo: Float[Tensor, "1 3 64 64"]
) -> None:
    # The reference was built from this very photo by this very backend, so the
    # affine fit is the identity and the residual is exactly nothing. Anything
    # else means the adapter is not comparing what it claims to.
    assert float(reward.score(room_photo)) == pytest.approx(0.0, abs=1e-9)


def test_a_drifted_room_scores_below_the_photo(
    reward: MfmGeometricReward, room_photo: Float[Tensor, "1 3 64 64"]
) -> None:
    drifted = room_photo.clone()
    drifted[:, :, :32, :] = 1.0

    assert float(reward.score(drifted)) < float(reward.score(room_photo))


def test_scale_multiplies_the_score_but_not_the_energy(
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
    generator: Generator,
) -> None:
    images = torch.rand((2, 3, 64, 64), generator=generator)
    plain = MfmGeometricReward(
        reference=reference, depth_estimator=fake_depth_estimator
    )
    tilted = MfmGeometricReward(
        reference=reference, depth_estimator=fake_depth_estimator, scale=7.0
    )

    assert torch.allclose(tilted.energy(images), plain.energy(images))
    assert torch.allclose(tilted.score(images), 7.0 * plain.score(images))


def test_the_score_is_per_image_not_a_batch_mean(
    reward: MfmGeometricReward, room_photo: Float[Tensor, "1 3 64 64"]
) -> None:
    drifted = room_photo.clone()
    drifted[:, :, :32, :] = 1.0
    batch = torch.cat([room_photo, drifted])

    scored = reward.score(batch)
    assert scored.shape == (2,)
    # MFM weights posterior draws against each other by their own scores, so a
    # reward that collapsed the batch would weight every draw identically.
    assert float(scored[0]) > float(scored[1])


def test_the_score_reaches_the_image_it_scored(
    reward: MfmGeometricReward, generator: Generator
) -> None:
    images = torch.rand((2, 3, 64, 64), generator=generator).requires_grad_(True)

    gradient = assert_gradient_is_finite_and_nonzero(reward.score(images).sum(), images)
    assert gradient.shape == images.shape


def test_a_prompt_is_accepted_and_changes_nothing(
    reward: MfmGeometricReward, generator: Generator
) -> None:
    images = torch.rand((2, 3, 64, 64), generator=generator)

    assert torch.equal(
        reward.score(images, ["a tabby cat", "a studio couch"]), reward.score(images)
    )


def test_calling_it_is_scoring_it(
    reward: MfmGeometricReward, generator: Generator
) -> None:
    images = torch.rand((2, 3, 64, 64), generator=generator)

    assert torch.equal(reward(images), reward.score(images))


def test_re_segmenting_narrows_what_is_scored(
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
    generator: Generator,
) -> None:
    images = torch.rand((2, 3, 64, 64), generator=generator)
    shell_only = MfmGeometricReward(
        reference=reference, depth_estimator=fake_depth_estimator
    )
    resegmenting = MfmGeometricReward(
        reference=reference,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )

    narrowed = resegmenting.scored_pixels(images)
    assert int(narrowed.sum()) < int(shell_only.scored_pixels(images).sum())
    assert not (narrowed & ~reference.mask).any()


def test_the_scored_mask_is_the_one_the_energy_used(
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
    generator: Generator,
) -> None:
    # The mask an eval quotes coverage over has to be the mask the reward
    # actually scored, or the coverage number is about something else.
    images = torch.rand((2, 3, 64, 64), generator=generator)
    reward = MfmGeometricReward(
        reference=reference,
        depth_estimator=fake_depth_estimator,
        segmenter=fake_segmenter,
        erosion_radius=_EROSION_RADIUS,
    )

    expected = masked_depth_error(
        fake_depth_estimator(images),
        reference.disparity.expand(2, -1, -1),
        reward.scored_pixels(images),
    )
    assert torch.allclose(reward.energy(images), expected)


def test_a_reference_at_another_resolution_is_refused(
    reward: MfmGeometricReward, generator: Generator
) -> None:
    with pytest.raises(ValueError, match="reference was built at"):
        reward.score(torch.rand((1, 3, 32, 32), generator=generator))


def test_a_half_configured_re_segmentation_is_refused(
    reference: GeometricReference,
    fake_depth_estimator: FakeDepthEstimator,
    fake_segmenter: FakeSegmenter,
) -> None:
    with pytest.raises(ValueError, match="erosion_radius"):
        MfmGeometricReward(
            reference=reference,
            depth_estimator=fake_depth_estimator,
            segmenter=fake_segmenter,
        )


def test_a_reward_that_steers_away_from_the_reference_is_refused(
    reference: GeometricReference, fake_depth_estimator: FakeDepthEstimator
) -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        MfmGeometricReward(
            reference=reference, depth_estimator=fake_depth_estimator, scale=-1.0
        )
