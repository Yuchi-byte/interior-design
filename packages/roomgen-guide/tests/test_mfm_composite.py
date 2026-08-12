from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import pytest
import torch
from jaxtyping import Bool, Float
from torch import Generator, Tensor

from roomgen_fakes import FakeDepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.adapters.mfm_composite import (
    MfmReward,
    MfmRewardTerm,
    MfmWeightedReward,
    gradient_scale,
)
from roomgen_guide.adapters.mfm_reward import MfmGeometricReward

_BATCH: Final = 3


@dataclass
class FakePreferenceReward:
    """A caption-conditioned preference score, as `imscore`'s HPSv2 presents one.

    Reads the caption — a preference model that did not would hide exactly the
    plumbing bug this adapter exists to avoid, a term scored against the wrong
    prompt or against none.
    """

    seen: list[Sequence[str] | None] = field(default_factory=list)

    def score(
        self,
        images: Float[Tensor, "batch 3 height width"],
        prompts: Sequence[str] | None,
    ) -> Float[Tensor, "batch"]:
        self.seen.append(prompts)
        if prompts is None:
            raise ValueError("a preference model cannot score without a caption")
        length = torch.tensor(
            [float(len(prompt)) for prompt in prompts], device=images.device
        )
        return length * images.flatten(1).mean(-1)


@pytest.fixture
def geometric(
    fake_depth_estimator: FakeDepthEstimator,
    room_photo: Float[Tensor, "1 3 64 64"],
    shell_mask: Bool[Tensor, "1 64 64"],
) -> MfmGeometricReward:
    return MfmGeometricReward(
        reference=GeometricReference(
            disparity=fake_depth_estimator(room_photo).detach(), mask=shell_mask
        ),
        depth_estimator=fake_depth_estimator,
    )


@pytest.fixture
def images(generator: Generator) -> Tensor:
    return torch.rand((_BATCH, 3, 64, 64), generator=generator)


def test_both_kinds_of_reward_are_the_same_kind_of_term(
    geometric: MfmGeometricReward,
) -> None:
    assert isinstance(geometric, MfmReward)
    assert isinstance(FakePreferenceReward(), MfmReward)


def test_the_combined_score_is_the_weighted_sum_of_its_terms(
    geometric: MfmGeometricReward, images: Tensor
) -> None:
    preference = FakePreferenceReward()
    combined = MfmWeightedReward(
        (
            MfmRewardTerm(geometric, weight=1.0),
            MfmRewardTerm(preference, weight=0.25, prompt="a home theater"),
        )
    )

    scored = combined.score(images)

    expected = geometric.score(images) + 0.25 * preference.score(
        images, ["a home theater"] * _BATCH
    )
    torch.testing.assert_close(scored, expected)


def test_the_combined_score_is_per_draw_and_not_a_batch_mean(
    geometric: MfmGeometricReward, images: Tensor
) -> None:
    # MFM reshapes what a reward returns to [batch, mc_samples] and weights the
    # draws against each other, so a combined reward that collapsed the batch
    # would not even reshape, let alone weight.
    combined = MfmWeightedReward(
        (
            MfmRewardTerm(geometric),
            MfmRewardTerm(FakePreferenceReward(), prompt="a studio couch"),
        )
    )

    scored = combined(images)

    assert scored.shape == (_BATCH,)
    assert float(scored.std()) > 0.0


def test_a_term_is_scored_against_its_own_caption_when_the_sampler_gives_none(
    geometric: MfmGeometricReward, images: Tensor
) -> None:
    # MFM's steering calls `reward_fn(images)` with no prompts at all, and the
    # class-conditional model has no prompt to give: the term's own caption is
    # the only one there will ever be.
    preference = FakePreferenceReward()
    combined = MfmWeightedReward(
        (MfmRewardTerm(geometric), MfmRewardTerm(preference, prompt="a dining table"))
    )

    combined(images)

    assert preference.seen == [["a dining table"] * _BATCH]


def test_the_samplers_own_captions_win_over_the_terms(images: Tensor) -> None:
    preference = FakePreferenceReward()
    combined = MfmWeightedReward((MfmRewardTerm(preference, prompt="a dining table"),))
    supplied = ["a loft"] * _BATCH

    combined.score(images, supplied)

    assert preference.seen == [supplied]


def test_a_term_at_weight_zero_leaves_the_rest_exactly_where_they_were(
    geometric: MfmGeometricReward, images: Tensor
) -> None:
    alone = MfmWeightedReward((MfmRewardTerm(geometric),))
    silenced = MfmWeightedReward(
        (
            MfmRewardTerm(geometric),
            MfmRewardTerm(FakePreferenceReward(), weight=0.0, prompt="a home theater"),
        )
    )

    torch.testing.assert_close(silenced(images), alone(images), rtol=0.0, atol=0.0)


def test_the_gradient_carries_both_terms_back_to_the_image(
    geometric: MfmGeometricReward, images: Tensor
) -> None:
    # The whole point of combining in score space: one backward, both pulls. A
    # term that arrived detached would leave the sampler steering by the other
    # one alone while every score still looked right.
    preference = FakePreferenceReward()
    combined = MfmWeightedReward(
        (
            MfmRewardTerm(geometric),
            MfmRewardTerm(preference, weight=0.5, prompt="a home theater"),
        )
    )
    leaf = images.clone().requires_grad_(True)

    (gradient,) = torch.autograd.grad(combined(leaf).sum(), leaf)

    depth_leaf = images.clone().requires_grad_(True)
    (depth_gradient,) = torch.autograd.grad(
        geometric.score(depth_leaf).sum(), depth_leaf
    )
    preference_leaf = images.clone().requires_grad_(True)
    (preference_gradient,) = torch.autograd.grad(
        preference.score(preference_leaf, ["a home theater"] * _BATCH).sum(),
        preference_leaf,
    )
    assert torch.isfinite(gradient).all()
    assert float(depth_gradient.abs().max()) > 0.0
    assert float(preference_gradient.abs().max()) > 0.0
    torch.testing.assert_close(gradient, depth_gradient + 0.5 * preference_gradient)


def test_a_negative_weight_is_rejected(geometric: MfmGeometricReward) -> None:
    with pytest.raises(ValueError, match="steers away from it"):
        MfmRewardTerm(geometric, weight=-1.0)


def test_a_combination_with_no_terms_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one term"):
        MfmWeightedReward(())


def test_the_gradient_scale_is_what_a_weight_is_read_against(
    geometric: MfmGeometricReward, images: Tensor
) -> None:
    # Scores in different units say nothing about relative pull; the gradients
    # do, and a weight is chosen so the two are the ratio the run wants.
    preference = FakePreferenceReward()

    depth = gradient_scale(geometric, images)
    plausibility = gradient_scale(preference, images, ["a home theater"] * _BATCH)

    assert depth.shape == (_BATCH,)
    assert (depth > 0.0).all()
    assert (plausibility > 0.0).all()
    # A weight built from that ratio equalises the two terms' pull, which is
    # what makes it a starting point rather than a guess.
    weight = float((depth / plausibility).mean())
    balanced = gradient_scale(
        MfmWeightedReward(
            (MfmRewardTerm(preference, weight=weight, prompt="a home theater"),)
        ),
        images,
    )
    torch.testing.assert_close(balanced, weight * plausibility)
