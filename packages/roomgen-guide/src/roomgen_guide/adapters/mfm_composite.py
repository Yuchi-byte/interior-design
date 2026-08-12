"""Several Meta Flow Maps rewards summed in MFM's own score space, per image.

`mfm_reward.MfmGeometricReward` holds a sample to one room's geometry and says
nothing about whether the sample is a plausible photograph — the failure the
steering runs surface is a sample that satisfies the depth energy with large
planar slabs that no longer read as a room. MFM's *native* rewards are
prompt-conditioned human-preference models (HPSv2 by default,
`conf/config_finetune.yaml`), which say exactly that and nothing about geometry.
:class:`MfmWeightedReward` runs both on the same draws and hands MFM one score,
which is the reweighting-family analogue of v1's two-sided depth/adherence gate.

Nothing here imports `mfm` or any preference model. The interface is structural
— `score(images, prompts)` over decoded `[0, 1]` NCHW tensors, higher is better,
one score per image — and it is the interface `imscore`'s `HPSv2.score`,
`ImageReward.score` and our own
:class:`~roomgen_guide.adapters.mfm_reward.MfmGeometricReward` already present,
so a term is anything of that shape.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import ASCENDED_SCORE, RewardDirection


@runtime_checkable
class MfmReward(Protocol):
    """What Meta Flow Maps calls a reward: one score per image, higher is better.

    Args:
        images: decoded samples, NCHW, floating point, values in ``[0, 1]``,
            still attached to the latent they came from.
        prompts: one caption per image, for a reward that is conditioned on one;
            a reward that is not takes it and ignores it.

    Per image and never reduced: MFM weights the `mc_samples` posterior draws of
    a step against each other by their own scores, so a reward that returned the
    batch's mean would weight every draw identically.
    """

    def score(
        self,
        images: Float[Tensor, "batch 3 height width"],
        prompts: Sequence[str] | None,
        /,
    ) -> Float[Tensor, "batch"]: ...


@dataclass(frozen=True)
class MfmRewardTerm:
    """One term of a combined MFM reward, in the units its own `reward` returns.

    Args:
        weight: what the term's score is multiplied by. This is the term's
            relative pull *and* its unit conversion, and two rewards combined
            here are nowhere near the same units — HPSv2.1 returns about 0.1 to 0.3
            while the negated depth energy spans 0 to about -1.5 at `scale` 1,
            and what actually reaches the sampler is neither, but the gradient.
            Size it from :func:`gradient_scale` on real samples rather than from
            the scores. Negative weights are rejected: a term is a reward to
            ascend, and flipping its sign here would steer away from it while
            still reading as a weight.
        prompt: the caption this term is scored against when the sampler hands
            the reward none of its own — which MFM's steering always does, since
            `utils/steering.py` calls `reward_fn(images)`. A class-conditional
            model has no prompt at all, so a preference term needs one supplied
            here; a term that ignores captions leaves this `None`.
    """

    reward: MfmReward
    weight: float = 1.0
    prompt: str | None = None

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(
                f"weight must be non-negative — a term is a reward the sampler "
                f"ascends, and a negative weight steers away from it rather than "
                f"scaling it — got {self.weight}"
            )

    def score(
        self,
        images: Float[Tensor, "batch 3 height width"],
        prompts: Sequence[str] | None,
    ) -> Float[Tensor, "batch"]:
        """This term's weighted score, under whichever caption applies to it."""
        return self.weight * self.reward.score(images, self._captions(images, prompts))

    def _captions(
        self,
        images: Float[Tensor, "batch 3 height width"],
        prompts: Sequence[str] | None,
    ) -> Sequence[str] | None:
        if prompts is not None:
            return prompts
        if self.prompt is None:
            return None
        return [self.prompt] * images.shape[0]


@dataclass(frozen=True)
class MfmWeightedReward:
    """Several rewards on the same draws, summed into the one score MFM ascends.

    Every term scores the same images and stays on their graph, so the gradient
    the sampler steers by is the weighted sum of the terms' own gradients; the
    sum is taken per image, so MFM's draw weighting still sees the draws apart.

    Args:
        terms: at least one, evaluated in the order given — which is the order
            they cost their forward passes in.
    """

    #: Ascended, like every term in it — see `MfmGeometricReward.direction` for
    #: what the declaration is for.
    direction: ClassVar[RewardDirection] = ASCENDED_SCORE

    terms: tuple[MfmRewardTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError(
                "a combined reward needs at least one term; an empty sum has no "
                "gradient to steer by"
            )

    def score(
        self,
        images: Float[Tensor, "batch 3 height width"],
        prompts: Sequence[str] | None = None,
    ) -> Float[Tensor, "batch"]:
        """The combined reward, per image — `sum(weight * term)` over the terms.

        Args:
            prompts: one caption per image, applied to *every* term that reads
                one. `None` — which is what MFM's sampler passes — leaves each
                term to its own configured `prompt`.
        """
        total = self.terms[0].score(images, prompts)
        for term in self.terms[1:]:
            total = total + term.score(images, prompts)
        return total

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        """`score` without the prompt, which is how MFM's samplers call a reward."""
        return self.score(images)


def gradient_scale(
    reward: MfmReward,
    images: Float[Tensor, "batch 3 height width"],
    prompts: Sequence[str] | None = None,
) -> Float[Tensor, "batch"]:
    """The RMS of ∂score/∂image per image: what a term's weight has to be read against.

    Two rewards at weight 1 pull on the sample in proportion to their gradients,
    not their scores, and nothing about a preference score in ``[0, 1]`` beside a
    depth energy in aligned-disparity units says what that ratio is. Measured on
    real samples, this is the number a weight is chosen from.

    `images` needs no gradient of its own: the score is differentiated against a
    leaf made from it here.
    """
    with torch.enable_grad():
        leaf = images.detach().requires_grad_(True)
        scored = reward.score(leaf, prompts)
        (gradient,) = torch.autograd.grad(scored.sum(), leaf)
    return gradient.flatten(1).square().mean(-1).sqrt()
