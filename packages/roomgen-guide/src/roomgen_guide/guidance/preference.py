"""A human-preference model as an energy, so it can ride beside the geometry."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class ImagePreferenceModel(Protocol):
    """One score per image, higher is better — `imscore`'s HPSv2/PickScore shape.

    Args:
        images: decoded samples, NCHW, floating point, values in ``[0, 1]``,
            still attached to the latent they came from.
        prompts: one caption per image, for a model conditioned on one; a model
            that is not takes it and ignores it.

    Per image and never reduced, for the reason every reward here is: a caller
    that weights the batch's members against each other by their own numbers
    cannot recover them from a mean.
    """

    def score(
        self,
        images: Float[Tensor, "batch 3 height width"],
        prompts: Sequence[str] | None,
        /,
    ) -> Float[Tensor, "batch"]: ...


@dataclass(frozen=True)
class PreferenceGuidance:
    """A preference score presented to this repo's samplers as an energy to descend.

    The depth energy cannot be trusted alone and the measurements say so: an
    unfurnished shell is the *global optimum* of a shell-masked depth reward, so
    a walk steered by geometry alone — and a warmup selection that judges by the
    same number — both prefer emptying the room, and every depth-only arm
    measured below the unguided control on preference. This term is what makes
    that trade visible to the sampler instead of only to the eval.

    It is the one direction conversion in the package: the model *ascends* its
    score, a `Guidance` is *descended*, and the negation here is what
    `WeightedGuidance` needs to sum it beside the geometric and structural terms
    without any of them changing. Size its weight from the terms' gradients on
    real samples rather than from their scores — a preference score in ``[0, 1]``
    beside a depth energy in aligned-disparity units says nothing about which
    pulls harder — and note that gradient RMS is not the whole story under a sign
    step, which spends the same magnitude on every element: matching the RMS of a
    dense preference gradient to a sparse masked depth one hands the preference
    term the walk.

    Args:
        model: the scorer, frozen and off the weight-gradient path.
        prompt: the caption every image in the batch is scored against, which is
            the prompt the sampler is already conditioned on.
    """

    model: ImagePreferenceModel
    prompt: str

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        prompts = [self.prompt] * predicted_image.shape[0]
        return -self.model.score(predicted_image, prompts)
