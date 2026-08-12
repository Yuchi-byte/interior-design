"""roomgen's depth energy as a Meta Flow Maps reward: `score(images, prompts) -> [batch]`.

MFM (`packages/mfm`, vendored upstream, never edited) scores decoded images with
an image-preference model and *ascends* what it returns — `sample_steered.py`
adds the gradient of the reward to the sampler's drift, and `losses/finetune.py`
tilts the training target by ``exp(reward)``. roomgen's
:func:`~roomgen_geometry.reward.masked_depth_error` is an *energy*: zero when
the geometry matches and growing as it drifts. :class:`MfmGeometricReward` is
the sign flip and the scale between the two conventions, and nothing else.

The *shape* needs no conversion, and must not get one: MFM's
`batched_reward_estimation` reshapes what a reward returns to `[batch,
mc_samples]` and weights the draws against each other by their own scores, so
every method here is per image — the same per-batch-element contract
:class:`~roomgen_guide.guidance.base.Guidance` carries.

Nothing here imports `mfm`. The interface is structural — the two methods MFM
calls, on tensors it already speaks — so the adapter is checked, tested and
gated like the rest of the repo while the vendored tree stays outside all three.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.masking import MaskMode, scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.reward import masked_depth_error
from roomgen_geometry.segmentation.base import Segmenter

# MFM samples under `torch.autocast(cuda, bfloat16)`, and bfloat16 carries about
# three significant decimal digits — fewer than the depth error's own dynamic
# range across a run. The energy is float32 by contract
# (`roomgen_geometry.reward`), so the disparity it reduces must be too, and the
# depth backend returns its prediction in the dtype it was *called* with. Hence
# the input is lifted here rather than trusted to arrive lifted.
_REWARD_DTYPE: Final = torch.float32


@dataclass(frozen=True)
class MfmGeometricReward:
    """Masked depth error against a reference photo, negated for a sampler that ascends.

    Args:
        reference: the photographed room whose shell the sample is held to,
            built at the sampler's own frame size — MFM's ImageNet models decode
            at 256x256, so a reference at any other resolution is a mismatch this
            rejects rather than silently resamples.
        depth_estimator: the backend `reference` was built with; re-run on every
            scored image, on the gradient path throughout.
        segmenter: re-segments each sample so the score falls on the reference's
            shell intersected with the sample's own, as
            :class:`~roomgen_guide.guidance.geometric.GeometricGuidance` does.
            `None` — the default here, unlike in guidance — scores the
            reference's shell alone: MFM evaluates the reward on several
            posterior draws per particle per step, so a segmentation forward per
            draw is a large constant on the sampling loop, and a mask that
            shrinks as the sample fills with furniture is a way for a *search*
            to score well without holding any geometry.
        erosion_radius: for the sample's mask; required with `segmenter` and
            meaningless without it.
        mask_mode: which way a sample's segmentation is read, as
            :func:`roomgen_geometry.masking.scorable_mask` defines it.
        scale: multiplies the energy before the sign flip. This is MFM's
            `image_reward._lambda` under another name, and it is not cosmetic:
            MFM-G weights its posterior draws by ``softmax`` over their rewards
            (`utils/steering.py`'s IWAE estimator takes a `logsumexp`), and
            fine-tuning tilts by ``exp(reward)``, so `scale` is the temperature
            of both. Gradient *magnitude* is a separate matter — the steering
            config renormalises the steering drift against the base drift, which
            divides `scale` back out.
        autocast: whether the reward may run under an ambient autocast. False —
            the default — evaluates depth and the energy in float32 inside a
            sampler that is otherwise bfloat16, which is the same float32-by-
            contract rule the guided sampler follows.

    The returned score is ``-scale * masked_depth_error``: at most zero, zero
    only for a sample whose masked disparity affine-fits the photo's exactly.
    That upper bound is what makes it safe for the fine-tuning path, where the
    target is tilted by ``exp(score)`` and a positive-unbounded reward would
    exponentiate away.
    """

    reference: GeometricReference
    depth_estimator: DepthEstimator
    segmenter: Segmenter | None = None
    erosion_radius: int | None = None
    mask_mode: MaskMode = "intersect_shell"
    scale: float = 1.0
    autocast: bool = False

    def __post_init__(self) -> None:
        if (self.segmenter is None) != (self.erosion_radius is None):
            raise ValueError(
                "re-segmenting the sample needs both a segmenter and an "
                "erosion_radius; pass neither to score the reference's shell alone"
            )
        if self.scale <= 0.0:
            raise ValueError(
                f"scale must be positive — it is the temperature of MFM's reward "
                f"weighting, and a negative one steers away from the reference — "
                f"got {self.scale}"
            )

    def score(
        self,
        images: Float[Tensor, "batch 3 height width"],
        prompts: Sequence[str] | None = None,
    ) -> Float[Tensor, "batch"]:
        """The reward MFM ascends: `-scale` times the masked depth error, per image.

        Args:
            images: decoded samples, NCHW, floating point, values in ``[0, 1]``,
                still attached to the latent they came from.
            prompts: accepted and ignored, because MFM's reward models take one
                and this has to stand where they do. Geometry is not a caption:
                the reference photo is the whole conditioning.

        Differentiable with respect to `images`; a caller that hands this a
        detached tensor gets a valid score and no steering.
        """
        del prompts
        return -self.scale * self.energy(images)

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        """`score` without the prompt, which is how MFM's samplers call a reward."""
        return self.score(images)

    def energy(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        """The unnegated, unscaled masked depth error — what an eval quotes."""
        with torch.autocast(images.device.type, enabled=self.autocast):
            scored = self.scored_pixels(images)
            prepared = images if self.autocast else images.to(_REWARD_DTYPE)
            disparity = self.depth_estimator(prepared)
            return masked_depth_error(
                disparity, self._reference_disparity(images), scored
            )

    def scored_pixels(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Bool[Tensor, "batch height width"]:
        """Which pixels this reward scored, for the coverage an eval reports beside it."""
        self._check_frame(images)
        shell = self._expand(self.reference.mask, images.shape[0])
        segmenter, erosion_radius = self.segmenter, self.erosion_radius
        if segmenter is None or erosion_radius is None:
            return shell
        # A constant of the step, like every segmentation in this repo: detached
        # and under `no_grad`, so no graph is built for it at all. The undetached
        # image is what the depth backend sees.
        with torch.no_grad():
            segmentation = segmenter(images.detach())
        return shell & scorable_mask(
            segmentation, mode=self.mask_mode, erosion_radius=erosion_radius
        )

    def _reference_disparity(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch height width"]:
        return self._expand(self.reference.disparity, images.shape[0]).to(_REWARD_DTYPE)

    def _check_frame(self, images: Float[Tensor, "batch 3 height width"]) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must be (batch, 3, height, width), got shape "
                f"{tuple(images.shape)}"
            )
        frame = tuple(images.shape[-2:])
        expected = tuple(self.reference.mask.shape[-2:])
        if frame != expected:
            raise ValueError(
                f"the sampler decodes at {frame[0]}x{frame[1]} but the reference was "
                f"built at {expected[0]}x{expected[1]}; build the reference at the "
                f"sampler's frame size rather than comparing depth across a resize"
            )

    @staticmethod
    def _expand(constant: Tensor, batch: int) -> Tensor:
        if constant.shape[0] == batch:
            return constant
        if constant.shape[0] != 1:
            raise ValueError(
                f"a reference over {constant.shape[0]} frames cannot score a batch of "
                f"{batch}; build it from a single photo and let it broadcast"
            )
        return constant.expand(batch, -1, -1)
