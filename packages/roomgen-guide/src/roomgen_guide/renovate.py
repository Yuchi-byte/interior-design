"""The end-to-end entry points: a photographed room in, a renovated one out.

Two samplers, one reward. :func:`renovate` steers Stable Diffusion 1.5 img2img
by the raw capped gradient of the geometric energy; :func:`renovate_flow_map`
steers a two-time flow map by FMRG-J's sign step. They are siblings, not a
default and a variant: everything that makes a renovation a renovation — the
reward, the shell mask, the reference, the eval that judges the result — is the
same object in both, and what differs is only which process the driver walks and
which tilt policy turns the reward's gradient into a step.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from jaxtyping import Float
from torch import Tensor

from roomgen_geometry.alignment import AlignmentMode
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_guide.adapters.flow_map import (
    FlowMapBackend,
    FlowMapProcess,
    FlowMapStepping,
    GuidanceWindow,
)
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.composite import WeightedGuidance, WeightedTerm
from roomgen_guide.guidance.geometric import GeometricGuidance, MaskObserver
from roomgen_guide.process import (
    AnnealedSignTilt,
    WarmupRestarts,
    sample_endpoint_process,
)
from roomgen_guide.sampling.guided import sample_guided
from roomgen_guide.sampling.pipeline import Img2ImgPipeline

# The canonical Stable Diffusion 1.5 mirror: the original `runwayml` repo no
# longer exists on the Hub (CreativeML OpenRAIL-M).
DEFAULT_CHECKPOINT: Final = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# The flow-map operating point roomgen's own tuning selected, at 512² against a
# FLUX.1-dev flow-map LoRA: eta 10 (2.5 never builds the room, 5 leaves cells
# unsteered), a 102-sigma grid stopped at 25 — t_stop 0.245, the value upstream
# reports and this reward reproduces, with adherence falling monotonically as the
# window lengthens — the final jump split in two, and 3 warmup particles walking
# half the guided window. Together that is ~101 NFE, which is the frontier: 53
# goes off-manifold and 205 wins raw depth error while hacking it (an 11x blur
# ratio), so more budget is not more geometry.
TUNED_STRENGTH: Final = 10.0
TUNED_STEPS: Final = 102
TUNED_WINDOW: Final = GuidanceWindow(early_stop=25, unguided_steps=2)
TUNED_RESTARTS: Final = WarmupRestarts(particles=3, steps=12)


def renovate(
    photo: Float[Tensor, "batch 3 height width"],
    prompt: str,
    *,
    pipeline: Img2ImgPipeline,
    depth_estimator: DepthEstimator,
    segmenter: Segmenter,
    strength: float,
    erosion_radius: int,
    steps: int,
    noise_strength: float,
    text_guidance_scale: float,
    generator: torch.Generator,
    clip_step_rms: float | None = None,
    reference: GeometricReference | None = None,
    resegment: bool = True,
    extra_rewards: Sequence[WeightedTerm] = (),
    on_mask: MaskObserver | None = None,
    alignment: AlignmentMode = "scale_shift",
) -> Float[Tensor, "batch 3 height width"]:
    """Restyle the room in `photo` under `prompt` while holding its geometry.

    Args:
        photo: NCHW, floating point, values in ``[0, 1]``; the image returned
            follows the same convention.
        strength: what the geometric energy's gradient is multiplied by before
            it moves the latent, at every step; 0 is the unguided control.
        clip_step_rms: ceiling on the RMS of the raw step, keeping the step that
            a well-matched scene earns while bounding the one a large energy
            asks for.
        reference: the photo's already-frozen geometry, which the caller must
            have built with this same `depth_estimator` and `segmenter` — a
            reference is only meaningful to the backends that produced it.
            `None` freezes the photo here, at the cost of a depth and a
            segmentation pass per call.
        resegment: score on the reference shell *and* the prediction's own,
            re-segmenting the sample at each guided step — occluding a wall with
            new furniture is then free, moving that wall is not. `False` scores
            the reference shell alone, which charges a renovation for every
            surface it covers up, and leaves `segmenter` unused whenever
            `reference` is already built.
        extra_rewards: further terms summed beside the geometric one, as
            :func:`renovate_flow_map` takes them.
        on_mask: called with the mask each guided step scored, for a caller
            recording what the reward actually saw.
        alignment: how much affine freedom the energy's depth comparison hands
            back, as :class:`~roomgen_guide.guidance.geometric.GeometricGuidance`
            takes it; only a metric `depth_estimator` may loosen the default.

    Samples against the photo's frozen
    :class:`~roomgen_geometry.reference.GeometricReference` with the same
    `depth_estimator` on both sides.
    """
    return sample_guided(
        pipeline,
        photo,
        prompt,
        guidance=geometric_reward(
            photo,
            depth_estimator=depth_estimator,
            segmenter=segmenter,
            erosion_radius=erosion_radius,
            reference=reference,
            resegment=resegment,
            extra_rewards=extra_rewards,
            on_mask=on_mask,
            alignment=alignment,
        ),
        strength=strength,
        steps=steps,
        noise_strength=noise_strength,
        text_guidance_scale=text_guidance_scale,
        generator=generator,
        clip_step_rms=clip_step_rms,
    )


def renovate_flow_map(
    photo: Float[Tensor, "batch 3 height width"],
    *,
    backend: FlowMapBackend,
    depth_estimator: DepthEstimator,
    segmenter: Segmenter,
    erosion_radius: int,
    generator: torch.Generator,
    start_sigma: float,
    strength: float = TUNED_STRENGTH,
    steps: int = TUNED_STEPS,
    window: GuidanceWindow | None = TUNED_WINDOW,
    restarts: WarmupRestarts | None = TUNED_RESTARTS,
    iterations: int = 1,
    stepping: FlowMapStepping = "two_time",
    differentiate_through_velocity: bool = True,
    reference: GeometricReference | None = None,
    resegment: bool = True,
    extra_rewards: Sequence[WeightedTerm] = (),
    on_mask: MaskObserver | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    """The same renovation on a two-time flow map, steered by FMRG-J.

    The defaults are the faithful configuration, and they are faithful together:
    the Jacobian pullback, the two-time step, the truncated window and the sign
    step are one method (arXiv:2604.27147), not four knobs. What roomgen adds to
    it is the reward and the img2img start — every published FMRG result begins
    at pure noise, and starting from the photographed room is what makes matching
    *this* room's geometry a matter of holding what is already there rather than
    of inventing texture the depth network misreads.

    Args:
        backend: the flow map, holding the prompt; there is no `prompt` argument
            because a two-time sampler encodes its conditioning once, at load.
        start_sigma: how far up the interpolant the photo is renoised; 1 discards
            it and is the text-to-image control. Measurements are run at 0.9 or
            at 1, never lower — below that the unguided control already sits at
            the metric's noise floor.
        strength: FMRG's `eta`, in latent units per unit sigma. Not comparable to
            :func:`renovate`'s `strength`, which multiplies a raw gradient: the
            sign step's displacement is `eta * dsigma * (1 - k/N)` per element
            whatever the reward's scale.
        steps: the resolution of the sigma grid, not the number of steps walked —
            `window` truncates it.
        window: FMRG's early stop, the grid truncation and the jump to data.
            `None` guides and walks the whole grid, which its own ablation and
            roomgen's both measure as worse.
        restarts: warmup particles; `None` runs a single trajectory.
        iterations: FMRG's `n_opt`, at one model call each.
        stepping: which velocity the step is taken off; `two_time` is the mode
            upstream uses above ~30 NFE and the one this operating point is
            measured under.
        differentiate_through_velocity: FMRG-J rather than FMRG-E — the reward
            gradient is pulled back through the transformer, which is what
            projects it onto the data manifold. Off, the walk is FMRG-E, cheaper
            per step and free to leave the manifold in exactly the direction a
            depth reward is satisfied by.
        extra_rewards: terms summed beside the geometric one. A preference model
            (:class:`~roomgen_guide.guidance.preference.PreferenceGuidance`)
            belongs here whenever the run is judged on more than depth: the depth
            energy's own optimum is an empty room, and every depth-only arm
            measured below the unguided control on preference.
    """
    process = FlowMapProcess(
        backend,
        photo,
        steps=steps,
        start_sigma=start_sigma,
        generator=generator,
        window=window,
        stepping=stepping,
        differentiate_through_velocity=differentiate_through_velocity,
    )
    return sample_endpoint_process(
        process,
        reward=geometric_reward(
            photo,
            depth_estimator=depth_estimator,
            segmenter=segmenter,
            erosion_radius=erosion_radius,
            reference=reference,
            resegment=resegment,
            extra_rewards=extra_rewards,
            on_mask=on_mask,
        ),
        tilt=AnnealedSignTilt(
            strength=strength,
            guided_steps=None if window is None else window.guided_steps,
            anneal_steps=None if window is None else window.early_stop,
            iterations=iterations,
        ),
        restarts=restarts,
    )


def geometric_reward(
    photo: Float[Tensor, "batch 3 height width"],
    *,
    depth_estimator: DepthEstimator,
    segmenter: Segmenter,
    erosion_radius: int,
    reference: GeometricReference | None = None,
    resegment: bool = True,
    extra_rewards: Sequence[WeightedTerm] = (),
    on_mask: MaskObserver | None = None,
    alignment: AlignmentMode = "scale_shift",
) -> Guidance:
    """The energy both samplers descend: the depth term, plus whatever rides beside it.

    The reference is frozen from `photo` when the caller has not already frozen
    it — with the same `depth_estimator` and `segmenter` the sample will be
    scored by, since a reference means nothing to any other backend.
    """
    if reference is None:
        reference = GeometricReference.from_photo(
            photo,
            depth_estimator=depth_estimator,
            segmenter=segmenter,
            erosion_radius=erosion_radius,
        )
    geometric = GeometricGuidance(
        depth_estimator=depth_estimator,
        reference=reference,
        segmenter=segmenter if resegment else None,
        erosion_radius=erosion_radius if resegment else None,
        on_mask=on_mask,
        alignment=alignment,
    )
    if not extra_rewards:
        return geometric
    return WeightedGuidance((WeightedTerm(geometric, 1.0), *extra_rewards))


def load_pipeline(
    checkpoint: str = DEFAULT_CHECKPOINT,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | None = None,
) -> Img2ImgPipeline:
    """Stable Diffusion img2img from the Hub, ready for `renovate` to drive.

    Args:
        checkpoint: Hugging Face Hub id; weights are fetched on demand, never
            vendored.
        device: where the weights live and where inputs are expected.
        dtype: parameter dtype.
        cache_dir: Hub snapshot cache the weights are read from and downloaded
            into (`RoomgenSettings.hub_cache`); `None` follows the Hub client's
            own environment.
    """
    # The safety checker is refused rather than merely skipped: its weights are
    # excluded from the seeded cache, so a pipeline that still wants them turns
    # an offline load into a download.
    #
    # diffusers ships `py.typed` but leaves `from_pretrained`'s `**kwargs`
    # unannotated, which makes the loader itself an untyped call.
    pipeline: Img2ImgPipeline = StableDiffusionImg2ImgPipeline.from_pretrained(  # type: ignore[no-untyped-call]
        checkpoint,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    return pipeline
