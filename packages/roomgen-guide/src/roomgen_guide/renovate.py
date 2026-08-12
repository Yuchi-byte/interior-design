"""The end-to-end entry point: a photographed room in, a renovated one out."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from jaxtyping import Float
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.masking import MaskMode
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_guide.guidance.geometric import GeometricGuidance, MaskObserver
from roomgen_guide.sampling.guided import sample_guided
from roomgen_guide.sampling.pipeline import Img2ImgPipeline

# The canonical Stable Diffusion 1.5 mirror: the original `runwayml` repo no
# longer exists on the Hub (CreativeML OpenRAIL-M).
DEFAULT_CHECKPOINT: Final = "stable-diffusion-v1-5/stable-diffusion-v1-5"


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
    differentiate_through_unet: bool = False,
    reference: GeometricReference | None = None,
    resegment: bool = True,
    mask_mode: MaskMode = "intersect_shell",
    on_mask: MaskObserver | None = None,
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
        differentiate_through_unet: DPS-style gradient — keep the graph through
            the UNet so the step carries the denoiser's Jacobian, at roughly
            one extra UNet backward per guided step.
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
        mask_mode: which way a prediction's segmentation is read when it
            narrows the reference shell — `intersect_shell` keeps what the
            prediction also calls shell, `subtract_contents` keeps everything it
            does not affirmatively call contents. Whatever judges the render
            afterwards must read it the same way or the two disagree about what
            was ever being scored.
        on_mask: called with the mask each guided step scored, for a caller
            recording what the reward actually saw.

    Samples against the photo's frozen
    :class:`~roomgen_geometry.reference.GeometricReference` with the same
    `depth_estimator` on both sides.
    """
    if reference is None:
        reference = GeometricReference.from_photo(
            photo,
            depth_estimator=depth_estimator,
            segmenter=segmenter,
            erosion_radius=erosion_radius,
        )
    return sample_guided(
        pipeline,
        photo,
        prompt,
        guidance=GeometricGuidance(
            depth_estimator=depth_estimator,
            reference=reference,
            segmenter=segmenter if resegment else None,
            erosion_radius=erosion_radius if resegment else None,
            mask_mode=mask_mode,
            on_mask=on_mask,
        ),
        strength=strength,
        steps=steps,
        noise_strength=noise_strength,
        text_guidance_scale=text_guidance_scale,
        generator=generator,
        clip_step_rms=clip_step_rms,
        differentiate_through_unet=differentiate_through_unet,
    )


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
