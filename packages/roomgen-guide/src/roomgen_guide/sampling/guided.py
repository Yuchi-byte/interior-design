"""The denoising loop that steers each step by an energy gradient."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.schedule import GuidanceSchedule

if TYPE_CHECKING:
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from diffusers.schedulers.scheduling_utils import SchedulerMixin


@runtime_checkable
class Img2ImgPipeline(Protocol):
    """The part of a diffusers img2img pipeline the guided loop drives directly.

    The loop steps the scheduler itself rather than calling the pipeline, so a
    test drives it with a small fake carrying these three components.
    """

    unet: UNet2DConditionModel
    vae: AutoencoderKL
    scheduler: SchedulerMixin


def sample_guided(
    pipeline: Img2ImgPipeline,
    photo: Float[Tensor, "batch 3 height width"],
    prompt: str,
    *,
    guidance: Guidance,
    schedule: GuidanceSchedule,
    steps: int,
    noise_strength: float,
    text_guidance_scale: float,
    generator: torch.Generator,
) -> Float[Tensor, "batch 3 height width"]:
    """Renoise `photo` and denoise it back under `prompt`, steered by `guidance`.

    Args:
        photo: NCHW, floating point, values in ``[0, 1]``; the image returned
            follows the same convention.
        noise_strength: how far up the trajectory the photo is renoised, in
            ``[0, 1]``; 1 discards it entirely.
        text_guidance_scale: classifier-free guidance on the prompt, unrelated
            to `guidance`.
        generator: the only source of randomness in the loop, so a run is
            reproducible from its seed alone.

    A guided step forms the predicted clean image from the current latent,
    decodes it, evaluates `guidance` on it, and shifts the latent along the
    gradient of that energy — scaled by `schedule` — before the scheduler takes
    its step. No weights are updated.
    """
    raise NotImplementedError
