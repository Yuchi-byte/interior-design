"""Removal by inpainting: repaint what a mask covers, keep every other pixel the photo's."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import torch
from diffusers import StableDiffusionInpaintPipeline
from jaxtyping import Bool, Float
from torch import Tensor

DEFAULT_CHECKPOINT: Final = "stable-diffusion-v1-5/stable-diffusion-inpainting"


class InpaintOutput(Protocol):
    """What a diffusers inpainting call returns: the batch under `.images`."""

    images: Tensor


@runtime_checkable
class InpaintPipeline(Protocol):
    """The part of a diffusers inpainting pipeline this module drives.

    Called with tensors and `output_type="pt"` throughout, so no image round-trips
    through PIL and the returned batch is comparable to the photo pixel for pixel.
    """

    def __call__(
        self,
        *,
        prompt: str,
        image: Float[Tensor, "batch 3 height width"],
        mask_image: Float[Tensor, "batch 1 height width"],
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator,
        output_type: str,
    ) -> InpaintOutput: ...


@dataclass(frozen=True)
class Inpainting:
    """A strip and the raw render behind it, so a caller can measure what compositing saved.

    Args:
        image: the photo everywhere the mask is False and the render everywhere
            it is True — the result the caller wants.
        generated: what the pipeline returned, before the photo was put back
            outside the mask. It differs from the photo *everywhere*, because a
            latent-diffusion pipeline decodes the whole frame through its VAE;
            the distance between the two outside the mask is the drift
            compositing is what removes.
    """

    image: Float[Tensor, "batch 3 height width"]
    generated: Float[Tensor, "batch 3 height width"]


def inpaint_masked_region(
    photo: Float[Tensor, "batch 3 height width"],
    mask: Bool[Tensor, "batch height width"],
    *,
    pipeline: InpaintPipeline,
    prompt: str,
    steps: int,
    guidance_scale: float,
    generator: torch.Generator,
) -> Inpainting:
    """Repaint `photo` under `prompt` wherever `mask` is True, leaving the rest untouched.

    Args:
        photo: NCHW, floating point, values in ``[0, 1]``; the images returned
            follow the same convention and dtype.
        mask: True where the pipeline may paint — what is being removed, grown by
            whatever margin its caller chose
            (:func:`roomgen_geometry.masking.contents_mask`).
        generator: the only source of randomness, so a strip is reproducible from
            its seed alone.

    Outside the mask the result is the input photograph bit for bit: the render
    is composited back rather than trusted, since the pipeline's VAE re-decodes
    the whole frame and would otherwise move pixels nothing asked it to touch.
    """
    output = pipeline(
        prompt=prompt,
        image=photo,
        mask_image=mask.unsqueeze(1).to(photo.dtype),
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
    )
    generated = output.images.to(device=photo.device, dtype=photo.dtype)
    return Inpainting(
        image=torch.where(mask.unsqueeze(1), generated, photo), generated=generated
    )


def load_inpaint_pipeline(
    checkpoint: str = DEFAULT_CHECKPOINT,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | None = None,
) -> InpaintPipeline:
    """The Stable Diffusion inpainting pipeline from the Hub, ready to strip a room.

    Args:
        checkpoint: Hugging Face Hub id; weights are fetched on demand, never
            vendored.
        cache_dir: Hub snapshot cache the weights are read from and downloaded
            into (`RoomgenSettings.hub_cache`); `None` follows the Hub client's
            own environment.
    """
    # The safety checker is refused rather than merely skipped, as in
    # `roomgen_guide.renovate.load_pipeline`: its weights are excluded from the
    # seeded cache, so a pipeline that still wants them turns an offline load
    # into a download.
    #
    # diffusers ships `py.typed` but leaves `from_pretrained`'s `**kwargs`
    # unannotated, which makes the loader itself an untyped call.
    pipeline: InpaintPipeline = StableDiffusionInpaintPipeline.from_pretrained(  # type: ignore[no-untyped-call]
        checkpoint,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    return pipeline
