"""Removal by inpainting: repaint what a mask covers, keep every other pixel the photo's."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import torch
from diffusers import StableDiffusionInpaintPipeline
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn.functional import avg_pool2d, interpolate

DEFAULT_CHECKPOINT: Final = "stable-diffusion-v1-5/stable-diffusion-inpainting"

# Below this share of a pooling window, a level has no measurement of its own
# and reads the level above instead. Small enough to trust a single known pixel
# in a 2x2, large enough to stay clear of float32's noise after ten poolings.
_WEIGHT_FLOOR: Final = 1e-4


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
        negative_prompt: str | None,
        image: Float[Tensor, "batch 3 height width"],
        mask_image: Float[Tensor, "batch 1 height width"],
        num_inference_steps: int,
        strength: float,
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
    negative_prompt: str | None = None,
    steps: int,
    fill_strength: float = 1.0,
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
        negative_prompt: what the fill must not contain. A large hole in a
            furnished room is filled from its own surroundings, so an inpainter
            asked only for a bare room will still put furniture back into it;
            naming the furniture on the negative side is what a removal has to
            say out loud.
        fill_strength: how much of the trajectory the hole is repainted over,
            having first been filled with the colours of the surfaces around it
            (:func:`fill_from_surroundings`). At 1 the fill is discarded and the
            model invents the region from the room around it — which is where an
            inpainter puts a television on a wall it was asked to leave bare;
            below 1 it starts from that fill and can only smooth what is already
            a wall or a floor, at the cost of a softer patch.
        generator: the only source of randomness, so a strip is reproducible from
            its seed alone.

    Outside the mask the result is the input photograph bit for bit: the render
    is composited back rather than trusted, since the pipeline's VAE re-decodes
    the whole frame and would otherwise move pixels nothing asked it to touch.
    """
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=fill_from_surroundings(photo, mask),
        mask_image=mask.unsqueeze(1).to(photo.dtype),
        num_inference_steps=steps,
        strength=fill_strength,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
    )
    generated = output.images.to(device=photo.device, dtype=photo.dtype)
    return Inpainting(
        image=torch.where(mask.unsqueeze(1), generated, photo), generated=generated
    )


def fill_from_surroundings(
    image: Float[Tensor, "batch 3 height width"],
    mask: Bool[Tensor, "batch height width"],
) -> Float[Tensor, "batch 3 height width"]:
    """Flood the masked region with the colours around it; leave every other pixel alone.

    A coarse fill, not a repair: the hole comes out as a smooth continuation of
    the wall and floor that surround it, with no structure of its own. What it is
    for is to give a partial-strength inpainting something to start from that is
    already a surface, so the model has a room to finish rather than a hole to
    fill with whatever a room usually contains.
    """
    known = (~mask).unsqueeze(1).to(image.dtype)
    # Pull-push (arXiv:2005.02436 §3.1 states the classic form): average the
    # known pixels up a power-of-two pyramid until the hole is covered at some
    # level, then read each level's own average back down wherever it has one.
    # A same-resolution diffusion would need hundreds of iterations to cross a
    # hole this size, and underflows its weights on the way.
    numerators = [image * known]
    weights = [known]
    while min(numerators[-1].shape[-2:]) > 1:
        numerators.append(avg_pool2d(numerators[-1], 2, ceil_mode=True))
        weights.append(avg_pool2d(weights[-1], 2, ceil_mode=True))
    filled = numerators[-1] / weights[-1].clamp(min=_WEIGHT_FLOOR)
    for level in range(len(numerators) - 2, -1, -1):
        coarser = interpolate(
            filled, size=numerators[level].shape[-2:], mode="bilinear"
        )
        weight = weights[level]
        filled = torch.where(
            weight > _WEIGHT_FLOOR,
            numerators[level] / weight.clamp(min=_WEIGHT_FLOOR),
            coarser,
        )
    return torch.where(mask.unsqueeze(1), filled, image).clamp(0.0, 1.0)


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
