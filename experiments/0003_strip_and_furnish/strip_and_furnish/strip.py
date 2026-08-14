"""Stage 1: the photographed room with its furniture cut out and painted over.

Segmentation decides what goes, inpainting fills the hole it leaves, and the
photograph itself is composited back everywhere else. No depth guidance is
involved at any point — the room is held by the pixels that were never
regenerated, which is what makes the stripped frame the *same* room rather than
an empty one that merely resembles it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.masking import contents_mask, scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.eval import depth_error, shell_coverage
from roomgen_guide.sampling.guided import decode_latent
from roomgen_guide.sampling.inpaint import InpaintPipeline, inpaint_masked_region
from roomgen_guide.sampling.pipeline import LatentAutoencoder
from strip_and_furnish.backends import Backends
from strip_and_furnish.config import SceneConfig, StudyConfig
from strip_and_furnish.images import (
    DisparityScale,
    write_disparity_map,
    write_image,
    write_mask_overlay,
)


@dataclass(frozen=True)
class StripNumbers:
    """What one strip measured, and nothing about whether it is good enough.

    Args:
        contents_fraction: share of the frame the dilated contents mask handed
            to the inpainter.
        shell_fraction: share of the frame the photograph's architectural shell
            covers.
        depth_error: the stripped room's masked depth error against the
            photograph, over the photograph's own shell — the evidence that the
            empty room is still this room. A strip that repainted the walls
            somewhere else shows up here and nowhere else.
        visible_depth_error: the same score narrowed to the shell the stripped
            render's *own* segmentation still calls scorable, beside the
            coverage that narrowing left.
        unrepainted_shell_depth_error: the same score again, on the shell the
            mask never reached — pixels that are the photograph's own, to the
            bit. Whatever it reports is what re-reading a room without its
            furniture does to a monocular depth network, since nothing there was
            repainted at all; the gap between it and `depth_error` is what the
            fill itself cost.
        reconstruction_depth_error: what a perfect strip would still pay for one
            VAE round trip of the photograph, on the same shell — the floor the
            two numbers above are read against.
        unmasked_max_absolute_difference: the largest pixel difference between
            the strip and the photograph *outside* the mask. Zero is the claim.
        generated_unmasked_max_absolute_difference: the same distance measured on
            the first pass's raw output, which is what the compositing removed.
        residual_contents_fraction: the share of the cut region the segmenter
            *still* calls furniture once the last pass is done — the number that
            says whether the inpainter put a table where the sofa was.
    """

    contents_fraction: float
    shell_fraction: float
    depth_error: float
    visible_depth_error: float
    visible_shell_coverage: float
    unrepainted_shell_depth_error: float
    unrepainted_shell_coverage: float
    reconstruction_depth_error: float
    unmasked_max_absolute_difference: float
    generated_unmasked_max_absolute_difference: float
    residual_contents_fraction: float
    seconds: float


@dataclass(frozen=True)
class StripResult:
    """One stripped scene: the images stage 2 starts from, and what stage 1 measured."""

    scene: str
    photo: Float[Tensor, "1 3 height width"]
    empty_room: Float[Tensor, "1 3 height width"]
    contents: Bool[Tensor, "1 height width"]
    reference: GeometricReference
    numbers: StripNumbers
    images: dict[str, str]

    def as_row(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "contents_fraction": self.numbers.contents_fraction,
            "shell_fraction": self.numbers.shell_fraction,
            "depth_error": self.numbers.depth_error,
            "visible_depth_error": self.numbers.visible_depth_error,
            "visible_shell_coverage": self.numbers.visible_shell_coverage,
            "unrepainted_shell_depth_error": (
                self.numbers.unrepainted_shell_depth_error
            ),
            "unrepainted_shell_coverage": self.numbers.unrepainted_shell_coverage,
            "reconstruction_depth_error": self.numbers.reconstruction_depth_error,
            "unmasked_max_absolute_difference": (
                self.numbers.unmasked_max_absolute_difference
            ),
            "generated_unmasked_max_absolute_difference": (
                self.numbers.generated_unmasked_max_absolute_difference
            ),
            "residual_contents_fraction": self.numbers.residual_contents_fraction,
            "seconds": self.numbers.seconds,
            "images": self.images,
        }


def strip_scene(
    scene: SceneConfig,
    photo: Float[Tensor, "1 3 height width"],
    *,
    config: StudyConfig,
    backends: Backends,
    inpainter: InpaintPipeline,
    seed: int,
    gallery: Path,
) -> StripResult:
    """Cut the furniture out of `photo`, paint the holes bare, and measure what moved."""
    reference = GeometricReference.from_photo(
        photo,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        erosion_radius=config.furnish.erosion_radius,
    )
    with torch.no_grad():
        contents = contents_mask(
            backends.segmenter(photo), dilation_radius=config.strip.dilation_radius
        )

    generator = torch.Generator(device=backends.device).manual_seed(seed)
    _synchronize(backends.device)
    started = time.perf_counter()
    empty_room, first_generated, residual = _repaint(
        photo,
        contents,
        config=config,
        backends=backends,
        inpainter=inpainter,
        generator=generator,
    )
    _synchronize(backends.device)
    seconds = time.perf_counter() - started

    # The shell the mask never reached: those pixels are the photograph's own,
    # so any depth error over them is the network reading a room that has lost
    # its furniture, not a fill that moved a wall.
    unrepainted = reference.mask & ~contents
    with torch.no_grad():
        visible = reference.mask & scorable_mask(
            backends.segmenter(empty_room),
            erosion_radius=config.furnish.erosion_radius,
        )
    numbers = StripNumbers(
        contents_fraction=float(contents.to(torch.float32).mean()),
        shell_fraction=float(reference.mask.to(torch.float32).mean()),
        depth_error=depth_error(
            empty_room, reference=reference, depth_estimator=backends.depth_estimator
        ),
        visible_depth_error=depth_error(
            empty_room,
            reference=reference,
            depth_estimator=backends.depth_estimator,
            mask=visible,
        ),
        visible_shell_coverage=shell_coverage(visible, reference=reference),
        unrepainted_shell_depth_error=depth_error(
            empty_room,
            reference=reference,
            depth_estimator=backends.depth_estimator,
            mask=unrepainted,
        ),
        unrepainted_shell_coverage=shell_coverage(unrepainted, reference=reference),
        reconstruction_depth_error=depth_error(
            vae_roundtrip(backends.pipeline.vae, photo, device=backends.device),
            reference=reference,
            depth_estimator=backends.depth_estimator,
        ),
        unmasked_max_absolute_difference=_outside_mask_distance(
            empty_room, photo, contents
        ),
        generated_unmasked_max_absolute_difference=_outside_mask_distance(
            first_generated, photo, contents
        ),
        residual_contents_fraction=residual,
        seconds=seconds,
    )
    return StripResult(
        scene=scene.photo,
        photo=photo,
        empty_room=empty_room,
        contents=contents,
        reference=reference,
        numbers=numbers,
        images=_write_gallery(
            scene.photo,
            photo=photo,
            empty_room=empty_room,
            contents=contents,
            reference=reference,
            backends=backends,
            gallery=gallery,
        ),
    )


def _repaint(
    photo: Float[Tensor, "1 3 height width"],
    contents: Bool[Tensor, "1 height width"],
    *,
    config: StudyConfig,
    backends: Backends,
    inpainter: InpaintPipeline,
    generator: torch.Generator,
) -> tuple[Float[Tensor, "1 3 height width"], Float[Tensor, "1 3 height width"], float]:
    """Repaint the cut region until the segmenter stops calling what is in it furniture.

    Returns the stripped room, the first pass's raw render, and the share of the
    cut region still read as contents at the end. Every pass repaints a subset of
    the *first* mask, so however many run, a pixel the first segmentation left
    alone is still the photograph's.
    """
    image, generated, remaining = photo, photo, contents
    for index in range(config.strip.passes):
        inpainting = inpaint_masked_region(
            image,
            remaining,
            pipeline=inpainter,
            prompt=config.strip.prompt,
            negative_prompt=config.strip.negative_prompt,
            steps=config.strip.steps,
            fill_strength=config.strip.fill_strength,
            guidance_scale=config.strip.guidance_scale,
            generator=generator,
        )
        image = inpainting.image
        if index == 0:
            generated = inpainting.generated
        with torch.no_grad():
            remaining = contents & contents_mask(
                backends.segmenter(image), dilation_radius=config.strip.dilation_radius
            )
        if not bool(remaining.any()):
            break
    return image, generated, _share_of(remaining, contents)


def _share_of(
    part: Bool[Tensor, "1 height width"], whole: Bool[Tensor, "1 height width"]
) -> float:
    total = int(whole.sum())
    return int(part.sum()) / total if total else 0.0


def vae_roundtrip(
    vae: LatentAutoencoder,
    photo: Float[Tensor, "1 3 height width"],
    *,
    device: torch.device,
) -> Float[Tensor, "1 3 height width"]:
    """The photo encoded and decoded once, which is the floor any latent sampler pays."""
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        distribution = vae.encode(photo * 2.0 - 1.0).latent_dist
        latent = distribution.sample(generator=generator) * vae.config.scaling_factor
        return decode_latent(vae, latent)


def _outside_mask_distance(
    image: Float[Tensor, "1 3 height width"],
    photo: Float[Tensor, "1 3 height width"],
    mask: Bool[Tensor, "1 height width"],
) -> float:
    outside = ~mask.unsqueeze(1).expand_as(photo)
    if not bool(outside.any()):
        return 0.0
    return float((image - photo).abs()[outside].max())


def _write_gallery(
    scene: str,
    *,
    photo: Float[Tensor, "1 3 height width"],
    empty_room: Float[Tensor, "1 3 height width"],
    contents: Bool[Tensor, "1 height width"],
    reference: GeometricReference,
    backends: Backends,
    gallery: Path,
) -> dict[str, str]:
    """Gallery A: the photo, what was cut out of it, the empty room, and both depth maps."""
    names = {
        "photo": f"{scene}_photo.png",
        "contents_overlay": f"{scene}_contents_overlay.png",
        "empty_room": f"{scene}_empty_room.png",
        "photo_depth": f"{scene}_photo_depth.png",
        "empty_room_depth": f"{scene}_empty_room_depth.png",
    }
    write_image(photo, gallery / names["photo"])
    write_mask_overlay(photo, contents, gallery / names["contents_overlay"])
    write_image(empty_room, gallery / names["empty_room"])
    # Both maps are painted against the photograph's own near/far range, and the
    # stripped room's is fitted onto it over the shell first: unaligned, two
    # inverse-depth predictions share no frame and the pair says nothing.
    scale = DisparityScale.of(reference.disparity)
    with torch.no_grad():
        write_disparity_map(
            reference.disparity,
            scale=scale,
            mask=reference.mask,
            path=gallery / names["photo_depth"],
        )
        write_disparity_map(
            backends.depth_estimator(empty_room),
            scale=scale,
            mask=reference.mask,
            path=gallery / names["empty_room_depth"],
        )
    return names


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
