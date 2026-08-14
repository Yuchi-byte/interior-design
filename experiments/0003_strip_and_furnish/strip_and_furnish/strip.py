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
from jaxtyping import Bool, Float, Int
from torch import Tensor

from roomgen_geometry.masking import (
    contents_mask,
    erode_boundaries,
    fill_mask,
    scorable_mask,
    structural_mask,
)
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.eval import depth_error, shell_coverage
from roomgen_guide.sampling.guided import decode_latent
from roomgen_guide.sampling.inpaint import InpaintPipeline, inpaint_masked_region
from roomgen_guide.sampling.pipeline import LatentAutoencoder
from strip_and_furnish.backends import Backends
from strip_and_furnish.config import SceneConfig, StripConfig
from strip_and_furnish.images import (
    DisparityScale,
    write_disparity_map,
    write_image,
    write_mask_overlay,
    write_segmentation_map,
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
        hole_not_fill_fraction: the requirement's own number — the share of the
            cut region the final segmentation does NOT read as an acceptable
            fill (`FILL_CLASSES`: wall, floor, ceiling, door, windowpane, and
            nothing else). A radiator, a vacuum-shaped blob, a table and an
            invented staircase all land here alike; only a fill the segmenter
            *mislabels* as one of the five escapes it, which no label-reading
            metric can catch.
        hole_not_shell_fraction: the same share against the wider
            `STRUCTURAL_CLASSES` — kept beside the whitelist number so the gap
            between them reports how much of the hole passed only as
            railing/column/stairs-shaped shell.
        residual_contents_fraction: the subset the segmenter affirmatively
            calls furniture (`CONTENTS_CLASSES`, undilated) — the "recognisable
            furniture" slice of the categorical numbers above.
        pass_residual_fractions: after each inpainting pass, the share of the
            cut region the acceptance still refused — not-whitelisted, grown by
            the dilation radius — which is exactly what the next pass repaints.
            How fast it falls is whether the pass budget is enough.
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
    hole_not_fill_fraction: float
    hole_not_shell_fraction: float
    residual_contents_fraction: float
    pass_residual_fractions: tuple[float, ...]
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
            "hole_not_fill_fraction": self.numbers.hole_not_fill_fraction,
            "hole_not_shell_fraction": self.numbers.hole_not_shell_fraction,
            "residual_contents_fraction": self.numbers.residual_contents_fraction,
            "pass_residual_fractions": list(self.numbers.pass_residual_fractions),
            "seconds": self.numbers.seconds,
            "images": self.images,
        }


def strip_scene(
    scene: SceneConfig,
    photo: Float[Tensor, "1 3 height width"],
    *,
    strip: StripConfig,
    erosion_radius: int,
    backends: Backends,
    inpainter: InpaintPipeline,
    seed: int,
    gallery: Path,
) -> StripResult:
    """Cut the furniture out of `photo`, paint the holes bare, and measure what moved.

    Args:
        strip: stage 1's own knobs; the stage reads nothing else of a study's
            config, so any furnish backend can sit downstream of it.
        erosion_radius: for the photo's shell reference and the visibility
            re-check, as the downstream reward masks take it.
    """
    reference = GeometricReference.from_photo(
        photo,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        erosion_radius=erosion_radius,
    )
    with torch.no_grad():
        contents = contents_mask(
            backends.segmenter(photo), dilation_radius=strip.dilation_radius
        )

    generator = torch.Generator(device=backends.device).manual_seed(seed)
    _synchronize(backends.device)
    started = time.perf_counter()
    empty_room, first_generated, final_segmentation, pass_residuals = _repaint(
        photo,
        contents,
        strip=strip,
        backends=backends,
        inpainter=inpainter,
        generator=generator,
    )
    _synchronize(backends.device)
    seconds = time.perf_counter() - started
    # The requirement's own number, undilated: of the pixels the strip cut, how
    # many does the final segmentation read as anything but an acceptable fill —
    # with the wider not-shell share and the recognisable-furniture slice beside
    # it, so what the whitelist alone refused is a gap and not a guess.
    hole_not_fill = contents & ~fill_mask(final_segmentation, erosion_radius=0)
    hole_not_shell = contents & ~structural_mask(final_segmentation, erosion_radius=0)
    residual_contents = contents & contents_mask(final_segmentation, dilation_radius=0)

    # The shell the mask never reached: those pixels are the photograph's own,
    # so any depth error over them is the network reading a room that has lost
    # its furniture, not a fill that moved a wall.
    unrepainted = reference.mask & ~contents
    with torch.no_grad():
        visible = reference.mask & scorable_mask(
            backends.segmenter(empty_room),
            erosion_radius=erosion_radius,
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
        hole_not_fill_fraction=_share_of(hole_not_fill, contents),
        hole_not_shell_fraction=_share_of(hole_not_shell, contents),
        residual_contents_fraction=_share_of(residual_contents, contents),
        pass_residual_fractions=pass_residuals,
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
            empty_room_segmentation=final_segmentation,
            reference=reference,
            backends=backends,
            gallery=gallery,
        ),
    )


def _repaint(
    photo: Float[Tensor, "1 3 height width"],
    contents: Bool[Tensor, "1 height width"],
    *,
    strip: StripConfig,
    backends: Backends,
    inpainter: InpaintPipeline,
    generator: torch.Generator,
) -> tuple[
    Float[Tensor, "1 3 height width"],
    Float[Tensor, "1 3 height width"],
    Int[Tensor, "1 height width"],
    tuple[float, ...],
]:
    """Repaint the cut region until everything in it is an acceptable fill, and nothing else.

    The acceptance test is inverted on purpose: a pass does not ask "is there
    still furniture here" but "is every pixel here an acceptable fill yet" —
    wall, floor, ceiling, door or windowpane (`FILL_CLASSES`), and *only* those
    five, by owner decision. A fill only had to dodge the named contents list to
    survive the old test; a radiator (listed or not) and a vacuum-shaped object
    ADE20K has no name for are non-shell either way, so they are repainted
    categorically — and shell that is not a bare surface (a hallucinated ledge
    or appliance the segmenter reads as railing, column or stairs) is refused
    with them rather than admitted as room structure the hole never had. The
    honest limit stays: the check reads the segmenter's labels, so a fill it
    MISlabels as wall passes, and only the negative prompt argues with that.
    Accepted fill must also clear the dilation radius of any refused pixel, for
    the same reason the contents mask was dilated: the fringe of an object is
    the evidence it grows back from.

    Returns the stripped room, the first pass's raw render, the final image's
    segmentation — from which the caller reads what is left in the hole — and
    each pass's refused share of the hole, which is how fast the acceptance
    converged. Every pass repaints a subset of the *first* mask, so however many
    run, a pixel the first segmentation left alone is still the photograph's.
    """
    image, generated, remaining = photo, photo, contents
    residuals: list[float] = []
    with torch.no_grad():
        segmentation = backends.segmenter(image)
    for index in range(strip.passes):
        inpainting = inpaint_masked_region(
            image,
            remaining,
            pipeline=inpainter,
            prompt=strip.prompt,
            negative_prompt=strip.negative_prompt,
            steps=strip.steps,
            fill_strength=strip.fill_strength,
            guidance_scale=strip.guidance_scale,
            generator=generator,
        )
        image = inpainting.image
        if index == 0:
            generated = inpainting.generated
        with torch.no_grad():
            segmentation = backends.segmenter(image)
        # `erode_boundaries` keeps only whitelisted pixels farther than the
        # radius from any refused one, so its complement is exactly the refused
        # region grown by the radius — the inverted, dilated acceptance test.
        confidently_fill = erode_boundaries(
            fill_mask(segmentation, erosion_radius=0),
            radius=strip.dilation_radius,
        )
        remaining = contents & ~confidently_fill
        residuals.append(_share_of(remaining, contents))
        if not bool(remaining.any()):
            break
    return image, generated, segmentation, tuple(residuals)


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
    empty_room_segmentation: Int[Tensor, "1 height width"],
    reference: GeometricReference,
    backends: Backends,
    gallery: Path,
) -> dict[str, str]:
    """Gallery A: photo and empty room, what was cut, both segmentations, both depth maps."""
    names = {
        "photo": f"{scene}_photo.png",
        "contents_overlay": f"{scene}_contents_overlay.png",
        "empty_room": f"{scene}_empty_room.png",
        "photo_segmentation": f"{scene}_photo_segmentation.png",
        "empty_room_segmentation": f"{scene}_empty_room_segmentation.png",
        "photo_depth": f"{scene}_photo_depth.png",
        "empty_room_depth": f"{scene}_empty_room_depth.png",
    }
    write_image(photo, gallery / names["photo"])
    write_mask_overlay(photo, contents, gallery / names["contents_overlay"])
    write_image(empty_room, gallery / names["empty_room"])
    # Class-coloured maps under one key (`SEGMENTATION_COLOR_KEY`, recorded in
    # results.json): the photo's segmentation beside the final empty room's, so
    # "did the fill become shell" is visible and not only a fraction.
    with torch.no_grad():
        write_segmentation_map(
            backends.segmenter(photo), gallery / names["photo_segmentation"]
        )
    write_segmentation_map(
        empty_room_segmentation, gallery / names["empty_room_segmentation"]
    )
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
