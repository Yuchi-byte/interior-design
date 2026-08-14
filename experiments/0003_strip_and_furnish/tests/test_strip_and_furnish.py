"""The framework's weightless surface: config guards, gallery helpers, reference wiring.

Everything here runs on a `tmp_path`, the shipped `conf/` and the fake stack,
so this file carries no `models` marker and costs seconds. The parts that need
weights are covered by running a variant, not by a test.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from jaxtyping import Bool, Float, Int
from strip_and_furnish.backends import Backends, build_adherence, build_backends
from strip_and_furnish.config import LATENT_ALIGNMENT, SceneConfig, StudyConfig
from strip_and_furnish.furnish import (
    build_energies,
    empty_room_reference,
    furnish_scene,
    guidance_reference,
    write_scene_gallery,
)
from strip_and_furnish.images import (
    SEGMENTATION_COLOR_KEY,
    DisparityScale,
    write_disparity_map,
    write_image,
    write_mask_overlay,
    write_segmentation_map,
)
from strip_and_furnish.strip import StripResult, strip_scene
from torch import Tensor
from torchvision.io import ImageReadMode, read_image

from roomgen_geometry.masking import architectural_mask
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.sampling.inpaint import InpaintOutput

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


@pytest.mark.parametrize(
    "variant", sorted(path.stem for path in CONF_DIR.glob("*.yaml"))
)
def test_every_shipped_variant_validates(variant: str) -> None:
    """A conf/ file that no longer matches the config is a run that dies on startup."""
    config = StudyConfig.from_file(CONF_DIR / f"{variant}.yaml")

    assert config.scenes
    assert config.seeds
    assert config.resolution % LATENT_ALIGNMENT == 0


def test_scoring_adherence_with_the_guided_checkpoint_is_refused() -> None:
    # Arm A descends this very checkpoint's cosine, so scoring adherence with it
    # would report how well the sampler hit its own target.
    with pytest.raises(ValueError, match="graded on the model it optimised"):
        StudyConfig.from_file(
            CONF_DIR / "study.yaml",
            ["adherence.checkpoint=openai/clip-vit-base-patch32"],
        )


def test_a_dotlist_override_reaches_a_nested_field() -> None:
    config = StudyConfig.from_file(
        CONF_DIR / "smoke.yaml", ["furnish.strength=12.5", "strip.dilation_radius=5"]
    )

    assert config.furnish.strength == pytest.approx(12.5)
    assert config.strip.dilation_radius == 5


def test_the_overlay_tints_the_masked_region_and_leaves_the_rest_of_the_photo(
    tmp_path: Path,
) -> None:
    photo = torch.full((1, 3, 16, 16), 0.5)
    mask = torch.zeros((1, 16, 16), dtype=torch.bool)
    mask[:, 4:8, 4:8] = True
    path = tmp_path / "overlay.png"

    write_mask_overlay(photo, mask, path)

    overlay = read_image(str(path), mode=ImageReadMode.RGB)
    assert not torch.equal(overlay[:, 4:8, 4:8], overlay[:, 12:16, 12:16])
    assert bool((overlay[:, 12:16, 12:16] == 128).all())


def test_two_depth_maps_of_the_same_room_are_painted_the_same_way(
    tmp_path: Path,
) -> None:
    # The point of fitting before painting: an affine-rescaled prediction is the
    # same geometry, and a gallery that paints it differently is reporting the
    # backend's arbitrary scale as a change in the room.
    disparity = torch.linspace(0.0, 1.0, 256).reshape(1, 16, 16)
    mask = torch.ones((1, 16, 16), dtype=torch.bool)
    scale = DisparityScale.of(disparity)
    paths = (tmp_path / "one.png", tmp_path / "two.png")

    write_disparity_map(disparity, scale=scale, mask=mask, path=paths[0])
    write_disparity_map(3.0 * disparity - 2.0, scale=scale, mask=mask, path=paths[1])

    painted = [
        read_image(str(path), mode=ImageReadMode.RGB).to(torch.int16) for path in paths
    ]
    # Within a quantisation step: the fit reproduces the map to float precision,
    # and 8-bit rounding puts a handful of pixels either side of a boundary.
    assert int((painted[0] - painted[1]).abs().max()) <= 1


def _fake_run(tmp_path: Path) -> tuple[StudyConfig, Backends, StripResult, SceneConfig]:
    """One scene stripped on the weightless stack, small and short enough for a test."""
    config = StudyConfig.from_file(
        CONF_DIR / "smoke.yaml",
        [
            "seeds=[0]",
            f"resolution={LATENT_ALIGNMENT}",
            "strip.passes=1",
            "strip.steps=2",
            "furnish.steps=4",
        ],
    )
    backends = build_backends(config.backends, cache_dir=tmp_path)
    scene = config.scenes[0]
    photo = torch.rand(
        (1, 3, config.resolution, config.resolution),
        generator=torch.Generator().manual_seed(0),
    )
    strip = strip_scene(
        scene,
        photo,
        config=config,
        backends=backends,
        inpainter=backends.build_inpainter(),
        seed=0,
        gallery=tmp_path,
    )
    return config, backends, strip, scene


def test_stage_two_guidance_consumes_the_photographs_reference(tmp_path: Path) -> None:
    """The energy stage 2 descends is built on the photo's retained reference.

    The photo-derived reference the strip froze — depth and shell mask of the
    *furnished* photograph — is exactly what reaches the depth term, never a
    reference re-estimated from the inpainted empty room.
    """
    config, backends, strip, scene = _fake_run(tmp_path)

    reference = guidance_reference(strip)

    assert reference is strip.reference
    with torch.no_grad():
        photo_disparity = backends.depth_estimator(strip.photo)
        photo_shell = architectural_mask(
            backends.segmenter(strip.photo),
            erosion_radius=config.furnish.erosion_radius,
        )
        empty_disparity = backends.depth_estimator(strip.empty_room)
    assert torch.equal(reference.disparity, photo_disparity)
    assert torch.equal(reference.mask, photo_shell)
    # The inpainted room reads differently to the depth backend, so a reference
    # silently recomputed from it could not pass the equality above.
    assert not torch.equal(reference.disparity, empty_disparity)

    energies = build_energies(
        backends.text_energy(scene.prompt),
        reference=reference,
        config=config,
        backends=backends,
        depth_weight=1.0,
    )
    assert energies.depth_only.reference is strip.reference
    composite_depth_terms = [
        term.guidance
        for term in energies.text_and_depth.terms
        if isinstance(term.guidance, GeometricGuidance)
    ]
    assert [term.reference is strip.reference for term in composite_depth_terms] == [
        True
    ]


def test_furnishing_scores_every_arm_against_both_references(tmp_path: Path) -> None:
    """Primary numbers against the photo's reference, the empty room's beside them.

    Each reference scores all three arms over its own common mask, and the row
    written to `results.json` carries both sets.
    """
    config, backends, strip, scene = _fake_run(tmp_path)
    reference = guidance_reference(strip)
    empty_reference = empty_room_reference(strip, config=config, backends=backends)
    energies = build_energies(
        backends.text_energy(scene.prompt),
        reference=reference,
        config=config,
        backends=backends,
        depth_weight=1.0,
    )

    results = furnish_scene(
        scene,
        strip,
        config=config,
        backends=backends,
        adherence=build_adherence(
            config.adherence, device=backends.device, cache_dir=None
        ),
        reference=reference,
        empty_reference=empty_reference,
        energies=energies,
        gallery=tmp_path,
    )

    (result,) = results
    row = result.as_row()
    for key in (
        "depth_error",
        "depth_error_empty_room",
        "common_coverage",
        "common_coverage_empty_room",
        "reconstruction_depth_error",
        "reconstruction_depth_error_empty_room",
    ):
        assert key in row
    for errors in (result.depth_errors, result.empty_room_depth_errors):
        assert set(errors) == {"text_only", "text_and_depth", "control"}
        assert all(value >= 0.0 for value in errors.values())
    # Two references, two masks: the scores are read over different pixel sets,
    # whose coverages are each against their own shell.
    assert 0.0 < result.common_coverage <= 1.0
    assert 0.0 < result.empty_room_common_coverage <= 1.0


_FLOOR = ADE20K_LABELS.index("floor")
_SOFA = ADE20K_LABELS.index("sofa")
_PERSON = ADE20K_LABELS.index("person")


@dataclass(frozen=True)
class _ColourCodedSegmenter:
    """A `Segmenter` reading class off pixel colour, so a repaint changes what it says.

    Green is a sofa (contents), strong red is a person — a class in *neither*
    list, standing in for the radiator/vacuum fills the contents-based re-check
    was blind to — and everything else is floor (shell).
    """

    def __call__(
        self, images: Float[Tensor, "batch 3 height width"]
    ) -> Int[Tensor, "batch height width"]:
        labels = torch.full(
            (images.shape[0], images.shape[2], images.shape[3]),
            _FLOOR,
            dtype=torch.int64,
            device=images.device,
        )
        labels[images[:, 1] > 0.9] = _SOFA
        labels[images[:, 0] > 0.9] = _PERSON
        return labels


@dataclass
class _Painted:
    images: Tensor


@dataclass
class _ScriptedInpainter:
    """An `InpaintPipeline` that paints the hole a scripted colour and records each mask.

    Args:
        red_fills: red-channel value painted into the mask on each call; the
            last entry repeats if more passes run than the script names.
    """

    red_fills: tuple[float, ...]
    masks: list[Tensor] = field(default_factory=list)

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
    ) -> InpaintOutput:
        self.masks.append(mask_image.detach().clone())
        red = self.red_fills[min(len(self.masks) - 1, len(self.red_fills) - 1)]
        colour = torch.tensor([red, 0.0, 0.2], dtype=image.dtype).view(1, 3, 1, 1)
        painted = mask_image > 0.5
        return _Painted(images=torch.where(painted, colour.expand_as(image), image))


def _colour_coded_strip(
    tmp_path: Path, red_fills: tuple[float, ...]
) -> tuple[StripResult, _ScriptedInpainter, Bool[Tensor, "1 height width"]]:
    """One strip over the colour-coded segmenter and the scripted inpainter."""
    config = StudyConfig.from_file(
        CONF_DIR / "smoke.yaml",
        ["seeds=[0]", f"resolution={LATENT_ALIGNMENT}", "strip.passes=3"],
    )
    backends = dataclasses.replace(
        build_backends(config.backends, cache_dir=tmp_path),
        segmenter=_ColourCodedSegmenter(),
    )
    photo = torch.full((1, 3, LATENT_ALIGNMENT, LATENT_ALIGNMENT), 0.3)
    photo[:, 1, 20:36, 20:36] = 0.95  # the sofa the strip is asked to remove
    inpainter = _ScriptedInpainter(red_fills=red_fills)
    strip = strip_scene(
        config.scenes[0],
        photo,
        config=config,
        backends=backends,
        inpainter=inpainter,
        seed=0,
        gallery=tmp_path,
    )
    return strip, inpainter, strip.contents


def test_the_acceptance_test_repaints_any_non_shell_fill_not_only_furniture(
    tmp_path: Path,
) -> None:
    """A fill the segmenter calls neither shell nor contents is repainted anyway.

    The first pass paints the hole a colour that segments as *person* — in
    neither class list, exactly like the observed radiator and vacuum-shaped
    hallucinations. The old contents-based re-check accepted that fill and
    stopped; the inverted acceptance repaints it, and only stops once the hole
    reads as shell.
    """
    strip, inpainter, contents = _colour_coded_strip(tmp_path, red_fills=(1.0, 0.0))

    # Pass 1's fill was not shell, so a second pass ran; its fill was floor, so
    # a third did not.
    assert len(inpainter.masks) == 2
    repainted = inpainter.masks[1][:, 0] > 0.5
    assert bool(repainted.any())
    # The re-check repaints inside the first mask only: the invariant that the
    # untouched photograph stays untouched, whatever the passes do.
    assert not bool((repainted & ~contents).any())
    assert strip.numbers.unmasked_max_absolute_difference == 0.0
    assert strip.numbers.hole_not_shell_fraction == 0.0
    assert strip.numbers.residual_contents_fraction == 0.0


def test_hole_not_shell_reports_what_the_furniture_metric_is_blind_to(
    tmp_path: Path,
) -> None:
    """A hole that never becomes shell scores 1.0 not-shell and 0.0 furniture.

    Every pass paints the hole the person-coloured fill: the furniture-only
    residual reads 0.0 — the loophole — while the categorical metric reports
    the entire hole, which is the number the requirement is checked by.
    """
    strip, inpainter, _ = _colour_coded_strip(tmp_path, red_fills=(1.0,))

    assert len(inpainter.masks) == 3  # never accepted, so every pass ran
    assert strip.numbers.hole_not_shell_fraction == 1.0
    assert strip.numbers.residual_contents_fraction == 0.0


def test_the_segmentation_map_paints_classes_under_the_recorded_key(
    tmp_path: Path,
) -> None:
    segmentation = torch.tensor(
        [[[0, 3], [23, 2]]], dtype=torch.int64
    )  # wall, floor / sofa (contents), sky (neither list)
    path = tmp_path / "segmentation.png"

    write_segmentation_map(segmentation, path)

    painted = read_image(str(path), mode=ImageReadMode.RGB)
    assert tuple(painted[:, 0, 0].tolist()) == SEGMENTATION_COLOR_KEY["wall"]
    assert tuple(painted[:, 0, 1].tolist()) == SEGMENTATION_COLOR_KEY["floor"]
    assert tuple(painted[:, 1, 0].tolist()) == SEGMENTATION_COLOR_KEY["contents"]
    assert tuple(painted[:, 1, 1].tolist()) == SEGMENTATION_COLOR_KEY["other"]


def test_gallery_a_records_a_segmentation_map_for_photo_and_empty_room(
    tmp_path: Path,
) -> None:
    _, _, strip, _ = _fake_run(tmp_path)

    row = strip.as_row()
    assert "hole_not_shell_fraction" in row
    images = strip.images
    for key in ("photo_segmentation", "empty_room_segmentation"):
        assert (tmp_path / images[key]).exists()


def test_gallery_b_renders_the_reward_area_as_overlays(tmp_path: Path) -> None:
    """The guidance mask, the common evaluation masks and each arm's scorable slice."""
    config, backends, strip, scene = _fake_run(tmp_path)
    reference = guidance_reference(strip)

    scene_images = write_scene_gallery(strip, reference=reference, gallery=tmp_path)
    for key in (
        "empty_room",
        "guidance_mask_on_photo",
        "guidance_mask_on_empty_room",
    ):
        assert (tmp_path / scene_images[key]).exists()

    (result,) = furnish_scene(
        scene,
        strip,
        config=config,
        backends=backends,
        adherence=build_adherence(
            config.adherence, device=backends.device, cache_dir=None
        ),
        reference=reference,
        empty_reference=empty_room_reference(strip, config=config, backends=backends),
        energies=build_energies(
            backends.text_energy(scene.prompt),
            reference=reference,
            config=config,
            backends=backends,
            depth_weight=1.0,
        ),
        gallery=tmp_path,
    )
    for key in (
        "common_mask_on_photo",
        "empty_room_common_mask_on_empty_room",
        "text_only_scorable",
        "text_and_depth_scorable",
        "control_scorable",
    ):
        assert (tmp_path / result.images[key]).exists()


def test_a_render_survives_the_round_trip_through_a_png(tmp_path: Path) -> None:
    render = torch.rand((1, 3, 16, 16), generator=torch.Generator().manual_seed(0))
    path = tmp_path / "render.png"

    write_image(render, path)

    decoded = read_image(str(path), mode=ImageReadMode.RGB).to(torch.float32) / 255.0
    torch.testing.assert_close(decoded.unsqueeze(0), render, atol=1 / 255, rtol=0)
