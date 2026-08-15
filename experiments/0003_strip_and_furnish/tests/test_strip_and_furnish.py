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
from strip_and_furnish.config import (
    LATENT_ALIGNMENT,
    FmrgReduxConfig,
    FmrgStudyConfig,
    SceneConfig,
    StudyConfig,
    load_study_config,
)
from strip_and_furnish.fmrg import (
    build_flux_stack,
    furnish_arms,
    furnish_reference,
)
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
from strip_and_furnish.redux import (
    REDUX_ARMS,
    build_image_energy_bank,
    instruction_prompt,
)
from strip_and_furnish.strip import StripResult, strip_scene
from torch import Tensor
from torchvision.io import ImageReadMode, read_image

from roomgen_geometry.masking import architectural_mask
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.sampling.inpaint import InpaintOutput
from roomgen_guide.settings import RoomgenSettings

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


@pytest.mark.parametrize(
    "variant", sorted(path.stem for path in CONF_DIR.glob("*.yaml"))
)
def test_every_shipped_variant_validates(variant: str) -> None:
    """A conf/ file that no longer matches the config is a run that dies on startup."""
    config = load_study_config(CONF_DIR / f"{variant}.yaml")

    assert config.scenes
    assert config.seeds
    assert config.resolution % LATENT_ALIGNMENT == 0
    # The `study: fmrg` key and the config shape must agree, since run.py
    # dispatches on the key alone.
    assert isinstance(config, FmrgStudyConfig) == variant.startswith("fmrg")


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
        strip=config.strip,
        erosion_radius=config.furnish.erosion_radius,
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
_RAILING = ADE20K_LABELS.index("railing")


@dataclass(frozen=True)
class _ColourCodedSegmenter:
    """A `Segmenter` reading class off pixel colour, so a repaint changes what it says.

    Green is a sofa (removable contents, which the blocklist acceptance
    repaints), strong red is a person — a class in *neither* list, standing in
    for the vacuum-shaped fills ADE20K cannot name — strong blue is a railing —
    shell, standing in for the ledge/machine fills EoMT reads as structure —
    and everything else is floor. The blocklist accepts all but the sofa; the
    not-shell diagnostic is what still reports the person-shaped fill.
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
        labels[images[:, 2] > 0.9] = _RAILING
        return labels


@dataclass
class _Painted:
    images: Tensor


@dataclass
class _ScriptedInpainter:
    """An `InpaintPipeline` that paints the hole a scripted colour and records each mask.

    Args:
        fills: RGB colour painted into the mask on each call; the last entry
            repeats if more passes run than the script names.
    """

    fills: tuple[tuple[float, float, float], ...]
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
        fill = self.fills[min(len(self.masks) - 1, len(self.fills) - 1)]
        colour = torch.tensor(fill, dtype=image.dtype).view(1, 3, 1, 1)
        painted = mask_image > 0.5
        return _Painted(images=torch.where(painted, colour.expand_as(image), image))


def _colour_coded_strip(
    tmp_path: Path, fills: tuple[tuple[float, float, float], ...]
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
    inpainter = _ScriptedInpainter(fills=fills)
    strip = strip_scene(
        config.scenes[0],
        photo,
        strip=config.strip,
        erosion_radius=config.furnish.erosion_radius,
        backends=backends,
        inpainter=inpainter,
        seed=0,
        gallery=tmp_path,
    )
    return strip, inpainter, strip.contents


def test_the_acceptance_repaints_a_fill_the_segmenter_still_calls_removable(
    tmp_path: Path,
) -> None:
    """A fill that segments as furniture is refused and repainted until clean.

    The first pass paints the hole sofa-green — recognisable contents — so the
    blocklist forces a second pass; that one paints floor, which is accepted,
    and the loop stops early with the budget unspent.
    """
    strip, inpainter, contents = _colour_coded_strip(
        tmp_path, fills=((0.0, 1.0, 0.2), (0.0, 0.0, 0.2))
    )

    # Pass 1's fill was still removable, so a second pass ran; its fill was
    # floor, so a third did not.
    assert len(inpainter.masks) == 2
    repainted = inpainter.masks[1][:, 0] > 0.5
    assert bool(repainted.any())
    # The re-check repaints inside the first mask only: the invariant that the
    # untouched photograph stays untouched, whatever the passes do.
    assert not bool((repainted & ~contents).any())
    assert strip.numbers.unmasked_max_absolute_difference == 0.0
    assert strip.numbers.hole_not_shell_fraction == 0.0
    assert strip.numbers.residual_contents_fraction == 0.0
    # The convergence trace the pass budget is judged by: refused, then clean.
    assert len(strip.numbers.pass_residual_fractions) == 2
    assert strip.numbers.pass_residual_fractions[0] > 0.0
    assert strip.numbers.pass_residual_fractions[-1] == 0.0


def test_a_hole_that_keeps_refilling_with_furniture_uses_every_pass(
    tmp_path: Path,
) -> None:
    """A hole that segments as furniture after every pass exhausts the budget.

    The residual trace stays at 1.0 throughout, and the final number reports
    the whole hole as recognisable contents — the failure mode the pass budget
    and `residual_contents_fraction` exist to expose.
    """
    strip, inpainter, _ = _colour_coded_strip(tmp_path, fills=((0.0, 1.0, 0.2),))

    assert len(inpainter.masks) == 3  # never accepted, so every pass ran
    assert strip.numbers.residual_contents_fraction == 1.0
    assert all(fraction == 1.0 for fraction in strip.numbers.pass_residual_fractions)


def test_a_fill_in_neither_class_list_is_accepted_the_blocklist_honest_limit(
    tmp_path: Path,
) -> None:
    """A fill the segmenter calls neither shell nor contents is accepted as-is.

    The pass paints the hole a colour that segments as *person* — in neither
    class list, exactly like the vacuum-shaped hallucinations ADE20K has no
    name for. The blocklist accepts it; `hole_not_shell_fraction` is the
    diagnostic that still reports it.
    """
    strip, inpainter, _ = _colour_coded_strip(tmp_path, fills=((1.0, 0.0, 0.2),))

    assert len(inpainter.masks) == 1  # accepted immediately, no repaint
    assert strip.numbers.pass_residual_fractions == (0.0,)
    assert strip.numbers.residual_contents_fraction == 0.0
    assert strip.numbers.hole_not_shell_fraction == 1.0


def test_a_shell_labelled_fill_is_accepted_and_invisible_to_both_numbers(
    tmp_path: Path,
) -> None:
    """A fill that segments as railing — shell — sails through the blocklist.

    A hallucinated ledge EoMT labels as railing/column/stairs is not
    removable, so the acceptance stops, and even the not-shell diagnostic reads
    it as a success. Only the negative prompt argues with such a fill.
    """
    strip, inpainter, _ = _colour_coded_strip(tmp_path, fills=((0.2, 0.0, 1.0),))

    assert len(inpainter.masks) == 1
    assert strip.numbers.pass_residual_fractions == (0.0,)
    assert strip.numbers.residual_contents_fraction == 0.0
    assert strip.numbers.hole_not_shell_fraction == 0.0


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


def _fake_fmrg_run(
    tmp_path: Path,
) -> tuple[FmrgStudyConfig, Backends, StripResult, SceneConfig]:
    """One scene stripped on the weightless stack, under the FMRG study shape."""
    config = load_study_config(
        CONF_DIR / "fmrg_smoke.yaml",
        [
            "seeds=[0]",
            f"resolution={LATENT_ALIGNMENT}",
            "strip.passes=1",
            "strip.steps=2",
            "furnish.grid_steps=4",
            "furnish.early_stop=3",
        ],
    )
    assert isinstance(config, FmrgStudyConfig)
    backends = build_backends(config.backends, cache_dir=tmp_path)
    scene = config.scenes[0]
    photo = torch.rand(
        (1, 3, config.resolution, config.resolution),
        generator=torch.Generator().manual_seed(0),
    )
    strip = strip_scene(
        scene,
        photo,
        strip=config.strip,
        erosion_radius=config.furnish.erosion_radius,
        backends=backends,
        inpainter=backends.build_inpainter(),
        seed=0,
        gallery=tmp_path,
    )
    return config, backends, strip, scene


def test_fmrg_guidance_descends_the_empty_rooms_own_reference(tmp_path: Path) -> None:
    """Part 2's wiring: what the FMRG furnish descends is the *stripped room's* geometry.

    The reference's disparity and mask are exactly a fresh depth read and shell
    segmentation of the empty room — never the photograph's retained reference,
    which is what `study.py`'s protocol steers by.
    """
    config, backends, strip, _ = _fake_fmrg_run(tmp_path)

    reference = furnish_reference(
        strip,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        erosion_radius=config.furnish.erosion_radius,
    )

    empty_room = strip.empty_room.to(torch.float32)
    with torch.no_grad():
        empty_disparity = backends.depth_estimator(empty_room)
        empty_shell = architectural_mask(
            backends.segmenter(empty_room),
            erosion_radius=config.furnish.erosion_radius,
        )
    assert torch.equal(reference.disparity, empty_disparity)
    assert torch.equal(reference.mask, empty_shell)
    # The photo's retained reference reads differently, so steering by it could
    # not pass the equalities above.
    assert not torch.equal(reference.disparity, strip.reference.disparity)


def test_fmrg_arms_are_control_guided_and_labelled_probes(tmp_path: Path) -> None:
    config, _, _, _ = _fake_fmrg_run(tmp_path)

    specs = furnish_arms(config)

    assert [spec.arm for spec in specs] == ["control", "guided", "guided_s0.75"]
    assert specs[0].eta == 0.0
    assert specs[1].eta == config.furnish.eta
    assert [spec.probe for spec in specs] == [False, False, True]
    assert specs[2].start_sigma == pytest.approx(0.75)


def test_fmrg_scene_scores_every_arm_on_the_intersected_mask(tmp_path: Path) -> None:
    """Eval mirrors guidance: empty-room shell ∩ every arm's own prediction shell.

    The common mask can only shrink the empty shell, each arm's own scorable
    mask sits between the two, and the row records the frame fractions the
    enlarged-mask claim is judged by.
    """
    from strip_and_furnish.fmrg import _run_scene, _scored  # noqa: PLC0415

    config, backends, strip, scene = _fake_fmrg_run(tmp_path)
    reference = furnish_reference(
        strip,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        erosion_radius=config.furnish.erosion_radius,
    )
    stacks = build_flux_stack(
        config,
        prompts=[scene.prompt],
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        device=backends.device,
        settings=RoomgenSettings(wdir=tmp_path, hf_home=tmp_path),
    )
    shell_fractions = {
        "photo": float(strip.reference.mask.to(torch.float32).mean()),
        "empty_room": float(reference.mask.to(torch.float32).mean()),
    }

    (pending,) = _run_scene(
        scene,
        strip,
        config=config,
        stacks=stacks,
        reference=reference,
        shell_fractions=shell_fractions,
        specs=furnish_arms(config),
        gallery=tmp_path,
        warm_up=False,
    )
    result = _scored(
        pending,
        adherence=build_adherence(
            config.adherence, device=backends.device, cache_dir=None
        ),
        device=backends.device,
    )

    row = result.as_row()
    for key in (
        "common_coverage",
        "common_frame_fraction",
        "empty_shell_frame_fraction",
        "photo_shell_frame_fraction",
        "reconstruction_depth_error",
    ):
        assert key in row
    # Intersection can only shrink: common ⊆ each arm's scorable ⊆ empty shell.
    assert 0.0 < result.common_coverage <= 1.0
    for cell in result.arms:
        assert result.common_coverage <= cell.shell_coverage + 1e-9
        assert cell.depth_error >= 0.0
        assert cell.adherence == cell.adherence  # scored, not the NaN sentinel
        assert (tmp_path / cell.render).exists()
        assert (tmp_path / cell.depth_map).exists()
        assert (tmp_path / cell.residual_map).exists()
        assert (tmp_path / cell.scorable_overlay).exists()
    assert result.common_frame_fraction <= result.empty_shell_frame_fraction + 1e-9
    assert (tmp_path / result.empty_depth_map).exists()
    assert (tmp_path / result.common_mask_overlay).exists()


def test_redux_arms_cover_the_owners_grid() -> None:
    assert [spec.arm for spec in REDUX_ARMS] == [
        "control",
        "redux",
        "redux_depth",
        "depth",
        "img",
        "redux_img",
        "redux_img_depth",
    ]
    assert [(spec.redux, spec.depth, spec.image) for spec in REDUX_ARMS] == [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (False, True, False),
        (False, False, True),
        (True, False, True),
        (True, True, True),
    ]
    # Redux is conditioning, not steering: an arm with no reward is unguided.
    assert [spec.guided for spec in REDUX_ARMS] == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_the_instruction_prompt_wraps_the_style_descriptors_verbatim() -> None:
    scene = SceneConfig(photo="scandi_living_room", prompt="a cosy den, warm wood")

    assert instruction_prompt(scene) == (
        "renovate this room into a cosy den, warm wood, keep the windows, "
        "doors, walls, floor and ceiling's positions"
    )


def test_a_judge_sharing_the_steered_dinov2_is_refused() -> None:
    with pytest.raises(ValueError, match="image reward descends"):
        FmrgReduxConfig.from_file(
            CONF_DIR / "fmrg_redux.yaml",
            ["image_similarity_checkpoint=facebook/dinov2-base"],
        )


def _fake_redux_run(
    tmp_path: Path,
) -> tuple[FmrgReduxConfig, Backends, StripResult, SceneConfig]:
    """One scene stripped on the weightless stack, under the Redux-grid shape."""
    config = load_study_config(
        CONF_DIR / "fmrg_redux_smoke.yaml",
        [
            "seeds=[0]",
            f"resolution={LATENT_ALIGNMENT}",
            "strip.passes=1",
            "strip.steps=2",
            "furnish.grid_steps=4",
            "furnish.early_stop=3",
        ],
    )
    assert isinstance(config, FmrgReduxConfig)
    backends = build_backends(config.backends, cache_dir=tmp_path)
    scene = config.scenes[0]
    photo = torch.rand(
        (1, 3, config.resolution, config.resolution),
        generator=torch.Generator().manual_seed(0),
    )
    strip = strip_scene(
        scene,
        photo,
        strip=config.strip,
        erosion_radius=config.furnish.erosion_radius,
        backends=backends,
        inpainter=backends.build_inpainter(),
        seed=0,
        gallery=tmp_path,
    )
    return config, backends, strip, scene


def test_redux_scene_wires_every_arm_and_scores_the_intersected_mask(
    tmp_path: Path,
) -> None:
    """The seven-arm grid: rewards wired per spec, one common mask, judges deferred.

    The image arms descend the shell-masked patch energy, the composite carries
    the measured weight, and every arm leaves a render, a segmentation map, a
    masked depth map, a signed residual map and its scorable overlay.
    """
    from strip_and_furnish.redux import (  # noqa: PLC0415
        _arm_reward,
        _run_scene,
        _score_pending,
    )

    from roomgen_guide.guidance.composite import WeightedGuidance  # noqa: PLC0415
    from roomgen_guide.guidance.image_reference import (  # noqa: PLC0415
        MaskedPatchReferenceGuidance,
    )

    config, backends, strip, scene = _fake_redux_run(tmp_path)
    settings = RoomgenSettings(wdir=tmp_path, hf_home=tmp_path)
    reference = furnish_reference(
        strip,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        erosion_radius=config.furnish.erosion_radius,
    )
    stacks = build_flux_stack(
        config,
        prompts=[scene.prompt],
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        device=backends.device,
        settings=settings,
    )
    bank = build_image_energy_bank(config, device=backends.device, settings=settings)

    (pending,) = _run_scene(
        scene,
        strip,
        config=config,
        stacks=stacks,
        bank=bank,
        reference=reference,
        shell_fractions={
            "photo": float(strip.reference.mask.to(torch.float32).mean()),
            "empty_room": float(reference.mask.to(torch.float32).mean()),
        },
        gallery=tmp_path,
        warm_up=False,
    )
    (result,) = _score_pending(
        [pending], config=config, device=backends.device, settings=settings
    )

    assert [cell.arm for cell in result.arms] == [spec.arm for spec in REDUX_ARMS]
    assert result.sizing is not None
    triple = result.arm("redux_img_depth")
    assert triple.depth_weight == pytest.approx(result.sizing.depth_weight)
    for cell in result.arms:
        assert result.common_coverage <= cell.shell_coverage + 1e-9
        assert cell.depth_error >= 0.0
        assert cell.adherence == cell.adherence  # scored, not the NaN sentinel
        assert cell.preference == cell.preference
        assert cell.similarity_empty_room is None  # smoke skips the judge
        for name in (
            cell.render,
            cell.segmentation_map,
            cell.depth_map,
            cell.residual_map,
            cell.scorable_overlay,
        ):
            assert (tmp_path / name).exists()
    assert (tmp_path / result.empty_segmentation_map).exists()
    row = result.as_row()
    for key in (
        "style_prompt",
        "instruction_prompt",
        "redux_scale",
        "sizing",
        "common_frame_fraction",
    ):
        assert key in row

    # The rewards behind the arms: image-only is the masked patch energy, the
    # triple is a weighted pair, and everything else defers to the walk's own
    # depth default.
    image_only = _arm_reward(
        REDUX_ARMS[4],
        empty_room=strip.empty_room.to(torch.float32),
        config=config,
        stacks=stacks,
        bank=bank,
        reference=reference,
        composite_weight=None,
    )
    assert isinstance(image_only, MaskedPatchReferenceGuidance)
    triple_reward = _arm_reward(
        REDUX_ARMS[6],
        empty_room=strip.empty_room.to(torch.float32),
        config=config,
        stacks=stacks,
        bank=bank,
        reference=reference,
        composite_weight=2.5,
    )
    assert isinstance(triple_reward, WeightedGuidance)
    assert [term.weight for term in triple_reward.terms] == [1.0, 2.5]
    for spec in REDUX_ARMS[:4]:
        assert (
            _arm_reward(
                spec,
                empty_room=strip.empty_room.to(torch.float32),
                config=config,
                stacks=stacks,
                bank=bank,
                reference=reference,
                composite_weight=None,
            )
            is None
        )
