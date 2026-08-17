"""The framework's weightless surface: the config bridge and the gallery helpers.

Everything here runs on a `tmp_path` and the shipped `conf/`, so this file
carries no `models` marker and costs milliseconds. The parts that need weights
are covered by running a variant, not by a test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from framework.alignment_study import CONTROL_ARM, arm_specs
from framework.config import (
    FLUX_LATENT_ALIGNMENT,
    AlignmentStudyConfig,
    BenchmarkConfig,
    FmrgAlignmentConfig,
    FmrgConfig,
    FmrgContextConfig,
    FmrgEtaConfig,
    FmrgImageConfig,
    FmrgReduxConfig,
    FramingConfig,
    KontextConfig,
    load_study_config,
)
from framework.context_study import BASELINE_ARM, context_arms
from framework.depth_maps import DisparityRange, write_disparity_map
from framework.eta_study import BASELINE_ARM as ETA_BASELINE_ARM
from framework.eta_study import eta_arms
from framework.framing_gallery import FRAMING_GALLERY_FILENAME, write_framing_gallery
from framework.gallery import GALLERY_FILENAME, write_gallery
from framework.preference import ContrastPreference
from framework.redux_study import BASELINE_ARM as REDUX_BASELINE_ARM
from framework.redux_study import redux_arms, redux_prompt_key

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


@pytest.mark.parametrize(
    "variant", sorted(path.stem for path in CONF_DIR.glob("*.yaml"))
)
def test_every_shipped_variant_validates(variant: str) -> None:
    """A conf/ file that no longer matches its config is a run that dies on startup."""
    config = load_study_config(CONF_DIR / f"{variant}.yaml")

    assert config.scenes
    assert config.seeds
    # Every stack's latent grid divides the SD one's, so the loosest alignment
    # any variant may claim is FLUX's; each config enforces its own on top.
    assert config.resolution % FLUX_LATENT_ALIGNMENT == 0
    assert isinstance(
        config,
        BenchmarkConfig
        | FmrgConfig
        | FmrgContextConfig
        | FmrgReduxConfig
        | FmrgEtaConfig
        | FmrgAlignmentConfig
        | FmrgImageConfig
        | KontextConfig
        | FramingConfig
        | AlignmentStudyConfig,
    )


def test_framing_scenes_vary_framing_alone() -> None:
    """The prompt axis is the framing: the style text is inside the renovation prompt verbatim."""
    config = load_study_config(CONF_DIR / "framing.yaml")

    assert isinstance(config, FramingConfig)
    for scene in config.scenes:
        assert scene.style_prompt != scene.renovation_prompt
        assert scene.style_prompt in scene.renovation_prompt
        # One neutral text judges both framings, and it carries the style
        # content rather than either arm's phrasing advantage.
        assert scene.eval_prompt == scene.style_prompt


def test_the_alignment_study_runs_one_guided_arm_per_mode_beside_one_control() -> None:
    config = load_study_config(CONF_DIR / "da3_alignment.yaml")

    assert isinstance(config, AlignmentStudyConfig)
    specs = arm_specs(config)
    assert specs[0].arm == CONTROL_ARM
    assert not specs[0].guided
    assert [spec.alignment for spec in specs[1:]] == list(config.guidance.arms)
    assert all(spec.guided for spec in specs[1:])
    # The claim under test is raw against shift-only; both must actually run.
    assert set(config.guidance.arms) == {"none", "shift"}


def test_the_fmrg_alignment_study_runs_one_guided_arm_per_mode_beside_a_control() -> (
    None
):
    """The FMRG re-run keeps the SD1.5 study's shape: control first, then one arm per mode."""
    from framework.fmrg_alignment_study import (  # noqa: PLC0415
        CONTROL_ARM as FMRG_CONTROL_ARM,
    )
    from framework.fmrg_alignment_study import alignment_arms  # noqa: PLC0415

    config = load_study_config(CONF_DIR / "da3_fmrg.yaml")

    assert isinstance(config, FmrgAlignmentConfig)
    specs = alignment_arms(config)
    assert specs[0].arm == FMRG_CONTROL_ARM
    assert not specs[0].guided
    assert [spec.alignment for spec in specs[1:]] == list(config.guidance.arms)
    assert all(spec.guided for spec in specs[1:])
    # The claim under test is raw against shift-only; both must actually run.
    assert set(config.guidance.arms) == {"none", "shift"}
    # The eval judge must never be the guidance CLIP the repo steers with.
    assert config.adherence.checkpoint == "openai/clip-vit-large-patch14"


def test_the_fmrg_study_measures_the_benchmarks_scenes_at_its_seeds() -> None:
    """The head-to-head claim: an FMRG row sits beside an SD1.5 row at the same (scene, seed)."""
    fmrg = load_study_config(CONF_DIR / "fmrg.yaml")
    benchmark = load_study_config(CONF_DIR / "benchmark.yaml")

    assert isinstance(fmrg, FmrgConfig)
    assert isinstance(benchmark, BenchmarkConfig)
    assert fmrg.scenes == benchmark.scenes
    assert fmrg.seeds == benchmark.seeds
    # Scored under the same adherence checkpoint, or the columns stop comparing.
    assert fmrg.adherence == benchmark.adherence


def test_context_scenes_vary_framing_alone() -> None:
    """The prompt axis is the framing: the style text is inside the instruction verbatim."""
    config = load_study_config(CONF_DIR / "context.yaml")

    assert isinstance(config, FmrgContextConfig)
    for scene in config.scenes:
        assert scene.style_prompt != scene.renovation_prompt
        assert scene.style_prompt in scene.renovation_prompt
        # One neutral text judges both framings, and it carries the style
        # content rather than either arm's phrasing advantage.
        assert scene.eval_prompt == scene.style_prompt
    # The eval judge must never be the guidance CLIP the repo steers with.
    assert config.adherence.checkpoint == "openai/clip-vit-large-patch14"


def test_context_arms_pair_each_sweep_sigma_and_lead_with_the_baseline() -> None:
    """A first (it sets the colour scale), C at A's sigma, one B and D per sweep value."""
    config = load_study_config(CONF_DIR / "context.yaml")

    assert isinstance(config, FmrgContextConfig)
    specs = context_arms(config)
    assert specs[0].arm == BASELINE_ARM
    assert not specs[0].guided
    assert specs[1].arm == "C"
    assert specs[1].guided
    assert specs[0].start_sigma == specs[1].start_sigma == config.sampler.start_sigma
    swept = list(specs[2:])
    assert len(swept) == 2 * len(config.context_sigmas)
    for index, sigma in enumerate(config.context_sigmas):
        unguided, guided = swept[2 * index], swept[2 * index + 1]
        assert (unguided.guided, guided.guided) == (False, True)
        assert unguided.start_sigma == guided.start_sigma == sigma
        assert unguided.framing == guided.framing == "renovation"


def test_redux_arms_pair_each_scale_and_lead_with_the_text_only_baseline() -> None:
    """The scale-0 unguided arm first (it sets the ratios and the colour scale)."""
    config = load_study_config(CONF_DIR / "redux.yaml")

    assert isinstance(config, FmrgReduxConfig)
    specs = redux_arms(config)
    assert specs[0].arm == REDUX_BASELINE_ARM
    assert (specs[0].guided, specs[0].redux_scale) == (False, 0.0)
    assert len(specs) == 2 * len(config.redux_scales)
    for index, scale in enumerate(config.redux_scales):
        unguided, guided = specs[2 * index], specs[2 * index + 1]
        assert (unguided.guided, guided.guided) == (False, True)
        assert unguided.redux_scale == guided.redux_scale == scale
        assert unguided.arm != guided.arm
    # The drowning probe is in the shipped config: the full-strength scale is
    # diffusers' own composition, beside the low ones that keep the text audible.
    assert 1.0 in config.redux_scales
    # Every prompt both renovates and removes: the removal instruction is what
    # the compliance metric quantifies against the dial.
    for scene in config.scenes:
        assert scene.removal_object in scene.renovation_prompt


def test_a_redux_sweep_without_its_text_only_baseline_is_a_config_error() -> None:
    """Scale 0 is the baseline pair every arm's delta divides by."""
    with pytest.raises(ValueError, match=r"must start at 0\.0"):
        load_study_config(CONF_DIR / "redux.yaml", ["redux_scales=[0.3, 1.0]"])


def test_an_unordered_redux_sweep_is_a_config_error() -> None:
    """The sweep is a dial, read left to right."""
    with pytest.raises(ValueError, match="strictly increasing"):
        load_study_config(CONF_DIR / "redux.yaml", ["redux_scales=[0.0, 1.0, 0.3]"])


def test_the_redux_cache_key_never_collides_with_a_text_prompt() -> None:
    """The combined entry's name embeds the scale, so two scales are two entries."""
    low = redux_prompt_key("renovate this room", 0.3)
    full = redux_prompt_key("renovate this room", 1.0)

    assert low != full
    assert low != "renovate this room"


def test_eta_arms_lead_with_the_baseline_and_sweep_every_eta_guided() -> None:
    """A first (it sets the colour scale and the ratios), then one guided arm per eta."""
    config = load_study_config(CONF_DIR / "eta.yaml")

    assert isinstance(config, FmrgEtaConfig)
    specs = eta_arms(config)
    assert specs[0].arm == ETA_BASELINE_ARM
    assert specs[0].eta == 0.0
    assert not specs[0].guided
    assert [spec.eta for spec in specs[1:]] == list(config.etas)
    assert all(spec.guided for spec in specs[1:])
    # The sweep exists to push above the never-tuned default, not around it.
    assert config.etas[0] == 5.0
    assert list(config.etas) == sorted(config.etas)


def test_an_unguided_eta_in_the_sweep_is_a_config_error() -> None:
    """Eta 0 in the sweep would duplicate the baseline under a guided arm's name."""
    with pytest.raises(ValueError, match="etas must be positive"):
        load_study_config(CONF_DIR / "eta.yaml", ["etas=[0.0, 5.0]"])


def test_image_arms_lead_with_the_control_then_anchor_then_models_then_composites() -> (
    None
):
    """The pairwise readings need all four kinds: control, depth, per-model, composite."""
    from framework.image_study import (  # noqa: PLC0415
        CONTROL_ARM as IMAGE_CONTROL_ARM,
    )
    from framework.image_study import (  # noqa: PLC0415
        CONTROL_INSTRUCTION_ARM,
        DEPTH_ARM,
        DEPTH_INSTRUCTION_ARM,
        image_arms,
    )

    config = load_study_config(CONF_DIR / "image.yaml")

    assert isinstance(config, FmrgImageConfig)
    specs = image_arms(config)
    assert specs[0].arm == IMAGE_CONTROL_ARM
    assert not specs[0].guided
    assert specs[1].arm == DEPTH_ARM
    # The prompt-matched baselines: without them, instruction-following on the
    # helper arms could not be attributed to the reward rather than the prompt.
    assert specs[2].arm == CONTROL_INSTRUCTION_ARM
    assert not specs[2].guided
    assert specs[3].arm == DEPTH_INSTRUCTION_ARM
    assert [spec.instruction for spec in specs[:4]] == [False, False, True, True]
    image_specs = [spec for spec in specs if spec.kind == "image"]
    composite_specs = [spec for spec in specs if spec.kind == "composite"]
    assert [spec.model for spec in image_specs] == list(config.image.models)
    assert [spec.model for spec in composite_specs] == list(
        config.image.composite_models
    )
    # Every helper arm steers under the instruction prompt.
    assert all(spec.instruction for spec in image_specs + composite_specs)
    # The completed grid: every helper appears both alone and beside depth.
    assert [spec.model for spec in composite_specs] == [
        spec.model for spec in image_specs
    ]
    # The measured-weight pattern: no configured weight means the ratio is read
    # off the gradients per scene.
    assert config.image.depth_weight is None


def test_the_instruction_prompt_wraps_the_style_descriptors_verbatim() -> None:
    """The owner's template: same style content, plus the keep and remove clauses."""
    config = load_study_config(CONF_DIR / "image.yaml")

    assert isinstance(config, FmrgImageConfig)
    for scene in config.scenes:
        prompt = scene.instruction_prompt
        assert prompt.startswith(f"renovate this room into {scene.style_prompt}, ")
        assert "keep the windows, walls, floor and ceiling the same" in prompt
        assert prompt.endswith(f"remove the {scene.removal_object} in the room")
        # The judge's text never changes with the steering prompt.
        assert scene.eval_prompt == scene.style_prompt


def test_kontext_arms_lead_with_the_prior_alone_and_frame_the_style_as_an_instruction() -> (
    None
):
    """The unguided prior is the baseline; the steering prompt wraps the style verbatim."""
    from framework.kontext_study import (  # noqa: PLC0415
        BASELINE_ARM as KONTEXT_BASELINE_ARM,
    )
    from framework.kontext_study import GUIDED_ARM, kontext_arms  # noqa: PLC0415

    config = load_study_config(CONF_DIR / "kontext.yaml")

    assert isinstance(config, KontextConfig)
    specs = kontext_arms(config)
    assert [spec.arm for spec in specs] == [KONTEXT_BASELINE_ARM, GUIDED_ARM]
    assert [spec.guided for spec in specs] == [False, True]
    for scene in config.scenes:
        prompt = scene.instruction_prompt
        assert prompt.startswith(f"renovate this room into {scene.style_prompt}, ")
        assert prompt.endswith("keep the windows, walls, floor and ceiling the same")
        # The judge's text never changes with the steering prompt.
        assert scene.eval_prompt == scene.style_prompt
    # Kontext-dev's own operating point, not FLUX.1-dev's.
    assert config.sampler.text_guidance_scale == pytest.approx(2.5)
    # The sharpness strip is the pilot's reason to exist.
    assert config.endpoint_strip_every >= 1


def test_no_kontext_judge_may_be_the_guidance_clip() -> None:
    """The self-grading rule holds for the pilot's two CLIP judges."""
    with pytest.raises(ValueError, match="guidance CLIP"):
        load_study_config(
            CONF_DIR / "kontext.yaml",
            ["adherence.checkpoint=openai/clip-vit-base-patch32"],
        )
    with pytest.raises(ValueError, match="guidance CLIP"):
        load_study_config(
            CONF_DIR / "kontext.yaml",
            ["image_similarity_checkpoint=openai/clip-vit-base-patch32"],
        )


def test_a_shell_class_as_a_removal_target_is_a_config_error() -> None:
    """Removing a piece of the shell would contradict the prompt's own keep clause."""
    from framework.config import ImageSceneConfig  # noqa: PLC0415

    with pytest.raises(ValueError, match="contents"):
        ImageSceneConfig(
            photo="scandi_living_room",
            style_prompt="a scandinavian style living room",
            eval_prompt="a scandinavian style living room",
            removal_object="wall",
            removal_classes=(0,),
        )


def test_class_area_fraction_counts_the_named_classes_alone() -> None:
    from framework.metrics import class_area_fraction  # noqa: PLC0415

    segmentation = torch.zeros(1, 4, 4, dtype=torch.int64)
    segmentation[0, :2] = 15  # table over the top half
    segmentation[0, 2, 0] = 33  # one desk pixel

    assert class_area_fraction(segmentation, [15]) == pytest.approx(0.5)
    assert class_area_fraction(segmentation, [15, 33]) == pytest.approx(9 / 16)
    assert class_area_fraction(segmentation, [64]) == 0.0


def test_masked_mean_square_reads_only_the_mask_and_refuses_an_empty_one() -> None:
    from framework.metrics import masked_mean_square  # noqa: PLC0415

    field = torch.tensor([[[3.0, -1.0], [100.0, 100.0]]])
    mask = torch.tensor([[[True, True], [False, False]]])

    assert masked_mean_square(field, mask) == pytest.approx(5.0)
    # An empty window mask is a scene with no windowpane, not a zero error.
    assert masked_mean_square(field, torch.zeros_like(mask)) is None


def test_a_composite_without_its_image_only_twin_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="image-only arm"):
        load_study_config(CONF_DIR / "image.yaml", ["image.models=[clip]"])


def test_no_image_study_judge_may_share_a_steered_checkpoint() -> None:
    """The self-grading rule: the checkpoint an energy descends never scores it."""
    with pytest.raises(ValueError, match="independent"):
        load_study_config(
            CONF_DIR / "image.yaml",
            ["adherence.checkpoint=openai/clip-vit-base-patch32"],
        )
    with pytest.raises(ValueError, match="independent"):
        load_study_config(
            CONF_DIR / "image.yaml",
            ["image_similarity_checkpoint=openai/clip-vit-base-patch32"],
        )
    # The rule binds through the steering config, not the literal default: an
    # eval judge promoted to a steering checkpoint is caught the same way.
    with pytest.raises(ValueError, match="independent"):
        load_study_config(
            CONF_DIR / "image.yaml",
            ["image.clip_checkpoint=openai/clip-vit-large-patch14"],
        )


def test_the_context_adherence_judge_may_not_be_the_guidance_clip() -> None:
    """Grading a guided arm on the model it optimised is a config error at load."""
    with pytest.raises(ValueError, match="guidance CLIP"):
        load_study_config(
            CONF_DIR / "context.yaml",
            ["adherence.checkpoint=openai/clip-vit-base-patch32"],
        )


def test_the_fmrg_window_stops_inside_the_grid_it_truncates() -> None:
    """A window past the grid's end is a config error at load, not a mid-run crash."""
    with pytest.raises(ValueError, match="early_stop"):
        load_study_config(
            CONF_DIR / "fmrg_smoke.yaml",
            ["sampler.grid_steps=4", "sampler.early_stop=6"],
        )


def test_a_dotlist_override_reaches_a_nested_field() -> None:
    config = BenchmarkConfig.from_file(
        CONF_DIR / "smoke.yaml", ["guidance.strength=12.5", "sampler.steps=3"]
    )

    assert config.guidance.strength == pytest.approx(12.5)
    assert config.sampler.steps == 3


def test_an_override_addressing_a_list_element_says_so() -> None:
    """OmegaConf's own error names neither the override nor the list it hit."""
    with pytest.raises(ValueError, match="cannot address list elements") as raised:
        BenchmarkConfig.from_file(CONF_DIR / "smoke.yaml", ["scenes.0.prompt=a loft"])

    assert "scenes.0.prompt" in str(raised.value)


def test_an_unknown_override_key_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="guidnce"):
        BenchmarkConfig.from_file(CONF_DIR / "smoke.yaml", ["guidnce.strength=1"])


def _results(tmp_path: Path) -> Path:
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "config": {"name": "0002_smoke", "guidance": {"strength": 500.0}},
                "git_commit": "0123456789abcdef",
                "runs": [
                    {
                        "scene": "scandi_living_room",
                        "prompt": "a scandinavian living room",
                        "seed": seed,
                        "guided_depth_error": 0.01,
                        "control_depth_error": 0.02,
                        "depth_error_reduction": 0.5,
                        "adherence_ratio": 0.9,
                        "common_coverage": 0.42,
                    }
                    for seed in (0, 1)
                ],
            }
        ),
        encoding="utf-8",
    )
    return results


def test_the_gallery_renders_from_the_numbers_a_run_recorded(tmp_path: Path) -> None:
    """No images beside the results: every figure is skipped, the page still stands."""
    page = write_gallery(_results(tmp_path))

    assert page == tmp_path / GALLERY_FILENAME
    rendered = page.read_text(encoding="ascii")
    assert "0002_smoke" in rendered
    assert "Scandi living room" in rendered
    # The numbers are the run's, read off the file rather than recomputed.
    assert "&minus;50%" in rendered
    assert "42%" in rendered
    assert "01234567" in rendered
    assert "<img" not in rendered


def test_the_gallery_refuses_a_results_file_with_no_runs(tmp_path: Path) -> None:
    empty = tmp_path / "results.json"
    empty.write_text(json.dumps({"config": {"name": "0002"}, "runs": []}), "utf-8")

    with pytest.raises(ValueError, match="no run entries"):
        write_gallery(empty)


def test_contrast_preference_scores_per_image_and_orders_by_spread() -> None:
    """The stand-in's contract: one score per image, higher for the image with more contrast."""
    flat = torch.full((1, 3, 8, 8), 0.5)
    contrasty = torch.zeros(1, 3, 8, 8)
    contrasty[..., ::2] = 1.0

    scores = ContrastPreference().score(torch.cat([flat, contrasty]), None)

    assert scores.shape == (2,)
    assert float(scores[1]) > float(scores[0])


def test_disparity_maps_write_even_for_a_constant_field(tmp_path: Path) -> None:
    ramp = torch.linspace(0.0, 1.0, 16).view(1, 1, 16).expand(1, 16, 16)

    write_disparity_map(ramp, tmp_path / "ramp.png")
    write_disparity_map(torch.full((1, 16, 16), 2.5), tmp_path / "flat.png")

    assert (tmp_path / "ramp.png").stat().st_size > 0
    assert (tmp_path / "flat.png").stat().st_size > 0


def test_a_shared_scale_keeps_an_absolute_offset_visible(tmp_path: Path) -> None:
    # The metric study's whole reason for a shared range: self-normalisation
    # paints a globally offset field identically, and an offset is exactly what
    # a metric backend determines and an affine-invariant one does not.
    from framework.depth_maps import write_scaled_disparity_map  # noqa: PLC0415
    from torchvision.io import ImageReadMode, read_image  # noqa: PLC0415

    ramp = torch.linspace(0.2, 0.8, 256).reshape(1, 16, 16)
    scale = DisparityRange.of(ramp)

    write_scaled_disparity_map(ramp, scale=scale, path=tmp_path / "reference.png")
    write_scaled_disparity_map(ramp, scale=scale, path=tmp_path / "same.png")
    write_scaled_disparity_map(ramp + 0.3, scale=scale, path=tmp_path / "offset.png")
    write_disparity_map(ramp, tmp_path / "self_a.png")
    write_disparity_map(ramp + 0.3, tmp_path / "self_b.png")

    def _decode(name: str) -> torch.Tensor:
        decoded: torch.Tensor = read_image(str(tmp_path / name), mode=ImageReadMode.RGB)
        return decoded

    assert torch.equal(_decode("reference.png"), _decode("same.png"))
    assert not torch.equal(_decode("reference.png"), _decode("offset.png"))
    assert torch.equal(_decode("self_a.png"), _decode("self_b.png"))


def _framing_results(tmp_path: Path) -> Path:
    arms = [
        {
            "arm": arm,
            "framing": framing,
            "guided": guided,
            "prompt": "renovate this furnished room into a scandinavian living room"
            if framing == "renovation"
            else "a scandinavian living room",
            "depth_error": 0.25 if not guided else 0.1,
            "shell_coverage": 0.8,
            "adherence": 0.3,
            "preference": 19.5,
            "render": f"scandi_living_room_seed0_{arm}.png",
            "depth_map": f"scandi_living_room_seed0_{arm}_depth.png",
            "residual_map": f"scandi_living_room_seed0_{arm}_residual.png",
            "seconds": 5.0,
        }
        for arm, framing, guided in (
            ("style_unguided", "style", False),
            ("renovation_unguided", "renovation", False),
            ("style_guided", "style", True),
            ("renovation_guided", "renovation", True),
        )
    ]
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "config": {"name": "0002_prompt_framing_smoke"},
                "git_commit": "0123456789abcdef",
                "comparisons": {
                    "unguided_style_vs_renovation": {
                        "depth_error": {
                            "style": 0.25,
                            "renovation": 0.22,
                            "delta": -0.03,
                        }
                    }
                },
                "runs": [
                    {
                        "scene": "scandi_living_room",
                        "seed": 0,
                        "style_prompt": "a scandinavian living room",
                        "renovation_prompt": (
                            "renovate this furnished room into "
                            "a scandinavian living room"
                        ),
                        "eval_prompt": "a scandinavian living room",
                        "common_coverage": 0.42,
                        "reconstruction_depth_error": 0.001,
                        "photo_image": "scandi_living_room_photo.png",
                        "photo_depth_map": "scandi_living_room_photo_depth.png",
                        "shell_mask": "scandi_living_room_shell_mask.png",
                        "eval_mask_common": "scandi_living_room_seed0_evalmask.png",
                        "arms": arms,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return results


def test_the_framing_gallery_renders_the_grid_and_both_prompts(tmp_path: Path) -> None:
    """No images beside the results: every figure is skipped, the page still stands."""
    page = write_framing_gallery(_framing_results(tmp_path))

    assert page == tmp_path / FRAMING_GALLERY_FILENAME
    rendered = page.read_text(encoding="ascii")
    assert "0002_prompt_framing_smoke" in rendered
    assert "Scandi living room" in rendered
    assert "renovate this furnished room" in rendered
    assert "42%" in rendered
    assert "01234567" in rendered
    assert "<img" not in rendered


def test_the_framing_gallery_refuses_a_results_file_with_no_runs(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "results.json"
    empty.write_text(json.dumps({"config": {"name": "0002"}, "runs": []}), "utf-8")

    with pytest.raises(ValueError, match="no run entries"):
        write_framing_gallery(empty)
