"""The typed configuration of this framework's studies; one variant per `conf/` file.

The study shapes share the machinery: `benchmark` is guided-vs-control at one
prompt per scene on the SD1.5 stack, `fmrg` is the same claim on FMRG's FLUX
flow map, `framing` is the prompt-framing 2x2 (style description vs renovation
instruction, each unguided and guided). A `conf/` file names its shape in a
top-level `study` key and `load_study_config` dispatches on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final, Literal

import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, field_validator, model_validator

from roomgen_geometry.alignment import AlignmentMode
from roomgen_geometry.depth.depth_anything import DEFAULT_CHECKPOINT as DEPTH_CHECKPOINT
from roomgen_geometry.depth.depth_anything_v3 import (
    DEFAULT_CHECKPOINT as METRIC_DEPTH_CHECKPOINT,
)
from roomgen_geometry.depth.depth_anything_v3 import (
    DEFAULT_INPUT_SIZE as METRIC_INPUT_SIZE,
)
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide.config import Config
from roomgen_guide.eval import DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT
from roomgen_guide.experiment import ExperimentConfig
from roomgen_guide.guidance.clip import DEFAULT_CHECKPOINT as GUIDANCE_CLIP_CHECKPOINT
from roomgen_guide.renovate import DEFAULT_CHECKPOINT as DIFFUSION_CHECKPOINT

DTypeName = Literal["float32", "float16", "bfloat16"]

TORCH_DTYPES: Final[dict[DTypeName, torch.dtype]] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Stable Diffusion 1.5's VAE downsamples by 8 and its UNet by a further 8, so a
# side that is not a multiple of 64 is silently rounded by the pipeline and the
# render no longer matches the reference the reward was built from.
LATENT_ALIGNMENT: Final = 64

# FLUX's VAE downsamples by 8 and its transformer packs the latents 2x2, so a
# side has to be a multiple of 16 to survive the round trip unrounded.
FLUX_LATENT_ALIGNMENT: Final = 16


class SceneConfig(Config):
    """One benchmark photo paired with the renovation asked of it.

    Args:
        photo: id of an entry in `experiments/assets/photos.json`.
        prompt: the renovation prompt, held fixed across seeds and variants so
            that a guided run and its control differ in guidance alone.
    """

    photo: Annotated[str, Field(min_length=1)]
    prompt: Annotated[str, Field(min_length=1)]


class BackendConfig(Config):
    """Which model stack the run drives, where it lives, and in what precision.

    Args:
        kind: `real` loads the three pretrained checkpoints; `fake` swaps in the
            weightless doubles the test suites run against, which is what
            `conf/smoke.yaml` selects to exercise this framework end to end on
            CPU in seconds.
        device: torch device string; the sampler's generator is created on it,
            so a CUDA run is reproducible from its seed the same way a CPU one is.
        dtype: parameter dtype of the UNet and VAE, and the dtype the photo is
            cast to. It is *also* the dtype the reward is evaluated in, since
            the depth backend returns its prediction in the input image's dtype
            — float16 here quantises the energy, not just the sampler.
        depth_dtype: parameter dtype of the depth backend, independent of
            `dtype`: a float32 depth network can sit above a float16 sampler.
        segmentation_dtype: parameter dtype of the segmenter, which is off the
            gradient path entirely and only decides the mask.
        deterministic: ask torch for deterministic kernels before the stack is
            built (`roomgen_guide.determinism`). Best effort: it narrows
            run-to-run variance on CUDA, it does not guarantee it away, and it
            must be set before the process's first CUDA allocation to take
            effect on cuBLAS.
        scheduler: `ddim` replaces the checkpoint's PNDM scheduler with the DDIM
            whose update is exactly the x̂₀ closed form the guided loop assumes;
            `pipeline` keeps whatever scheduler the pipeline was built with,
            which is the only option the weightless fake stack has.
    """

    kind: Literal["real", "fake"]
    device: Annotated[str, Field(min_length=1)] = "cpu"
    deterministic: bool = False
    dtype: DTypeName = "float32"
    depth_dtype: DTypeName = "float32"
    segmentation_dtype: DTypeName = "float32"
    scheduler: Literal["ddim", "pipeline"] = "ddim"
    diffusion_checkpoint: Annotated[str, Field(min_length=1)] = DIFFUSION_CHECKPOINT
    depth_checkpoint: Annotated[str, Field(min_length=1)] = DEPTH_CHECKPOINT
    segmentation_checkpoint: Annotated[str, Field(min_length=1)] = (
        SEGMENTATION_CHECKPOINT
    )


class SamplerConfig(Config):
    """The img2img budget, identical for the guided run and its control.

    Args:
        steps: denoising steps the scheduler is set to.
        noise_strength: how far up the trajectory the photo is renoised (SDEdit).
        text_guidance_scale: classifier-free guidance on the prompt, unrelated
            to the geometric guidance this framework is testing.
    """

    steps: Annotated[int, Field(ge=1)] = 30
    noise_strength: Annotated[float, Field(gt=0.0, le=1.0)] = 0.6
    text_guidance_scale: Annotated[float, Field(ge=1.0)] = 7.5


class GuidanceConfig(Config):
    """The geometric guidance term under test.

    Args:
        strength: what the energy gradient is multiplied by at every step before
            it is applied to the latent — the knob a strength sweep varies, in
            the squared-disparity units the depth backend defines.
        clip: ceiling on the RMS of the raw step, in latent-RMS units: the raw
            step keeps shrinking as the geometry is satisfied, and the cap is
            what stops a large energy from running away.
        erosion_radius: pixels trimmed either side of a segment edge before the
            shell mask is trusted.
        resegment: segment each predicted image and score depth only on
            reference-shell AND prediction-shell, so furniture the renovation
            adds may occlude the shell for free; false scores the reference
            mask alone (pre-occlusion semantics, the control arm for A/B).
    """

    strength: Annotated[float, Field(ge=0.0)]
    clip: Annotated[float, Field(gt=0.0)] | None = None
    erosion_radius: Annotated[int, Field(ge=0)] = 2
    resegment: bool = True


class AdherenceConfig(Config):
    """Which metric reports how much of the renovation survived the guidance.

    Args:
        kind: `clip` is the CLIP prompt score, read against the prompt itself.
            `contents_change` is checkpoint-free: it measures how far the render
            moved from the photo *outside* the shell mask, which is weaker (it
            cannot read the prompt) but still separates a renovated room from an
            untouched one, and is what the smoke variant uses.
        checkpoint: Hugging Face Hub id, read only by `clip`.
    """

    kind: Literal["clip", "contents_change"]
    checkpoint: Annotated[str, Field(min_length=1)] = ADHERENCE_CHECKPOINT


class BenchmarkConfig(ExperimentConfig):
    """One variant of 0002: a scene set, a sampler budget, and what is measured on them.

    Args:
        scenes: the (photo, prompt) pairs the run measures, each at every seed.
        resolution: side of the square the photos are centre-cropped and resized
            to, a multiple of 64.
    """

    study: Literal["benchmark"] = "benchmark"
    scenes: Annotated[tuple[SceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=LATENT_ALIGNMENT)] = 512
    backends: BackendConfig
    sampler: SamplerConfig = SamplerConfig()
    guidance: GuidanceConfig
    adherence: AdherenceConfig

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_latent_grid(cls, resolution: int) -> int:
        return _validate_resolution(resolution)


class FmrgBackendConfig(Config):
    """The FLUX flow-map stack an `fmrg` study drives, and where its pieces live.

    Args:
        kind: `real` loads FLUX.1-dev with the FlowMap LoRA through the vendored
            FMRG sampler; `fake` swaps in the weightless `ToyFlowMapBackend` the
            adapter's tests walk, which is what `conf/fmrg_smoke.yaml` selects.
        dtype: the transformer's and VAE's parameter dtype. The walk's state is
            float32 regardless (`FlowMapProcess`), and the depth reward is
            evaluated in the depth backend's own dtype.
        flux_checkpoint: Hub id (or local path) of the FLUX base weights,
            resolved against the `ROOMGEN_HF_HOME` cache offline — the run never
            re-downloads a 33GB checkpoint mid-benchmark.
        lora_dir: directory holding the FlowMap LoRA's
            `pytorch_lora_weights.safetensors` for the sampled resolution;
            `None` means `<ROOMGEN_HF_HOME>/fmrg/flux-flowmap-lora-512`, where
            the repo's download instructions put it.
        fmrg_path: the vendored FMRG checkout, reached through `sys.path`
            because upstream ships no installable package; `None` means this
            repo's `packages/fmrg`.
    """

    kind: Literal["real", "fake"]
    device: Annotated[str, Field(min_length=1)] = "cpu"
    deterministic: bool = False
    dtype: DTypeName = "bfloat16"
    depth_dtype: DTypeName = "float32"
    segmentation_dtype: DTypeName = "float32"
    flux_checkpoint: Annotated[str, Field(min_length=1)] = (
        "black-forest-labs/FLUX.1-dev"
    )
    lora_dir: Path | None = None
    fmrg_path: Path | None = None
    depth_checkpoint: Annotated[str, Field(min_length=1)] = DEPTH_CHECKPOINT
    segmentation_checkpoint: Annotated[str, Field(min_length=1)] = (
        SEGMENTATION_CHECKPOINT
    )


class FmrgSamplerConfig(Config):
    """FMRG's sampling budget, identical for the guided walk and its control.

    The defaults are upstream's `fmrgj_nfe100` reward-guidance operating point
    (`packages/fmrg/experiments/geneval_nfe_sweep/generate_fmrg.sh`), which the
    overnight pod runs validated on this reward: a dense sigma grid, the walk
    truncated to its head, and one flow-map jump covering the rest.

    Args:
        grid_steps: resolution of the dense sigma grid (`num_steps`); the walk
            never visits most of it.
        early_stop: how many of the grid's sigmas the walk traverses before the
            jump to data — FMRG's `early_stop`, the guidance window.
        unguided_steps: how many steps the final jump is split into, unguided.
        start_sigma: how far up the interpolant the photo is renoised, pre-shift
            (`FlowMapProcess` warps it by resolution); 1.0 discards the photo.
        text_guidance_scale: FLUX.1-dev's embedded guidance — conditioning, not
            reward steering.
        stepping: which velocity the step is taken off
            (`roomgen_guide.adapters.flow_map.FlowMapStepping`).
        differentiate_through_velocity: FMRG-J's pullback through the
            transformer rather than the cheap decoder-only one.
    """

    grid_steps: Annotated[int, Field(ge=1)] = 208
    early_stop: Annotated[int, Field(ge=2)] = 51
    unguided_steps: Annotated[int, Field(ge=1)] = 1
    start_sigma: Annotated[float, Field(gt=0.0, le=1.0)] = 0.9
    text_guidance_scale: Annotated[float, Field(ge=0.0)] = 3.5
    stepping: Literal["endpoint_reuse", "two_time"] = "endpoint_reuse"
    differentiate_through_velocity: bool = False

    @model_validator(mode="after")
    def _the_window_stops_inside_the_grid(self) -> FmrgSamplerConfig:
        # `GuidanceWindow.truncate` enforces the same bound, but only once a
        # 24GB transformer is already loaded; a config error should not cost one.
        if self.early_stop > self.grid_steps:
            raise ValueError(
                f"early_stop {self.early_stop} is past the end of a "
                f"{self.grid_steps}-step grid: the window has to stop inside "
                f"the walk it truncates"
            )
        return self


class FmrgGuidanceConfig(Config):
    """The geometric guidance term, in FMRG's own step rule.

    Args:
        eta: FMRG's `step_size` — the annealed per-unit-time displacement of
            `AnnealedSignTilt`, not comparable to the benchmark's raw-gradient
            `strength`; 0 is the unguided control.
        iterations: FMRG's `n_opt` inner loop, model calls per guided step.
        erosion_radius: pixels trimmed either side of a segment edge before a
            shell mask is trusted.
        resegment: score depth only on reference-shell AND prediction-shell, as
            the benchmark study does.
    """

    eta: Annotated[float, Field(ge=0.0)] = 5.0
    iterations: Annotated[int, Field(ge=1)] = 1
    erosion_radius: Annotated[int, Field(ge=0)] = 2
    resegment: bool = True


class FmrgConfig(ExperimentConfig):
    """Guided-vs-control on FMRG's FLUX flow map: the benchmark claim, new sampler.

    Args:
        scenes: the (photo, prompt) pairs the run measures, each at every seed.
        resolution: side of the square the photos are centre-cropped and resized
            to, a multiple of 16 (FLUX's VAE downsamples by 8 and packs 2x2).
    """

    study: Literal["fmrg"]
    scenes: Annotated[tuple[SceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=FLUX_LATENT_ALIGNMENT)] = 512
    backends: FmrgBackendConfig
    sampler: FmrgSamplerConfig = FmrgSamplerConfig()
    guidance: FmrgGuidanceConfig = FmrgGuidanceConfig()
    adherence: AdherenceConfig

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_packed_latent_grid(cls, resolution: int) -> int:
        if resolution % FLUX_LATENT_ALIGNMENT:
            raise ValueError(
                f"resolution must be a multiple of {FLUX_LATENT_ALIGNMENT} for "
                f"FLUX's packed latent grid, got {resolution}"
            )
        return resolution


class FramingSceneConfig(Config):
    """One photo, the two framings of one renovation, and the text every arm is judged on.

    Args:
        photo: id of an entry in `experiments/assets/photos.json`.
        style_prompt: the plain style description — the prompt shape this repo
            has sampled with since the start.
        renovation_prompt: the same style content phrased as an explicit
            renovation instruction over the furnished photo. Only the framing
            may differ from `style_prompt`, or the comparison stops being about
            framing.
        eval_prompt: the one neutral style description the adherence and
            preference metrics score *every* arm against — never the arm's own
            prompt, which would grade the two framings on different text.
    """

    photo: Annotated[str, Field(min_length=1)]
    style_prompt: Annotated[str, Field(min_length=1)]
    renovation_prompt: Annotated[str, Field(min_length=1)]
    eval_prompt: Annotated[str, Field(min_length=1)]


class FmrgContextConfig(ExperimentConfig):
    """The renovation-context study: does image context alone keep the layout?

    Per scene, two prompt framings of the identical style content — the plain
    style description and an explicit renovation instruction — each sampled
    with and without the depth reward, the instruction arms swept over SDEdit
    init strengths. Arm A (style, unguided, at the sampler's own `start_sigma`)
    is the baseline every other arm's depth error and adherence are read
    against.

    Args:
        scenes: each photo with both framings; adherence is scored against the
            scene's one neutral `eval_prompt` for every arm, never an arm's own
            prompt.
        context_sigmas: the pre-shift `start_sigma` values the renovation-framed
            arms are swept over, one unguided (B) and one guided (D) arm per
            value. Pre-shift on purpose — FLUX warps the grid by resolution, and
            the study records the effective first sigma read off the actual
            schedule beside each arm.
        resolution: side of the square the photos are centre-cropped and resized
            to, a multiple of 16 (FLUX's VAE downsamples by 8 and packs 2x2).
    """

    study: Literal["fmrg_context"]
    scenes: Annotated[tuple[FramingSceneConfig, ...], Field(min_length=1)]
    context_sigmas: Annotated[tuple[float, ...], Field(min_length=1)] = (
        0.6,
        0.75,
        0.9,
    )
    resolution: Annotated[int, Field(ge=FLUX_LATENT_ALIGNMENT)] = 512
    backends: FmrgBackendConfig
    sampler: FmrgSamplerConfig = FmrgSamplerConfig()
    guidance: FmrgGuidanceConfig = FmrgGuidanceConfig()
    adherence: AdherenceConfig

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_packed_latent_grid(cls, resolution: int) -> int:
        if resolution % FLUX_LATENT_ALIGNMENT:
            raise ValueError(
                f"resolution must be a multiple of {FLUX_LATENT_ALIGNMENT} for "
                f"FLUX's packed latent grid, got {resolution}"
            )
        return resolution

    @field_validator("context_sigmas")
    @classmethod
    def _sigmas_are_renoise_depths(cls, sigmas: tuple[float, ...]) -> tuple[float, ...]:
        for sigma in sigmas:
            if not 0.0 < sigma <= 1.0:
                raise ValueError(
                    f"context_sigmas must lie in (0, 1], got {sigma}: each is how "
                    f"far up the interpolant the photo is renoised"
                )
        return sigmas

    @model_validator(mode="after")
    def _adherence_is_never_the_guidance_clip(self) -> FmrgContextConfig:
        # The eval CLIP must stay independent of the CLIP any guidance energy in
        # this repo descends, or a guided arm is graded on the model it optimised.
        if (
            self.adherence.kind == "clip"
            and self.adherence.checkpoint == GUIDANCE_CLIP_CHECKPOINT
        ):
            raise ValueError(
                f"adherence.checkpoint is {self.adherence.checkpoint!r}, the "
                f"guidance CLIP: score adherence with an independent checkpoint "
                f"(e.g. openai/clip-vit-large-patch14)."
            )
        return self


class PreferenceConfig(Config):
    """Which human-preference model scores the finished renders — eval only, never guidance.

    Args:
        kind: `pickscore` loads PickScore v1, a CLIP-H fine-tuned on human
            choices, through transformers; `contrast` is a checkpoint-free
            stand-in (per-image pixel spread) that lets the smoke variant
            exercise the plumbing offline — it cannot read beauty.
        checkpoint: Hugging Face Hub id of the scoring model, read only by
            `pickscore`.
        processor_checkpoint: Hub id the tokenizer and image statistics come
            from — PickScore ships no processor of its own, so upstream points
            at the CLIP-H it was fine-tuned from.
    """

    kind: Literal["pickscore", "contrast"]
    checkpoint: Annotated[str, Field(min_length=1)] = "yuvalkirstain/PickScore_v1"
    processor_checkpoint: Annotated[str, Field(min_length=1)] = (
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )


class FramingConfig(ExperimentConfig):
    """The prompt-framing 2x2: {style, renovation} x {unguided, guided} at matched seeds.

    Args:
        scenes: each photo with both framings of one renovation; every scene
            runs all four arms at every seed.
        resolution: side of the square the photos are centre-cropped and resized
            to, a multiple of 64.
    """

    study: Literal["framing"]
    scenes: Annotated[tuple[FramingSceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=LATENT_ALIGNMENT)] = 512
    backends: BackendConfig
    sampler: SamplerConfig = SamplerConfig()
    guidance: GuidanceConfig
    adherence: AdherenceConfig
    preference: PreferenceConfig

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_latent_grid(cls, resolution: int) -> int:
        return _validate_resolution(resolution)


class AlignmentBackendConfig(BackendConfig):
    """The benchmark's stack plus the metric depth judge the alignment study steers by.

    Args:
        metric_depth_checkpoint: the Depth Anything 3 *metric* tier — the only
            backend for which raw or shift-only comparison is meaningful at all.
            `depth_checkpoint` (V2, affine-invariant) stays loaded beside it as
            the neutral second judge every arm is also scored under.
        metric_input_size: the square edge DA3 resamples to, a multiple of its
            14-pixel patch.
    """

    metric_depth_checkpoint: Annotated[str, Field(min_length=1)] = (
        METRIC_DEPTH_CHECKPOINT
    )
    metric_input_size: Annotated[int, Field(ge=14)] = METRIC_INPUT_SIZE


class AlignmentGuidanceConfig(GuidanceConfig):
    """The benchmark's guidance term, run once per alignment mode under test.

    Args:
        arms: the alignment modes the guided arms are steered under, one arm
            per mode at every (scene, seed), beside the shared unguided control.
            Loosening the fit is only meaningful for the metric backend the
            study steers by; the affine-invariant second judge always scores
            under the full fit.
    """

    arms: Annotated[tuple[AlignmentMode, ...], Field(min_length=1)] = ("none", "shift")


class AlignmentStudyConfig(ExperimentConfig):
    """The DA3 alignment-mode study: raw against shift-only, beside one control.

    Tests the claim that a metric backend guides better under a shift-only fit
    than under no fit at all — until now a code comment, not a measurement.
    Every arm is scored under *all three* modes plus the affine-invariant
    second judge, so no arm is read solely through its own objective.

    Args:
        scenes: the (photo, prompt) pairs the study measures, each at every seed.
        resolution: side of the square the photos are centre-cropped and resized
            to, a multiple of 64.
    """

    study: Literal["da3_alignment"]
    scenes: Annotated[tuple[SceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=LATENT_ALIGNMENT)] = 512
    backends: AlignmentBackendConfig
    sampler: SamplerConfig = SamplerConfig()
    guidance: AlignmentGuidanceConfig
    adherence: AdherenceConfig

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_latent_grid(cls, resolution: int) -> int:
        return _validate_resolution(resolution)


def _validate_resolution(resolution: int) -> int:
    if resolution % LATENT_ALIGNMENT:
        raise ValueError(
            f"resolution must be a multiple of {LATENT_ALIGNMENT} for Stable "
            f"Diffusion 1.5's latent grid, got {resolution}"
        )
    return resolution


def load_study_config(
    path: Path, overrides: Sequence[str] | None = None
) -> (
    BenchmarkConfig
    | FmrgConfig
    | FmrgContextConfig
    | FramingConfig
    | AlignmentStudyConfig
):
    """Load whichever study shape `path` declares in its top-level `study` key.

    The key is peeked before validation because the shapes are different
    pydantic models; a missing key means `benchmark`, which is what every
    variant predating the other studies is.
    """
    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise TypeError(f"Expected a mapping at {path}, got {type(raw).__name__}")
    study = raw.get("study", "benchmark")
    match study:
        case "benchmark":
            return BenchmarkConfig.from_file(path, overrides)
        case "fmrg":
            return FmrgConfig.from_file(path, overrides)
        case "fmrg_context":
            return FmrgContextConfig.from_file(path, overrides)
        case "framing":
            return FramingConfig.from_file(path, overrides)
        case "da3_alignment":
            return AlignmentStudyConfig.from_file(path, overrides)
        case _:
            raise ValueError(
                f"{path} declares unknown study {study!r}; "
                "known studies: benchmark, fmrg, fmrg_context, framing, "
                "da3_alignment"
            )
