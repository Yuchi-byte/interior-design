"""The typed configuration of this framework's studies; one variant per `conf/` file.

The study shapes share the machinery: `benchmark` is guided-vs-control at one
prompt per scene on the SD1.5 stack, `fmrg` is the same claim on FMRG's FLUX
flow map, `framing` is the prompt-framing 2x2 (style description vs renovation
instruction, each unguided and guided). A `conf/` file names its shape in a
top-level `study` key and `load_study_config` dispatches on it.
"""

from __future__ import annotations

import itertools
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
from roomgen_geometry.segmentation.ade20k import CONTENTS_CLASSES
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide.adapters.fmrg_flux import (
    DEFAULT_REDUX_CHECKPOINT as REDUX_CHECKPOINT,
)
from roomgen_guide.adapters.kontext_flux import (
    DEFAULT_KONTEXT_CHECKPOINT,
    DEFAULT_KONTEXT_GUIDANCE_SCALE,
)
from roomgen_guide.config import Config
from roomgen_guide.eval import DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT
from roomgen_guide.eval import (
    DEFAULT_IMAGE_SIMILARITY_CHECKPOINT as IMAGE_SIMILARITY_CHECKPOINT,
)
from roomgen_guide.experiment import ExperimentConfig
from roomgen_guide.guidance.clip import DEFAULT_CHECKPOINT as GUIDANCE_CLIP_CHECKPOINT
from roomgen_guide.guidance.image_reference import (
    DEFAULT_DINOV2_CHECKPOINT,
    DEFAULT_SIGLIP_CHECKPOINT,
)
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


class ReduxSceneConfig(FramingSceneConfig):
    """A framing scene plus the one object its renovation prompt removes.

    The Redux sweep's renovation prompts carry an explicit removal instruction
    ("remove the <object> in the room") so the scale dial measures where the
    image tokens start to drown the text; the scene names the object and the
    ADE20K ids the removal-compliance metric counts it under.

    Args:
        removal_object: the object the renovation prompt removes, named as it
            appears in the photo (e.g. `chair`); must appear verbatim in
            `renovation_prompt`, or the prompt and the metric drift apart.
        removal_classes: ADE20K-150 model-output ids the removal-compliance
            metric counts as that object; every id must be a contents class,
            because removing a piece of the shell would contradict the same
            prompt's "keep" clause and the depth reward.
        removal_substituted: true when the photo holds no table or desk at all
            and the prompt removes the scene's most removable chair instead —
            recorded so the report can say so.
    """

    removal_object: Annotated[str, Field(min_length=1)]
    removal_classes: Annotated[tuple[int, ...], Field(min_length=1)]
    removal_substituted: bool = False

    @field_validator("removal_classes")
    @classmethod
    def _removal_targets_are_contents(cls, ids: tuple[int, ...]) -> tuple[int, ...]:
        outside = [
            identifier for identifier in ids if identifier not in CONTENTS_CLASSES
        ]
        if outside:
            raise ValueError(
                f"removal_classes {outside} are not ADE20K contents classes: the "
                f"instruction removes furniture, never shell, so a non-contents "
                f"id would score the wrong pixels as compliance"
            )
        return ids

    @model_validator(mode="after")
    def _the_prompt_removes_what_the_metric_counts(self) -> ReduxSceneConfig:
        if self.removal_object not in self.renovation_prompt:
            raise ValueError(
                f"renovation_prompt never names {self.removal_object!r}: the "
                f"removal-compliance metric counts an object the prompt has to "
                f"actually ask to remove"
            )
        return self


class ReduxStripConfig(Config):
    """Stage zero of the Redux sweep: strip each photo bare before anything samples.

    The fields mirror 0003's `StripConfig` plus the SD stack dtype the strip
    runs in; the machinery itself is imported from
    `experiments/0003_strip_and_furnish` at run time rather than reimplemented,
    and the defaults are that study's owner-validated `fmrg.yaml` values
    verbatim. The stripped empty room is what the depth reference is built
    from — its shell is nearly the whole frame, which is the larger comparison
    area the reference switch buys — while Redux still encodes the furnished
    photo.

    Args:
        prompt: what the inpainter is asked to fill the cut region with.
        negative_prompt: what the fill must not become; the only guard against
            fills ADE20K has no name for.
        dilation_radius: pixels the contents mask is grown by before the cut.
        passes: inpainting passes; each repaints what the segmenter still calls
            removable, and the loop stops early once the hole segments clean.
        dtype: the SD1.5 inpainting stack's parameter dtype; the strip stack is
            released in full before the FLUX transformer loads.
    """

    prompt: Annotated[str, Field(min_length=1)] = (
        "empty room, bare floor, bare wall, no furniture, plain interior"
    )
    negative_prompt: Annotated[str, Field(min_length=1)] | None = (
        "furniture, chair, table, desk, sofa, bed, radiator, heater, appliance, "
        "vent, air conditioner, fireplace, television, screen, shelf, ledge, "
        "counter, cabinet, box, machine, vacuum cleaner, cart, trolley, wheels, "
        "plant, lamp, curtain, rug, decoration, clutter, objects, people, text"
    )
    dilation_radius: Annotated[int, Field(ge=0)] = 8
    passes: Annotated[int, Field(ge=1)] = 5
    steps: Annotated[int, Field(ge=1)] = 40
    fill_strength: Annotated[float, Field(gt=0.0, le=1.0)] = 0.85
    guidance_scale: Annotated[float, Field(ge=1.0)] = 12.0
    dtype: DTypeName = "float16"


class FmrgReduxBackendConfig(FmrgBackendConfig):
    """The FMRG stack plus the Redux image-context encoders the study caches with.

    Args:
        redux_checkpoint: Hub id of FLUX.1-Redux-dev — the SigLIP image encoder
            and the adapter projecting its patch grid into T5 embedding space.
            Loaded once per run to encode the reference photos, freed before the
            transformer's walks begin.
    """

    redux_checkpoint: Annotated[str, Field(min_length=1)] = REDUX_CHECKPOINT


class FmrgReduxConfig(ExperimentConfig):
    """The Redux-scale sweep: the furnished photo's image tokens dialled against the text.

    Per scene at one seed, one unguided and one depth-guided arm per
    `redux_scales` value, every arm conditioned on the scene's one renovation
    prompt (style descriptors, a keep-the-structure clause, and an explicit
    remove-the-object instruction). Scale 0 is the text-only pair — the plain
    prompt entry, never a zero-scaled token copy — and its unguided arm is the
    baseline every delta divides by. All arms run at the sampler's one
    `start_sigma`; the axis under test is the image context's strength alone.

    The depth reference is the *stripped empty room's* shell depth (stage
    zero, `strip`), by owner decision: the empty shell is nearly the whole
    frame, so the valid comparison area is far larger than the furnished
    photo's shell — and it contains none of the furniture the removal
    instruction targets, so the depth reward never defends the removed
    object's mass. The Redux tokens still encode the furnished photo; that is
    the axis under test.

    Args:
        scenes: each photo with its prompts and its removal target; adherence
            and preference are scored against the scene's one neutral
            `eval_prompt` for every arm, never an arm's own prompt — and never
            the image, which most arms were conditioned on.
        redux_scales: multipliers on the image tokens alone (the text embeds
            are never scaled), strictly increasing from the mandatory 0.0
            baseline. 1.0 is diffusers' composition and the known drowning
            regime; ~0.3 the community operating point that keeps the text
            audible.
        image_similarity_checkpoint: the eval judge of render-to-photo cosine
            (`None` skips it, the offline smoke setting) — read beside
            adherence, since a photo copy scores near 1 on it.
    """

    study: Literal["fmrg_redux"]
    scenes: Annotated[tuple[ReduxSceneConfig, ...], Field(min_length=1)]
    redux_scales: Annotated[tuple[float, ...], Field(min_length=2)] = (
        0.0,
        0.1,
        0.3,
        0.7,
        1.0,
    )
    resolution: Annotated[int, Field(ge=FLUX_LATENT_ALIGNMENT)] = 512
    backends: FmrgReduxBackendConfig
    strip: ReduxStripConfig = ReduxStripConfig()
    sampler: FmrgSamplerConfig = FmrgSamplerConfig()
    guidance: FmrgGuidanceConfig = FmrgGuidanceConfig()
    adherence: AdherenceConfig
    image_similarity_checkpoint: Annotated[str, Field(min_length=1)] | None = (
        IMAGE_SIMILARITY_CHECKPOINT
    )
    preference: PreferenceConfig

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_packed_latent_grid(cls, resolution: int) -> int:
        if resolution % FLUX_LATENT_ALIGNMENT:
            raise ValueError(
                f"resolution must be a multiple of {FLUX_LATENT_ALIGNMENT} for "
                f"FLUX's packed latent grid, got {resolution}"
            )
        return resolution

    @field_validator("redux_scales")
    @classmethod
    def _scales_dial_up_from_the_text_only_baseline(
        cls, scales: tuple[float, ...]
    ) -> tuple[float, ...]:
        if scales[0] != 0.0:
            raise ValueError(
                f"redux_scales must start at 0.0, got {scales[0]}: the text-only "
                f"pair is the baseline every arm's delta divides by"
            )
        for lower, upper in itertools.pairwise(scales):
            if upper <= lower:
                raise ValueError(
                    f"redux_scales must be strictly increasing, got {upper} "
                    f"after {lower}: the sweep is a dial, read left to right"
                )
        return scales

    @model_validator(mode="after")
    def _no_judge_is_the_guidance_clip(self) -> FmrgReduxConfig:
        # The eval CLIPs must stay independent of the CLIP any guidance energy in
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
        if self.image_similarity_checkpoint == GUIDANCE_CLIP_CHECKPOINT:
            raise ValueError(
                f"image_similarity_checkpoint is "
                f"{self.image_similarity_checkpoint!r}, the guidance CLIP: "
                f"judge image similarity with an independent checkpoint "
                f"(e.g. openai/clip-vit-large-patch14)."
            )
        return self


class EtaSceneConfig(Config):
    """One photo, the style prompt every arm samples with, and the judging text.

    Args:
        photo: id of an entry in `experiments/assets/photos.json`.
        style_prompt: the plain style description — the one prompt every arm of
            the eta sweep is sampled with.
        eval_prompt: the neutral style text the adherence metric scores every
            arm against, never varied across arms.
    """

    photo: Annotated[str, Field(min_length=1)]
    style_prompt: Annotated[str, Field(min_length=1)]
    eval_prompt: Annotated[str, Field(min_length=1)]


class FmrgEtaConfig(ExperimentConfig):
    """The guidance-strength sweep: one unguided baseline beside guided arms per eta.

    Everything but eta is held at the sampler's one operating point — same
    prompt, same seed, same init strength — so the columns read as the price
    and the payoff of pushing the depth reward harder.

    Args:
        scenes: each photo with its style prompt; adherence is scored against
            the scene's neutral `eval_prompt` for every arm.
        etas: the `AnnealedSignTilt` strengths the guided arms sweep, each
            positive; the unguided baseline always runs beside them at eta 0.
        resolution: side of the square the photos are centre-cropped and resized
            to, a multiple of 16 (FLUX's VAE downsamples by 8 and packs 2x2).
    """

    study: Literal["fmrg_eta"]
    scenes: Annotated[tuple[EtaSceneConfig, ...], Field(min_length=1)]
    etas: Annotated[tuple[float, ...], Field(min_length=1)] = (5.0, 15.0, 45.0)
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

    @field_validator("etas")
    @classmethod
    def _etas_are_guided_strengths(cls, etas: tuple[float, ...]) -> tuple[float, ...]:
        for eta in etas:
            if eta <= 0.0:
                raise ValueError(
                    f"etas must be positive, got {eta}: the unguided eta-0 "
                    f"baseline already runs beside every sweep"
                )
        return etas

    @model_validator(mode="after")
    def _adherence_is_never_the_guidance_clip(self) -> FmrgEtaConfig:
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


class KontextSceneConfig(EtaSceneConfig):
    """An eta-study scene, edited under Kontext's instruction framing.

    The steering prompt is the owner's instruction template around the scene's
    style descriptors — renovate into the style, keep the shell — while the
    judges read the neutral `eval_prompt` as every study's do.
    """

    @property
    def instruction_prompt(self) -> str:
        """The owner's instruction template around the scene's style descriptors."""
        return (
            f"renovate this room into {self.style_prompt}, keep the windows, "
            f"walls, floor and ceiling the same"
        )


class KontextBackendConfig(Config):
    """The FLUX.1-Kontext-dev stack a `kontext` study drives.

    Args:
        kind: `real` loads Kontext-dev through the staged
            `roomgen_guide.adapters.kontext_flux` loader; `fake` swaps in the
            weightless `ToyFlowMapBackend` the flow-map tests walk, which is
            what `conf/kontext_smoke.yaml` selects — the toy has no reference
            conditioning, so the smoke exercises the harness, not the editing.
        dtype: the transformer's and VAE's parameter dtype. The walk's state is
            float32 regardless (`FlowMapProcess`), and the depth reward is
            evaluated in the depth backend's own dtype.
        kontext_checkpoint: Hub id (or local path) of the Kontext-dev weights,
            resolved against the `ROOMGEN_HF_HOME` cache offline — the run
            never re-downloads a 24GB checkpoint mid-study.
    """

    kind: Literal["real", "fake"]
    device: Annotated[str, Field(min_length=1)] = "cpu"
    deterministic: bool = False
    dtype: DTypeName = "bfloat16"
    depth_dtype: DTypeName = "float32"
    segmentation_dtype: DTypeName = "float32"
    kontext_checkpoint: Annotated[str, Field(min_length=1)] = DEFAULT_KONTEXT_CHECKPOINT
    depth_checkpoint: Annotated[str, Field(min_length=1)] = DEPTH_CHECKPOINT
    segmentation_checkpoint: Annotated[str, Field(min_length=1)] = (
        SEGMENTATION_CHECKPOINT
    )


class KontextSamplerConfig(Config):
    """Kontext-dev's native sampling budget, identical for both arms.

    Args:
        steps: rectified-flow Euler steps from pure noise; 28 is the pipeline's
            own default and the walk truncates nothing — Kontext's trajectory
            is anchored by the reference tokens, not by an SDEdit init.
        text_guidance_scale: Kontext's embedded guidance — conditioning, not
            reward steering. 2.5 is Black Forest Labs' editing operating point
            (diffusers' signature default of 3.5 is inherited from FLUX.1-dev).
    """

    steps: Annotated[int, Field(ge=1)] = 28
    text_guidance_scale: Annotated[float, Field(ge=0.0)] = (
        DEFAULT_KONTEXT_GUIDANCE_SCALE
    )


class KontextGuidanceConfig(Config):
    """The geometric guidance term on Kontext, in FMRG's annealed step rule.

    Args:
        eta: `AnnealedSignTilt`'s strength; 0 is the unguided arm.
        iterations: FMRG's `n_opt` inner loop, model calls per guided step.
        erosion_radius: pixels trimmed either side of a segment edge before a
            shell mask is trusted.
        resegment: score depth only on reference-shell AND prediction-shell.
        guided_steps: steer only the first this many steps; `None` steers all.
        max_sigma: steer only the steps whose sigma is at or below this — the
            fallback the endpoint-sharpness strip is there to calibrate: if the
            early one-step endpoints turn out to be garbage, the reward starts
            where they sharpen. `None` applies no sigma window.
    """

    eta: Annotated[float, Field(ge=0.0)] = 5.0
    iterations: Annotated[int, Field(ge=1)] = 1
    erosion_radius: Annotated[int, Field(ge=0)] = 2
    resegment: bool = True
    guided_steps: Annotated[int, Field(ge=0)] | None = None
    max_sigma: Annotated[float, Field(gt=0.0, le=1.0)] | None = None


class KontextConfig(ExperimentConfig):
    """The Kontext pilot: a strong instruction-editing prior, with and without depth.

    Per scene at each seed off one transformer load: `kontext` — the
    instruction prompt with the photo as reference tokens, no reward — and
    `kontext_depth`, the same walk steered by the geometric reward against the
    furnished photo's shell depth. The unguided arm is the powerful-prior
    baseline every delta divides by. Both arms additionally leave an
    endpoint-sharpness strip: the decoded one-step endpoint estimate every
    `endpoint_strip_every` steps with its depth error against the reference
    shell — the direct evidence on whether the estimate is a usable steering
    signal at each noise level, and the curve of the geometry converging
    under the reward beside the prior's own.

    Args:
        scenes: each photo with its style text; adherence and preference are
            scored against the scene's neutral `eval_prompt` for every arm.
        resolution: side of the square the photos are centre-cropped and
            resized to, a multiple of 16 (FLUX's VAE downsamples by 8 and
            packs 2x2). Kontext trained around 1024; 512 is the pilot setting
            and the report says which survived.
        image_similarity_checkpoint: the eval judge of render-to-photo cosine
            (`None` skips it) — read beside adherence, since a photo copy
            scores near 1 on it.
        endpoint_strip_every: save each arm's decoded endpoint estimate every
            this many steps.
        fmrg_comparison_run: name of a finished `fmrg_redux` run under the
            working directory whose text-only (`s0u`/`s0g`) arms are quoted
            beside this study's, with their depth errors recomputed against
            the furnished photo's shell so the column compares; `None` skips
            the comparison (the smoke setting).
    """

    study: Literal["kontext"]
    scenes: Annotated[tuple[KontextSceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=FLUX_LATENT_ALIGNMENT)] = 512
    backends: KontextBackendConfig
    sampler: KontextSamplerConfig = KontextSamplerConfig()
    guidance: KontextGuidanceConfig = KontextGuidanceConfig()
    adherence: AdherenceConfig
    image_similarity_checkpoint: Annotated[str, Field(min_length=1)] | None = (
        IMAGE_SIMILARITY_CHECKPOINT
    )
    preference: PreferenceConfig
    endpoint_strip_every: Annotated[int, Field(ge=1)] = 4
    fmrg_comparison_run: Annotated[str, Field(min_length=1)] | None = "0002_fmrg_redux"

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_packed_latent_grid(cls, resolution: int) -> int:
        if resolution % FLUX_LATENT_ALIGNMENT:
            raise ValueError(
                f"resolution must be a multiple of {FLUX_LATENT_ALIGNMENT} for "
                f"FLUX's packed latent grid, got {resolution}"
            )
        return resolution

    @model_validator(mode="after")
    def _no_judge_is_the_guidance_clip(self) -> KontextConfig:
        # The eval CLIPs must stay independent of the CLIP any guidance energy in
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
        if self.image_similarity_checkpoint == GUIDANCE_CLIP_CHECKPOINT:
            raise ValueError(
                f"image_similarity_checkpoint is "
                f"{self.image_similarity_checkpoint!r}, the guidance CLIP: "
                f"judge image similarity with an independent checkpoint "
                f"(e.g. openai/clip-vit-large-patch14)."
            )
        return self


class FmrgAlignmentBackendConfig(FmrgBackendConfig):
    """The FMRG stack plus the metric depth backend the alignment arms steer by.

    Args:
        metric_depth_checkpoint: the Depth Anything 3 *metric* tier — the only
            backend for which raw or shift-only comparison is meaningful at all.
            `depth_checkpoint` (V2, affine-invariant) stays loaded beside it as
            the neutral second judge every arm is also scored under.
        metric_input_size: the square edge DA3 resamples to, a multiple of its
            14-pixel patch.
        metric_depth_dtype: DA3's parameter dtype, separate from `depth_dtype`
            so the affine-invariant judge can stay float32 while the metric
            backend — whose guided backward has to share a 32GB card with
            FLUX's — drops to bfloat16. The backend casts its output back to
            the image's dtype before the reciprocal, so the reward's energy
            stays float32 either way.
    """

    metric_depth_checkpoint: Annotated[str, Field(min_length=1)] = (
        METRIC_DEPTH_CHECKPOINT
    )
    metric_input_size: Annotated[int, Field(ge=14)] = METRIC_INPUT_SIZE
    metric_depth_dtype: DTypeName = "float32"


class FmrgAlignmentGuidanceConfig(FmrgGuidanceConfig):
    """FMRG's guidance term, run once per alignment mode under test.

    Args:
        arms: the alignment modes the guided arms are steered under, one arm
            per mode at every (scene, seed), beside the shared unguided control.
            Loosening the fit is only meaningful for the metric backend the
            study steers by; the affine-invariant second judge always scores
            under the full fit.
    """

    arms: Annotated[tuple[AlignmentMode, ...], Field(min_length=1)] = ("none", "shift")


class FmrgAlignmentConfig(ExperimentConfig):
    """The DA3 alignment-mode study on FMRG: raw against shift-only, beside one control.

    The SD1.5 run of this claim predates the FMRG-only ruling, so the study is
    re-measured on the FLUX flow map: per (scene, seed) one unguided control and
    one guided arm per alignment mode, all steered by DA3 metric at one eta.
    Every arm is scored under *all three* modes plus the affine-invariant DA2
    judge, so no arm is read solely through its own objective.

    Args:
        scenes: each photo with its style prompt; adherence is scored against
            the scene's neutral `eval_prompt` for every arm.
        resolution: side of the square the photos are centre-cropped and resized
            to, a multiple of 16 (FLUX's VAE downsamples by 8 and packs 2x2).
    """

    study: Literal["fmrg_da3_alignment"]
    scenes: Annotated[tuple[EtaSceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=FLUX_LATENT_ALIGNMENT)] = 512
    backends: FmrgAlignmentBackendConfig
    sampler: FmrgSamplerConfig = FmrgSamplerConfig()
    guidance: FmrgAlignmentGuidanceConfig = FmrgAlignmentGuidanceConfig()
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

    @model_validator(mode="after")
    def _adherence_is_never_the_guidance_clip(self) -> FmrgAlignmentConfig:
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


ImageModelName = Literal["clip", "dinov2", "siglip"]


class ImageSceneConfig(EtaSceneConfig):
    """An eta-study scene plus the instruction-prompt arm's removal target.

    The image study's instruction arms steer with the scene's style content
    rephrased as an explicit renovation instruction — keep the shell, remove one
    named object — so each scene names the object and the ADE20K ids the
    removal-compliance metric counts it under.

    Args:
        removal_object: the object the instruction prompt asks the renovation to
            remove, named as it appears in the photo (e.g. `coffee table`).
        removal_classes: ADE20K-150 model-output ids the removal-compliance
            metric counts as that object; every id must be a contents class,
            because asking a renovation to remove a piece of the shell would
            contradict the same prompt's "keep" clause and the depth reward.
        removal_substituted: true when the photo holds no table or desk at all
            and the prompt removes the scene's most removable large furniture
            item instead — recorded so the report can say so.
    """

    removal_object: Annotated[str, Field(min_length=1)]
    removal_classes: Annotated[tuple[int, ...], Field(min_length=1)]
    removal_substituted: bool = False

    @field_validator("removal_classes")
    @classmethod
    def _removal_targets_are_contents(cls, ids: tuple[int, ...]) -> tuple[int, ...]:
        outside = [
            identifier for identifier in ids if identifier not in CONTENTS_CLASSES
        ]
        if outside:
            raise ValueError(
                f"removal_classes {outside} are not ADE20K contents classes: the "
                f"instruction removes furniture, never shell, so a non-contents "
                f"id would score the wrong pixels as compliance"
            )
        return ids

    @property
    def instruction_prompt(self) -> str:
        """The owner's template around the scene's style descriptors, verbatim."""
        return (
            f"renovate this room into {self.style_prompt}, keep the windows, "
            f"walls, floor and ceiling the same, remove the "
            f"{self.removal_object} in the room"
        )


class ImageGuidanceConfig(Config):
    """Which frozen vision models steer the render toward the reference photo.

    Args:
        models: the helper models under comparison, one image-only guided arm
            each. `clip`/`siglip` descend a pooled-embedding cosine, `dinov2` a
            cosine over its spatial patch-token grid.
        composite_models: the models that also run an image-plus-depth arm; a
            subset of `models`, so every composite has its image-only twin.
        depth_weight: the depth term's weight beside an image term held at 1.0;
            `None` measures it per (scene, model) as the ratio of the two
            terms' gradient RMS at a midpoint step, the unit conversion between
            a cosine and a masked squared disparity.
        sizing_fraction: how far into the guided window the sizing gradient is
            taken, on a real partially denoised sample.
        siglip_checkpoint: defaults to FLUX.1-Redux-dev, whose snapshot already
            carries the SigLIP tower — no extra download.
        siglip_dtype: the so400m tower's parameter dtype, separate from the
            other helpers' (which run in `backends.depth_dtype`): at float32
            its composite arm — the 428M tower beside FLUX, the depth term's
            backward and the per-step resegmentation — allocates past a 32GB
            card's ceiling, measured, not estimated. Half precision costs the
            walk nothing it can see: the energy still scores and reduces in
            float32, and FMRG's tilt keeps only the gradient's sign.
    """

    models: Annotated[tuple[ImageModelName, ...], Field(min_length=1)] = (
        "clip",
        "dinov2",
        "siglip",
    )
    composite_models: tuple[ImageModelName, ...] = ("clip", "dinov2")
    depth_weight: Annotated[float, Field(gt=0.0)] | None = None
    sizing_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    clip_checkpoint: Annotated[str, Field(min_length=1)] = GUIDANCE_CLIP_CHECKPOINT
    dinov2_checkpoint: Annotated[str, Field(min_length=1)] = DEFAULT_DINOV2_CHECKPOINT
    siglip_checkpoint: Annotated[str, Field(min_length=1)] = DEFAULT_SIGLIP_CHECKPOINT
    siglip_dtype: DTypeName = "bfloat16"

    @model_validator(mode="after")
    def _every_composite_has_its_image_only_twin(self) -> ImageGuidanceConfig:
        missing = [model for model in self.composite_models if model not in self.models]
        if missing:
            raise ValueError(
                f"composite_models {missing} are not in models: an image-plus-"
                f"depth arm is only readable beside the same model's image-only "
                f"arm"
            )
        return self

    def steering_checkpoints(self) -> tuple[str, ...]:
        """Every checkpoint some arm descends — what no eval judge may share."""
        return (self.clip_checkpoint, self.dinov2_checkpoint, self.siglip_checkpoint)


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


class FmrgImageConfig(ExperimentConfig):
    """The image-reward study: the user's photo at the reward point, per helper model.

    Per scene at one seed off one transformer load: an unguided control, a
    depth-only anchor, one image-only guided arm per helper model, and an
    image-plus-depth composite (measured weight) for each `composite_models`
    entry. Every arm is scored on the one pixel set all arms left visible, by
    judges independent of every feature space any arm descended.

    Args:
        image_similarity_checkpoint: the eval judge of render-to-photo cosine;
            `None` skips the metric, which is what the offline smoke variant
            does. Never a checkpoint any arm steers by.
    """

    study: Literal["fmrg_image"]
    scenes: Annotated[tuple[ImageSceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=FLUX_LATENT_ALIGNMENT)] = 512
    backends: FmrgBackendConfig
    sampler: FmrgSamplerConfig = FmrgSamplerConfig()
    guidance: FmrgGuidanceConfig = FmrgGuidanceConfig()
    image: ImageGuidanceConfig = ImageGuidanceConfig()
    adherence: AdherenceConfig
    image_similarity_checkpoint: Annotated[str, Field(min_length=1)] | None = (
        IMAGE_SIMILARITY_CHECKPOINT
    )
    preference: PreferenceConfig

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_packed_latent_grid(cls, resolution: int) -> int:
        if resolution % FLUX_LATENT_ALIGNMENT:
            raise ValueError(
                f"resolution must be a multiple of {FLUX_LATENT_ALIGNMENT} for "
                f"FLUX's packed latent grid, got {resolution}"
            )
        return resolution

    @model_validator(mode="after")
    def _no_judge_shares_a_steered_checkpoint(self) -> FmrgImageConfig:
        # The self-grading rule: the checkpoint an energy descends must never be
        # the checkpoint that scores it. Both judges are checked against every
        # steering checkpoint, plus the repo's guidance CLIP as the standing
        # pattern from every other study.
        steered = {*self.image.steering_checkpoints(), GUIDANCE_CLIP_CHECKPOINT}
        if self.adherence.kind == "clip" and self.adherence.checkpoint in steered:
            raise ValueError(
                f"adherence.checkpoint is {self.adherence.checkpoint!r}, a "
                f"checkpoint some arm descends: score adherence with an "
                f"independent checkpoint (e.g. openai/clip-vit-large-patch14)."
            )
        if (
            self.image_similarity_checkpoint is not None
            and self.image_similarity_checkpoint in steered
        ):
            raise ValueError(
                f"image_similarity_checkpoint is "
                f"{self.image_similarity_checkpoint!r}, a checkpoint some arm "
                f"descends: judge image similarity with an independent "
                f"checkpoint (e.g. openai/clip-vit-large-patch14)."
            )
        return self


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
    | FmrgReduxConfig
    | FmrgEtaConfig
    | FmrgAlignmentConfig
    | FmrgImageConfig
    | KontextConfig
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
        case "fmrg_redux":
            return FmrgReduxConfig.from_file(path, overrides)
        case "fmrg_eta":
            return FmrgEtaConfig.from_file(path, overrides)
        case "fmrg_da3_alignment":
            return FmrgAlignmentConfig.from_file(path, overrides)
        case "fmrg_image":
            return FmrgImageConfig.from_file(path, overrides)
        case "kontext":
            return KontextConfig.from_file(path, overrides)
        case "framing":
            return FramingConfig.from_file(path, overrides)
        case "da3_alignment":
            return AlignmentStudyConfig.from_file(path, overrides)
        case _:
            raise ValueError(
                f"{path} declares unknown study {study!r}; "
                "known studies: benchmark, fmrg, fmrg_context, fmrg_redux, "
                "fmrg_eta, fmrg_da3_alignment, fmrg_image, kontext, framing, "
                "da3_alignment"
            )
