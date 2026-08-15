"""The typed configuration of the strip-and-furnish study; one variant per `conf/` file."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import Field, field_validator, model_validator

from roomgen_geometry.depth.depth_anything import DEFAULT_CHECKPOINT as DEPTH_CHECKPOINT
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide.adapters.fmrg_flux import DEFAULT_REDUX_CHECKPOINT
from roomgen_guide.config import Config
from roomgen_guide.eval import (
    DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT,
)
from roomgen_guide.eval import (
    DEFAULT_IMAGE_SIMILARITY_CHECKPOINT,
)
from roomgen_guide.experiment import ExperimentConfig
from roomgen_guide.guidance.clip import DEFAULT_CHECKPOINT as TEXT_ENERGY_CHECKPOINT
from roomgen_guide.guidance.image_reference import DEFAULT_DINOV2_CHECKPOINT
from roomgen_guide.renovate import DEFAULT_CHECKPOINT as DIFFUSION_CHECKPOINT
from roomgen_guide.sampling.inpaint import DEFAULT_CHECKPOINT as INPAINTING_CHECKPOINT

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


class SceneConfig(Config):
    """One photograph, stripped bare and then furnished under one prompt.

    Args:
        photo: id of an entry in `experiments/assets/photos.json`.
        prompt: what stage 2 furnishes the stripped room with, held fixed across
            seeds and arms so the arms differ in their guidance energy alone.
    """

    photo: Annotated[str, Field(min_length=1)]
    prompt: Annotated[str, Field(min_length=1)]


class BackendConfig(Config):
    """Which model stack the study drives, where it lives, and in what precision.

    Args:
        kind: `real` loads the pretrained checkpoints; `fake` swaps in the
            weightless doubles the test suites run against, which is what
            `conf/smoke.yaml` selects to exercise both stages on CPU in seconds.
        device: torch device string; every generator is created on it, so a CUDA
            run is reproducible from its seeds the same way a CPU one is.
        dtype: parameter dtype of the UNet and VAE, and the dtype the photo is
            cast to.
        depth_dtype: parameter dtype of the depth backend, independent of
            `dtype`.
        segmentation_dtype: parameter dtype of the segmenter, which is off the
            gradient path entirely and only decides the masks.
        deterministic: ask torch for deterministic kernels before the stack is
            built (`roomgen_guide.determinism`), which must happen before the
            process's first CUDA allocation to take effect on cuBLAS.
        scheduler: `ddim` replaces the checkpoint's PNDM with the DDIM whose
            update is exactly the x̂₀ closed form the guided loop assumes;
            `pipeline` keeps what the pipeline was built with, which is the only
            option the weightless fake stack has.
        text_energy_checkpoint: the CLIP the *guidance* is steered by. It must
            differ from the adherence checkpoint, which is what the study
            asserts rather than trusts.
    """

    kind: Literal["real", "fake"]
    device: Annotated[str, Field(min_length=1)] = "cpu"
    deterministic: bool = False
    dtype: DTypeName = "float32"
    depth_dtype: DTypeName = "float32"
    segmentation_dtype: DTypeName = "float32"
    scheduler: Literal["ddim", "pipeline"] = "ddim"
    diffusion_checkpoint: Annotated[str, Field(min_length=1)] = DIFFUSION_CHECKPOINT
    inpainting_checkpoint: Annotated[str, Field(min_length=1)] = INPAINTING_CHECKPOINT
    depth_checkpoint: Annotated[str, Field(min_length=1)] = DEPTH_CHECKPOINT
    segmentation_checkpoint: Annotated[str, Field(min_length=1)] = (
        SEGMENTATION_CHECKPOINT
    )
    text_energy_checkpoint: Annotated[str, Field(min_length=1)] = TEXT_ENERGY_CHECKPOINT


class StripConfig(Config):
    """Stage 1: what the segmentation hands the inpainter, and how it repaints it.

    Args:
        prompt: what the holes are filled with — a bare room, never a renovation.
        negative_prompt: what the fill must not contain. A hole this size is
            filled from the furnished room around it, so an inpainter asked only
            for a bare room paints the furniture straight back in.
        dilation_radius: pixels the contents mask is grown by before it is
            inpainted, and the margin a later pass's non-shell region is grown
            by before it is repainted. Contact shadows and segment fringes sit
            just outside the segment, so a mask that stops at the boundary
            leaves the inpainter the evidence it needs to paint the object
            straight back.
        passes: how many times the segmenter is asked again. An inpainter fills
            a hole from the room around it and puts objects back into it, so one
            pass is a strip that leaves a table where a sofa was; each further
            pass repaints whatever *inside the first mask* the segmenter does
            not yet read as shell — wall, floor, ceiling, door, windowpane. The
            acceptance is inverted on purpose: a fill only had to dodge the
            named contents list to survive a furniture check, and a radiator or
            a vacuum-shaped object slipped through; not-yet-shell is
            categorical. Repainting stays inside the first mask, which is what
            keeps the untouched part of the photograph untouched however many
            passes run.
        steps: denoising steps of the inpainting pipeline, per pass.
        fill_strength: how much of each pass's trajectory repaints the hole,
            which is first flooded with the colours of the surfaces around it.
            At 1 the model invents the region from the furnished room around it
            and puts a television on the wall it was told to leave bare; below 1
            it starts from that flood and can only smooth what is already a wall
            or a floor, at the cost of a softer patch.
        guidance_scale: classifier-free guidance on `prompt`. There is no
            geometric guidance in this stage at all: the room is held by the
            pixels outside the mask, which are never regenerated.
    """

    prompt: Annotated[str, Field(min_length=1)]
    negative_prompt: Annotated[str, Field(min_length=1)] | None = None
    dilation_radius: Annotated[int, Field(ge=0)] = 8
    passes: Annotated[int, Field(ge=1)] = 3
    steps: Annotated[int, Field(ge=1)] = 40
    fill_strength: Annotated[float, Field(gt=0.0, le=1.0)] = 0.6
    guidance_scale: Annotated[float, Field(ge=1.0)] = 7.5


class FurnishConfig(Config):
    """Stage 2: the img2img budget and the two energies compared over it.

    Args:
        steps: denoising steps the scheduler is set to.
        noise_strength: how far up the trajectory the *stripped* room is
            renoised (SDEdit). Furnishing an empty room needs enough of the
            trajectory to put objects in it, which is what makes holding the
            shell a real constraint rather than a formality.
        text_guidance_scale: classifier-free guidance on the scene prompt,
            identical in every arm — the prompt is in the conditioning of all
            three, and only the guidance energy differs.
        strength: what the energy gradient is multiplied by before it moves the
            latent, at every step. Shared by both guided arms, so arm B is arm A
            plus a depth term rather than arm A at a different temperature.
        clip_step_rms: ceiling on the RMS of the raw step, in latent-RMS units.
        depth_weight: the depth term's weight in the composite energy, the text
            term's being 1. `null` measures it instead: the ratio of the two
            terms' gradient RMS on a real sample at a representative sigma, so
            the two pull equally hard rather than in whatever units they happen
            to be denominated in.
        sizing_fraction: how far into the walk the sizing measurement is taken,
            as a share of the walked steps.
        erosion_radius: pixels trimmed either side of a segment edge before a
            shell mask is trusted.
        resegment: score depth on the empty room's shell AND the prediction's
            own, so furniture the furnishing adds may occlude the shell for free.
    """

    steps: Annotated[int, Field(ge=1)] = 50
    noise_strength: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
    text_guidance_scale: Annotated[float, Field(ge=1.0)] = 7.5
    strength: Annotated[float, Field(ge=0.0)]
    clip_step_rms: Annotated[float, Field(gt=0.0)] | None = None
    depth_weight: Annotated[float, Field(ge=0.0)] | None = None
    sizing_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    erosion_radius: Annotated[int, Field(ge=0)] = 2
    resegment: bool = True


class AdherenceConfig(Config):
    """Which metric reports how much of the furnishing the guidance left standing.

    Args:
        kind: `clip` is the CLIP prompt score; `contents_change` is
            checkpoint-free and is what the offline smoke variant uses.
        checkpoint: Hugging Face Hub id, read only by `clip`, and never the
            checkpoint the text energy steers by.
    """

    kind: Literal["clip", "contents_change"]
    checkpoint: Annotated[str, Field(min_length=1)] = ADHERENCE_CHECKPOINT


class StudyConfig(ExperimentConfig):
    """One variant of 0003: scenes stripped by segmentation, then furnished by two energies.

    Args:
        scenes: the (photo, prompt) pairs the study measures, each at every seed.
        resolution: side of the square the photos are centre-cropped and resized
            to, a multiple of 64.
    """

    scenes: Annotated[tuple[SceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=LATENT_ALIGNMENT)] = 512
    backends: BackendConfig
    strip: StripConfig
    furnish: FurnishConfig
    adherence: AdherenceConfig

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_latent_grid(cls, resolution: int) -> int:
        if resolution % LATENT_ALIGNMENT:
            raise ValueError(
                f"resolution must be a multiple of {LATENT_ALIGNMENT} for Stable "
                f"Diffusion 1.5's latent grid, got {resolution}"
            )
        return resolution

    @model_validator(mode="after")
    def _adherence_is_not_graded_by_the_model_it_optimised(self) -> Self:
        # Arm A descends a CLIP energy directly, so scoring adherence with that
        # same checkpoint would report how well the sampler hit its own target
        # and call it prompt adherence.
        if (
            self.adherence.kind == "clip"
            and self.adherence.checkpoint == self.backends.text_energy_checkpoint
        ):
            raise ValueError(
                f"adherence.checkpoint is {self.adherence.checkpoint!r}, the very "
                f"checkpoint the text energy is steered by: a guided arm would be "
                f"graded on the model it optimised. Score with a different CLIP."
            )
        return self


class FluxConfig(Config):
    """The FLUX flow-map stack the FMRG furnish stage drives, and where its pieces live.

    Only the furnish stage touches it: stage 1 keeps the SD-inpainting stack of
    `BackendConfig`, whose `kind` also decides whether this one loads real
    weights or the weightless `ToyFlowMapBackend` the adapter's tests walk.

    Args:
        dtype: the transformer's and VAE's parameter dtype; the walk's state is
            float32 regardless (`FlowMapProcess`).
        flux_checkpoint: Hub id (or local path) of the FLUX base weights,
            resolved against the `ROOMGEN_HF_HOME` cache offline.
        lora_dir: directory holding the FlowMap LoRA's weights for the sampled
            resolution; `None` means `<ROOMGEN_HF_HOME>/fmrg/flux-flowmap-lora-512`.
        fmrg_path: the vendored FMRG checkout, reached through `sys.path`;
            `None` means this repo's `packages/fmrg`.
        text_guidance_scale: FLUX.1-dev's embedded guidance — conditioning, not
            reward steering.
    """

    dtype: DTypeName = "bfloat16"
    flux_checkpoint: Annotated[str, Field(min_length=1)] = (
        "black-forest-labs/FLUX.1-dev"
    )
    lora_dir: Path | None = None
    fmrg_path: Path | None = None
    text_guidance_scale: Annotated[float, Field(ge=0.0)] = 3.5


class FmrgFurnishConfig(Config):
    """Stage 2 on FMRG: the flow-map walk over the empty room, and the depth reward on it.

    The reference the guided walk descends is the *stripped empty room's* shell
    depth — segmented and depth-read after stage 1 — because the empty room's
    shell is nearly the whole frame, so the valid comparison area is much larger
    than the photo-shell ∩ prediction-shell intersection the retained-photo
    protocol scored. FMRG's text conditioning is native, so the SD1.5 study's
    separate CLIP text energy has no counterpart here and is deliberately absent.

    Args:
        grid_steps: resolution of the dense sigma grid; the walk never visits
            most of it.
        early_stop: how many of the grid's sigmas the walk traverses before the
            jump to data — FMRG's guidance window.
        unguided_steps: how many steps the final jump is split into, unguided.
        start_sigma: how far up the interpolant the *empty room* is renoised,
            pre-shift; high enough that the room is actually furnished (lower
            sigmas mostly copy the input back).
        probe_sigmas: extra guided walks per scene at these start sigmas,
            clearly labelled as probes; the control runs at `start_sigma` only.
        eta: FMRG's step size — `AnnealedSignTilt.strength`; 0 would be the
            control, which the study runs beside every guided arm anyway.
        iterations: FMRG's `n_opt` inner loop, model calls per guided step.
        erosion_radius: pixels trimmed either side of a segment edge before a
            shell mask is trusted, on the reference and the per-step prediction
            alike.
        resegment: score depth only on empty-room-shell AND prediction-shell,
            so furniture the furnishing adds is excused rather than punished —
            scoring furniture pixels against empty-room depth would penalise
            furnishing itself.
    """

    grid_steps: Annotated[int, Field(ge=1)] = 208
    early_stop: Annotated[int, Field(ge=2)] = 51
    unguided_steps: Annotated[int, Field(ge=1)] = 1
    start_sigma: Annotated[float, Field(gt=0.0, le=1.0)] = 0.9
    probe_sigmas: tuple[Annotated[float, Field(gt=0.0, le=1.0)], ...] = ()
    stepping: Literal["endpoint_reuse", "two_time"] = "endpoint_reuse"
    differentiate_through_velocity: bool = False
    eta: Annotated[float, Field(gt=0.0)] = 5.0
    iterations: Annotated[int, Field(ge=1)] = 1
    erosion_radius: Annotated[int, Field(ge=0)] = 2
    resegment: bool = True

    @model_validator(mode="after")
    def _the_window_stops_inside_the_grid(self) -> Self:
        if self.early_stop > self.grid_steps:
            raise ValueError(
                f"early_stop {self.early_stop} is past the end of a "
                f"{self.grid_steps}-step grid: the window has to stop inside "
                f"the walk it truncates"
            )
        return self


class FmrgStudyConfig(ExperimentConfig):
    """The two-stage study with its furnish stage on FMRG's FLUX flow map.

    Stage 1 is byte-identical to `StudyConfig`'s: EoMT decides, SD-inpainting
    fills, the acceptance is the inverted shell check. Stage 2 swaps SD1.5's
    guided img2img for the FMRG walk of `FmrgFurnishConfig`, steered by the
    stripped empty room's shell depth.

    Args:
        scenes: the (photo, prompt) pairs the study measures, each at every seed.
        resolution: side of the square the photos are centre-cropped and resized
            to; a multiple of 64 satisfies both SD-inpainting's latent grid and
            FLUX's packed one.
        backends: stage 1's stack — and its `kind` decides real-vs-fake for the
            FLUX stack too.
    """

    study: Literal["fmrg"]
    scenes: Annotated[tuple[SceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=LATENT_ALIGNMENT)] = 512
    backends: BackendConfig
    strip: StripConfig
    furnish: FmrgFurnishConfig
    flux: FluxConfig = FluxConfig()
    adherence: AdherenceConfig

    @field_validator("resolution")
    @classmethod
    def _aligned_to_the_latent_grid(cls, resolution: int) -> int:
        if resolution % LATENT_ALIGNMENT:
            raise ValueError(
                f"resolution must be a multiple of {LATENT_ALIGNMENT} for the "
                f"inpainter's latent grid, got {resolution}"
            )
        return resolution


class ReduxContextConfig(Config):
    """The stripped empty room handed to the *prior* as FLUX.1-Redux image context.

    Args:
        scale: multiplier on the image tokens alone, the text tokens untouched
            (`register_image_context_prompt`). 1.0 is diffusers' composition and
            is known to drown the prompt; 0.3 is the operating point the Redux
            study found keeps the text audible.
        checkpoint: the SigLIP + Redux-adapter weights, gated like FLUX.1-dev
            and resolved from the same cache.
    """

    scale: Annotated[float, Field(ge=0.0)] = 0.3
    checkpoint: Annotated[str, Field(min_length=1)] = DEFAULT_REDUX_CHECKPOINT


class ImageRewardConfig(Config):
    """The image-reward channel: DINOv2 patch cosine toward the empty room, shell patches only.

    The reference is the *empty room*, which contains no furniture, so a
    full-frame similarity would punish furnishing itself; the energy is
    restricted to patch positions the empty room's shell mask names, and
    intersected per step with the prediction's own resegmented shell exactly as
    the depth reward's pixel mask is.

    Args:
        depth_weight: the depth term's weight beside the image term's 1.0 in
            the composite arm; `None` measures it as the ratio of the two
            terms' gradient RMS at a midpoint step of the real walk — the unit
            conversion between a patch cosine and a masked squared disparity.
        sizing_fraction: how far into the guided window that measurement is
            taken.
        min_patch_coverage: share of a patch's pixels the shell must cover
            before the patch is scored, on the reference and the per-step
            prediction alike.
    """

    dinov2_checkpoint: Annotated[str, Field(min_length=1)] = DEFAULT_DINOV2_CHECKPOINT
    depth_weight: Annotated[float, Field(gt=0.0)] | None = None
    sizing_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    min_patch_coverage: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5


class PreferenceConfig(Config):
    """Which human-preference model scores the finished renders — eval only, never guidance.

    Args:
        kind: `pickscore` loads PickScore v1, a CLIP-H fine-tuned on human
            choices; `contrast` is a checkpoint-free stand-in (per-image pixel
            spread) that lets the smoke variant exercise the plumbing offline.
        checkpoint: Hub id of the scoring model, read only by `pickscore`.
        processor_checkpoint: Hub id the tokenizer and image statistics come
            from — PickScore ships no processor of its own.
    """

    kind: Literal["pickscore", "contrast"]
    checkpoint: Annotated[str, Field(min_length=1)] = "yuvalkirstain/PickScore_v1"
    processor_checkpoint: Annotated[str, Field(min_length=1)] = (
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )


class FmrgReduxConfig(FmrgStudyConfig):
    """The Redux x depth x image-reward comparison over the FMRG furnish stage.

    The strip and the walk are `FmrgStudyConfig`'s unchanged; what this variant
    adds is the arm grid — the stripped empty room injected as Redux context
    into the prior, the depth reward, and the shell-masked DINOv2 image reward,
    in every combination the owner named — plus the judges the extra readouts
    need. Every furnish arm steers under one instruction prompt built from the
    scene's style descriptors; the neutral style text stays the eval prompt.

    Args:
        image_similarity_checkpoint: the eval judge of render-to-empty-room and
            render-to-photo cosine; `None` skips the metric (the offline smoke
            variant). Never the checkpoint the image reward descends.
    """

    study: Literal["fmrg_redux"]  # type: ignore[assignment]
    redux: ReduxContextConfig = ReduxContextConfig()
    image: ImageRewardConfig = ImageRewardConfig()
    image_similarity_checkpoint: Annotated[str, Field(min_length=1)] | None = (
        DEFAULT_IMAGE_SIMILARITY_CHECKPOINT
    )
    preference: PreferenceConfig

    @model_validator(mode="after")
    def _no_judge_shares_the_steered_tower(self) -> Self:
        # The self-grading rule: the DINOv2 the image arms descend must never be
        # the model that scores them.
        steered = self.image.dinov2_checkpoint
        if self.adherence.kind == "clip" and self.adherence.checkpoint == steered:
            raise ValueError(
                f"adherence.checkpoint is {steered!r}, the tower the image "
                f"reward descends: score adherence with an independent judge."
            )
        if self.image_similarity_checkpoint == steered:
            raise ValueError(
                f"image_similarity_checkpoint is {steered!r}, the tower the "
                f"image reward descends: judge similarity independently."
            )
        return self


def load_study_config(
    path: Path, overrides: Sequence[str] | None = None
) -> StudyConfig | FmrgStudyConfig:
    """Load whichever study shape `path` declares in its top-level `study` key.

    The key is peeked before validation because the shapes are different
    pydantic models; a missing key means the SD1.5-era `StudyConfig`, which is
    what every variant predating the FMRG furnish stage is.
    """
    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise TypeError(f"Expected a mapping at {path}, got {type(raw).__name__}")
    match raw.get("study"):
        case "fmrg_redux":
            return FmrgReduxConfig.from_file(path, overrides)
        case "fmrg":
            return FmrgStudyConfig.from_file(path, overrides)
        case _:
            return StudyConfig.from_file(path, overrides)
