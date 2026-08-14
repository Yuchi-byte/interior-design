"""The typed configuration of the strip-and-furnish study; one variant per `conf/` file."""

from __future__ import annotations

from typing import Annotated, Final, Literal, Self

import torch
from pydantic import Field, field_validator, model_validator

from roomgen_geometry.depth.depth_anything import DEFAULT_CHECKPOINT as DEPTH_CHECKPOINT
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide.config import Config
from roomgen_guide.eval import DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT
from roomgen_guide.experiment import ExperimentConfig
from roomgen_guide.guidance.clip import DEFAULT_CHECKPOINT as TEXT_ENERGY_CHECKPOINT
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
