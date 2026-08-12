"""The typed configuration of the geometric-guidance benchmark; one variant per `conf/` file."""

from __future__ import annotations

from typing import Annotated, Final, Literal

import torch
from pydantic import Field, field_validator

from roomgen_geometry.depth.depth_anything import DEFAULT_CHECKPOINT as DEPTH_CHECKPOINT
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide.config import Config
from roomgen_guide.eval import DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT
from roomgen_guide.experiment import ExperimentConfig
from roomgen_guide.guidance.schedule import (
    ConstantStrength,
    DecayingStrength,
    GuidanceSchedule,
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
        scheduler: `ddim` replaces the checkpoint's PNDM scheduler with the DDIM
            whose update is exactly the x̂₀ closed form the guided loop assumes;
            `pipeline` keeps whatever scheduler the pipeline was built with,
            which is the only option the weightless fake stack has.
    """

    kind: Literal["real", "fake"]
    device: Annotated[str, Field(min_length=1)] = "cpu"
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
        strength: what the energy gradient is scaled by before it is applied to
            the latent, and the value a `decaying` schedule starts from — the
            knob a strength sweep varies. `normalize` decides its units: a
            multiplier on a gradient in the squared-disparity units the depth
            backend defines, or a step size in latent-RMS units.
        normalize: divide the gradient by its own RMS before the step, bounding
            the per-step push so a large energy cannot run the sampler away into
            a depth-satisfying texture.
        clip: ceiling on the RMS of the raw step, in latent-RMS units: the raw
            step keeps shrinking as the geometry is satisfied, and the cap is
            what stops a large energy from running away. Read only when
            `normalize` is false, which the sampler enforces.
        schedule: `constant` holds `strength` for the whole trajectory;
            `decaying` falls from it to `final_strength` on a cosine, spending
            the steering on the early steps that decide layout.
        final_strength: where a `decaying` schedule lands on the last step;
            read only by that schedule.
        erosion_radius: pixels trimmed either side of a segment edge before the
            shell mask is trusted.
        resegment: segment each predicted image and score depth only on
            reference-shell AND prediction-shell, so furniture the renovation
            adds may occlude the shell for free; false scores the reference
            mask alone (pre-occlusion semantics, the control arm for A/B).
    """

    strength: Annotated[float, Field(ge=0.0)]
    normalize: bool = False
    clip: Annotated[float, Field(gt=0.0)] | None = None
    schedule: Literal["constant", "decaying"] = "constant"
    final_strength: Annotated[float, Field(ge=0.0)] = 0.0
    erosion_radius: Annotated[int, Field(ge=0)] = 2
    resegment: bool = True


def build_schedule(config: GuidanceConfig) -> GuidanceSchedule:
    """The strength schedule `config` selects, for the guided arm of a pair."""
    match config.schedule:
        case "constant":
            return ConstantStrength(strength=config.strength)
        case "decaying":
            return DecayingStrength(
                initial=config.strength, final=config.final_strength
            )


class AdherenceConfig(Config):
    """Which metric stands on the second side of the gate.

    Args:
        kind: `clip` is the CLIP prompt score — the metric gate (b) is meant to
            be. `contents_change` is checkpoint-free: it measures how far the
            render moved from the photo *outside* the shell mask, which is
            weaker (it cannot read the prompt) but still fails the degenerate
            run that renovates nothing, and is what the smoke variant uses.
        checkpoint: Hugging Face Hub id, read only by `clip`.
    """

    kind: Literal["clip", "contents_change"]
    checkpoint: Annotated[str, Field(min_length=1)] = ADHERENCE_CHECKPOINT


class BenchmarkConfig(ExperimentConfig):
    """One variant of 0002: a scene set, a sampler budget, and the gate it is judged on.

    Args:
        scenes: the (photo, prompt) pairs, every one of which the gate must hold on.
        resolution: side of the square the photos are centre-cropped and resized
            to, a multiple of 64.
    """

    scenes: Annotated[tuple[SceneConfig, ...], Field(min_length=1)]
    resolution: Annotated[int, Field(ge=LATENT_ALIGNMENT)] = 512
    backends: BackendConfig
    sampler: SamplerConfig = SamplerConfig()
    guidance: GuidanceConfig
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
