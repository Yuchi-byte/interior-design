"""Stage 2: furnishing a stripped room, under a text energy and under text plus depth.

Three arms per (scene, seed) over one sampler and one prompt: text alone, text
with the masked depth energy beside it, and no guidance at all. The prompt sits
in the classifier-free conditioning of all three, so the only thing that differs
between them is the energy the walk descends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.eval import PromptAdherence, depth_error, shell_coverage
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.clip import ClipTextGuidance
from roomgen_guide.guidance.composite import WeightedGuidance, WeightedTerm
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.metrics import RunMetrics
from roomgen_guide.process.endpoint import StepContext
from roomgen_guide.sampling.guided import LatentDiffusionProcess, sample_guided
from strip_and_furnish.backends import Backends
from strip_and_furnish.config import SceneConfig, StudyConfig
from strip_and_furnish.images import DisparityScale, write_disparity_map, write_image
from strip_and_furnish.strip import StripResult, vae_roundtrip

type Arm = Literal["text_only", "text_and_depth", "control"]

ARMS: Final[tuple[Arm, ...]] = ("text_only", "text_and_depth", "control")


@dataclass(frozen=True)
class Energies:
    """The two energies under test, and the depth term on its own for sizing.

    Arm B is arm A *plus* `depth`, at one weight and at the same strength, so
    what separates the two renders is the geometry term and not a temperature.
    """

    text_only: ClipTextGuidance
    depth_only: GeometricGuidance
    text_and_depth: WeightedGuidance


@dataclass(frozen=True)
class GradientSizing:
    """What each term's pull actually measured, and the weight read off it.

    Args:
        step_index: where in the walk the measurement was taken, on a real
            partially denoised sample rather than on the photograph.
        text_gradient_rms: RMS of ∂(text energy)/∂latent at that step.
        depth_gradient_rms: the same for the depth energy, whose units — masked
            squared aligned disparity — have nothing to do with a cosine's.
        weight: `text_gradient_rms / depth_gradient_rms`, so the two terms pull
            equally hard at this step. Equal RMS is not equal *influence*: the
            depth gradient is sparse, carrying its magnitude on the shell alone,
            while the text gradient is dense over the frame, so matching the two
            globally leaves the depth term stronger where it acts.
        composite_gradient_rms: the weighted sum's own pull, measured after the
            weight was fixed.
    """

    step_index: int
    total_steps: int
    text_gradient_rms: float
    depth_gradient_rms: float
    weight: float
    composite_gradient_rms: float | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "text_gradient_rms": self.text_gradient_rms,
            "depth_gradient_rms": self.depth_gradient_rms,
            "weight": self.weight,
            "composite_gradient_rms": self.composite_gradient_rms,
        }


@dataclass(frozen=True)
class ArmRender:
    """One arm's finished render and the pixels it left the score to read."""

    image: Float[Tensor, "1 3 height width"]
    scorable: Bool[Tensor, "1 height width"]
    seconds: float


@dataclass(frozen=True)
class FurnishResult:
    """One (scene, seed): three renders, the one mask they were compared over, the numbers."""

    scene: str
    prompt: str
    seed: int
    depth_errors: dict[str, float]
    adherences: dict[str, float]
    arm_coverages: dict[str, float]
    seconds: dict[str, float]
    common_coverage: float
    reconstruction_depth_error: float
    images: dict[str, str]

    def metrics(self, guided: Arm, control: Arm) -> RunMetrics:
        """One arm read against another over the pair's common mask."""
        return RunMetrics(
            guided_depth_error=self.depth_errors[guided],
            control_depth_error=self.depth_errors[control],
            guided_prompt_adherence=self.adherences[guided],
            control_prompt_adherence=self.adherences[control],
            reconstruction_depth_error=self.reconstruction_depth_error,
            common_coverage=self.common_coverage,
            guided_shell_coverage=self.arm_coverages[guided],
            control_shell_coverage=self.arm_coverages[control],
        )

    def as_row(self) -> dict[str, object]:
        comparisons = {
            f"{guided}_vs_{control}": {
                "depth_error_reduction": self.metrics(
                    guided, control
                ).depth_error_reduction,
                "adherence_ratio": self.metrics(guided, control).adherence_ratio,
            }
            for guided, control in _COMPARISONS
        }
        return {
            "scene": self.scene,
            "prompt": self.prompt,
            "seed": self.seed,
            "depth_error": self.depth_errors,
            "adherence": self.adherences,
            "arm_shell_coverage": self.arm_coverages,
            "common_coverage": self.common_coverage,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "seconds": self.seconds,
            "comparisons": comparisons,
            "images": self.images,
        }


_COMPARISONS: Final[tuple[tuple[Arm, Arm], ...]] = (
    ("text_only", "control"),
    ("text_and_depth", "control"),
    ("text_and_depth", "text_only"),
)


def empty_room_reference(
    strip: StripResult, *, config: StudyConfig, backends: Backends
) -> GeometricReference:
    """The geometry the furnishing must keep: the *stripped* room, not the photograph.

    Stage 2 is asked to put furniture into an empty room without moving it, so
    the reference is frozen from the empty room the strip produced — the same
    frame, one VAE round trip and one repaint later.
    """
    return GeometricReference.from_photo(
        strip.empty_room,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        erosion_radius=config.furnish.erosion_radius,
    )


def build_energies(
    text: ClipTextGuidance,
    *,
    reference: GeometricReference,
    config: StudyConfig,
    backends: Backends,
    depth_weight: float,
) -> Energies:
    """The three guidance objects one scene is furnished under.

    Args:
        text: built once per scene by the caller, since loading a CLIP is what
            it costs and the weight sizing needs the same object the arms run.
    """
    depth = GeometricGuidance(
        depth_estimator=backends.depth_estimator,
        reference=reference,
        segmenter=backends.segmenter if config.furnish.resegment else None,
        erosion_radius=(
            config.furnish.erosion_radius if config.furnish.resegment else None
        ),
    )
    return Energies(
        text_only=text,
        depth_only=depth,
        text_and_depth=WeightedGuidance(
            (WeightedTerm(text, 1.0), WeightedTerm(depth, depth_weight))
        ),
    )


def measure_gradient_sizing(
    empty_room: Float[Tensor, "1 3 height width"],
    prompt: str,
    *,
    text: Guidance,
    depth: Guidance,
    composite: Guidance | None = None,
    weight: float = 1.0,
    config: StudyConfig,
    backends: Backends,
    seed: int,
) -> GradientSizing:
    """Each term's gradient RMS at one real, partially denoised sample.

    Guessing a weight between a cosine and a masked squared disparity is
    guessing a unit conversion; this measures it instead, on the very walk the
    arms run, at a step `furnish.sizing_fraction` of the way into it.
    """
    process = _process(empty_room, prompt, config=config, backends=backends, seed=seed)
    index = min(
        int(process.steps * config.furnish.sizing_fraction), max(process.steps - 1, 0)
    )
    state = process.initial_state()
    for walked in range(index):
        state = process.advance(
            state, _context(process, walked), prediction=None, steering=None
        )
    return GradientSizing(
        step_index=index,
        total_steps=process.steps,
        text_gradient_rms=_gradient_rms(process, state, index, text),
        depth_gradient_rms=_gradient_rms(process, state, index, depth),
        weight=weight,
        composite_gradient_rms=(
            None
            if composite is None
            else _gradient_rms(process, state, index, composite)
        ),
    )


def furnish_scene(
    scene: SceneConfig,
    strip: StripResult,
    *,
    config: StudyConfig,
    backends: Backends,
    adherence: PromptAdherence,
    reference: GeometricReference,
    energies: Energies,
    gallery: Path,
) -> list[FurnishResult]:
    """Every seed of one scene, all three arms, scored over one common mask per seed."""
    reconstruction = depth_error(
        vae_roundtrip(backends.pipeline.vae, strip.empty_room, device=backends.device),
        reference=reference,
        depth_estimator=backends.depth_estimator,
    )
    guidance: dict[Arm, Guidance] = {
        "text_only": energies.text_only,
        "text_and_depth": energies.text_and_depth,
        # Strength 0 never evaluates the energy, so which one the control is
        # handed cannot reach its trajectory; it is the same walk with the
        # steering switched off.
        "control": energies.text_only,
    }
    scale = DisparityScale.of(reference.disparity)

    results: list[FurnishResult] = []
    for seed in config.seeds:
        renders = {
            arm: _render(
                strip.empty_room,
                scene.prompt,
                guidance=guidance[arm],
                strength=0.0 if arm == "control" else config.furnish.strength,
                config=config,
                backends=backends,
                reference=reference,
                seed=seed,
            )
            for arm in ARMS
        }
        # One pixel set for all three arms: scoring each on its own shell makes
        # the numbers incommensurable and pays an arm for hiding the geometry it
        # moved, since a mask that shrinks takes the evidence with it.
        common = _common_mask(renders, reference=reference)
        results.append(
            FurnishResult(
                scene=scene.photo,
                prompt=scene.prompt,
                seed=seed,
                depth_errors={
                    arm: depth_error(
                        render.image,
                        reference=reference,
                        depth_estimator=backends.depth_estimator,
                        mask=common,
                    )
                    for arm, render in renders.items()
                },
                adherences={
                    arm: adherence(
                        render.image,
                        prompt=scene.prompt,
                        photo=strip.empty_room,
                        reference=reference,
                    )
                    for arm, render in renders.items()
                },
                arm_coverages={
                    arm: shell_coverage(render.scorable, reference=reference)
                    for arm, render in renders.items()
                },
                seconds={arm: render.seconds for arm, render in renders.items()},
                common_coverage=shell_coverage(common, reference=reference),
                reconstruction_depth_error=reconstruction,
                images=_write_gallery(
                    scene.photo,
                    seed=seed,
                    renders=renders,
                    scale=scale,
                    reference=reference,
                    backends=backends,
                    gallery=gallery,
                ),
            )
        )
    return results


def _render(
    empty_room: Float[Tensor, "1 3 height width"],
    prompt: str,
    *,
    guidance: Guidance,
    strength: float,
    config: StudyConfig,
    backends: Backends,
    reference: GeometricReference,
    seed: int,
) -> ArmRender:
    # The generator lives on the sampling device: the loop draws its noise with
    # `torch.randn(..., generator=..., device=latent.device)`, which refuses a
    # CPU generator for a CUDA latent.
    generator = torch.Generator(device=backends.device).manual_seed(seed)
    # CUDA kernels are launched asynchronously, so a wall clock taken around the
    # call measures queueing unless both ends are synchronised.
    _synchronize(backends.device)
    started = time.perf_counter()
    image = sample_guided(
        backends.pipeline,
        empty_room,
        prompt,
        guidance=guidance,
        strength=strength,
        steps=config.furnish.steps,
        noise_strength=config.furnish.noise_strength,
        text_guidance_scale=config.furnish.text_guidance_scale,
        generator=generator,
        clip_step_rms=config.furnish.clip_step_rms,
    )
    _synchronize(backends.device)
    seconds = time.perf_counter() - started
    with torch.no_grad():
        scorable = reference.mask & scorable_mask(
            backends.segmenter(image), erosion_radius=config.furnish.erosion_radius
        )
    return ArmRender(image=image, scorable=scorable, seconds=seconds)


def _common_mask(
    renders: dict[Arm, ArmRender], *, reference: GeometricReference
) -> Bool[Tensor, "1 height width"]:
    common = reference.mask
    for render in renders.values():
        common = common & render.scorable
    return common


def _process(
    empty_room: Float[Tensor, "1 3 height width"],
    prompt: str,
    *,
    config: StudyConfig,
    backends: Backends,
    seed: int,
) -> LatentDiffusionProcess:
    return LatentDiffusionProcess(
        backends.pipeline,
        empty_room,
        prompt,
        steps=config.furnish.steps,
        noise_strength=config.furnish.noise_strength,
        text_guidance_scale=config.furnish.text_guidance_scale,
        generator=torch.Generator(device=backends.device).manual_seed(seed),
    )


def _context(process: LatentDiffusionProcess, index: int) -> StepContext:
    return StepContext(index=index, total=process.steps, span=process.step_span(index))


def _gradient_rms(
    process: LatentDiffusionProcess,
    state: Float[Tensor, "1 channels height width"],
    index: int,
    energy: Guidance,
) -> float:
    # `enable_grad` rather than an ambient mode, exactly as the driver takes its
    # own gradient: this is the same evaluation a guided step makes, taken once
    # for its magnitude instead of for a step.
    with torch.enable_grad():
        endpoint = process.endpoint(state, _context(process, index))
        (gradient,) = torch.autograd.grad(energy(endpoint.image).mean(), endpoint.state)
    return float(gradient.to(torch.float32).square().mean().sqrt())


def _write_gallery(
    scene: str,
    *,
    seed: int,
    renders: dict[Arm, ArmRender],
    scale: DisparityScale,
    reference: GeometricReference,
    backends: Backends,
    gallery: Path,
) -> dict[str, str]:
    """Gallery B: each arm's render at this seed, and the depth map behind it."""
    names: dict[str, str] = {}
    for arm, render in renders.items():
        names[arm] = f"{scene}_seed{seed}_{arm}.png"
        names[f"{arm}_depth"] = f"{scene}_seed{seed}_{arm}_depth.png"
        write_image(render.image, gallery / names[arm])
        with torch.no_grad():
            write_disparity_map(
                backends.depth_estimator(render.image),
                scale=scale,
                mask=reference.mask,
                path=gallery / names[f"{arm}_depth"],
            )
    return names


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
