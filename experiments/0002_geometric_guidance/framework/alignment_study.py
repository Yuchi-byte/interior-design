"""The DA3 alignment-mode study: raw residuals against shift-only, beside one control.

The claim under test has only ever been a code comment: that a metric depth
backend guides better under a shift-only fit than under no fit at all. Here it
is a measurement — per (scene, seed), one unguided control and one guided arm
per alignment mode, all from the same seed and prompt, steered by DA3 metric.
No arm is read through its own objective alone: every render is scored under
all three alignment modes of the metric judge *and* under the affine-invariant
V2 judge's full fit, beside prompt adherence and shell coverage. Every render
and every reference photo leaves behind its depth maps, painted on one shared
scale per scene so an absolute offset stays visible.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from framework.backends import AlignmentBackends, build_alignment_backends
from framework.benchmark import (
    PHOTO_SUBDIRECTORY,
    _git_commit,
    _synchronize,
    _vae_roundtrip,
    _write_image,
    _write_mask,
)
from framework.config import AlignmentStudyConfig, SceneConfig
from framework.depth_maps import DisparityRange, write_scaled_disparity_map
from framework.metrics import build_adherence
from framework.photos import ensure_photos, load_photo
from roomgen_geometry.alignment import (
    AlignmentMode,
    align_scale_shift,
    apply_scale_shift,
)
from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.eval import PromptAdherence, depth_error, shell_coverage
from roomgen_guide.renovate import renovate

# `_renoised_timesteps` is imported rather than restated so the number of steps
# a run records having walked cannot drift from the number the sampler walks.
from roomgen_guide.sampling.guided import _renoised_timesteps
from roomgen_guide.settings import RoomgenSettings

# Every render is scored under the full matrix whatever mode steered it, so the
# comparison is never read through one arm's own objective.
EVAL_MODES: Final[tuple[AlignmentMode, ...]] = ("none", "shift", "scale_shift")

CONTROL_ARM: Final = "control"

_WARM_UP_STEPS: Final = 1


@dataclass(frozen=True)
class ArmSpec:
    """One arm: the strength it is steered at and the fit its energy runs through."""

    arm: str
    guided: bool
    alignment: AlignmentMode


def arm_specs(config: AlignmentStudyConfig) -> tuple[ArmSpec, ...]:
    """The control first, then one guided arm per configured alignment mode.

    The control's alignment is never exercised: at strength 0 the energy is not
    evaluated at all, so the value is a placeholder, not a fourth condition.
    """
    return (
        ArmSpec(CONTROL_ARM, False, "scale_shift"),
        *(ArmSpec(f"guided_{mode}", True, mode) for mode in config.guidance.arms),
    )


@dataclass(frozen=True)
class ArmMeasurement:
    """One arm of one (scene, seed): its render, its depth fields, and the full score matrix."""

    arm: str
    guided: bool
    alignment: AlignmentMode
    metric_depth_errors: dict[str, float]
    judge_depth_error: float
    adherence: float
    shell_coverage: float
    render: str
    metric_depth_map: str
    judge_depth_map: str
    seconds: float

    def as_row(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "guided": self.guided,
            "alignment": self.alignment,
            "metric_depth_errors": self.metric_depth_errors,
            "judge_depth_error": self.judge_depth_error,
            "adherence": self.adherence,
            "shell_coverage": self.shell_coverage,
            "render": self.render,
            "metric_depth_map": self.metric_depth_map,
            "judge_depth_map": self.judge_depth_map,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class AlignmentResult:
    """All arms of one (scene, seed), scored on one common pixel set."""

    scene: str
    prompt: str
    seed: int
    arms: tuple[ArmMeasurement, ...]
    common_coverage: float
    photo_image: str
    photo_metric_depth_map: str
    photo_judge_depth_map: str
    shell_mask: str
    eval_mask_common: str
    reconstruction_metric_depth_errors: dict[str, float]
    reconstruction_judge_depth_error: float

    def arm(self, arm_id: str) -> ArmMeasurement:
        for measurement in self.arms:
            if measurement.arm == arm_id:
                return measurement
        raise KeyError(arm_id)

    def as_row(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "prompt": self.prompt,
            "seed": self.seed,
            "common_coverage": self.common_coverage,
            "photo_image": self.photo_image,
            "photo_metric_depth_map": self.photo_metric_depth_map,
            "photo_judge_depth_map": self.photo_judge_depth_map,
            "shell_mask": self.shell_mask,
            "eval_mask_common": self.eval_mask_common,
            "reconstruction_metric_depth_errors": (
                self.reconstruction_metric_depth_errors
            ),
            "reconstruction_judge_depth_error": self.reconstruction_judge_depth_error,
            "arms": [measurement.as_row() for measurement in self.arms],
        }


@dataclass(frozen=True)
class AlignmentSummary:
    """What the study measured, where its artifacts landed, and the code that produced them."""

    config: AlignmentStudyConfig
    results: tuple[AlignmentResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


def run_alignment_study(
    config: AlignmentStudyConfig, settings: RoomgenSettings
) -> AlignmentSummary:
    """Run every arm over every (scene, seed) of `config`, measure it, write every artifact."""
    started_at = datetime.now(UTC).isoformat()
    output_dir = settings.wdir / config.name
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = ensure_photos(settings.wdir / PHOTO_SUBDIRECTORY)
    backends = build_alignment_backends(config.backends, cache_dir=settings.hub_cache)
    adherence = build_adherence(
        config.adherence, device=backends.device, cache_dir=settings.hub_cache
    )
    if backends.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backends.device)

    results: list[AlignmentResult] = []
    for index, scene in enumerate(config.scenes):
        if scene.photo not in photo_paths:
            raise ValueError(
                f"scene photo {scene.photo!r} is not in experiments/assets/photos.json; "
                f"known ids: {', '.join(sorted(photo_paths))}"
            )
        results.extend(
            _run_scene(
                scene,
                config=config,
                backends=backends,
                adherence=adherence,
                photo_path=photo_paths[scene.photo],
                image_dir=image_dir,
                warm_up=index == 0,
            )
        )

    on_cuda = backends.device.type == "cuda"
    summary = AlignmentSummary(
        config=config,
        results=tuple(results),
        output_dir=output_dir,
        peak_vram_bytes=(
            torch.cuda.max_memory_allocated(backends.device) if on_cuda else None
        ),
        started_at=started_at,
        git_commit=_git_commit(),
        walked_steps=_walked_steps(config),
        device_name=(torch.cuda.get_device_name(backends.device) if on_cuda else None),
    )
    _write_results(summary, settings)
    return summary


def _run_scene(
    scene: SceneConfig,
    *,
    config: AlignmentStudyConfig,
    backends: AlignmentBackends,
    adherence: PromptAdherence,
    photo_path: Path,
    image_dir: Path,
    warm_up: bool,
) -> list[AlignmentResult]:
    photo = load_photo(
        photo_path,
        resolution=config.resolution,
        device=backends.device,
        dtype=backends.dtype,
    )
    # One segmentation for both references: the two judges must disagree only
    # about depth, never about which pixels are the room.
    shell = scorable_mask(
        backends.segmenter(photo), erosion_radius=config.guidance.erosion_radius
    )
    with torch.no_grad():
        metric_reference = GeometricReference(
            disparity=backends.metric_depth(photo).detach(), mask=shell
        )
        judge_reference = GeometricReference(
            disparity=backends.judge_depth(photo).detach(), mask=shell
        )
    # The scene's two colour scales, both taken from the photo: every map of the
    # scene is painted against the reference's own span, so a render that moved
    # the metric backend's absolute level shows up as a brighter or darker room
    # rather than being re-normalised away.
    metric_scale = DisparityRange.of(metric_reference.disparity)
    judge_scale = DisparityRange.of(judge_reference.disparity)

    photo_image = f"{scene.photo}_photo.png"
    shell_mask = f"{scene.photo}_shell_mask.png"
    photo_metric_depth_map = f"{scene.photo}_photo_depth_da3.png"
    photo_judge_depth_map = f"{scene.photo}_photo_depth_da2.png"
    _write_image(photo, image_dir / photo_image)
    _write_mask(shell, image_dir / shell_mask)
    write_scaled_disparity_map(
        metric_reference.disparity,
        scale=metric_scale,
        path=image_dir / photo_metric_depth_map,
    )
    write_scaled_disparity_map(
        judge_reference.disparity,
        scale=judge_scale,
        path=image_dir / photo_judge_depth_map,
    )

    # The floor under every arm's error: what a perfect geometry-preserving
    # renovation still pays for the VAE round trip alone, under each judge and
    # mode the arms are scored under.
    roundtrip = _vae_roundtrip(backends.pipeline.vae, photo, device=backends.device)
    reconstruction_metric: dict[str, float] = {
        mode: depth_error(
            roundtrip,
            reference=metric_reference,
            depth_estimator=backends.metric_depth,
            alignment=mode,
        )
        for mode in EVAL_MODES
    }
    reconstruction_judge = depth_error(
        roundtrip, reference=judge_reference, depth_estimator=backends.judge_depth
    )
    print(
        f"[scene] {scene.photo}: shell covers "
        f"{float(shell.to(torch.float32).mean()):.1%} of the frame, "
        f"reconstruction floor DA3 none {reconstruction_metric['none']:.6f} / "
        f"shift {reconstruction_metric['shift']:.6f} / "
        f"ssi {reconstruction_metric['scale_shift']:.6f}, "
        f"DA2 ssi {reconstruction_judge:.6f}"
    )

    if warm_up:
        print(f"[warmup] {_WARM_UP_STEPS} throwaway step(s), untimed")
        _sample(
            photo,
            scene.prompt,
            backends=backends,
            config=config,
            reference=metric_reference,
            strength=config.guidance.strength,
            alignment=config.guidance.arms[0],
            seed=config.seeds[0],
            steps=_WARM_UP_STEPS,
        )

    specs = arm_specs(config)
    results: list[AlignmentResult] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        for spec in specs:
            image, elapsed = _sample(
                photo,
                scene.prompt,
                backends=backends,
                config=config,
                reference=metric_reference,
                strength=config.guidance.strength if spec.guided else 0.0,
                alignment=spec.alignment,
                seed=seed,
            )
            renders[spec.arm] = image
            seconds[spec.arm] = elapsed
            _write_image(image, image_dir / f"{scene.photo}_seed{seed}_{spec.arm}.png")

        # One pixel set for every arm: the reference shell narrowed by each
        # arm's own segmentation, intersected across all of them, so the score
        # matrix is commensurable and covering shell up buys nothing.
        arm_masks = {
            spec.arm: _scorable(
                renders[spec.arm], backends=backends, config=config, shell=shell
            )
            for spec in specs
        }
        common = shell.clone()
        for mask in arm_masks.values():
            common &= mask
        eval_mask_common = f"{scene.photo}_seed{seed}_evalmask_common.png"
        _write_mask(common, image_dir / eval_mask_common)

        measurements: list[ArmMeasurement] = []
        for spec in specs:
            render = renders[spec.arm]
            metric_depth_map = f"{scene.photo}_seed{seed}_{spec.arm}_depth_da3.png"
            judge_depth_map = f"{scene.photo}_seed{seed}_{spec.arm}_depth_da2.png"
            with torch.no_grad():
                # The metric map is painted raw on the photo's span — absolute
                # offset and rescale are signal for this backend. The V2 map is
                # affine-invariant and must be fitted onto its reference's frame
                # (over the shell) before its colours mean anything at all.
                write_scaled_disparity_map(
                    backends.metric_depth(render),
                    scale=metric_scale,
                    path=image_dir / metric_depth_map,
                )
                judge_disparity = backends.judge_depth(render)
                write_scaled_disparity_map(
                    apply_scale_shift(
                        judge_disparity,
                        align_scale_shift(
                            judge_disparity, judge_reference.disparity, shell
                        ),
                    ),
                    scale=judge_scale,
                    path=image_dir / judge_depth_map,
                )
            measurements.append(
                ArmMeasurement(
                    arm=spec.arm,
                    guided=spec.guided,
                    alignment=spec.alignment,
                    metric_depth_errors={
                        mode: depth_error(
                            render,
                            reference=metric_reference,
                            depth_estimator=backends.metric_depth,
                            mask=common,
                            alignment=mode,
                        )
                        for mode in EVAL_MODES
                    },
                    judge_depth_error=depth_error(
                        render,
                        reference=judge_reference,
                        depth_estimator=backends.judge_depth,
                        mask=common,
                    ),
                    adherence=adherence(
                        render,
                        prompt=scene.prompt,
                        photo=photo,
                        reference=metric_reference,
                    ),
                    shell_coverage=shell_coverage(
                        arm_masks[spec.arm], reference=metric_reference
                    ),
                    render=f"{scene.photo}_seed{seed}_{spec.arm}.png",
                    metric_depth_map=metric_depth_map,
                    judge_depth_map=judge_depth_map,
                    seconds=seconds[spec.arm],
                )
            )

        result = AlignmentResult(
            scene=scene.photo,
            prompt=scene.prompt,
            seed=seed,
            arms=tuple(measurements),
            common_coverage=shell_coverage(common, reference=metric_reference),
            photo_image=photo_image,
            photo_metric_depth_map=photo_metric_depth_map,
            photo_judge_depth_map=photo_judge_depth_map,
            shell_mask=shell_mask,
            eval_mask_common=eval_mask_common,
            reconstruction_metric_depth_errors=dict(reconstruction_metric),
            reconstruction_judge_depth_error=reconstruction_judge,
        )
        print(f"  {_format_result(result)}")
        results.append(result)
        # A tuple's eval interleaves CLIP, segmenter and depth allocations with
        # the sampler's; at float32 the walk peaks within a few GiB of the card,
        # and what those leave behind is what ran the first full grid out of
        # memory one scene in. The collect sweeps reference cycles that may pin
        # tensors, the cache release returns the freed blocks, and the live
        # figure is printed so a leak shows up as growth across tuples rather
        # than as an OOM two scenes later.
        if backends.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            live = torch.cuda.memory_allocated(backends.device) / 2**30
            print(f"  [memory] {live:.2f} GiB live after seed {seed}")
    return results


def _sample(
    photo: Float[Tensor, "1 3 height width"],
    prompt: str,
    *,
    backends: AlignmentBackends,
    config: AlignmentStudyConfig,
    reference: GeometricReference,
    strength: float,
    alignment: AlignmentMode,
    seed: int,
    steps: int | None = None,
) -> tuple[Float[Tensor, "1 3 height width"], float]:
    # The generator lives on the sampling device, and every arm gets a fresh one
    # at the same seed: matched noise, so the arms differ in their guidance
    # energy alone.
    generator = torch.Generator(device=backends.device).manual_seed(seed)
    _synchronize(backends.device)
    started = time.perf_counter()
    image = renovate(
        photo,
        prompt,
        pipeline=backends.pipeline,
        depth_estimator=backends.metric_depth,
        segmenter=backends.segmenter,
        strength=strength,
        erosion_radius=config.guidance.erosion_radius,
        steps=config.sampler.steps if steps is None else steps,
        noise_strength=config.sampler.noise_strength,
        text_guidance_scale=config.sampler.text_guidance_scale,
        generator=generator,
        clip_step_rms=config.guidance.clip,
        resegment=config.guidance.resegment,
        reference=reference,
        alignment=alignment,
    )
    _synchronize(backends.device)
    return image, time.perf_counter() - started


def _scorable(
    image: Float[Tensor, "1 3 height width"],
    *,
    backends: AlignmentBackends,
    config: AlignmentStudyConfig,
    shell: Bool[Tensor, "1 height width"],
) -> Bool[Tensor, "1 height width"]:
    """The reference shell narrowed by one finished render's own segmentation."""
    if not config.guidance.resegment:
        return shell
    with torch.no_grad():
        segmentation = backends.segmenter(image)
    return shell & scorable_mask(
        segmentation, erosion_radius=config.guidance.erosion_radius
    )


def _format_result(result: AlignmentResult) -> str:
    cells = ", ".join(
        f"{measurement.arm} "
        f"none={measurement.metric_depth_errors['none']:.4f} "
        f"shift={measurement.metric_depth_errors['shift']:.4f} "
        f"ssi={measurement.metric_depth_errors['scale_shift']:.4f} "
        f"da2={measurement.judge_depth_error:.4f} "
        f"adh={measurement.adherence:.4f}"
        for measurement in result.arms
    )
    return (
        f"seed {result.seed}: {cells}, shell scored "
        f"{result.common_coverage:.0%} in common"
    )


def _walked_steps(config: AlignmentStudyConfig) -> int:
    """How many of the scheduler's steps SDEdit actually walks, by the sampler's own arithmetic."""
    return len(
        _renoised_timesteps(
            torch.arange(config.sampler.steps),
            noise_strength=config.sampler.noise_strength,
        )
    )


def arm_means(
    results: tuple[AlignmentResult, ...], specs: tuple[ArmSpec, ...]
) -> dict[str, dict[str, float]]:
    """Each arm's mean over every (scene, seed): the score matrix, coverage and adherence."""
    means: dict[str, dict[str, float]] = {}
    for spec in specs:
        cells = [result.arm(spec.arm) for result in results]
        means[spec.arm] = {
            **{
                f"metric_{mode}": _mean(
                    [cell.metric_depth_errors[mode] for cell in cells]
                )
                for mode in EVAL_MODES
            },
            "judge_scale_shift": _mean([cell.judge_depth_error for cell in cells]),
            "adherence": _mean([cell.adherence for cell in cells]),
            "shell_coverage": _mean([cell.shell_coverage for cell in cells]),
            "seconds": _mean([cell.seconds for cell in cells]),
        }
    return means


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_results(summary: AlignmentSummary, settings: RoomgenSettings) -> None:
    results = summary.output_dir / "results.json"
    results.write_text(
        json.dumps(
            {
                "config": summary.config.model_dump(mode="json"),
                "settings": {
                    "wdir": str(settings.wdir),
                    "hub_cache": str(settings.hub_cache),
                },
                "started_at": summary.started_at,
                "git_commit": summary.git_commit,
                "torch_version": torch.__version__,
                "device_name": summary.device_name,
                "walked_steps": summary.walked_steps,
                "peak_vram_bytes": summary.peak_vram_bytes,
                "arm_means": arm_means(summary.results, arm_specs(summary.config)),
                "runs": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
