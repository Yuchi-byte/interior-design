"""The DA3 alignment-mode study on FMRG: raw against shift-only, beside one control.

The SD1.5 run of this claim predates the FMRG-only ruling, so it is re-measured
on the FLUX flow map: per (scene, seed) one unguided control and one guided arm
per alignment mode, all from one seed and one style prompt, steered by DA3
metric at one eta. The anti-self-grading design is kept verbatim — every render
is scored under all three alignment modes of the metric judge *and* under the
affine-invariant V2 judge's full fit, beside prompt adherence and shell
coverage — and every arm leaves a masked depth map on the scene's shared metric
colour scale plus a signed raw-residual heatmap against the reference.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Float
from torch import Tensor

from framework.config import TORCH_DTYPES, EtaSceneConfig, FmrgAlignmentConfig
from framework.depth_maps import (
    DisparityRange,
    write_disparity_map,
    write_masked_scaled_disparity_map,
    write_scaled_disparity_map,
)
from framework.fmrg_study import (
    PHOTO_SUBDIRECTORY,
    Stacks,
    _git_commit,
    _vae_roundtrip,
    _write_image,
    _write_mask,
    build_stacks,
    sample_walk,
    scorable,
    walked_steps,
)
from framework.metrics import build_adherence
from framework.photos import ensure_photos, load_photo
from framework.residual_maps import mean_absolute_residual, write_signed_residual_map
from roomgen_geometry.alignment import AlignmentMode
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.depth.depth_anything_v3 import DepthAnythingV3Metric
from roomgen_geometry.masking import architectural_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS, ARCHITECTURAL_CLASSES
from roomgen_guide.adapters.fmrg_flux import select_cached_prompt
from roomgen_guide.eval import (
    PromptAdherence,
    depth_error,
    shell_coverage,
    signed_depth_residual_map,
)
from roomgen_guide.settings import RoomgenSettings

# Every render is scored under the full matrix whatever mode steered it, so the
# comparison is never read through one arm's own objective.
EVAL_MODES: Final[tuple[AlignmentMode, ...]] = ("none", "shift", "scale_shift")

CONTROL_ARM: Final = "control"

# The residual heatmaps are painted in the identity frame on purpose: raw
# residuals are the one reading where a metric render's absolute offset stays
# visible, and they sit beside the `metric_none` column they square to.
RESIDUAL_MODE: Final[AlignmentMode] = "none"


@dataclass(frozen=True)
class ArmSpec:
    """One arm: whether it is steered, and the fit its energy runs through."""

    arm: str
    guided: bool
    alignment: AlignmentMode


def alignment_arms(config: FmrgAlignmentConfig) -> tuple[ArmSpec, ...]:
    """The control first, then one guided arm per configured alignment mode.

    The control's alignment is never exercised: at eta 0 the energy is not
    evaluated at all, so the value is a placeholder, not a third condition.
    """
    return (
        ArmSpec(CONTROL_ARM, False, "scale_shift"),
        *(ArmSpec(f"guided_{mode}", True, mode) for mode in config.guidance.arms),
    )


@dataclass(frozen=True)
class ArmMeasurement:
    """One arm of one (scene, seed): its render, its maps, and the full score matrix."""

    arm: str
    guided: bool
    alignment: AlignmentMode
    metric_depth_errors: dict[str, float]
    judge_depth_error: float
    adherence: float
    shell_coverage: float
    render: str
    metric_depth_map: str
    residual_map: str
    seconds: float
    peak_vram_bytes: int | None

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
            "residual_map": self.residual_map,
            "seconds": self.seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


@dataclass(frozen=True)
class AlignmentSceneResult:
    """All arms of one (scene, seed), scored on one common pixel set."""

    scene: str
    seed: int
    style_prompt: str
    eval_prompt: str
    start_sigma: float
    effective_first_sigma: float
    arms: tuple[ArmMeasurement, ...]
    common_coverage: float
    reconstruction_metric_depth_errors: dict[str, float]
    reconstruction_judge_depth_error: float
    mean_abs_control_residual: float
    photo_image: str
    photo_metric_depth_map: str
    photo_metric_depth_map_masked: str
    photo_judge_depth_map: str
    shell_mask: str
    eval_mask_common: str

    def arm(self, arm_id: str) -> ArmMeasurement:
        for measurement in self.arms:
            if measurement.arm == arm_id:
                return measurement
        raise KeyError(arm_id)

    def as_row(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "seed": self.seed,
            "style_prompt": self.style_prompt,
            "eval_prompt": self.eval_prompt,
            "start_sigma": self.start_sigma,
            "effective_first_sigma": self.effective_first_sigma,
            "common_coverage": self.common_coverage,
            "reconstruction_metric_depth_errors": (
                self.reconstruction_metric_depth_errors
            ),
            "reconstruction_judge_depth_error": self.reconstruction_judge_depth_error,
            "mean_abs_control_residual": self.mean_abs_control_residual,
            "photo_image": self.photo_image,
            "photo_metric_depth_map": self.photo_metric_depth_map,
            "photo_metric_depth_map_masked": self.photo_metric_depth_map_masked,
            "photo_judge_depth_map": self.photo_judge_depth_map,
            "shell_mask": self.shell_mask,
            "eval_mask_common": self.eval_mask_common,
            "arms": [measurement.as_row() for measurement in self.arms],
        }


@dataclass(frozen=True)
class AlignmentSummary:
    """What the study measured, where its artifacts landed, and the code behind them."""

    config: FmrgAlignmentConfig
    results: tuple[AlignmentSceneResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None
    shell_classes: dict[int, str]


def shell_class_table() -> dict[int, str]:
    """The shell definition this run scored under, read off the package at runtime.

    Recorded because the class lists are under active revision (windows and
    doors joining the shell): the results must say which definition produced
    them, not point at whatever the package says later.
    """
    return {index: ADE20K_LABELS[index] for index in sorted(ARCHITECTURAL_CLASSES)}


def run_fmrg_alignment_study(
    config: FmrgAlignmentConfig, settings: RoomgenSettings
) -> AlignmentSummary:
    """Run every arm over every (scene, seed) of `config`, measure it, write every artifact."""
    started_at = datetime.now(UTC).isoformat()
    output_dir = settings.wdir / config.name
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = ensure_photos(settings.wdir / PHOTO_SUBDIRECTORY)
    stacks = build_stacks(
        config.backends,
        prompts=list(dict.fromkeys(scene.style_prompt for scene in config.scenes)),
        text_guidance_scale=config.sampler.text_guidance_scale,
        settings=settings,
    )
    metric_depth = _build_metric_depth(config, stacks=stacks, settings=settings)
    if stacks.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(stacks.device)

    pending: list[_PendingScene] = []
    for index, scene in enumerate(config.scenes):
        if scene.photo not in photo_paths:
            raise ValueError(
                f"scene photo {scene.photo!r} is not in experiments/assets/"
                f"photos.json; known ids: {', '.join(sorted(photo_paths))}"
            )
        pending.extend(
            _run_scene(
                scene,
                config=config,
                stacks=stacks,
                metric_depth=metric_depth,
                photo_path=photo_paths[scene.photo],
                image_dir=image_dir,
                warm_up=index == 0,
            )
        )

    # Loaded only after the last walk: beside the transformer and the guided
    # decode's backward transient, a resident ViT-L/14 is exactly what tipped a
    # 32GB card into out-of-memory once already. Renders wait on the CPU;
    # scoring one at a time through a frozen encoder is cheap.
    adherence = build_adherence(
        config.adherence, device=stacks.device, cache_dir=settings.hub_cache
    )
    results = [
        _scored(entry, adherence=adherence, device=stacks.device) for entry in pending
    ]
    for result in results:
        print(f"  {_format_result(result)}")

    on_cuda = stacks.device.type == "cuda"
    summary = AlignmentSummary(
        config=config,
        results=tuple(results),
        output_dir=output_dir,
        peak_vram_bytes=_peak_vram(results),
        started_at=started_at,
        git_commit=_git_commit(),
        walked_steps=walked_steps(config.sampler),
        device_name=(torch.cuda.get_device_name(stacks.device) if on_cuda else None),
        shell_classes=shell_class_table(),
    )
    _write_results(summary, settings)
    return summary


def _build_metric_depth(
    config: FmrgAlignmentConfig, *, stacks: Stacks, settings: RoomgenSettings
) -> DepthEstimator:
    """The DA3 metric backend the guided arms descend, beside the stack's V2 judge."""
    match config.backends.kind:
        case "real":
            return DepthAnythingV3Metric(
                config.backends.metric_depth_checkpoint,
                device=stacks.device,
                dtype=TORCH_DTYPES[config.backends.metric_depth_dtype],
                cache_dir=settings.hub_cache,
                input_size=config.backends.metric_input_size,
            )
        case "fake":
            # `build_stacks` has already put the doubles on the path. A
            # deliberately different affine frame from the judge's double, so a
            # smoke run exercises exactly the disagreement the modes are about.
            from roomgen_fakes import FakeDepthEstimator  # noqa: PLC0415

            fake: DepthEstimator = FakeDepthEstimator(scale=2.0, shift=0.25)
            return fake


@dataclass(frozen=True)
class _PendingScene:
    """One scene measured on everything but adherence, its renders parked on the CPU."""

    result: AlignmentSceneResult
    renders: dict[str, Float[Tensor, "1 3 height width"]]
    photo: Float[Tensor, "1 3 height width"]
    reference: GeometricReference


def _scored(
    entry: _PendingScene, *, adherence: PromptAdherence, device: torch.device
) -> AlignmentSceneResult:
    """The scene with its adherence filled in, every arm against the neutral text."""
    result = entry.result
    arms = tuple(
        replace(
            cell,
            adherence=adherence(
                entry.renders[cell.arm].to(device),
                prompt=result.eval_prompt,
                photo=entry.photo,
                reference=entry.reference,
            ),
        )
        for cell in result.arms
    )
    return replace(result, arms=arms)


def _run_scene(
    scene: EtaSceneConfig,
    *,
    config: FmrgAlignmentConfig,
    stacks: Stacks,
    metric_depth: DepthEstimator,
    photo_path: Path,
    image_dir: Path,
    warm_up: bool,
) -> list[_PendingScene]:
    photo = load_photo(
        photo_path,
        resolution=config.resolution,
        device=stacks.device,
        dtype=torch.float32,
    )
    if stacks.sampler is not None:
        select_cached_prompt(stacks.sampler, scene.style_prompt)
    # One segmentation for both references: the two judges must disagree only
    # about depth, never about which pixels are the room.
    with torch.no_grad():
        shell = architectural_mask(
            stacks.segmenter(photo), erosion_radius=config.guidance.erosion_radius
        )
        metric_reference = GeometricReference(
            disparity=metric_depth(photo).detach(), mask=shell
        )
        judge_reference = GeometricReference(
            disparity=stacks.depth_estimator(photo).detach(), mask=shell
        )
    # The scene's metric colour scale, taken from the photo: every metric map of
    # the scene is painted against the reference's own span, so a render that
    # moved the backend's absolute level shows up as a brighter or darker room
    # rather than being re-normalised away.
    metric_scale = DisparityRange.of(metric_reference.disparity)

    photo_image = f"{scene.photo}_photo.png"
    shell_mask = f"{scene.photo}_shell_mask.png"
    photo_metric_depth_map = f"{scene.photo}_photo_depth_da3.png"
    photo_metric_depth_map_masked = f"{scene.photo}_photo_depth_da3_masked.png"
    photo_judge_depth_map = f"{scene.photo}_photo_depth_da2.png"
    _write_image(photo, image_dir / photo_image)
    _write_mask(shell, image_dir / shell_mask)
    write_scaled_disparity_map(
        metric_reference.disparity,
        scale=metric_scale,
        path=image_dir / photo_metric_depth_map,
    )
    write_masked_scaled_disparity_map(
        metric_reference.disparity,
        mask=shell,
        scale=metric_scale,
        path=image_dir / photo_metric_depth_map_masked,
    )
    write_disparity_map(judge_reference.disparity, image_dir / photo_judge_depth_map)

    # The floor under every arm's error: what a perfect geometry-preserving
    # renovation still pays for FLUX's VAE round trip alone, under each judge
    # and mode the arms are scored under.
    roundtrip = _vae_roundtrip(stacks.flow_backend, photo, device=stacks.device)
    reconstruction_metric: dict[str, float] = {
        mode: depth_error(
            roundtrip,
            reference=metric_reference,
            depth_estimator=metric_depth,
            alignment=mode,
        )
        for mode in EVAL_MODES
    }
    reconstruction_judge = depth_error(
        roundtrip, reference=judge_reference, depth_estimator=stacks.depth_estimator
    )
    effective_sigma = _effective_first_sigma(photo, config=config, stacks=stacks)
    print(
        f"[scene] {scene.photo}: shell covers "
        f"{float(shell.to(torch.float32).mean()):.1%} of the frame, "
        f"reconstruction floor DA3 none {reconstruction_metric['none']:.6f} / "
        f"shift {reconstruction_metric['shift']:.6f} / "
        f"ssi {reconstruction_metric['scale_shift']:.6f}, "
        f"DA2 ssi {reconstruction_judge:.6f}, effective first sigma "
        f"{config.sampler.start_sigma:g}->{effective_sigma:.4f}"
    )

    if warm_up:
        print("[warmup] 2 throwaway steps (1 guided), untimed")
        sample_walk(
            photo,
            stacks=stacks,
            sampler=config.sampler,
            guidance=config.guidance,
            reference=metric_reference,
            eta=max(config.guidance.eta, 1.0),
            seed=config.seeds[0],
            grid_steps=2,
            window=None,
            depth_estimator=metric_depth,
            alignment=config.guidance.arms[0],
        )

    specs = alignment_arms(config)
    results: list[_PendingScene] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        peaks: dict[str, int | None] = {}
        for spec in specs:
            image, elapsed, peak = sample_walk(
                photo,
                stacks=stacks,
                sampler=config.sampler,
                guidance=config.guidance,
                reference=metric_reference,
                eta=config.guidance.eta if spec.guided else 0.0,
                seed=seed,
                depth_estimator=metric_depth,
                alignment=spec.alignment,
            )
            renders[spec.arm] = image
            seconds[spec.arm] = elapsed
            peaks[spec.arm] = peak
            _write_image(image, image_dir / f"{scene.photo}_seed{seed}_{spec.arm}.png")

        # One pixel set for every arm: the reference shell narrowed by each
        # arm's own segmentation, intersected across all of them, so the score
        # matrix is commensurable and covering shell up buys nothing.
        arm_masks = {
            spec.arm: scorable(
                renders[spec.arm],
                stacks=stacks,
                reference=metric_reference,
                resegment=config.guidance.resegment,
                erosion_radius=config.guidance.erosion_radius,
            )
            for spec in specs
        }
        common = shell.clone()
        for mask in arm_masks.values():
            common &= mask
        eval_mask_common = f"{scene.photo}_seed{seed}_evalmask_common.png"
        _write_mask(common, image_dir / eval_mask_common)

        # Raw metric residuals, all on the control's scale: the identity frame
        # is the one where an arm that moved the absolute level stays visible.
        residuals = {
            spec.arm: signed_depth_residual_map(
                renders[spec.arm],
                reference=metric_reference,
                depth_estimator=metric_depth,
                mask=common,
                alignment=RESIDUAL_MODE,
            )
            for spec in specs
        }
        control_scale = mean_absolute_residual(residuals[CONTROL_ARM], common)

        measurements: list[ArmMeasurement] = []
        for spec in specs:
            render = renders[spec.arm]
            metric_depth_map = f"{scene.photo}_seed{seed}_{spec.arm}_depth_da3.png"
            residual_map = f"{scene.photo}_seed{seed}_{spec.arm}_residual.png"
            with torch.no_grad():
                write_masked_scaled_disparity_map(
                    metric_depth(render),
                    mask=common,
                    scale=metric_scale,
                    path=image_dir / metric_depth_map,
                )
            write_signed_residual_map(
                residuals[spec.arm],
                mask=common,
                normalizer=control_scale,
                path=image_dir / residual_map,
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
                            depth_estimator=metric_depth,
                            mask=common,
                            alignment=mode,
                        )
                        for mode in EVAL_MODES
                    },
                    judge_depth_error=depth_error(
                        render,
                        reference=judge_reference,
                        depth_estimator=stacks.depth_estimator,
                        mask=common,
                    ),
                    # Filled in by `_scored` once sampling is over and the
                    # adherence CLIP can afford to be resident.
                    adherence=math.nan,
                    shell_coverage=shell_coverage(
                        arm_masks[spec.arm], reference=metric_reference
                    ),
                    render=f"{scene.photo}_seed{seed}_{spec.arm}.png",
                    metric_depth_map=metric_depth_map,
                    residual_map=residual_map,
                    seconds=seconds[spec.arm],
                    peak_vram_bytes=peaks[spec.arm],
                )
            )

        result = AlignmentSceneResult(
            scene=scene.photo,
            seed=seed,
            style_prompt=scene.style_prompt,
            eval_prompt=scene.eval_prompt,
            start_sigma=config.sampler.start_sigma,
            effective_first_sigma=effective_sigma,
            arms=tuple(measurements),
            common_coverage=shell_coverage(common, reference=metric_reference),
            reconstruction_metric_depth_errors=dict(reconstruction_metric),
            reconstruction_judge_depth_error=reconstruction_judge,
            mean_abs_control_residual=control_scale,
            photo_image=photo_image,
            photo_metric_depth_map=photo_metric_depth_map,
            photo_metric_depth_map_masked=photo_metric_depth_map_masked,
            photo_judge_depth_map=photo_judge_depth_map,
            shell_mask=shell_mask,
            eval_mask_common=eval_mask_common,
        )
        print(f"  [sampled] {result.scene} seed {result.seed}, adherence pending")
        results.append(
            _PendingScene(
                result=result,
                renders={
                    arm: image.detach().to(device="cpu", dtype=torch.float32)
                    for arm, image in renders.items()
                },
                photo=photo,
                reference=metric_reference,
            )
        )
        # The walks interleave the transformer's transients with the two depth
        # backends' and the segmenter's; sweeping what a finished tuple left
        # behind keeps the next one from inheriting an allocator pool grown to
        # the sum of both.
        if stacks.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            live = torch.cuda.memory_allocated(stacks.device) / 2**30
            print(f"  [memory] {live:.2f} GiB live after seed {seed}")
    return results


def _effective_first_sigma(
    photo: Float[Tensor, "1 3 height width"],
    *,
    config: FmrgAlignmentConfig,
    stacks: Stacks,
) -> float:
    """The sigma every walk actually starts at, read off the backend's own schedule.

    FLUX's mu-shift warps the grid by resolution, so the configured
    `start_sigma` is a request, not the truth — 0.9 at 512² begins near 0.944.
    One value serves the whole study: neither eta nor the alignment mode
    touches the schedule.
    """
    generator = torch.Generator(device=stacks.device).manual_seed(0)
    with torch.no_grad():
        encoded = stacks.flow_backend.encode(photo, generator=generator)
    return float(
        stacks.flow_backend.schedule(
            steps=config.sampler.grid_steps,
            start_sigma=config.sampler.start_sigma,
            latent=encoded,
        )[0]
    )


def _format_result(result: AlignmentSceneResult) -> str:
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


def arm_means(
    results: tuple[AlignmentSceneResult, ...], specs: tuple[ArmSpec, ...]
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


def _peak_vram(results: list[AlignmentSceneResult]) -> int | None:
    """The largest peak any walk recorded; the per-walk resets clobber the global stat."""
    peaks = [
        cell.peak_vram_bytes
        for result in results
        for cell in result.arms
        if cell.peak_vram_bytes is not None
    ]
    return max(peaks) if peaks else None


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
                "shell_classes": {
                    str(index): label for index, label in summary.shell_classes.items()
                },
                "arm_means": arm_means(summary.results, alignment_arms(summary.config)),
                "runs": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
