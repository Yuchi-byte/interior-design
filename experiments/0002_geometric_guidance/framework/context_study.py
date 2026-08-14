"""The renovation-context study: does telling FLUX it is renovating keep the room?

Per scene at one seed, eight arms off one transformer load: the style framing at
the sampler's own init strength, unguided (A, the baseline) and guided (C), and
the renovation-instruction framing swept over SDEdit init strengths, unguided
(B@sigma) and guided (D@sigma). Every arm is scored on the one pixel set all
arms left visible and judged against the scene's one neutral style text, so the
columns compare across framings and init strengths. Beside the numbers, every
arm leaves a masked depth map and a signed depth-error map against the
reference, all on arm A's colour scale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from framework.config import FmrgContextConfig, FramingSceneConfig
from framework.depth_maps import write_disparity_map, write_masked_disparity_map
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
    warm_up_walk,
)
from framework.metrics import build_adherence
from framework.photos import ensure_photos, load_photo
from framework.residual_maps import mean_absolute_residual, write_signed_residual_map
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.adapters.fmrg_flux import select_cached_prompt
from roomgen_guide.eval import (
    PromptAdherence,
    depth_error,
    shell_coverage,
    signed_depth_residual_map,
)
from roomgen_guide.settings import RoomgenSettings

Framing = Literal["style", "renovation"]

# The baseline arm's id: style framing, unguided, at the sampler's own
# start_sigma. Every other arm's depth delta and adherence ratio divide by it,
# and its mean absolute residual sets the scene's residual-map colour scale.
BASELINE_ARM: Final = "A"


@dataclass(frozen=True)
class ContextArmSpec:
    """One arm of the study, named by all three axes it varies."""

    arm: str
    framing: Framing
    guided: bool
    start_sigma: float


def context_arms(config: FmrgContextConfig) -> tuple[ContextArmSpec, ...]:
    """The study's arms, baseline first: A, C, then B@sigma and D@sigma per sweep value."""
    base = config.sampler.start_sigma
    arms = [
        ContextArmSpec(BASELINE_ARM, "style", False, base),
        ContextArmSpec("C", "style", True, base),
    ]
    for sigma in config.context_sigmas:
        arms.append(ContextArmSpec(f"B_{sigma:g}", "renovation", False, sigma))
        arms.append(ContextArmSpec(f"D_{sigma:g}", "renovation", True, sigma))
    return tuple(arms)


@dataclass(frozen=True)
class ContextArmResult:
    """One arm's render, its depth reading, and its scores against the baseline's."""

    arm: str
    framing: Framing
    guided: bool
    prompt: str
    start_sigma: float
    effective_first_sigma: float
    depth_error: float
    shell_coverage: float
    adherence: float
    render: str
    depth_map: str
    residual_map: str
    seconds: float
    peak_vram_bytes: int | None

    def as_row(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "framing": self.framing,
            "guided": self.guided,
            "prompt": self.prompt,
            "start_sigma": self.start_sigma,
            "effective_first_sigma": self.effective_first_sigma,
            "depth_error": self.depth_error,
            "shell_coverage": self.shell_coverage,
            "adherence": self.adherence,
            "render": self.render,
            "depth_map": self.depth_map,
            "residual_map": self.residual_map,
            "seconds": self.seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


@dataclass(frozen=True)
class ContextSceneResult:
    """Every arm of one (scene, seed), scored on one common pixel set."""

    scene: str
    seed: int
    style_prompt: str
    renovation_prompt: str
    eval_prompt: str
    arms: tuple[ContextArmResult, ...]
    common_coverage: float
    reconstruction_depth_error: float
    mean_abs_baseline_residual: float
    photo_image: str
    photo_depth_map: str
    photo_depth_map_masked: str
    shell_mask: str
    eval_mask_common: str

    def arm(self, arm_id: str) -> ContextArmResult:
        for measurement in self.arms:
            if measurement.arm == arm_id:
                return measurement
        raise KeyError(arm_id)

    def depth_delta_vs_baseline(self, arm_id: str) -> float | None:
        """The arm's depth error relative to A's, signed; `None` if A scored zero."""
        baseline = self.arm(BASELINE_ARM).depth_error
        if baseline == 0.0:
            return None
        return self.arm(arm_id).depth_error / baseline - 1.0

    def adherence_ratio_vs_baseline(self, arm_id: str) -> float | None:
        """The arm's adherence as a share of A's; `None` if A scored zero."""
        baseline = self.arm(BASELINE_ARM).adherence
        if baseline == 0.0:
            return None
        return self.arm(arm_id).adherence / baseline

    def as_row(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "seed": self.seed,
            "style_prompt": self.style_prompt,
            "renovation_prompt": self.renovation_prompt,
            "eval_prompt": self.eval_prompt,
            "common_coverage": self.common_coverage,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "mean_abs_baseline_residual": self.mean_abs_baseline_residual,
            "photo_image": self.photo_image,
            "photo_depth_map": self.photo_depth_map,
            "photo_depth_map_masked": self.photo_depth_map_masked,
            "shell_mask": self.shell_mask,
            "eval_mask_common": self.eval_mask_common,
            "arms": [
                {
                    **measurement.as_row(),
                    "depth_delta_vs_baseline": self.depth_delta_vs_baseline(
                        measurement.arm
                    ),
                    "adherence_ratio_vs_baseline": self.adherence_ratio_vs_baseline(
                        measurement.arm
                    ),
                }
                for measurement in self.arms
            ],
        }


@dataclass(frozen=True)
class ContextSummary:
    """What the study measured, where its artifacts landed, and the code behind them."""

    config: FmrgContextConfig
    results: tuple[ContextSceneResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


def run_context_study(
    config: FmrgContextConfig, settings: RoomgenSettings
) -> ContextSummary:
    """Run every arm of every (scene, seed), measure them all, write the artifacts."""
    started_at = datetime.now(UTC).isoformat()
    output_dir = settings.wdir / config.name
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = ensure_photos(settings.wdir / PHOTO_SUBDIRECTORY)
    prompts = list(
        dict.fromkeys(
            prompt
            for scene in config.scenes
            for prompt in (scene.style_prompt, scene.renovation_prompt)
        )
    )
    stacks = build_stacks(
        config.backends,
        prompts=prompts,
        text_guidance_scale=config.sampler.text_guidance_scale,
        settings=settings,
    )
    adherence = build_adherence(
        config.adherence, device=stacks.device, cache_dir=settings.hub_cache
    )
    if stacks.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(stacks.device)

    results: list[ContextSceneResult] = []
    for index, scene in enumerate(config.scenes):
        if scene.photo not in photo_paths:
            raise ValueError(
                f"scene photo {scene.photo!r} is not in experiments/assets/"
                f"photos.json; known ids: {', '.join(sorted(photo_paths))}"
            )
        results.extend(
            _run_scene(
                scene,
                config=config,
                stacks=stacks,
                adherence=adherence,
                photo_path=photo_paths[scene.photo],
                image_dir=image_dir,
                warm_up=index == 0,
            )
        )

    on_cuda = stacks.device.type == "cuda"
    summary = ContextSummary(
        config=config,
        results=tuple(results),
        output_dir=output_dir,
        peak_vram_bytes=(
            torch.cuda.max_memory_allocated(stacks.device) if on_cuda else None
        ),
        started_at=started_at,
        git_commit=_git_commit(),
        walked_steps=walked_steps(config.sampler),
        device_name=(torch.cuda.get_device_name(stacks.device) if on_cuda else None),
    )
    _write_results(summary, settings)
    return summary


def _run_scene(
    scene: FramingSceneConfig,
    *,
    config: FmrgContextConfig,
    stacks: Stacks,
    adherence: PromptAdherence,
    photo_path: Path,
    image_dir: Path,
    warm_up: bool,
) -> list[ContextSceneResult]:
    photo = load_photo(
        photo_path,
        resolution=config.resolution,
        device=stacks.device,
        dtype=torch.float32,
    )
    reference = GeometricReference.from_photo(
        photo,
        depth_estimator=stacks.depth_estimator,
        segmenter=stacks.segmenter,
        erosion_radius=config.guidance.erosion_radius,
    )
    photo_image = f"{scene.photo}_photo.png"
    shell_mask = f"{scene.photo}_shell_mask.png"
    photo_depth_map = f"{scene.photo}_photo_depth.png"
    photo_depth_map_masked = f"{scene.photo}_photo_depth_masked.png"
    _write_image(photo, image_dir / photo_image)
    _write_mask(reference.mask, image_dir / shell_mask)
    write_disparity_map(reference.disparity, image_dir / photo_depth_map)
    write_masked_disparity_map(
        reference.disparity,
        mask=reference.mask,
        path=image_dir / photo_depth_map_masked,
    )

    reconstruction = depth_error(
        _vae_roundtrip(stacks.flow_backend, photo, device=stacks.device),
        reference=reference,
        depth_estimator=stacks.depth_estimator,
    )
    effective_sigmas = _effective_first_sigmas(photo, config=config, stacks=stacks)
    print(
        f"[scene] {scene.photo}: shell covers "
        f"{float(reference.mask.to(torch.float32).mean()):.1%} of the frame, "
        f"reconstruction floor {reconstruction:.6f}, effective first sigmas "
        + ", ".join(
            f"{sigma:g}->{effective_sigmas[sigma]:.4f}" for sigma in effective_sigmas
        )
    )

    if warm_up:
        if stacks.sampler is not None:
            select_cached_prompt(stacks.sampler, scene.style_prompt)
        warm_up_walk(
            photo,
            stacks=stacks,
            sampler=config.sampler,
            guidance=config.guidance,
            reference=reference,
            seed=config.seeds[0],
        )

    specs = context_arms(config)
    results: list[ContextSceneResult] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        peaks: dict[str, int | None] = {}
        for spec in specs:
            if stacks.sampler is not None:
                select_cached_prompt(stacks.sampler, _prompt(scene, spec.framing))
            image, elapsed, peak = sample_walk(
                photo,
                stacks=stacks,
                sampler=config.sampler,
                guidance=config.guidance,
                reference=reference,
                eta=config.guidance.eta if spec.guided else 0.0,
                seed=seed,
                start_sigma=spec.start_sigma,
            )
            renders[spec.arm] = image
            seconds[spec.arm] = elapsed
            peaks[spec.arm] = peak
            _write_image(image, image_dir / f"{scene.photo}_seed{seed}_{spec.arm}.png")

        # One pixel set for every arm: the reference shell narrowed by each
        # arm's own segmentation, intersected across all of them, so the depth
        # errors are commensurable and covering shell up buys nothing.
        arm_masks = {
            spec.arm: scorable(
                renders[spec.arm],
                stacks=stacks,
                reference=reference,
                resegment=config.guidance.resegment,
                erosion_radius=config.guidance.erosion_radius,
            )
            for spec in specs
        }
        common = reference.mask.clone()
        for mask in arm_masks.values():
            common &= mask
        eval_mask_common = f"{scene.photo}_seed{seed}_evalmask_common.png"
        _write_mask(common, image_dir / eval_mask_common)

        residuals = {
            spec.arm: signed_depth_residual_map(
                renders[spec.arm],
                reference=reference,
                depth_estimator=stacks.depth_estimator,
                mask=common,
            )
            for spec in specs
        }
        baseline_scale = mean_absolute_residual(residuals[BASELINE_ARM], common)

        measurements: list[ContextArmResult] = []
        for spec in specs:
            render = renders[spec.arm]
            depth_map = f"{scene.photo}_seed{seed}_{spec.arm}_depth.png"
            residual_map = f"{scene.photo}_seed{seed}_{spec.arm}_residual.png"
            with torch.no_grad():
                write_masked_disparity_map(
                    stacks.depth_estimator(render),
                    mask=common,
                    path=image_dir / depth_map,
                )
            write_signed_residual_map(
                residuals[spec.arm],
                mask=common,
                normalizer=baseline_scale,
                path=image_dir / residual_map,
            )
            measurements.append(
                ContextArmResult(
                    arm=spec.arm,
                    framing=spec.framing,
                    guided=spec.guided,
                    prompt=_prompt(scene, spec.framing),
                    start_sigma=spec.start_sigma,
                    effective_first_sigma=effective_sigmas[spec.start_sigma],
                    depth_error=depth_error(
                        render,
                        reference=reference,
                        depth_estimator=stacks.depth_estimator,
                        mask=common,
                    ),
                    shell_coverage=shell_coverage(
                        arm_masks[spec.arm], reference=reference
                    ),
                    adherence=adherence(
                        render,
                        prompt=scene.eval_prompt,
                        photo=photo,
                        reference=reference,
                    ),
                    render=f"{scene.photo}_seed{seed}_{spec.arm}.png",
                    depth_map=depth_map,
                    residual_map=residual_map,
                    seconds=seconds[spec.arm],
                    peak_vram_bytes=peaks[spec.arm],
                )
            )

        result = ContextSceneResult(
            scene=scene.photo,
            seed=seed,
            style_prompt=scene.style_prompt,
            renovation_prompt=scene.renovation_prompt,
            eval_prompt=scene.eval_prompt,
            arms=tuple(measurements),
            common_coverage=shell_coverage(common, reference=reference),
            reconstruction_depth_error=reconstruction,
            mean_abs_baseline_residual=baseline_scale,
            photo_image=photo_image,
            photo_depth_map=photo_depth_map,
            photo_depth_map_masked=photo_depth_map_masked,
            shell_mask=shell_mask,
            eval_mask_common=eval_mask_common,
        )
        print(f"  {_format_result(result)}")
        results.append(result)
    return results


def _effective_first_sigmas(
    photo: Float[Tensor, "1 3 height width"],
    *,
    config: FmrgContextConfig,
    stacks: Stacks,
) -> dict[float, float]:
    """The sigma each walk actually starts at, read off the backend's own schedule.

    FLUX's mu-shift warps the grid by resolution, so the configured
    `start_sigma` is a request, not the truth — 0.9 at 512² begins near 0.944.
    The window truncation keeps the schedule's head, so the first entry here is
    the first sigma of the walk whatever the window does.
    """
    generator = torch.Generator(device=stacks.device).manual_seed(0)
    with torch.no_grad():
        encoded = stacks.flow_backend.encode(photo, generator=generator)
    requested = dict.fromkeys(spec.start_sigma for spec in context_arms(config))
    return {
        sigma: float(
            stacks.flow_backend.schedule(
                steps=config.sampler.grid_steps, start_sigma=sigma, latent=encoded
            )[0]
        )
        for sigma in requested
    }


def _prompt(scene: FramingSceneConfig, framing: Framing) -> str:
    return scene.style_prompt if framing == "style" else scene.renovation_prompt


def _format_result(result: ContextSceneResult) -> str:
    cells = ", ".join(
        f"{cell.arm} E={cell.depth_error:.4f} adh={cell.adherence:.4f}"
        for cell in result.arms
    )
    return (
        f"seed {result.seed}: {cells}, shell scored "
        f"{result.common_coverage:.0%} in common"
    )


def arm_means(
    results: tuple[ContextSceneResult, ...], specs: tuple[ContextArmSpec, ...]
) -> dict[str, dict[str, float]]:
    """Each arm's mean depth error, coverage and adherence over every (scene, seed)."""
    means: dict[str, dict[str, float]] = {}
    for spec in specs:
        cells = [result.arm(spec.arm) for result in results]
        means[spec.arm] = {
            "depth_error": _mean([cell.depth_error for cell in cells]),
            "shell_coverage": _mean([cell.shell_coverage for cell in cells]),
            "adherence": _mean([cell.adherence for cell in cells]),
            "seconds": _mean([cell.seconds for cell in cells]),
        }
    return means


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_results(summary: ContextSummary, settings: RoomgenSettings) -> None:
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
                "arm_means": arm_means(summary.results, context_arms(summary.config)),
                "runs": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
