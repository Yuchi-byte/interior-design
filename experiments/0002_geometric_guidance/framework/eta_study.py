"""The eta sweep: how hard can the depth reward push before it costs the renovation?

FMRG's headline step size (eta 5.0) was inherited, never tuned in this stack,
and the owner reads the guided-vs-unguided visual difference as too small. Per
scene at one seed, one unguided baseline A and one guided arm per swept eta,
all off one transformer load at the sampler's one operating point, every arm
scored on the one pixel set all arms left visible and judged against the
scene's neutral style text. Beside the numbers, every arm leaves a masked depth
map and a signed depth-error map against the reference, on arm A's colour scale.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Float
from torch import Tensor

from framework.config import EtaSceneConfig, FmrgEtaConfig
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

# The baseline arm's id: unguided (eta 0) at the sampler's operating point.
# Every guided arm's depth delta and adherence ratio divide by it, and its mean
# absolute residual sets the scene's residual-map colour scale.
BASELINE_ARM: Final = "A"


@dataclass(frozen=True)
class EtaArmSpec:
    """One arm of the sweep: a guidance strength, everything else held."""

    arm: str
    eta: float

    @property
    def guided(self) -> bool:
        return self.eta > 0.0


def eta_arms(config: FmrgEtaConfig) -> tuple[EtaArmSpec, ...]:
    """The sweep's arms, baseline first: A at eta 0, then one guided arm per eta."""
    return (
        EtaArmSpec(BASELINE_ARM, 0.0),
        *(EtaArmSpec(f"G{eta:g}", eta) for eta in config.etas),
    )


@dataclass(frozen=True)
class EtaArmResult:
    """One arm's render, its depth reading, and its scores against the baseline's."""

    arm: str
    eta: float
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
            "eta": self.eta,
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
class EtaSceneResult:
    """Every arm of one (scene, seed), scored on one common pixel set."""

    scene: str
    seed: int
    style_prompt: str
    eval_prompt: str
    start_sigma: float
    effective_first_sigma: float
    arms: tuple[EtaArmResult, ...]
    common_coverage: float
    reconstruction_depth_error: float
    mean_abs_baseline_residual: float
    photo_image: str
    photo_depth_map: str
    photo_depth_map_masked: str
    shell_mask: str
    eval_mask_common: str

    def arm(self, arm_id: str) -> EtaArmResult:
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
            "eval_prompt": self.eval_prompt,
            "start_sigma": self.start_sigma,
            "effective_first_sigma": self.effective_first_sigma,
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
class EtaSummary:
    """What the sweep measured, where its artifacts landed, and the code behind them."""

    config: FmrgEtaConfig
    results: tuple[EtaSceneResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


def run_eta_study(config: FmrgEtaConfig, settings: RoomgenSettings) -> EtaSummary:
    """Run every arm of every (scene, seed), measure them all, write the artifacts."""
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
                photo_path=photo_paths[scene.photo],
                image_dir=image_dir,
                warm_up=index == 0,
            )
        )

    # Loaded only after the last walk: beside the transformer and the guided
    # decode's backward transient, a resident ViT-L/14 is exactly what tipped a
    # 32GB card into out-of-memory. Renders wait on the CPU; scoring one at a
    # time through a frozen encoder is cheap.
    adherence = build_adherence(
        config.adherence, device=stacks.device, cache_dir=settings.hub_cache
    )
    results = [
        _scored(entry, adherence=adherence, device=stacks.device) for entry in pending
    ]
    for result in results:
        print(f"  {_format_result(result)}")

    on_cuda = stacks.device.type == "cuda"
    summary = EtaSummary(
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


@dataclass(frozen=True)
class _PendingScene:
    """One scene measured on everything but adherence, its renders parked on the CPU."""

    result: EtaSceneResult
    renders: dict[str, Float[Tensor, "1 3 height width"]]
    photo: Float[Tensor, "1 3 height width"]
    reference: GeometricReference


def _scored(
    entry: _PendingScene, *, adherence: PromptAdherence, device: torch.device
) -> EtaSceneResult:
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
    config: FmrgEtaConfig,
    stacks: Stacks,
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
    effective_sigma = _effective_first_sigma(photo, config=config, stacks=stacks)
    print(
        f"[scene] {scene.photo}: shell covers "
        f"{float(reference.mask.to(torch.float32).mean()):.1%} of the frame, "
        f"reconstruction floor {reconstruction:.6f}, effective first sigma "
        f"{config.sampler.start_sigma:g}->{effective_sigma:.4f}"
    )

    if warm_up:
        warm_up_walk(
            photo,
            stacks=stacks,
            sampler=config.sampler,
            guidance=config.guidance,
            reference=reference,
            seed=config.seeds[0],
        )

    specs = eta_arms(config)
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
                reference=reference,
                eta=spec.eta,
                seed=seed,
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

        measurements: list[EtaArmResult] = []
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
                EtaArmResult(
                    arm=spec.arm,
                    eta=spec.eta,
                    depth_error=depth_error(
                        render,
                        reference=reference,
                        depth_estimator=stacks.depth_estimator,
                        mask=common,
                    ),
                    shell_coverage=shell_coverage(
                        arm_masks[spec.arm], reference=reference
                    ),
                    # Filled in by `_scored` once sampling is over and the
                    # adherence CLIP can afford to be resident.
                    adherence=math.nan,
                    render=f"{scene.photo}_seed{seed}_{spec.arm}.png",
                    depth_map=depth_map,
                    residual_map=residual_map,
                    seconds=seconds[spec.arm],
                    peak_vram_bytes=peaks[spec.arm],
                )
            )

        result = EtaSceneResult(
            scene=scene.photo,
            seed=seed,
            style_prompt=scene.style_prompt,
            eval_prompt=scene.eval_prompt,
            start_sigma=config.sampler.start_sigma,
            effective_first_sigma=effective_sigma,
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
        print(f"  [sampled] {result.scene} seed {result.seed}, adherence pending")
        results.append(
            _PendingScene(
                result=result,
                renders={
                    arm: image.detach().to(device="cpu", dtype=torch.float32)
                    for arm, image in renders.items()
                },
                photo=photo,
                reference=reference,
            )
        )
    return results


def _effective_first_sigma(
    photo: Float[Tensor, "1 3 height width"],
    *,
    config: FmrgEtaConfig,
    stacks: Stacks,
) -> float:
    """The sigma every walk actually starts at, read off the backend's own schedule.

    FLUX's mu-shift warps the grid by resolution, so the configured
    `start_sigma` is a request, not the truth — 0.9 at 512² begins near 0.944.
    One value serves the whole sweep: eta never touches the schedule.
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


def _format_result(result: EtaSceneResult) -> str:
    cells = ", ".join(
        f"{cell.arm} E={cell.depth_error:.4f} adh={cell.adherence:.4f}"
        for cell in result.arms
    )
    return (
        f"seed {result.seed}: {cells}, shell scored "
        f"{result.common_coverage:.0%} in common"
    )


def arm_means(
    results: tuple[EtaSceneResult, ...], specs: tuple[EtaArmSpec, ...]
) -> dict[str, dict[str, float]]:
    """Each arm's mean depth error, coverage and adherence over every (scene, seed)."""
    means: dict[str, dict[str, float]] = {}
    for spec in specs:
        cells = [result.arm(spec.arm) for result in results]
        means[spec.arm] = {
            "eta": spec.eta,
            "depth_error": _mean([cell.depth_error for cell in cells]),
            "shell_coverage": _mean([cell.shell_coverage for cell in cells]),
            "adherence": _mean([cell.adherence for cell in cells]),
            "seconds": _mean([cell.seconds for cell in cells]),
        }
    return means


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_results(summary: EtaSummary, settings: RoomgenSettings) -> None:
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
                "arm_means": arm_means(summary.results, eta_arms(summary.config)),
                "runs": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
