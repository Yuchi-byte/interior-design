"""The prompt-framing study: one renovation, two phrasings, with and without geometry.

A 2x2 per (scene, seed) at matched seeds — {style description, renovation
instruction} x {unguided, guided} — so each axis varies exactly the thing it
names. All four arms are scored on the one pixel set all four left visible, and
judged against one neutral style text, never an arm's own prompt. Beside the
numbers, every render and the reference photo leave behind an absolute
disparity map, which is the geometry claim made visible.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from framework.backends import Backends, build_backends
from framework.benchmark import (
    PHOTO_SUBDIRECTORY,
    _git_commit,
    _synchronize,
    _vae_roundtrip,
    _write_image,
    _write_mask,
)
from framework.config import FramingConfig, FramingSceneConfig
from framework.depth_maps import write_disparity_map
from framework.metrics import build_adherence
from framework.photos import ensure_photos, load_photo
from framework.preference import build_preference
from framework.residual_maps import mean_absolute_residual, write_signed_residual_map
from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.eval import (
    PromptAdherence,
    depth_error,
    shell_coverage,
    signed_depth_residual_map,
)
from roomgen_guide.guidance.preference import ImagePreferenceModel
from roomgen_guide.renovate import renovate

# `_renoised_timesteps` is imported rather than restated so the number of steps
# a run records having walked cannot drift from the number the sampler walks.
from roomgen_guide.sampling.guided import _renoised_timesteps
from roomgen_guide.settings import RoomgenSettings

Framing = Literal["style", "renovation"]

_WARM_UP_STEPS: Final = 1


@dataclass(frozen=True)
class ArmSpec:
    """One cell of the 2x2, named by both of its axes."""

    arm: str
    framing: Framing
    guided: bool


# The baseline arm comes first: the residual-map colour scale of a (scene, seed)
# is denominated in its mean absolute residual, so it must be measured first.
ARMS: Final[tuple[ArmSpec, ...]] = (
    ArmSpec("style_unguided", "style", False),
    ArmSpec("renovation_unguided", "renovation", False),
    ArmSpec("style_guided", "style", True),
    ArmSpec("renovation_guided", "renovation", True),
)


@dataclass(frozen=True)
class ArmMeasurement:
    """One arm of one (scene, seed): its render, its depth field, and its three scores."""

    arm: str
    framing: Framing
    guided: bool
    prompt: str
    depth_error: float
    shell_coverage: float
    adherence: float
    preference: float
    render: str
    depth_map: str
    residual_map: str
    seconds: float

    def as_row(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "framing": self.framing,
            "guided": self.guided,
            "prompt": self.prompt,
            "depth_error": self.depth_error,
            "shell_coverage": self.shell_coverage,
            "adherence": self.adherence,
            "preference": self.preference,
            "render": self.render,
            "depth_map": self.depth_map,
            "residual_map": self.residual_map,
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class FramingResult:
    """The full 2x2 of one (scene, seed), scored on one common pixel set."""

    scene: str
    seed: int
    style_prompt: str
    renovation_prompt: str
    eval_prompt: str
    arms: tuple[ArmMeasurement, ...]
    common_coverage: float
    reconstruction_depth_error: float
    mean_abs_baseline_residual: float
    photo_image: str
    photo_depth_map: str
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
            "renovation_prompt": self.renovation_prompt,
            "eval_prompt": self.eval_prompt,
            "common_coverage": self.common_coverage,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "mean_abs_baseline_residual": self.mean_abs_baseline_residual,
            "photo_image": self.photo_image,
            "photo_depth_map": self.photo_depth_map,
            "shell_mask": self.shell_mask,
            "eval_mask_common": self.eval_mask_common,
            "arms": [measurement.as_row() for measurement in self.arms],
        }


@dataclass(frozen=True)
class FramingSummary:
    """What the study measured, where its artifacts landed, and the code that produced them."""

    config: FramingConfig
    results: tuple[FramingResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


def run_framing(config: FramingConfig, settings: RoomgenSettings) -> FramingSummary:
    """Run the 2x2 over every (scene, seed) of `config`, measure it, write every artifact."""
    started_at = datetime.now(UTC).isoformat()
    output_dir = settings.wdir / config.name
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = ensure_photos(settings.wdir / PHOTO_SUBDIRECTORY)
    backends = build_backends(config.backends, cache_dir=settings.hub_cache)
    adherence = build_adherence(
        config.adherence, device=backends.device, cache_dir=settings.hub_cache
    )
    if backends.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backends.device)

    pending: list[_PendingResult] = []
    for index, scene in enumerate(config.scenes):
        if scene.photo not in photo_paths:
            raise ValueError(
                f"scene photo {scene.photo!r} is not in experiments/assets/photos.json; "
                f"known ids: {', '.join(sorted(photo_paths))}"
            )
        pending.extend(
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

    # Loaded only after the last sample: PickScore's CLIP-H is ~4GB resident,
    # and beside the guided VAE backward's transient peak that is what turned a
    # 32GB card into an out-of-memory error. Renders wait on the CPU; scoring
    # one at a time through a frozen encoder is cheap.
    preference = build_preference(
        config.preference, device=backends.device, cache_dir=settings.hub_cache
    )
    results = [
        _scored(entry, preference=preference, device=backends.device)
        for entry in pending
    ]
    for result in results:
        print(f"  {_format_result(result)}")

    on_cuda = backends.device.type == "cuda"
    summary = FramingSummary(
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


@dataclass(frozen=True)
class _PendingResult:
    """One grid measured on everything but preference, its renders parked on the CPU."""

    result: FramingResult
    renders: dict[str, Float[Tensor, "1 3 height width"]]


def _scored(
    entry: _PendingResult, *, preference: ImagePreferenceModel, device: torch.device
) -> FramingResult:
    """The grid with its preference scores filled in, each arm against the neutral text."""
    with torch.no_grad():
        arms = tuple(
            replace(
                cell,
                preference=float(
                    preference.score(
                        entry.renders[cell.arm].to(device),
                        [entry.result.eval_prompt],
                    )[0]
                ),
            )
            for cell in entry.result.arms
        )
    return replace(entry.result, arms=arms)


def _run_scene(
    scene: FramingSceneConfig,
    *,
    config: FramingConfig,
    backends: Backends,
    adherence: PromptAdherence,
    photo_path: Path,
    image_dir: Path,
    warm_up: bool,
) -> list[_PendingResult]:
    photo = load_photo(
        photo_path,
        resolution=config.resolution,
        device=backends.device,
        dtype=backends.dtype,
    )
    # Frozen once and handed to all four arms and the eval, so every sample is
    # steered by — and scored against — the identical (masked depth, mask) pair.
    reference = GeometricReference.from_photo(
        photo,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        erosion_radius=config.guidance.erosion_radius,
    )
    photo_image = f"{scene.photo}_photo.png"
    shell_mask = f"{scene.photo}_shell_mask.png"
    photo_depth_map = f"{scene.photo}_photo_depth.png"
    _write_image(photo, image_dir / photo_image)
    _write_mask(reference.mask, image_dir / shell_mask)
    # The reference's disparity is the full-frame prediction on the photo — the
    # owner-facing "what the depth model saw in the room being renovated".
    write_disparity_map(reference.disparity, image_dir / photo_depth_map)

    reconstruction = depth_error(
        _vae_roundtrip(backends.pipeline.vae, photo, device=backends.device),
        reference=reference,
        depth_estimator=backends.depth_estimator,
    )
    print(
        f"[scene] {scene.photo}: shell covers "
        f"{float(reference.mask.to(torch.float32).mean()):.1%} of the frame, "
        f"reconstruction floor {reconstruction:.6f}"
    )

    if warm_up:
        print(f"[warmup] {_WARM_UP_STEPS} throwaway step(s), untimed")
        _sample(
            photo,
            scene.style_prompt,
            backends=backends,
            config=config,
            reference=reference,
            strength=config.guidance.strength,
            seed=config.seeds[0],
            steps=_WARM_UP_STEPS,
        )

    results: list[_PendingResult] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        for spec in ARMS:
            image, elapsed = _sample(
                photo,
                _prompt(scene, spec.framing),
                backends=backends,
                config=config,
                reference=reference,
                strength=config.guidance.strength if spec.guided else 0.0,
                seed=seed,
            )
            renders[spec.arm] = image
            seconds[spec.arm] = elapsed
            _write_image(image, image_dir / f"{scene.photo}_seed{seed}_{spec.arm}.png")

        # One pixel set for the whole 2x2: the reference shell narrowed by each
        # arm's own segmentation, intersected across all four, so the four depth
        # errors are commensurable and covering shell up buys nothing.
        arm_masks = {
            spec.arm: _scorable(
                renders[spec.arm], backends=backends, config=config, reference=reference
            )
            for spec in ARMS
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
                depth_estimator=backends.depth_estimator,
                mask=common,
            )
            for spec in ARMS
        }
        # The style-description unguided arm is the study's baseline: it sets
        # one colour scale per (scene, seed), so a colour means the same
        # deviation in all four maps.
        baseline_scale = mean_absolute_residual(residuals[ARMS[0].arm], common)

        measurements: list[ArmMeasurement] = []
        for spec in ARMS:
            render = renders[spec.arm]
            depth_map = f"{scene.photo}_seed{seed}_{spec.arm}_depth.png"
            residual_map = f"{scene.photo}_seed{seed}_{spec.arm}_residual.png"
            with torch.no_grad():
                write_disparity_map(
                    backends.depth_estimator(render), image_dir / depth_map
                )
            write_signed_residual_map(
                residuals[spec.arm],
                mask=common,
                normalizer=baseline_scale,
                path=image_dir / residual_map,
            )
            measurements.append(
                ArmMeasurement(
                    arm=spec.arm,
                    framing=spec.framing,
                    guided=spec.guided,
                    prompt=_prompt(scene, spec.framing),
                    depth_error=depth_error(
                        render,
                        reference=reference,
                        depth_estimator=backends.depth_estimator,
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
                    # Filled in by `_scored` once sampling is over and the
                    # preference model can afford to be resident.
                    preference=math.nan,
                    render=f"{scene.photo}_seed{seed}_{spec.arm}.png",
                    depth_map=depth_map,
                    residual_map=residual_map,
                    seconds=seconds[spec.arm],
                )
            )

        result = FramingResult(
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
            shell_mask=shell_mask,
            eval_mask_common=eval_mask_common,
        )
        print(f"  [sampled] {result.scene} seed {result.seed}, preference pending")
        results.append(
            _PendingResult(
                result=result,
                renders={
                    arm: image.detach().to(device="cpu", dtype=torch.float32)
                    for arm, image in renders.items()
                },
            )
        )
    return results


def _prompt(scene: FramingSceneConfig, framing: Framing) -> str:
    return scene.style_prompt if framing == "style" else scene.renovation_prompt


def _sample(
    photo: Float[Tensor, "1 3 height width"],
    prompt: str,
    *,
    backends: Backends,
    config: FramingConfig,
    reference: GeometricReference,
    strength: float,
    seed: int,
    steps: int | None = None,
) -> tuple[Float[Tensor, "1 3 height width"], float]:
    # The generator lives on the sampling device, and every arm gets a fresh one
    # at the same seed: matched noise, so the arms differ in prompt and guidance
    # alone.
    generator = torch.Generator(device=backends.device).manual_seed(seed)
    _synchronize(backends.device)
    started = time.perf_counter()
    image = renovate(
        photo,
        prompt,
        pipeline=backends.pipeline,
        depth_estimator=backends.depth_estimator,
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
    )
    _synchronize(backends.device)
    return image, time.perf_counter() - started


def _scorable(
    image: Float[Tensor, "1 3 height width"],
    *,
    backends: Backends,
    config: FramingConfig,
    reference: GeometricReference,
) -> Bool[Tensor, "1 height width"]:
    """The reference shell narrowed by one finished render's own segmentation."""
    if not config.guidance.resegment:
        return reference.mask
    with torch.no_grad():
        segmentation = backends.segmenter(image)
    return reference.mask & scorable_mask(
        segmentation, erosion_radius=config.guidance.erosion_radius
    )


def _format_result(result: FramingResult) -> str:
    cells = ", ".join(
        f"{measurement.arm} E={measurement.depth_error:.4f} "
        f"adh={measurement.adherence:.4f} pref={measurement.preference:.4f}"
        for measurement in result.arms
    )
    return (
        f"seed {result.seed}: {cells}, shell scored "
        f"{result.common_coverage:.0%} in common"
    )


def _walked_steps(config: FramingConfig) -> int:
    """How many of the scheduler's steps SDEdit actually walks, by the sampler's own arithmetic."""
    return len(
        _renoised_timesteps(
            torch.arange(config.sampler.steps),
            noise_strength=config.sampler.noise_strength,
        )
    )


def arm_means(results: tuple[FramingResult, ...]) -> dict[str, dict[str, float]]:
    """Each arm's mean depth error, coverage, adherence and preference over every (scene, seed)."""
    means: dict[str, dict[str, float]] = {}
    for spec in ARMS:
        cells = [result.arm(spec.arm) for result in results]
        means[spec.arm] = {
            "depth_error": _mean([cell.depth_error for cell in cells]),
            "shell_coverage": _mean([cell.shell_coverage for cell in cells]),
            "adherence": _mean([cell.adherence for cell in cells]),
            "preference": _mean([cell.preference for cell in cells]),
        }
    return means


def framing_comparisons(
    results: tuple[FramingResult, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    """The owner's two comparisons: style vs renovation framing, unguided and guided.

    Per metric, the two arm means and their delta (renovation minus style). A
    delta is a measurement, not a verdict: lower is the direction geometry
    improves for `depth_error`, higher for the other two.
    """
    means = arm_means(results)
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    for label, style_arm, renovation_arm in (
        ("unguided_style_vs_renovation", "style_unguided", "renovation_unguided"),
        ("guided_style_vs_renovation", "style_guided", "renovation_guided"),
    ):
        comparisons[label] = {
            metric: {
                "style": means[style_arm][metric],
                "renovation": means[renovation_arm][metric],
                "delta": means[renovation_arm][metric] - means[style_arm][metric],
            }
            for metric in ("depth_error", "adherence", "preference")
        }
    return comparisons


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_results(summary: FramingSummary, settings: RoomgenSettings) -> None:
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
                "arm_means": arm_means(summary.results),
                "comparisons": framing_comparisons(summary.results),
                "runs": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
