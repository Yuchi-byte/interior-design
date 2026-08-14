"""The image-as-context study: the reference photo appended to the T5 context.

Per scene at one seed, six arms off one transformer load, all at the sampler's
one init strength: the style prompt with no image tokens, unguided (A, the
baseline) and guided (C, the anchor matching prior studies), then the
renovation instruction with the photo's FLUX.1-Redux tokens appended at each
configured scale, unguided and guided. The renovation prompt's "this room"
finally binds to actual image tokens rather than dangling. Every arm is scored
on the one pixel set all arms left visible and judged against the scene's one
neutral style text; beside the numbers, every arm leaves a masked depth map and
a signed depth-error map against the reference, all on arm A's colour scale.

The Redux encoders (SigLIP + adapter) are loaded first, while the GPU is
otherwise empty, and freed before the transformer loads; the combined
embeddings then live in the sampler's CPU prompt cache like any text prompt.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from framework.config import TORCH_DTYPES, FmrgReduxConfig, FramingSceneConfig
from framework.depth_maps import write_disparity_map, write_masked_disparity_map
from framework.fmrg_alignment_study import shell_class_table
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
from roomgen_guide.adapters.fmrg_flux import (
    encode_redux_image_tokens,
    register_image_context_prompt,
    select_cached_prompt,
)
from roomgen_guide.eval import (
    PromptAdherence,
    depth_error,
    shell_coverage,
    signed_depth_residual_map,
)
from roomgen_guide.settings import RoomgenSettings

Framing = Literal["style", "renovation"]

# The baseline arm's id: style prompt, no image tokens, unguided. Every other
# arm's depth delta and adherence ratio divide by it, and its mean absolute
# residual sets the scene's residual-map colour scale.
BASELINE_ARM: Final = "A"


def redux_prompt_key(prompt: str, scale: float) -> str:
    """The cache name of `prompt` with the scene's image tokens appended at `scale`."""
    return f"{prompt} || redux tokens x{scale:g}"


@dataclass(frozen=True)
class ReduxArmSpec:
    """One arm of the study; `redux_scale` `None` is a text-only arm."""

    arm: str
    framing: Framing
    guided: bool
    redux_scale: float | None


def redux_arms(config: FmrgReduxConfig) -> tuple[ReduxArmSpec, ...]:
    """The study's arms, baseline first: A, C, then an unguided and a guided arm per scale."""
    arms = [
        ReduxArmSpec(BASELINE_ARM, "style", False, None),
        ReduxArmSpec("C", "style", True, None),
    ]
    index = 1
    for scale in config.redux_scales:
        arms.append(ReduxArmSpec(f"R{index}", "renovation", False, scale))
        arms.append(ReduxArmSpec(f"R{index + 1}", "renovation", True, scale))
        index += 2
    return tuple(arms)


@dataclass(frozen=True)
class ReduxArmResult:
    """One arm's render, its depth reading, and its scores against the baseline's."""

    arm: str
    framing: Framing
    guided: bool
    redux_scale: float | None
    prompt: str
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
            "redux_scale": self.redux_scale,
            "prompt": self.prompt,
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
class ReduxSceneResult:
    """Every arm of one (scene, seed), scored on one common pixel set."""

    scene: str
    seed: int
    style_prompt: str
    renovation_prompt: str
    eval_prompt: str
    arms: tuple[ReduxArmResult, ...]
    common_coverage: float
    reconstruction_depth_error: float
    mean_abs_baseline_residual: float
    photo_image: str
    photo_depth_map: str
    photo_depth_map_masked: str
    shell_mask: str
    eval_mask_common: str

    def arm(self, arm_id: str) -> ReduxArmResult:
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
class ReduxSummary:
    """What the study measured, where its artifacts landed, and the code behind them."""

    config: FmrgReduxConfig
    results: tuple[ReduxSceneResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    caching_peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


def run_redux_study(config: FmrgReduxConfig, settings: RoomgenSettings) -> ReduxSummary:
    """Run every arm of every (scene, seed), measure them all, write the artifacts."""
    started_at = datetime.now(UTC).isoformat()
    output_dir = settings.wdir / config.name
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = ensure_photos(settings.wdir / PHOTO_SUBDIRECTORY)
    for scene in config.scenes:
        if scene.photo not in photo_paths:
            raise ValueError(
                f"scene photo {scene.photo!r} is not in experiments/assets/"
                f"photos.json; known ids: {', '.join(sorted(photo_paths))}"
            )

    # Phase one, on an otherwise empty GPU: each reference photo through SigLIP
    # and the Redux adapter, tokens parked on the CPU, encoders freed — all
    # before the ~24GB transformer exists.
    device = torch.device(config.backends.device)
    on_cuda = device.type == "cuda"
    redux_tokens: dict[str, Float[Tensor, "1 tokens embedding"]] = {}
    caching_peak: int | None = None
    if config.backends.kind == "real":
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        for scene in config.scenes:
            photo = load_photo(
                photo_paths[scene.photo],
                resolution=config.resolution,
                device=device,
                dtype=torch.float32,
            )
            redux_tokens[scene.photo] = encode_redux_image_tokens(
                photo,
                device=device,
                dtype=TORCH_DTYPES[config.backends.dtype],
                cache_dir=settings.hub_cache,
                checkpoint=config.backends.redux_checkpoint,
            )
            tokens = redux_tokens[scene.photo]
            print(
                f"[redux] {scene.photo}: {tokens.shape[1]} image tokens x "
                f"{tokens.shape[2]} wide, cached on CPU"
            )
        if on_cuda:
            caching_peak = torch.cuda.max_memory_allocated(device)

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
    if stacks.sampler is not None:
        for scene in config.scenes:
            for scale in config.redux_scales:
                register_image_context_prompt(
                    stacks.sampler,
                    name=redux_prompt_key(scene.renovation_prompt, scale),
                    prompt=scene.renovation_prompt,
                    image_tokens=redux_tokens[scene.photo],
                    scale=scale,
                )
    if stacks.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(stacks.device)

    pending: list[_PendingScene] = []
    for index, scene in enumerate(config.scenes):
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

    walking_on_cuda = stacks.device.type == "cuda"
    summary = ReduxSummary(
        config=config,
        results=tuple(results),
        output_dir=output_dir,
        peak_vram_bytes=(
            torch.cuda.max_memory_allocated(stacks.device) if walking_on_cuda else None
        ),
        caching_peak_vram_bytes=caching_peak,
        started_at=started_at,
        git_commit=_git_commit(),
        walked_steps=walked_steps(config.sampler),
        device_name=(
            torch.cuda.get_device_name(stacks.device) if walking_on_cuda else None
        ),
    )
    _write_results(summary, settings)
    return summary


@dataclass(frozen=True)
class _PendingScene:
    """One scene measured on everything but adherence, its renders parked on the CPU."""

    result: ReduxSceneResult
    renders: dict[str, Float[Tensor, "1 3 height width"]]
    photo: Float[Tensor, "1 3 height width"]
    reference: GeometricReference


def _scored(
    entry: _PendingScene, *, adherence: PromptAdherence, device: torch.device
) -> ReduxSceneResult:
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
    scene: FramingSceneConfig,
    *,
    config: FmrgReduxConfig,
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

    specs = redux_arms(config)
    results: list[_PendingScene] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        peaks: dict[str, int | None] = {}
        for spec in specs:
            if stacks.sampler is not None:
                select_cached_prompt(stacks.sampler, _prompt_key(scene, spec))
            image, elapsed, peak = sample_walk(
                photo,
                stacks=stacks,
                sampler=config.sampler,
                guidance=config.guidance,
                reference=reference,
                eta=config.guidance.eta if spec.guided else 0.0,
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

        measurements: list[ReduxArmResult] = []
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
                ReduxArmResult(
                    arm=spec.arm,
                    framing=spec.framing,
                    guided=spec.guided,
                    redux_scale=spec.redux_scale,
                    prompt=_prompt_text(scene, spec),
                    effective_first_sigma=effective_sigma,
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

        result = ReduxSceneResult(
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
    config: FmrgReduxConfig,
    stacks: Stacks,
) -> float:
    """The sigma every walk actually starts at, read off the backend's own schedule.

    FLUX's mu-shift warps the grid by resolution, so the configured
    `start_sigma` is a request, not the truth — 0.9 at 512² begins near 0.944.
    One value for the whole study, since every arm shares the one init.
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


def _prompt_key(scene: FramingSceneConfig, spec: ReduxArmSpec) -> str:
    """The prompt-cache entry the arm conditions on; image arms select a combined one."""
    if spec.redux_scale is None:
        return scene.style_prompt
    return redux_prompt_key(scene.renovation_prompt, spec.redux_scale)


def _prompt_text(scene: FramingSceneConfig, spec: ReduxArmSpec) -> str:
    return scene.style_prompt if spec.framing == "style" else scene.renovation_prompt


def _format_result(result: ReduxSceneResult) -> str:
    cells = ", ".join(
        f"{cell.arm} E={cell.depth_error:.4f} adh={cell.adherence:.4f}"
        for cell in result.arms
    )
    return (
        f"seed {result.seed}: {cells}, shell scored "
        f"{result.common_coverage:.0%} in common"
    )


def arm_means(
    results: tuple[ReduxSceneResult, ...], specs: tuple[ReduxArmSpec, ...]
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


def _write_results(summary: ReduxSummary, settings: RoomgenSettings) -> None:
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
                # The shell definition is under revision (windows, doors, soft
                # furnishings joining): the results must say which one scored
                # them, not point at whatever the package says later.
                "shell_classes": shell_class_table(),
                "eta": summary.config.guidance.eta,
                "torch_version": torch.__version__,
                "device_name": summary.device_name,
                "walked_steps": summary.walked_steps,
                "peak_vram_bytes": summary.peak_vram_bytes,
                "caching_peak_vram_bytes": summary.caching_peak_vram_bytes,
                "arm_means": arm_means(summary.results, redux_arms(summary.config)),
                "runs": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
