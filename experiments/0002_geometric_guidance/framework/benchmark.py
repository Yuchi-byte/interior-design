"""The run itself: guided against unguided control at identical seeds, and what it measured."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torchvision.io import write_png

from framework.backends import Backends, build_backends
from framework.config import BenchmarkConfig, SceneConfig, build_schedule
from framework.metrics import build_adherence
from framework.photos import ensure_photos, load_photo
from framework.residual_maps import mean_absolute_residual, write_signed_residual_map
from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.eval import (
    PromptAdherence,
    depth_error,
    shell_coverage,
    signed_depth_residual_map,
)
from roomgen_guide.guidance.schedule import ConstantStrength, GuidanceSchedule
from roomgen_guide.metrics import RunMetrics
from roomgen_guide.renovate import renovate

# `_renoised_timesteps` is imported rather than restated so the number of steps
# a run records having walked cannot drift from the number the sampler walks.
from roomgen_guide.sampling.guided import _renoised_timesteps, decode_latent
from roomgen_guide.sampling.pipeline import LatentAutoencoder
from roomgen_guide.settings import RoomgenSettings

PHOTO_SUBDIRECTORY = "benchmark_photos"

# One throwaway sample, this many steps, before the first timed arm of a run:
# the first call through a fresh device context pays for autotuning and lazy
# initialisation, which would otherwise be charged to whichever arm ran first.
_WARM_UP_STEPS: Final = 1

# The three points of the walk the mask snapshots are taken at, in order: what
# the reward scored at the first guided step, the middle one, and the last.
_MASK_SNAPSHOTS: Final = ("first", "mid", "last")


@dataclass
class _MaskLog:
    """Each guided step's effective mask, for the snapshots written after the run.

    Appending a bool tensor the energy already built is all the loop pays; the
    quantising and the PNG encoding happen once sampling is over, so recording
    what the reward saw does not show up in the timing of what it steered.
    """

    masks: list[Bool[Tensor, "1 height width"]] = field(default_factory=list)

    def __call__(self, progress: float, mask: Bool[Tensor, "1 height width"]) -> None:
        self.masks.append(mask)

    def snapshots(self) -> list[Bool[Tensor, "1 height width"]]:
        """The mask at the first, middle and last guided step; empty if none was guided."""
        if not self.masks:
            return []
        return [self.masks[0], self.masks[len(self.masks) // 2], self.masks[-1]]


@dataclass(frozen=True)
class PairResult:
    """One guided run, its control at the same seed, and what the pair measured."""

    scene: str
    prompt: str
    seed: int
    metrics: RunMetrics
    mean_abs_control_residual: float
    depth_map_control: str
    depth_map_guided: str
    eval_mask_common: str
    eval_mask_guided: str
    eval_mask_control: str
    mask_first: str | None
    mask_mid: str | None
    mask_last: str | None
    guided_seconds: float
    control_seconds: float

    def as_row(self) -> dict[str, object]:
        metrics = self.metrics
        return {
            "scene": self.scene,
            "prompt": self.prompt,
            "seed": self.seed,
            "guided_depth_error": metrics.guided_depth_error,
            "control_depth_error": metrics.control_depth_error,
            "reconstruction_depth_error": metrics.reconstruction_depth_error,
            "depth_error_reduction": metrics.depth_error_reduction,
            "mean_abs_control_residual": self.mean_abs_control_residual,
            "depth_map_control": self.depth_map_control,
            "depth_map_guided": self.depth_map_guided,
            "common_coverage": metrics.common_coverage,
            "guided_shell_coverage": metrics.guided_shell_coverage,
            "control_shell_coverage": metrics.control_shell_coverage,
            "eval_mask_common": self.eval_mask_common,
            "eval_mask_guided": self.eval_mask_guided,
            "eval_mask_control": self.eval_mask_control,
            "mask_first": self.mask_first,
            "mask_mid": self.mask_mid,
            "mask_last": self.mask_last,
            "guided_prompt_adherence": metrics.guided_prompt_adherence,
            "control_prompt_adherence": metrics.control_prompt_adherence,
            "adherence_ratio": metrics.adherence_ratio,
            "guided_seconds": self.guided_seconds,
            "control_seconds": self.control_seconds,
        }


@dataclass(frozen=True)
class Summary:
    """What the run measured, where its artifacts landed, and the code that produced them."""

    config: BenchmarkConfig
    results: tuple[PairResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


def run_benchmark(config: BenchmarkConfig, settings: RoomgenSettings) -> Summary:
    """Run every (scene, seed) of `config` guided and unguided, measure both, write the artifacts."""
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

    results: list[PairResult] = []
    for index, scene in enumerate(config.scenes):
        results.extend(
            _run_scene(
                scene,
                config=config,
                backends=backends,
                adherence=adherence,
                photo_path=_photo_path(scene, photo_paths),
                image_dir=image_dir,
                warm_up=index == 0,
            )
        )

    on_cuda = backends.device.type == "cuda"
    summary = Summary(
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
    config: BenchmarkConfig,
    backends: Backends,
    adherence: PromptAdherence,
    photo_path: Path,
    image_dir: Path,
    warm_up: bool = False,
) -> list[PairResult]:
    photo = load_photo(
        photo_path,
        resolution=config.resolution,
        device=backends.device,
        dtype=backends.dtype,
    )
    # Frozen once and handed to both samples and the eval, so guidance is
    # steered by the identical (masked depth, mask) pair the score is read
    # against, and the scene pays one segmentation and one depth pass.
    reference = GeometricReference.from_photo(
        photo,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        erosion_radius=config.guidance.erosion_radius,
    )
    _write_image(photo, image_dir / f"{scene.photo}_photo.png")
    _write_mask(reference.mask, image_dir / f"{scene.photo}_shell_mask.png")

    # The floor the two arms' depth errors are read against: what a *perfect*
    # geometry-preserving renovation still pays for the VAE round trip alone. On
    # the reference's own shell, unlike the two arms: the round trip renovates
    # nothing, so it has no furniture of its own to be excused for and no
    # segmentation of its own to narrow what it is scored on.
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
        _warm_up(
            photo, scene.prompt, backends=backends, config=config, reference=reference
        )

    results: list[PairResult] = []
    for seed in config.seeds:
        # Only the guided arm has masks to record: the control's schedule is
        # flat zero, so its energy is never evaluated.
        mask_log = _MaskLog()
        guided, guided_seconds = _sample(
            photo,
            scene.prompt,
            backends=backends,
            config=config,
            reference=reference,
            schedule=build_schedule(config.guidance),
            seed=seed,
            on_mask=mask_log,
        )
        # Strength 0 is the control: the same photo, prompt, seed, sampler and
        # trajectory, with the guidance term switched off — every difference
        # between the two images is guidance.
        control, control_seconds = _sample(
            photo,
            scene.prompt,
            backends=backends,
            config=config,
            reference=reference,
            schedule=ConstantStrength(strength=0.0),
            seed=seed,
        )
        _write_image(guided, image_dir / f"{scene.photo}_seed{seed}_guided.png")
        _write_image(control, image_dir / f"{scene.photo}_seed{seed}_control.png")
        mask_first, mask_mid, mask_last = _write_mask_snapshots(
            mask_log, image_dir=image_dir, scene=scene.photo, seed=seed
        )

        masks = _evaluation_masks(
            guided, control, backends=backends, config=config, reference=reference
        )
        eval_mask_common = f"{scene.photo}_seed{seed}_evalmask_common.png"
        eval_mask_guided = f"{scene.photo}_seed{seed}_evalmask_guided.png"
        eval_mask_control = f"{scene.photo}_seed{seed}_evalmask_control.png"
        _write_mask(masks.common, image_dir / eval_mask_common)
        _write_mask(masks.guided, image_dir / eval_mask_guided)
        _write_mask(masks.control, image_dir / eval_mask_control)

        control_residual = signed_depth_residual_map(
            control,
            reference=reference,
            depth_estimator=backends.depth_estimator,
            mask=masks.common,
        )
        guided_residual = signed_depth_residual_map(
            guided,
            reference=reference,
            depth_estimator=backends.depth_estimator,
            mask=masks.common,
        )
        # The control sets the scale for both arms, so a colour means the same
        # deviation in either map and the pair can be read against each other.
        mean_abs_control_residual = mean_absolute_residual(
            control_residual, masks.common
        )
        depth_map_control = f"{scene.photo}_seed{seed}_depth_control.png"
        depth_map_guided = f"{scene.photo}_seed{seed}_depth_guided.png"
        for residual, name in (
            (control_residual, depth_map_control),
            (guided_residual, depth_map_guided),
        ):
            write_signed_residual_map(
                residual,
                mask=masks.common,
                normalizer=mean_abs_control_residual,
                path=image_dir / name,
            )

        result = PairResult(
            scene=scene.photo,
            prompt=scene.prompt,
            seed=seed,
            metrics=RunMetrics(
                guided_depth_error=depth_error(
                    guided,
                    reference=reference,
                    depth_estimator=backends.depth_estimator,
                    mask=masks.common,
                ),
                control_depth_error=depth_error(
                    control,
                    reference=reference,
                    depth_estimator=backends.depth_estimator,
                    mask=masks.common,
                ),
                guided_prompt_adherence=adherence(
                    guided, prompt=scene.prompt, photo=photo, reference=reference
                ),
                control_prompt_adherence=adherence(
                    control, prompt=scene.prompt, photo=photo, reference=reference
                ),
                reconstruction_depth_error=reconstruction,
                common_coverage=shell_coverage(masks.common, reference=reference),
                guided_shell_coverage=shell_coverage(masks.guided, reference=reference),
                control_shell_coverage=shell_coverage(
                    masks.control, reference=reference
                ),
            ),
            mean_abs_control_residual=mean_abs_control_residual,
            depth_map_control=depth_map_control,
            depth_map_guided=depth_map_guided,
            eval_mask_common=eval_mask_common,
            eval_mask_guided=eval_mask_guided,
            eval_mask_control=eval_mask_control,
            mask_first=mask_first,
            mask_mid=mask_mid,
            mask_last=mask_last,
            guided_seconds=guided_seconds,
            control_seconds=control_seconds,
        )
        print(f"  {_format_result(result)}")
        results.append(result)
    return results


def _sample(
    photo: Float[Tensor, "1 3 height width"],
    prompt: str,
    *,
    backends: Backends,
    config: BenchmarkConfig,
    reference: GeometricReference,
    schedule: GuidanceSchedule,
    seed: int,
    steps: int | None = None,
    on_mask: _MaskLog | None = None,
) -> tuple[Float[Tensor, "1 3 height width"], float]:
    # The generator lives on the sampling device: the loop draws its noise with
    # `torch.randn(..., generator=..., device=latent.device)`, which refuses a
    # CPU generator for a CUDA latent.
    generator = torch.Generator(device=backends.device).manual_seed(seed)
    # CUDA kernels are launched asynchronously, so a wall clock taken around the
    # call measures queueing unless both ends are synchronised: without the
    # first, the previous arm's tail lands inside this one's window.
    _synchronize(backends.device)
    started = time.perf_counter()
    image = renovate(
        photo,
        prompt,
        pipeline=backends.pipeline,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        schedule=schedule,
        erosion_radius=config.guidance.erosion_radius,
        steps=config.sampler.steps if steps is None else steps,
        noise_strength=config.sampler.noise_strength,
        text_guidance_scale=config.sampler.text_guidance_scale,
        generator=generator,
        normalize_gradient=config.guidance.normalize,
        clip_step_rms=config.guidance.clip,
        differentiate_through_unet=config.guidance.through_unet,
        resegment=config.guidance.resegment,
        resegment_after=config.guidance.resegment_after,
        mask_mode=config.guidance.mask_mode,
        on_mask=on_mask,
        reference=reference,
    )
    _synchronize(backends.device)
    return image, time.perf_counter() - started


def _warm_up(
    photo: Float[Tensor, "1 3 height width"],
    prompt: str,
    *,
    backends: Backends,
    config: BenchmarkConfig,
    reference: GeometricReference,
) -> None:
    """One throwaway guided step before the run's first timed arm; its output is discarded.

    Everything the first sample of a process pays once — kernel autotuning, lazy
    module initialisation, the allocator growing its pools — would otherwise be
    billed to whichever arm happened to run first, which is the guided one.
    """
    print(f"[warmup] {_WARM_UP_STEPS} throwaway step(s), untimed")
    _sample(
        photo,
        prompt,
        backends=backends,
        config=config,
        reference=reference,
        schedule=build_schedule(config.guidance),
        seed=config.seeds[0],
        steps=_WARM_UP_STEPS,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class _EvaluationMasks:
    """The pixels both arms are scored on, beside each arm's own for diagnosis."""

    common: Bool[Tensor, "1 height width"]
    guided: Bool[Tensor, "1 height width"]
    control: Bool[Tensor, "1 height width"]


def _evaluation_masks(
    guided: Float[Tensor, "1 3 height width"],
    control: Float[Tensor, "1 3 height width"],
    *,
    backends: Backends,
    config: BenchmarkConfig,
    reference: GeometricReference,
) -> _EvaluationMasks:
    """The one pixel set the pair is scored on: the shell *both* renders left visible.

    Scoring each arm on its own shell makes the two errors incommensurable —
    different pixel sets, different alignment support — and pays an arm for
    hiding the geometry it moved, since a mask that shrinks takes the evidence
    with it. Intersecting the two makes that trade worthless: covering the shell
    up shrinks the comparison for the control as well, and `common_coverage`
    reports how much of the room the pair was compared over at all.
    """
    guided_scorable = _scorable(
        guided, backends=backends, config=config, reference=reference
    )
    control_scorable = _scorable(
        control, backends=backends, config=config, reference=reference
    )
    return _EvaluationMasks(
        common=guided_scorable & control_scorable,
        guided=guided_scorable,
        control=control_scorable,
    )


def _scorable(
    image: Float[Tensor, "1 3 height width"],
    *,
    backends: Backends,
    config: BenchmarkConfig,
    reference: GeometricReference,
) -> Bool[Tensor, "1 height width"]:
    """The reference shell narrowed by one finished render's own segmentation.

    One segmentation pass per arm, under `no_grad`: nothing downstream of a
    finished render is differentiated, and the same segmenter, erosion and mask
    polarity the guidance used are the ones the score is read under — including
    `resegment: false`, which steers on the reference shell alone and so is
    judged on it alone, rather than under semantics the sampler never saw.
    """
    if not config.guidance.resegment:
        return reference.mask
    with torch.no_grad():
        segmentation = backends.segmenter(image)
    return reference.mask & scorable_mask(
        segmentation,
        mode=config.guidance.mask_mode,
        erosion_radius=config.guidance.erosion_radius,
    )


def _vae_roundtrip(
    vae: LatentAutoencoder,
    photo: Float[Tensor, "1 3 height width"],
    *,
    device: torch.device,
) -> Float[Tensor, "1 3 height width"]:
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        distribution = vae.encode(photo * 2.0 - 1.0).latent_dist
        latent = distribution.sample(generator=generator) * vae.config.scaling_factor
        return decode_latent(vae, latent)


def _photo_path(scene: SceneConfig, photo_paths: dict[str, Path]) -> Path:
    if scene.photo not in photo_paths:
        raise ValueError(
            f"scene photo {scene.photo!r} is not in experiments/assets/photos.json; "
            f"known ids: {', '.join(sorted(photo_paths))}"
        )
    return photo_paths[scene.photo]


def _write_image(image: Float[Tensor, "1 3 height width"], path: Path) -> None:
    quantised = (image[0].detach().to(torch.float32).clamp(0.0, 1.0) * 255.0).round()
    write_png(quantised.to(torch.uint8).cpu(), str(path))


def _write_mask(mask: Bool[Tensor, "1 height width"], path: Path) -> None:
    write_png((mask[0].to(torch.uint8) * 255).unsqueeze(0).cpu(), str(path))


def _write_mask_snapshots(
    log: _MaskLog, *, image_dir: Path, scene: str, seed: int
) -> tuple[str | None, str | None, str | None]:
    """Save what the reward scored at the ends and the middle of the walk; name the files."""
    snapshots = log.snapshots()
    if not snapshots:
        return None, None, None
    names = [f"{scene}_seed{seed}_mask_{point}.png" for point in _MASK_SNAPSHOTS]
    for mask, name in zip(snapshots, names, strict=True):
        _write_mask(mask, image_dir / name)
    return names[0], names[1], names[2]


def _format_result(result: PairResult) -> str:
    metrics = result.metrics
    return (
        f"seed {result.seed}: depth {metrics.guided_depth_error:.6f} vs "
        f"{metrics.control_depth_error:.6f} "
        f"({_signed_share(metrics.depth_error_reduction)}), adherence "
        f"{metrics.guided_prompt_adherence:.4f} vs "
        f"{metrics.control_prompt_adherence:.4f} "
        f"({_share(metrics.adherence_ratio)}), shell scored "
        f"{metrics.common_coverage:.0%} in common "
        f"(guided {metrics.guided_shell_coverage:.0%}, "
        f"control {metrics.control_shell_coverage:.0%})"
    )


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _signed_share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def _git_commit() -> str:
    """The commit this run's code came from, so a number can be traced back to it."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _walked_steps(config: BenchmarkConfig) -> int:
    """How many of the scheduler's steps SDEdit actually walks, by the sampler's own arithmetic."""
    return len(
        _renoised_timesteps(
            torch.arange(config.sampler.steps),
            noise_strength=config.sampler.noise_strength,
        )
    )


def _write_results(summary: Summary, settings: RoomgenSettings) -> None:
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
                "runs": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
