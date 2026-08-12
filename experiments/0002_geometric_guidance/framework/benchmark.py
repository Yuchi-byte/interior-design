"""The run itself: guided against unguided control at identical seeds, then the gate."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor
from torchvision.io import write_png

from framework.backends import Backends, build_backends
from framework.config import BenchmarkConfig, SceneConfig, build_schedule
from framework.metrics import build_adherence
from framework.photos import ensure_photos, load_photo
from framework.residual_maps import (
    mean_masked_residual,
    write_residual_comparison,
    write_residual_difference,
)
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.eval import PromptAdherence, depth_error, depth_residual_map
from roomgen_guide.guidance.schedule import ConstantStrength, GuidanceSchedule
from roomgen_guide.renovate import renovate
from roomgen_guide.sampling.guided import decode_latent
from roomgen_guide.sampling.pipeline import LatentAutoencoder
from roomgen_guide.settings import RoomgenSettings
from roomgen_guide.verdict import Verdict, score_run

PHOTO_SUBDIRECTORY = "benchmark_photos"


@dataclass(frozen=True)
class PairResult:
    """One guided run, its control at the same seed, and the verdict between them."""

    scene: str
    prompt: str
    seed: int
    verdict: Verdict
    reconstruction_depth_error: float
    mean_control_residual: float
    residual_comparison_map: str
    residual_difference_map: str
    guided_seconds: float
    control_seconds: float

    @property
    def depth_error_reduction(self) -> float:
        """Fraction of the control's depth error that guidance removed."""
        if self.verdict.control_depth_error == 0.0:
            return 0.0
        return 1.0 - self.verdict.guided_depth_error / self.verdict.control_depth_error

    @property
    def adherence_ratio(self) -> float:
        if self.verdict.control_prompt_adherence == 0.0:
            return 0.0
        return (
            self.verdict.guided_prompt_adherence / self.verdict.control_prompt_adherence
        )

    def as_row(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "prompt": self.prompt,
            "seed": self.seed,
            "guided_depth_error": self.verdict.guided_depth_error,
            "control_depth_error": self.verdict.control_depth_error,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "depth_error_reduction": self.depth_error_reduction,
            "mean_control_residual": self.mean_control_residual,
            "residual_comparison_map": self.residual_comparison_map,
            "residual_difference_map": self.residual_difference_map,
            "guided_prompt_adherence": self.verdict.guided_prompt_adherence,
            "control_prompt_adherence": self.verdict.control_prompt_adherence,
            "adherence_ratio": self.adherence_ratio,
            "geometry_improved": self.verdict.geometry_improved,
            "adherence_held": self.verdict.adherence_held,
            "passed": self.verdict.passed,
            "guided_seconds": self.guided_seconds,
            "control_seconds": self.control_seconds,
        }


@dataclass(frozen=True)
class Summary:
    """What the run concluded, and where its artifacts landed."""

    config: BenchmarkConfig
    results: tuple[PairResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None

    @property
    def passed(self) -> bool:
        """The gate is two-sided *and* every-run: one failing seed fails the claim."""
        return all(result.verdict.passed for result in self.results)

    @property
    def failures(self) -> tuple[PairResult, ...]:
        return tuple(result for result in self.results if not result.verdict.passed)


def run_benchmark(config: BenchmarkConfig, settings: RoomgenSettings) -> Summary:
    """Run every (scene, seed) of `config` guided and unguided, score both, write the artifacts."""
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
    for scene in config.scenes:
        results.extend(
            _run_scene(
                scene,
                config=config,
                backends=backends,
                adherence=adherence,
                photo_path=_photo_path(scene, photo_paths),
                image_dir=image_dir,
            )
        )

    peak_vram = (
        torch.cuda.max_memory_allocated(backends.device)
        if backends.device.type == "cuda"
        else None
    )
    summary = Summary(
        config=config,
        results=tuple(results),
        output_dir=output_dir,
        peak_vram_bytes=peak_vram,
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
    _write_mask(reference, image_dir / f"{scene.photo}_shell_mask.png")

    # The floor gate (a) is read against: what a *perfect* geometry-preserving
    # renovation still pays for the VAE round trip alone.
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

    results: list[PairResult] = []
    for seed in config.seeds:
        guided, guided_seconds = _sample(
            photo,
            scene.prompt,
            backends=backends,
            config=config,
            reference=reference,
            schedule=build_schedule(config.guidance),
            seed=seed,
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

        guided_residual = depth_residual_map(
            guided, reference=reference, depth_estimator=backends.depth_estimator
        )
        control_residual = depth_residual_map(
            control, reference=reference, depth_estimator=backends.depth_estimator
        )
        mean_control_residual = mean_masked_residual(control_residual, reference.mask)
        comparison_map = f"{scene.photo}_seed{seed}_deptherr.png"
        difference_map = f"{scene.photo}_seed{seed}_depthdiff.png"
        write_residual_comparison(
            control_residual,
            guided_residual,
            mask=reference.mask,
            normalizer=mean_control_residual,
            path=image_dir / comparison_map,
        )
        write_residual_difference(
            control_residual,
            guided_residual,
            mask=reference.mask,
            normalizer=mean_control_residual,
            path=image_dir / difference_map,
        )

        result = PairResult(
            scene=scene.photo,
            prompt=scene.prompt,
            seed=seed,
            verdict=score_run(
                guided_depth_error=depth_error(
                    guided,
                    reference=reference,
                    depth_estimator=backends.depth_estimator,
                ),
                control_depth_error=depth_error(
                    control,
                    reference=reference,
                    depth_estimator=backends.depth_estimator,
                ),
                guided_prompt_adherence=adherence(
                    guided, prompt=scene.prompt, photo=photo, reference=reference
                ),
                control_prompt_adherence=adherence(
                    control, prompt=scene.prompt, photo=photo, reference=reference
                ),
                gate=config.gate,
            ),
            reconstruction_depth_error=reconstruction,
            mean_control_residual=mean_control_residual,
            residual_comparison_map=comparison_map,
            residual_difference_map=difference_map,
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
) -> tuple[Float[Tensor, "1 3 height width"], float]:
    # The generator lives on the sampling device: the loop draws its noise with
    # `torch.randn(..., generator=..., device=latent.device)`, which refuses a
    # CPU generator for a CUDA latent.
    generator = torch.Generator(device=backends.device).manual_seed(seed)
    started = time.perf_counter()
    image = renovate(
        photo,
        prompt,
        pipeline=backends.pipeline,
        depth_estimator=backends.depth_estimator,
        segmenter=backends.segmenter,
        schedule=schedule,
        erosion_radius=config.guidance.erosion_radius,
        steps=config.sampler.steps,
        noise_strength=config.sampler.noise_strength,
        text_guidance_scale=config.sampler.text_guidance_scale,
        generator=generator,
        normalize_gradient=config.guidance.normalize,
        clip_step_rms=config.guidance.clip,
        reference=reference,
    )
    return image, time.perf_counter() - started


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


def _write_mask(reference: GeometricReference, path: Path) -> None:
    mask = reference.mask[0].to(torch.uint8) * 255
    write_png(mask.unsqueeze(0).cpu(), str(path))


def _format_result(result: PairResult) -> str:
    verdict = result.verdict
    return (
        f"seed {result.seed}: depth {verdict.guided_depth_error:.6f} vs "
        f"{verdict.control_depth_error:.6f} ({result.depth_error_reduction:+.1%}), "
        f"adherence {verdict.guided_prompt_adherence:.4f} vs "
        f"{verdict.control_prompt_adherence:.4f} ({result.adherence_ratio:.1%}) "
        f"-> {'PASS' if verdict.passed else 'FAIL'}"
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
                    "wandb_mode": settings.wandb_mode,
                },
                "torch_version": torch.__version__,
                "peak_vram_bytes": summary.peak_vram_bytes,
                "runs": [result.as_row() for result in summary.results],
                "verdict": {
                    "passed": summary.passed,
                    "runs": len(summary.results),
                    "failed": len(summary.failures),
                },
            },
            indent=2,
        )
        + "\n"
    )
