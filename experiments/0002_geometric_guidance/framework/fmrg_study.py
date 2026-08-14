"""The benchmark claim on FMRG's FLUX flow map: guided against control at matched seeds.

The run loop mirrors `framework.benchmark` — same scenes, same reference, same
eval — with the sampler swapped: the walk is `FlowMapProcess` over the vendored
FMRG FLUX sampler, truncated by FMRG's guidance window and steered by its own
annealed-Adam step rule rather than SD1.5's clipped raw gradient. One
transformer load serves every scene: the prompts are all encoded up front and
switched between from the sampler's CPU cache.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torchvision.io import write_png

from framework.config import (
    TORCH_DTYPES,
    FmrgBackendConfig,
    FmrgConfig,
    FmrgGuidanceConfig,
    FmrgSamplerConfig,
    SceneConfig,
)
from framework.metrics import build_adherence
from framework.photos import ensure_photos, load_photo
from framework.residual_maps import mean_absolute_residual, write_signed_residual_map
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.depth.depth_anything import DepthAnythingV2
from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_geometry.segmentation.eomt import EoMT
from roomgen_guide.adapters.flow_map import (
    FlowMapBackend,
    FlowMapProcess,
    GuidanceWindow,
)
from roomgen_guide.adapters.fmrg_flux import (
    FluxFlowMapBackend,
    TwoTimeFlowMapSampler,
    load_flow_map_sampler,
    select_cached_prompt,
)
from roomgen_guide.determinism import enable_determinism
from roomgen_guide.eval import (
    PromptAdherence,
    depth_error,
    shell_coverage,
    signed_depth_residual_map,
)
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.metrics import RunMetrics
from roomgen_guide.process.driver import sample_endpoint_process
from roomgen_guide.process.tilt import AnnealedSignTilt
from roomgen_guide.settings import RoomgenSettings

PHOTO_SUBDIRECTORY: Final = "benchmark_photos"

# Where the repo's download instructions put the 512-res FlowMap LoRA, relative
# to `RoomgenSettings.hf_home`; a variant may point `lora_dir` anywhere else.
DEFAULT_LORA_SUBPATH: Final = Path("fmrg") / "flux-flowmap-lora-512"
LORA_WEIGHT_NAME: Final = "pytorch_lora_weights.safetensors"

# The diffusers-format FLUX load reads none of these, and the two root files
# alone are 24GB: the completeness check must not fail over files that were
# deliberately never fetched.
FLUX_IGNORE: Final = (
    "*.bin",
    "flux1-dev.safetensors",
    "ae.safetensors",
    "*.onnx",
    "*.msgpack",
    "*.h5",
    "*.md",
    ".gitattributes",
)

_REPO_ROOT: Final = Path(__file__).resolve().parents[3]
_FAKE_PACKAGES: Final = (
    _REPO_ROOT / "packages" / "roomgen-geometry" / "tests",
    _REPO_ROOT / "packages" / "roomgen-guide" / "tests",
)

# The three points of the walk the mask snapshots are taken at, in order.
_MASK_SNAPSHOTS: Final = ("first", "mid", "last")


@dataclass
class _MaskLog:
    """Each guided step's effective mask, recorded off the energy's own tensor."""

    masks: list[Bool[Tensor, "1 height width"]] = field(default_factory=list)

    def __call__(self, mask: Bool[Tensor, "1 height width"]) -> None:
        self.masks.append(mask)

    def snapshots(self) -> list[Bool[Tensor, "1 height width"]]:
        if not self.masks:
            return []
        return [self.masks[0], self.masks[len(self.masks) // 2], self.masks[-1]]


@dataclass(frozen=True)
class FmrgPairResult:
    """One guided FMRG walk, its control at the same seed, and what the pair measured.

    The row shape matches the benchmark study's `PairResult.as_row` so the same
    gallery renders both, plus the per-arm peak VRAM an SD1.5 run never needed
    to watch.
    """

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
    guided_peak_vram_bytes: int | None
    control_peak_vram_bytes: int | None

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
            "guided_peak_vram_bytes": self.guided_peak_vram_bytes,
            "control_peak_vram_bytes": self.control_peak_vram_bytes,
        }


@dataclass(frozen=True)
class FmrgSummary:
    """What the run measured, where its artifacts landed, and the code behind them."""

    config: FmrgConfig
    results: tuple[FmrgPairResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


@dataclass(frozen=True)
class Stacks:
    """Everything an FMRG run samples and scores with, already on the target device."""

    flow_backend: FlowMapBackend
    sampler: TwoTimeFlowMapSampler | None
    depth_estimator: DepthEstimator
    segmenter: Segmenter
    device: torch.device


def run_fmrg_study(config: FmrgConfig, settings: RoomgenSettings) -> FmrgSummary:
    """Run every (scene, seed) guided and unguided, measure both, write the artifacts."""
    started_at = datetime.now(UTC).isoformat()
    output_dir = settings.wdir / config.name
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    photo_paths = ensure_photos(settings.wdir / PHOTO_SUBDIRECTORY)
    stacks = build_stacks(
        config.backends,
        prompts=list(dict.fromkeys(scene.prompt for scene in config.scenes)),
        text_guidance_scale=config.sampler.text_guidance_scale,
        settings=settings,
    )
    adherence = build_adherence(
        config.adherence, device=stacks.device, cache_dir=settings.hub_cache
    )
    if stacks.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(stacks.device)

    results: list[FmrgPairResult] = []
    for index, scene in enumerate(config.scenes):
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
    summary = FmrgSummary(
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


def build_stacks(
    backends: FmrgBackendConfig,
    *,
    prompts: Sequence[str],
    text_guidance_scale: float,
    settings: RoomgenSettings,
) -> Stacks:
    """One FLUX flow-map stack serving every prompt in `prompts`, encoded up front.

    The sampler's text encoders are freed the moment it finishes loading, so
    every prompt any walk will ever condition on must be named here; a study
    switches between them with `select_cached_prompt`.
    """
    if backends.deterministic:
        enable_determinism()
    device = torch.device(backends.device)
    match backends.kind:
        case "real":
            sampler = load_flow_map_sampler(
                fmrg_path=backends.fmrg_path or _REPO_ROOT / "packages" / "fmrg",
                lora_path=_lora_dir(backends, settings),
                prompt=prompts[0],
                additional_prompts=list(prompts[1:]),
                device=device,
                model_id=_flux_path(backends, settings),
                dtype=TORCH_DTYPES[backends.dtype],
            )
            return Stacks(
                flow_backend=FluxFlowMapBackend(
                    sampler=sampler,
                    text_guidance_scale=text_guidance_scale,
                ),
                sampler=sampler,
                depth_estimator=DepthAnythingV2(
                    backends.depth_checkpoint,
                    device=device,
                    dtype=TORCH_DTYPES[backends.depth_dtype],
                    cache_dir=settings.hub_cache,
                ),
                segmenter=EoMT(
                    backends.segmentation_checkpoint,
                    device=device,
                    dtype=TORCH_DTYPES[backends.segmentation_dtype],
                    cache_dir=settings.hub_cache,
                ),
                device=device,
            )
        case "fake":
            for package in _FAKE_PACKAGES:
                if str(package) not in sys.path:
                    sys.path.insert(0, str(package))
            # Importable only once the fakes are on the path, exactly as the
            # benchmark study's fake stack is reached.
            from roomgen_fakes import (  # noqa: PLC0415
                FakeDepthEstimator,
                FakeSegmenter,
            )
            from roomgen_guide_fakes import ToyFlowMapBackend  # noqa: PLC0415

            return Stacks(
                flow_backend=ToyFlowMapBackend(dtype=TORCH_DTYPES[backends.dtype]),
                sampler=None,
                depth_estimator=FakeDepthEstimator(),
                segmenter=FakeSegmenter(),
                device=device,
            )


def _lora_dir(backends: FmrgBackendConfig, settings: RoomgenSettings) -> Path:
    directory = backends.lora_dir or settings.hf_home / DEFAULT_LORA_SUBPATH
    if not (directory / LORA_WEIGHT_NAME).exists():
        raise RuntimeError(
            f"no {LORA_WEIGHT_NAME} under {directory}. Fetch the FlowMap LoRA "
            f"(the 512-res one reward-guided generation uses; see "
            f"packages/fmrg/checkpoints_download.sh for the repo and file path) "
            f"into that directory, or point backends.lora_dir at where it lives."
        )
    return directory


def _flux_path(backends: FmrgBackendConfig, settings: RoomgenSettings) -> str:
    """The cached snapshot directory, resolved offline — never the repo id.

    Handed the id, `from_pretrained` re-resolves the whole repo and pulls the
    24GB single-file roots the diffusers-format load never reads.
    """
    checkpoint = backends.flux_checkpoint
    if Path(checkpoint).exists():
        return checkpoint
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    try:
        return str(
            snapshot_download(
                checkpoint,
                cache_dir=str(settings.hub_cache),
                local_files_only=True,
                ignore_patterns=list(FLUX_IGNORE),
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"{checkpoint} is not in the cache at {settings.hub_cache}. Download "
            f"its diffusers-format weights first (skip the single-file roots):\n"
            f'    uv run --no-sync python -c "from huggingface_hub import '
            f"snapshot_download; snapshot_download('{checkpoint}', "
            f"cache_dir='{settings.hub_cache}', "
            f'ignore_patterns={list(FLUX_IGNORE)})"'
        ) from error


def _run_scene(
    scene: SceneConfig,
    *,
    config: FmrgConfig,
    stacks: Stacks,
    adherence: PromptAdherence,
    photo_path: Path,
    image_dir: Path,
    warm_up: bool = False,
) -> list[FmrgPairResult]:
    photo = load_photo(
        photo_path,
        resolution=config.resolution,
        device=stacks.device,
        dtype=torch.float32,
    )
    if stacks.sampler is not None:
        select_cached_prompt(stacks.sampler, scene.prompt)
    reference = GeometricReference.from_photo(
        photo,
        depth_estimator=stacks.depth_estimator,
        segmenter=stacks.segmenter,
        erosion_radius=config.guidance.erosion_radius,
    )
    _write_image(photo, image_dir / f"{scene.photo}_photo.png")
    _write_mask(reference.mask, image_dir / f"{scene.photo}_shell_mask.png")

    # The floor both arms' depth errors are read against: what a perfect
    # geometry-preserving renovation still pays for FLUX's VAE round trip alone.
    reconstruction = depth_error(
        _vae_roundtrip(stacks.flow_backend, photo, device=stacks.device),
        reference=reference,
        depth_estimator=stacks.depth_estimator,
    )
    print(
        f"[scene] {scene.photo}: shell covers "
        f"{float(reference.mask.to(torch.float32).mean()):.1%} of the frame, "
        f"reconstruction floor {reconstruction:.6f}"
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

    results: list[FmrgPairResult] = []
    for seed in config.seeds:
        mask_log = _MaskLog()
        guided, guided_seconds, guided_peak = sample_walk(
            photo,
            stacks=stacks,
            sampler=config.sampler,
            guidance=config.guidance,
            reference=reference,
            eta=config.guidance.eta,
            seed=seed,
            on_mask=mask_log,
        )
        control, control_seconds, control_peak = sample_walk(
            photo,
            stacks=stacks,
            sampler=config.sampler,
            guidance=config.guidance,
            reference=reference,
            eta=0.0,
            seed=seed,
        )
        _write_image(guided, image_dir / f"{scene.photo}_seed{seed}_guided.png")
        _write_image(control, image_dir / f"{scene.photo}_seed{seed}_control.png")
        mask_names = _write_mask_snapshots(
            mask_log, image_dir=image_dir, scene=scene.photo, seed=seed
        )

        common, guided_own, control_own = _evaluation_masks(
            guided, control, config=config, stacks=stacks, reference=reference
        )
        eval_mask_common = f"{scene.photo}_seed{seed}_evalmask_common.png"
        eval_mask_guided = f"{scene.photo}_seed{seed}_evalmask_guided.png"
        eval_mask_control = f"{scene.photo}_seed{seed}_evalmask_control.png"
        _write_mask(common, image_dir / eval_mask_common)
        _write_mask(guided_own, image_dir / eval_mask_guided)
        _write_mask(control_own, image_dir / eval_mask_control)

        control_residual = signed_depth_residual_map(
            control,
            reference=reference,
            depth_estimator=stacks.depth_estimator,
            mask=common,
        )
        guided_residual = signed_depth_residual_map(
            guided,
            reference=reference,
            depth_estimator=stacks.depth_estimator,
            mask=common,
        )
        mean_abs_control_residual = mean_absolute_residual(control_residual, common)
        depth_map_control = f"{scene.photo}_seed{seed}_depth_control.png"
        depth_map_guided = f"{scene.photo}_seed{seed}_depth_guided.png"
        for residual, name in (
            (control_residual, depth_map_control),
            (guided_residual, depth_map_guided),
        ):
            write_signed_residual_map(
                residual,
                mask=common,
                normalizer=mean_abs_control_residual,
                path=image_dir / name,
            )

        result = FmrgPairResult(
            scene=scene.photo,
            prompt=scene.prompt,
            seed=seed,
            metrics=RunMetrics(
                guided_depth_error=depth_error(
                    guided,
                    reference=reference,
                    depth_estimator=stacks.depth_estimator,
                    mask=common,
                ),
                control_depth_error=depth_error(
                    control,
                    reference=reference,
                    depth_estimator=stacks.depth_estimator,
                    mask=common,
                ),
                guided_prompt_adherence=adherence(
                    guided, prompt=scene.prompt, photo=photo, reference=reference
                ),
                control_prompt_adherence=adherence(
                    control, prompt=scene.prompt, photo=photo, reference=reference
                ),
                reconstruction_depth_error=reconstruction,
                common_coverage=shell_coverage(common, reference=reference),
                guided_shell_coverage=shell_coverage(guided_own, reference=reference),
                control_shell_coverage=shell_coverage(control_own, reference=reference),
            ),
            mean_abs_control_residual=mean_abs_control_residual,
            depth_map_control=depth_map_control,
            depth_map_guided=depth_map_guided,
            eval_mask_common=eval_mask_common,
            eval_mask_guided=eval_mask_guided,
            eval_mask_control=eval_mask_control,
            mask_first=mask_names[0],
            mask_mid=mask_names[1],
            mask_last=mask_names[2],
            guided_seconds=guided_seconds,
            control_seconds=control_seconds,
            guided_peak_vram_bytes=guided_peak,
            control_peak_vram_bytes=control_peak,
        )
        print(f"  {_format_result(result)}")
        results.append(result)
    return results


def sample_walk(
    photo: Float[Tensor, "1 3 height width"],
    *,
    stacks: Stacks,
    sampler: FmrgSamplerConfig,
    guidance: FmrgGuidanceConfig,
    reference: GeometricReference,
    eta: float,
    seed: int,
    start_sigma: float | None = None,
    on_mask: _MaskLog | None = None,
    grid_steps: int | None = None,
    window: GuidanceWindow | bool | None = True,
) -> tuple[Float[Tensor, "1 3 height width"], float, int | None]:
    """One walk at `seed`; `eta` 0 is the control on the identical trajectory.

    Args:
        start_sigma: overrides `sampler.start_sigma` for this walk alone, which
            is how an init-strength sweep varies it without reloading anything —
            the schedule is re-derived per `FlowMapProcess`.
    """
    generator = torch.Generator(device=stacks.device).manual_seed(seed)
    resolved_window = (
        GuidanceWindow(
            early_stop=sampler.early_stop,
            unguided_steps=sampler.unguided_steps,
        )
        if window is True
        else (window or None)
    )
    process = FlowMapProcess(
        stacks.flow_backend,
        photo,
        steps=grid_steps if grid_steps is not None else sampler.grid_steps,
        start_sigma=sampler.start_sigma if start_sigma is None else start_sigma,
        generator=generator,
        window=resolved_window,
        stepping=sampler.stepping,
        differentiate_through_velocity=sampler.differentiate_through_velocity,
    )
    tilt = AnnealedSignTilt(
        strength=eta,
        guided_steps=(
            resolved_window.guided_steps if resolved_window is not None else None
        ),
        # Upstream anneals over the whole truncated grid — the guided steps plus
        # the terminal jump it never guides — not the guided window alone.
        anneal_steps=process.steps,
        iterations=guidance.iterations,
    )
    reward = GeometricGuidance(
        depth_estimator=stacks.depth_estimator,
        reference=reference,
        segmenter=stacks.segmenter if guidance.resegment else None,
        erosion_radius=(guidance.erosion_radius if guidance.resegment else None),
        on_mask=on_mask,
    )
    on_cuda = stacks.device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize(stacks.device)
        torch.cuda.reset_peak_memory_stats(stacks.device)
    started = time.perf_counter()
    image = sample_endpoint_process(process, reward=reward, tilt=tilt)
    if on_cuda:
        torch.cuda.synchronize(stacks.device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(stacks.device) if on_cuda else None
    return image.to(torch.float32), seconds, peak


def warm_up_walk(
    photo: Float[Tensor, "1 3 height width"],
    *,
    stacks: Stacks,
    sampler: FmrgSamplerConfig,
    guidance: FmrgGuidanceConfig,
    reference: GeometricReference,
    seed: int,
) -> None:
    """One tiny guided walk before the first timed arm; its output is discarded.

    Two steps with one of them guided pays for everything the first sample of a
    process pays once — kernel autotuning, lazy initialisation, the allocator
    growing its pools — on both the forward and the reward's backward path.
    """
    print("[warmup] 2 throwaway steps (1 guided), untimed")
    sample_walk(
        photo,
        stacks=stacks,
        sampler=sampler,
        guidance=guidance,
        reference=reference,
        eta=max(guidance.eta, 1.0),
        seed=seed,
        grid_steps=2,
        window=None,
    )


def _evaluation_masks(
    guided: Float[Tensor, "1 3 height width"],
    control: Float[Tensor, "1 3 height width"],
    *,
    config: FmrgConfig,
    stacks: Stacks,
    reference: GeometricReference,
) -> tuple[
    Bool[Tensor, "1 height width"],
    Bool[Tensor, "1 height width"],
    Bool[Tensor, "1 height width"],
]:
    """The one pixel set the pair is scored on, beside each arm's own for diagnosis."""
    guided_own = scorable(
        guided,
        stacks=stacks,
        reference=reference,
        resegment=config.guidance.resegment,
        erosion_radius=config.guidance.erosion_radius,
    )
    control_own = scorable(
        control,
        stacks=stacks,
        reference=reference,
        resegment=config.guidance.resegment,
        erosion_radius=config.guidance.erosion_radius,
    )
    return guided_own & control_own, guided_own, control_own


def scorable(
    image: Float[Tensor, "1 3 height width"],
    *,
    stacks: Stacks,
    reference: GeometricReference,
    resegment: bool,
    erosion_radius: int,
) -> Bool[Tensor, "1 height width"]:
    """The reference shell narrowed by one finished render's own segmentation."""
    if not resegment:
        return reference.mask
    with torch.no_grad():
        segmentation = stacks.segmenter(image)
    return reference.mask & scorable_mask(segmentation, erosion_radius=erosion_radius)


def _vae_roundtrip(
    backend: FlowMapBackend,
    photo: Float[Tensor, "1 3 height width"],
    *,
    device: torch.device,
) -> Float[Tensor, "1 3 height width"]:
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        return backend.decode(backend.encode(photo, generator=generator)).to(
            torch.float32
        )


def _write_image(image: Float[Tensor, "1 3 height width"], path: Path) -> None:
    quantised = (image[0].detach().to(torch.float32).clamp(0.0, 1.0) * 255.0).round()
    write_png(quantised.to(torch.uint8).cpu(), str(path))


def _write_mask(mask: Bool[Tensor, "1 height width"], path: Path) -> None:
    write_png((mask[0].to(torch.uint8) * 255).unsqueeze(0).cpu(), str(path))


def _write_mask_snapshots(
    log: _MaskLog, *, image_dir: Path, scene: str, seed: int
) -> tuple[str | None, str | None, str | None]:
    snapshots = log.snapshots()
    if not snapshots:
        return None, None, None
    names = [f"{scene}_seed{seed}_mask_{point}.png" for point in _MASK_SNAPSHOTS]
    for mask, name in zip(snapshots, names, strict=True):
        _write_mask(mask, image_dir / name)
    return names[0], names[1], names[2]


def _format_result(result: FmrgPairResult) -> str:
    metrics = result.metrics
    return (
        f"seed {result.seed}: depth {metrics.guided_depth_error:.6f} vs "
        f"{metrics.control_depth_error:.6f} "
        f"({_signed_share(metrics.depth_error_reduction)}), adherence "
        f"{metrics.guided_prompt_adherence:.4f} vs "
        f"{metrics.control_prompt_adherence:.4f} "
        f"({_share(metrics.adherence_ratio)}), shell scored "
        f"{metrics.common_coverage:.0%} in common, "
        f"{result.guided_seconds:.1f}s/{result.control_seconds:.1f}s, "
        f"peak {_gib(result.guided_peak_vram_bytes)}/"
        f"{_gib(result.control_peak_vram_bytes)}"
    )


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _signed_share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def _gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1024**3:.2f}GiB"


def _git_commit() -> str:
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


def walked_steps(sampler: FmrgSamplerConfig) -> int:
    """The steps the truncated walk actually takes, by the window's own arithmetic."""
    window = GuidanceWindow(
        early_stop=sampler.early_stop,
        unguided_steps=sampler.unguided_steps,
    )
    return window.guided_steps + sampler.unguided_steps


def _write_results(summary: FmrgSummary, settings: RoomgenSettings) -> None:
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
