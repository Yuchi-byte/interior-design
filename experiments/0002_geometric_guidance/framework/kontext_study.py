"""The Kontext pilot: a strong instruction-editing prior, with and without the depth reward.

FLUX.1-Kontext-dev holds the reference photo as attention context — the walk
starts from pure noise and the room enters through the tokens, never through
the init — so the unguided arm is the strongest layout-holding prior this repo
has sampled. Per scene at each seed off one transformer load: `kontext` (the
instruction prompt, no reward) and `kontext_depth` (the same walk steered by
the geometric reward against the *furnished photo's* shell depth). Both are
scored on the one pixel set both left visible, judged against the scene's
neutral style text by an adherence CLIP, a render-to-photo image similarity
and a preference model — all loaded only after the last walk.

Two extra readouts this pilot exists for:

- both walks' decoded one-step endpoint estimates every few steps, each with
  its depth error against the reference shell — the sharpness strip that says
  directly whether ``x0 = z - sigma * v`` is a usable steering signal at each
  noise level (on SD1.5 the analogous Tweedie estimate was not; Kontext's
  reference anchoring is the reason to re-ask), and the error curve that shows
  the geometry converging, or not, under the reward;
- the finished FMRG text-only pair from a previous `fmrg_redux` run, quoted
  beside these arms with its depth errors *recomputed against the furnished
  photo's shell* — that run steered and scored against the stripped empty
  room, so its own numbers do not compare directly and the recomputation is
  what makes the column honest.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torchvision.io import ImageReadMode, read_image

from framework.config import TORCH_DTYPES, KontextConfig, KontextSceneConfig
from framework.depth_maps import write_disparity_map, write_masked_disparity_map
from framework.fmrg_study import (
    FLUX_IGNORE,
    PHOTO_SUBDIRECTORY,
    _git_commit,
    _vae_roundtrip,
    _write_image,
    _write_mask,
)
from framework.metrics import build_adherence
from framework.photos import ensure_photos, load_photo
from framework.preference import build_preference
from framework.residual_maps import mean_absolute_residual, write_signed_residual_map
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.depth.depth_anything import DepthAnythingV2
from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.segmentation.base import Segmenter
from roomgen_geometry.segmentation.eomt import EoMT
from roomgen_guide.adapters.flow_map import FlowMapBackend, FlowMapProcess
from roomgen_guide.adapters.kontext_flux import (
    KontextStack,
    build_kontext_backend,
    load_kontext_stack,
)
from roomgen_guide.determinism import enable_determinism
from roomgen_guide.eval import (
    ClipImageSimilarity,
    PromptAdherence,
    depth_error,
    shell_coverage,
    signed_depth_residual_map,
)
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.guidance.preference import ImagePreferenceModel
from roomgen_guide.process.driver import sample_endpoint_process
from roomgen_guide.process.endpoint import StepContext
from roomgen_guide.process.tilt import AnnealedSignTilt, EnergyGradient, TiltPolicy
from roomgen_guide.settings import RoomgenSettings

# The powerful-prior baseline every delta divides by, and the guided arm.
BASELINE_ARM: Final = "kontext"
GUIDED_ARM: Final = "kontext_depth"

# The height the endpoint-strip thumbnails are saved at: evidence, not renders.
_STRIP_HEIGHT: Final = 256

# Kontext's checkpoint layout mirrors FLUX.1-dev's plus its own single-file
# root, which the diffusers-format load never reads.
KONTEXT_IGNORE: Final = (*FLUX_IGNORE, "flux1-kontext-dev.safetensors")


@dataclass(frozen=True)
class KontextArmSpec:
    arm: str
    guided: bool


def kontext_arms(config: KontextConfig) -> tuple[KontextArmSpec, ...]:
    """Baseline first: the unguided prior, then the same walk under the reward."""
    del config
    return (KontextArmSpec(BASELINE_ARM, False), KontextArmSpec(GUIDED_ARM, True))


@dataclass(frozen=True)
class KontextArmResult:
    """One arm's render, its depth reading, and its judged scores."""

    arm: str
    guided: bool
    prompt: str
    depth_error: float
    shell_coverage: float
    adherence: float
    image_similarity: float | None
    preference: float
    render: str
    depth_map: str
    residual_map: str
    seconds: float
    peak_vram_bytes: int | None

    def as_row(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "guided": self.guided,
            "prompt": self.prompt,
            "depth_error": self.depth_error,
            "shell_coverage": self.shell_coverage,
            "adherence": self.adherence,
            "image_similarity": self.image_similarity,
            "preference": self.preference,
            "render": self.render,
            "depth_map": self.depth_map,
            "residual_map": self.residual_map,
            "seconds": self.seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


@dataclass(frozen=True)
class EndpointSnapshot:
    """One decoded one-step endpoint estimate of a walk, with its depth reading.

    Args:
        depth_error: the estimate's aligned depth error against the reference,
            scored on the full reference shell — never a resegmentation of a
            half-formed estimate — so the curve is one mask across all steps
            and both arms.
    """

    step: int
    sigma: float
    image: str
    depth_error: float

    def as_row(self) -> dict[str, object]:
        return {
            "step": self.step,
            "sigma": self.sigma,
            "image": self.image,
            "depth_error": self.depth_error,
        }


@dataclass(frozen=True)
class KontextSceneResult:
    """Both arms of one (scene, seed), scored on one common pixel set.

    Args:
        endpoint_strips: each arm's decoded endpoint estimates in step order —
            the sharpness evidence, and through each snapshot's depth error the
            convergence curve.
        fmrg_comparison: the quoted FMRG text-only pair with recomputed depth
            errors, or `None` when no finished run was named or found.
    """

    scene: str
    seed: int
    style_prompt: str
    instruction_prompt: str
    eval_prompt: str
    effective_first_sigma: float
    arms: tuple[KontextArmResult, ...]
    common_coverage: float
    reconstruction_depth_error: float
    mean_abs_baseline_residual: float
    photo_image: str
    photo_depth_map: str
    photo_depth_map_masked: str
    shell_mask: str
    eval_mask_common: str
    endpoint_strips: dict[str, tuple[EndpointSnapshot, ...]]
    fmrg_comparison: dict[str, object] | None

    def arm(self, arm_id: str) -> KontextArmResult:
        for measurement in self.arms:
            if measurement.arm == arm_id:
                return measurement
        raise KeyError(arm_id)

    def depth_delta_vs_baseline(self, arm_id: str) -> float | None:
        """The arm's depth error relative to the prior alone, signed."""
        baseline = self.arm(BASELINE_ARM).depth_error
        if baseline == 0.0:
            return None
        return self.arm(arm_id).depth_error / baseline - 1.0

    def adherence_ratio_vs_baseline(self, arm_id: str) -> float | None:
        baseline = self.arm(BASELINE_ARM).adherence
        if baseline == 0.0:
            return None
        return self.arm(arm_id).adherence / baseline

    def as_row(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "seed": self.seed,
            "style_prompt": self.style_prompt,
            "instruction_prompt": self.instruction_prompt,
            "eval_prompt": self.eval_prompt,
            "effective_first_sigma": self.effective_first_sigma,
            "common_coverage": self.common_coverage,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "mean_abs_baseline_residual": self.mean_abs_baseline_residual,
            "photo_image": self.photo_image,
            "photo_depth_map": self.photo_depth_map,
            "photo_depth_map_masked": self.photo_depth_map_masked,
            "shell_mask": self.shell_mask,
            "eval_mask_common": self.eval_mask_common,
            "endpoint_strips": {
                arm: [snapshot.as_row() for snapshot in strip]
                for arm, strip in self.endpoint_strips.items()
            },
            "fmrg_comparison": self.fmrg_comparison,
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
class KontextSummary:
    """What the pilot measured, where its artifacts landed, and the code behind them."""

    config: KontextConfig
    results: tuple[KontextSceneResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


@dataclass(frozen=True)
class KontextStacks:
    """Everything the pilot samples and scores with, already on the target device.

    `stack` is the loaded Kontext transformer for the real path; the fake path
    carries a `FlowMapBackend` double instead, shared across scenes, and the
    reference conditioning simply does not exist for it.
    """

    stack: KontextStack | None
    fake_backend: FlowMapBackend | None
    depth_estimator: DepthEstimator
    segmenter: Segmenter
    device: torch.device


def run_kontext_study(
    config: KontextConfig, settings: RoomgenSettings
) -> KontextSummary:
    """Run both arms of every (scene, seed), measure them all, write the artifacts."""
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
    stacks = build_kontext_stacks(config, settings=settings)
    if stacks.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(stacks.device)

    pending: list[_PendingScene] = []
    for index, scene in enumerate(config.scenes):
        pending.extend(
            _run_scene(
                scene,
                config=config,
                stacks=stacks,
                settings=settings,
                photo_path=photo_paths[scene.photo],
                image_dir=image_dir,
                warm_up=index == 0,
            )
        )

    # Loaded only after the last walk, one judge resident at a time: beside the
    # transformer, a resident eval tower is exactly what tipped a 32GB card
    # into out-of-memory. Renders wait on the CPU.
    results = _score_pending(pending, config=config, stacks=stacks, settings=settings)
    for result in results:
        print(f"  {_format_result(result)}")

    on_cuda = stacks.device.type == "cuda"
    summary = KontextSummary(
        config=config,
        results=tuple(results),
        output_dir=output_dir,
        peak_vram_bytes=(
            torch.cuda.max_memory_allocated(stacks.device) if on_cuda else None
        ),
        started_at=started_at,
        git_commit=_git_commit(),
        walked_steps=config.sampler.steps,
        device_name=(torch.cuda.get_device_name(stacks.device) if on_cuda else None),
    )
    _write_results(summary, settings)
    return summary


def build_kontext_stacks(
    config: KontextConfig, *, settings: RoomgenSettings
) -> KontextStacks:
    """One Kontext transformer load serving every scene, prompts encoded up front."""
    backends = config.backends
    if backends.deterministic:
        enable_determinism()
    device = torch.device(backends.device)
    match backends.kind:
        case "real":
            stack = load_kontext_stack(
                checkpoint=_kontext_path(backends.kontext_checkpoint, settings),
                prompts=[scene.instruction_prompt for scene in config.scenes],
                device=device,
                dtype=TORCH_DTYPES[backends.dtype],
                cache_dir=settings.hub_cache,
            )
            return KontextStacks(
                stack=stack,
                fake_backend=None,
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
            from framework.fmrg_study import _FAKE_PACKAGES  # noqa: PLC0415

            for package in _FAKE_PACKAGES:
                if str(package) not in sys.path:
                    sys.path.insert(0, str(package))
            from roomgen_fakes import (  # noqa: PLC0415
                FakeDepthEstimator,
                FakeSegmenter,
            )
            from roomgen_guide_fakes import ToyFlowMapBackend  # noqa: PLC0415

            return KontextStacks(
                stack=None,
                fake_backend=ToyFlowMapBackend(dtype=TORCH_DTYPES[backends.dtype]),
                depth_estimator=FakeDepthEstimator(),
                segmenter=FakeSegmenter(),
                device=device,
            )


def _kontext_path(checkpoint: str, settings: RoomgenSettings) -> str:
    """The cached snapshot directory, resolved offline — never the repo id.

    Handed the id, `from_pretrained` re-resolves the whole repo and pulls the
    24GB single-file root the diffusers-format load never reads.
    """
    if Path(checkpoint).exists():
        return checkpoint
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    try:
        return str(
            snapshot_download(
                checkpoint,
                cache_dir=str(settings.hub_cache),
                local_files_only=True,
                ignore_patterns=list(KONTEXT_IGNORE),
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"{checkpoint} is not in the cache at {settings.hub_cache}. Download "
            f"its diffusers-format weights first (skip the single-file roots):\n"
            f'    uv run --no-sync python -c "from huggingface_hub import '
            f"snapshot_download; snapshot_download('{checkpoint}', "
            f"cache_dir='{settings.hub_cache}', "
            f'ignore_patterns={list(KONTEXT_IGNORE)})"'
        ) from error


@dataclass
class _StepCell:
    """The guided step the walk is currently inside, shared tilt-to-reward."""

    index: int = -1


@dataclass(frozen=True)
class _SigmaWindowedTilt:
    """FMRG's step rule, told which step it is and gated by the sigma window.

    The wrapper is also what stamps the current step index into the shared
    cell *before* the reward is evaluated, which is how the endpoint recorder
    knows which step a decoded estimate belongs to.
    """

    inner: AnnealedSignTilt
    sigmas: tuple[float, ...]
    max_sigma: float | None
    cell: _StepCell

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, batch * state],
        step: StepContext,
    ) -> Float[Tensor, batch * state] | None:
        self.cell.index = step.index
        if self.max_sigma is not None and self.sigmas[step.index] > self.max_sigma:
            return None
        return self.inner(energy_gradient, state=state, step=step)


@dataclass(frozen=True)
class _ObservationTilt:
    """The unguided arm's camera: evaluates the reward on recorded steps, steers nothing.

    `AnnealedSignTilt` at strength 0 never calls the reward, so the recorder
    wrapped around it would see nothing; this policy evaluates the reward on
    every Nth step purely so the recorder keeps the decoded endpoint, discards
    the gradient, and always leaves the step alone — the walk is the prior's
    own, at the cost of one backward per recorded step.
    """

    cell: _StepCell
    every: int

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, batch * state],
        step: StepContext,
    ) -> Float[Tensor, batch * state] | None:
        self.cell.index = step.index
        if step.index % self.every == 0:
            energy_gradient(state)
        return None


@dataclass
class _EndpointRecorder:
    """The reward with a camera on it: every Nth guided step's endpoint, kept.

    Observation only — the energy returned is the inner reward's, on the same
    attached tensor, so the recording cannot change a gradient. Only the first
    evaluation of a step is kept: an inner-loop iteration re-evaluates at a
    displaced state, which is not the walk's own endpoint estimate.
    """

    inner: Guidance
    cell: _StepCell
    every: int
    frames: list[tuple[int, Float[Tensor, "batch 3 height width"]]] = field(
        default_factory=list
    )

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, batch]:
        index = self.cell.index
        if index % self.every == 0 and (not self.frames or self.frames[-1][0] != index):
            self.frames.append(
                (index, predicted_image.detach().to(device="cpu", dtype=torch.float32))
            )
        return self.inner(predicted_image)


def _scene_backend(
    scene: KontextSceneConfig,
    *,
    config: KontextConfig,
    stacks: KontextStacks,
    photo: Float[Tensor, "1 3 height width"],
) -> FlowMapBackend:
    if stacks.stack is None:
        assert stacks.fake_backend is not None
        return stacks.fake_backend
    return build_kontext_backend(
        stacks.stack,
        prompt=scene.instruction_prompt,
        reference_image=photo,
        text_guidance_scale=config.sampler.text_guidance_scale,
    )


def _sample_walk(
    photo: Float[Tensor, "1 3 height width"],
    backend: FlowMapBackend,
    *,
    config: KontextConfig,
    stacks: KontextStacks,
    reference: GeometricReference,
    eta: float,
    seed: int,
    steps: int | None = None,
    recorder_every: int | None = None,
) -> tuple[Float[Tensor, "1 3 height width"], float, int | None, list[EndpointFrame]]:
    """One walk at `seed` from pure noise; `eta` 0 is the prior-only control."""
    walked = steps if steps is not None else config.sampler.steps
    generator = torch.Generator(device=stacks.device).manual_seed(seed)
    process = FlowMapProcess(
        backend,
        photo,
        steps=walked,
        # Kontext's operating point: the photo enters through the reference
        # tokens, never through the init — pure noise is the trained walk.
        start_sigma=1.0,
        generator=generator,
    )
    cell = _StepCell()
    sigmas = _walk_sigmas(photo, backend, steps=walked, device=stacks.device)
    tilt: TiltPolicy
    if eta == 0.0 and recorder_every is not None:
        tilt = _ObservationTilt(cell=cell, every=recorder_every)
    else:
        tilt = _SigmaWindowedTilt(
            inner=AnnealedSignTilt(
                strength=eta,
                guided_steps=config.guidance.guided_steps,
                anneal_steps=process.steps,
                iterations=config.guidance.iterations,
            ),
            sigmas=sigmas,
            max_sigma=config.guidance.max_sigma,
            cell=cell,
        )
    reward: Guidance = GeometricGuidance(
        depth_estimator=stacks.depth_estimator,
        reference=reference,
        segmenter=stacks.segmenter if config.guidance.resegment else None,
        erosion_radius=(
            config.guidance.erosion_radius if config.guidance.resegment else None
        ),
    )
    recorder: _EndpointRecorder | None = None
    if recorder_every is not None:
        recorder = _EndpointRecorder(inner=reward, cell=cell, every=recorder_every)
        reward = recorder
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
    frames = [] if recorder is None else recorder.frames
    return (
        image.to(torch.float32),
        seconds,
        peak,
        [EndpointFrame(index, sigmas[index], frame) for index, frame in frames],
    )


@dataclass(frozen=True)
class EndpointFrame:
    """A decoded endpoint estimate still in memory, before it is written out."""

    step: int
    sigma: float
    image: Float[Tensor, "1 3 height width"]


def _walk_sigmas(
    photo: Float[Tensor, "1 3 height width"],
    backend: FlowMapBackend,
    *,
    steps: int,
    device: torch.device,
) -> tuple[float, ...]:
    """The sigma before each step, read off the backend's own resolved schedule."""
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        encoded = backend.encode(photo, generator=generator)
    return tuple(backend.schedule(steps=steps, start_sigma=1.0, latent=encoded))


def _run_scene(
    scene: KontextSceneConfig,
    *,
    config: KontextConfig,
    stacks: KontextStacks,
    settings: RoomgenSettings,
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
    # The furnished photo's shell: the reference the reward defends and every
    # depth number is read against — the owner's setting for this pilot.
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

    backend = _scene_backend(scene, config=config, stacks=stacks, photo=photo)
    reconstruction = depth_error(
        _vae_roundtrip(backend, photo, device=stacks.device),
        reference=reference,
        depth_estimator=stacks.depth_estimator,
    )
    sigmas = _walk_sigmas(
        photo, backend, steps=config.sampler.steps, device=stacks.device
    )
    print(
        f"[scene] {scene.photo}: shell covers "
        f"{float(reference.mask.to(torch.float32).mean()):.1%} of the frame, "
        f"reconstruction floor {reconstruction:.6f}, first sigma {sigmas[0]:.4f}"
    )

    if warm_up:
        print("[warmup] 2 throwaway steps (guided), untimed")
        _sample_walk(
            photo,
            backend,
            config=config,
            stacks=stacks,
            reference=reference,
            eta=max(config.guidance.eta, 1.0),
            seed=config.seeds[0],
            steps=2,
        )

    fmrg_comparison = _fmrg_comparison(
        scene, config=config, stacks=stacks, settings=settings, reference=reference
    )

    specs = kontext_arms(config)
    results: list[_PendingScene] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        peaks: dict[str, int | None] = {}
        strips: dict[str, tuple[EndpointSnapshot, ...]] = {}
        for spec in specs:
            image, elapsed, peak, frames = _sample_walk(
                photo,
                backend,
                config=config,
                stacks=stacks,
                reference=reference,
                eta=config.guidance.eta if spec.guided else 0.0,
                seed=seed,
                recorder_every=config.endpoint_strip_every,
            )
            renders[spec.arm] = image
            seconds[spec.arm] = elapsed
            peaks[spec.arm] = peak
            strips[spec.arm] = _write_endpoint_strip(
                frames,
                image_dir=image_dir,
                scene=scene.photo,
                seed=seed,
                arm=spec.arm,
                stacks=stacks,
                reference=reference,
            )
            _write_image(image, image_dir / f"{scene.photo}_seed{seed}_{spec.arm}.png")
            # Sweep the allocator between arms: the guided backward's transient
            # should never be the reason the next arm fragments.
            gc.collect()
            if stacks.device.type == "cuda":
                torch.cuda.empty_cache()

        # One pixel set for both arms: the reference shell narrowed by each
        # arm's own segmentation, intersected, so the depth errors are
        # commensurable and covering shell up buys nothing.
        arm_masks = {
            spec.arm: _scorable(
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

        measurements: list[KontextArmResult] = []
        for spec in specs:
            render = renders[spec.arm]
            # `_depthmap`, not `_depth`: the guided arm is named `kontext_depth`,
            # so the unguided arm's `..._kontext_depth.png` map would overwrite
            # the guided arm's render of exactly that name.
            depth_map = f"{scene.photo}_seed{seed}_{spec.arm}_depthmap.png"
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
                KontextArmResult(
                    arm=spec.arm,
                    guided=spec.guided,
                    prompt=scene.instruction_prompt,
                    depth_error=depth_error(
                        render,
                        reference=reference,
                        depth_estimator=stacks.depth_estimator,
                        mask=common,
                    ),
                    shell_coverage=shell_coverage(
                        arm_masks[spec.arm], reference=reference
                    ),
                    # Filled in by `_score_pending` once sampling is over and
                    # the judges can afford to be resident, one at a time.
                    adherence=math.nan,
                    image_similarity=None,
                    preference=math.nan,
                    render=f"{scene.photo}_seed{seed}_{spec.arm}.png",
                    depth_map=depth_map,
                    residual_map=residual_map,
                    seconds=seconds[spec.arm],
                    peak_vram_bytes=peaks[spec.arm],
                )
            )

        result = KontextSceneResult(
            scene=scene.photo,
            seed=seed,
            style_prompt=scene.style_prompt,
            instruction_prompt=scene.instruction_prompt,
            eval_prompt=scene.eval_prompt,
            effective_first_sigma=sigmas[0],
            arms=tuple(measurements),
            common_coverage=shell_coverage(common, reference=reference),
            reconstruction_depth_error=reconstruction,
            mean_abs_baseline_residual=baseline_scale,
            photo_image=photo_image,
            photo_depth_map=photo_depth_map,
            photo_depth_map_masked=photo_depth_map_masked,
            shell_mask=shell_mask,
            eval_mask_common=eval_mask_common,
            endpoint_strips=strips,
            fmrg_comparison=fmrg_comparison,
        )
        print(f"  [sampled] {result.scene} seed {result.seed}, judges pending")
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


def _scorable(
    image: Float[Tensor, "1 3 height width"],
    *,
    stacks: KontextStacks,
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


def _write_endpoint_strip(
    frames: list[EndpointFrame],
    *,
    image_dir: Path,
    scene: str,
    seed: int,
    arm: str,
    stacks: KontextStacks,
    reference: GeometricReference,
) -> tuple[EndpointSnapshot, ...]:
    """Each frame written small, and read for depth against the full reference shell."""
    snapshots = []
    for frame in frames:
        name = f"{scene}_seed{seed}_{arm}_endpoint_step{frame.step:02d}.png"
        error = depth_error(
            frame.image.to(stacks.device),
            reference=reference,
            depth_estimator=stacks.depth_estimator,
        )
        small = torch.nn.functional.interpolate(
            frame.image,
            size=(_STRIP_HEIGHT, _STRIP_HEIGHT),
            mode="area",
        )
        _write_image(small, image_dir / name)
        snapshots.append(EndpointSnapshot(frame.step, frame.sigma, name, error))
    return tuple(snapshots)


def _fmrg_comparison(
    scene: KontextSceneConfig,
    *,
    config: KontextConfig,
    stacks: KontextStacks,
    settings: RoomgenSettings,
    reference: GeometricReference,
) -> dict[str, object] | None:
    """The finished FMRG run's text-only pair, made comparable to this study's arms.

    That run steered and scored against the *stripped empty room's* shell, so
    its recorded depth errors are read against a different reference over a
    different pixel set. The renders it saved are re-scored here against the
    furnished photo's shell (reference-shell AND render-shell, this study's own
    semantics), which is the number the comparison column quotes; the original
    empty-reference error rides along, labelled as what it is.
    """
    if config.fmrg_comparison_run is None or config.backends.kind != "real":
        return None
    results_path = settings.wdir / config.fmrg_comparison_run / "results.json"
    if not results_path.exists():
        print(f"[fmrg] no results.json under {results_path.parent}; comparison skipped")
        return None
    data = json.loads(results_path.read_text())
    run = next(
        (entry for entry in data.get("runs", []) if entry.get("scene") == scene.photo),
        None,
    )
    if run is None:
        print(f"[fmrg] {scene.photo} not in {results_path}; comparison skipped")
        return None
    rows: list[dict[str, object]] = []
    for arm_id in ("s0u", "s0g"):
        arm = next(
            (cell for cell in run.get("arms", []) if cell.get("arm") == arm_id), None
        )
        if arm is None:
            continue
        render_path = results_path.parent / "images" / str(arm["render"])
        if not render_path.exists():
            continue
        decoded = read_image(str(render_path), mode=ImageReadMode.RGB)
        render = (
            decoded.to(device=stacks.device, dtype=torch.float32) / 255.0
        ).unsqueeze(0)
        mask = _scorable(
            render,
            stacks=stacks,
            reference=reference,
            resegment=config.guidance.resegment,
            erosion_radius=config.guidance.erosion_radius,
        )
        rows.append(
            {
                "arm": arm_id,
                "guided": bool(arm.get("guided")),
                "depth_error_vs_photo_shell": depth_error(
                    render,
                    reference=reference,
                    depth_estimator=stacks.depth_estimator,
                    mask=mask,
                ),
                "shell_coverage_vs_photo_shell": shell_coverage(
                    mask, reference=reference
                ),
                "depth_error_vs_empty_reference": arm.get("depth_error"),
                "adherence": arm.get("adherence"),
                "image_similarity": arm.get("image_similarity"),
                "preference": arm.get("preference"),
                "seconds": arm.get("seconds"),
                "render": str(render_path),
            }
        )
    if not rows:
        return None
    return {
        "run": config.fmrg_comparison_run,
        "guidance_reference": data.get("guidance_reference"),
        "note": (
            "depth_error_vs_photo_shell is recomputed here on the run's saved "
            "renders against this study's furnished-photo reference and each "
            "render's own scorable mask; it is not the common-mask number of "
            "either study. depth_error_vs_empty_reference is the source run's "
            "own reading against the stripped empty room."
        ),
        "rows": rows,
    }


@dataclass(frozen=True)
class _PendingScene:
    """One scene measured on everything but the deferred judges, renders on the CPU."""

    result: KontextSceneResult
    renders: dict[str, Float[Tensor, "1 3 height width"]]
    photo: Float[Tensor, "1 3 height width"]
    reference: GeometricReference


def _score_pending(
    pending: list[_PendingScene],
    *,
    config: KontextConfig,
    stacks: KontextStacks,
    settings: RoomgenSettings,
) -> list[KontextSceneResult]:
    device = stacks.device
    adherence = build_adherence(
        config.adherence, device=device, cache_dir=settings.hub_cache
    )
    scored = [
        _with_adherence(entry, adherence=adherence, device=device) for entry in pending
    ]
    del adherence
    _release(device)

    if config.image_similarity_checkpoint is not None:
        similarity = ClipImageSimilarity(
            config.image_similarity_checkpoint,
            device=device,
            cache_dir=settings.hub_cache,
        )
        scored = [
            _with_image_similarity(entry, similarity=similarity, device=device)
            for entry in scored
        ]
        del similarity
        _release(device)

    preference = build_preference(
        config.preference, device=device, cache_dir=settings.hub_cache
    )
    scored = [
        _with_preference(entry, preference=preference, device=device)
        for entry in scored
    ]
    del preference
    _release(device)
    return [entry.result for entry in scored]


def _with_adherence(
    entry: _PendingScene, *, adherence: PromptAdherence, device: torch.device
) -> _PendingScene:
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
    return replace(entry, result=replace(result, arms=arms))


def _with_image_similarity(
    entry: _PendingScene, *, similarity: ClipImageSimilarity, device: torch.device
) -> _PendingScene:
    result = entry.result
    arms = tuple(
        replace(
            cell,
            image_similarity=similarity(
                entry.renders[cell.arm].to(device), photo=entry.photo.to(device)
            ),
        )
        for cell in result.arms
    )
    return replace(entry, result=replace(result, arms=arms))


def _with_preference(
    entry: _PendingScene, *, preference: ImagePreferenceModel, device: torch.device
) -> _PendingScene:
    result = entry.result
    arms = []
    for cell in result.arms:
        with torch.no_grad():
            score = preference.score(
                entry.renders[cell.arm].to(device), [result.eval_prompt]
            )
        arms.append(replace(cell, preference=float(score.mean())))
    return replace(entry, result=replace(result, arms=tuple(arms)))


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _format_result(result: KontextSceneResult) -> str:
    cells = ", ".join(
        f"{cell.arm} E={cell.depth_error:.4f} adh={cell.adherence:.4f}"
        for cell in result.arms
    )
    return (
        f"seed {result.seed}: {cells}, shell scored "
        f"{result.common_coverage:.0%} in common"
    )


def arm_means(
    results: tuple[KontextSceneResult, ...], specs: tuple[KontextArmSpec, ...]
) -> dict[str, dict[str, float | None]]:
    """Each arm's mean readings over every (scene, seed)."""
    means: dict[str, dict[str, float | None]] = {}
    for spec in specs:
        cells = [result.arm(spec.arm) for result in results]
        similarities = [
            cell.image_similarity for cell in cells if cell.image_similarity is not None
        ]
        means[spec.arm] = {
            "depth_error": _mean([cell.depth_error for cell in cells]),
            "shell_coverage": _mean([cell.shell_coverage for cell in cells]),
            "adherence": _mean([cell.adherence for cell in cells]),
            "image_similarity": _mean(similarities) if similarities else None,
            "preference": _mean([cell.preference for cell in cells]),
            "seconds": _mean([cell.seconds for cell in cells]),
        }
    return means


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_results(summary: KontextSummary, settings: RoomgenSettings) -> None:
    from framework.fmrg_alignment_study import shell_class_table  # noqa: PLC0415

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
                "shell_classes": shell_class_table(),
                # What steered the guided arm and what every depth number is
                # read against: the furnished photo's own shell depth.
                "guidance_reference": "photo",
                # The strip's depth errors are read on the full reference shell
                # (one mask across all steps and both arms), never a
                # resegmentation of a half-formed estimate.
                "endpoint_strip_mask": "reference_shell",
                "eta": summary.config.guidance.eta,
                "torch_version": torch.__version__,
                "device_name": summary.device_name,
                "walked_steps": summary.walked_steps,
                "peak_vram_bytes": summary.peak_vram_bytes,
                "arm_means": arm_means(summary.results, kontext_arms(summary.config)),
                "runs": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
