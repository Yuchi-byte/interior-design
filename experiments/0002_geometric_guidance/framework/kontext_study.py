"""The Kontext pilot: a strong instruction-editing prior, with and without the depth reward.

FLUX.1-Kontext-dev holds the reference photo as attention context — the walk
starts from pure noise and the room enters through the tokens, never through
the init — so the unguided arm is the strongest layout-holding prior this repo
has sampled. Stage zero strips every photo bare (0003's strip machinery,
released in full before the transformer loads) and the depth reference is the
*stripped empty room's* shell depth, by owner decision: the empty shell is
nearly the whole frame, a far larger valid comparison area than the furnished
shell. The furnished photo's shell is kept as a secondary depth column so the
reference switch itself is quantified. Per scene at each seed off one
transformer load: `kontext` (the instruction prompt, no reward),
`kontext_depth` (the same walk steered by the geometric reward against the
empty room's shell depth), and one extra guided arm per configured grid point
— the small operating-point grid a diagnosis sweeps. All arms are scored on
the one pixel set every arm left visible, judged against the scene's neutral
style text by an adherence CLIP, a render-to-photo image similarity and a
preference model — all loaded only after the last walk.

Three extra readouts this pilot exists for:

- the baseline's and `kontext_depth`'s decoded one-step endpoint estimates
  every few steps, each with its depth error against the empty reference
  shell, its segmentation map and its masked depth map — the strip that says
  directly whether ``x0 = z - sigma * v`` is a usable steering signal at each
  noise level, what the segmenter and the depth network actually read off it,
  and the error curve of the geometry converging (or not) under the reward;
- every guided walk's per-step instrumentation — the reward's energy, the
  gradient's RMS and the applied displacement's RMS — the trace that says
  where steering is actually spent versus where the error actually drops;
- the finished FMRG text-only pair from a previous `fmrg_redux` run, quoted
  beside these arms with its depth errors recomputed against both of this
  study's references on its saved renders, which is what makes the column
  honest.
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

from framework.config import (
    TORCH_DTYPES,
    KontextConfig,
    KontextGridPointConfig,
    KontextSceneConfig,
)
from framework.depth_maps import write_masked_disparity_map
from framework.fmrg_study import (
    _REPO_ROOT,
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

# The powerful-prior baseline every delta divides by, and the primary guided arm.
BASELINE_ARM: Final = "kontext"
GUIDED_ARM: Final = "kontext_depth"

# The height the endpoint-strip thumbnails are saved at: evidence, not renders.
_STRIP_HEIGHT: Final = 256

# Kontext's checkpoint layout mirrors FLUX.1-dev's plus its own single-file
# root, which the diffusers-format load never reads.
KONTEXT_IGNORE: Final = (*FLUX_IGNORE, "flux1-kontext-dev.safetensors")

# 0003's strip machinery is reused, not reimplemented; the frameworks are
# separate top-level directories, so it is reached the way the fakes are.
_STRIP_STUDY_DIR: Final = _REPO_ROOT / "experiments" / "0003_strip_and_furnish"


def _ensure_strip_study_on_path() -> None:
    if str(_STRIP_STUDY_DIR) not in sys.path:
        sys.path.insert(0, str(_STRIP_STUDY_DIR))


@dataclass(frozen=True)
class KontextArmSpec:
    """One arm of the study: its steering knobs and whether its walk is filmed."""

    arm: str
    guided: bool
    eta: float
    first_guided_step: int
    guided_steps: int | None
    records_strip: bool


def grid_arm_id(point: KontextGridPointConfig) -> str:
    """The grid arm's name, read straight off its knobs: eta then the step window."""
    end = "all" if point.guided_steps is None else str(point.guided_steps)
    return f"kontext_e{point.eta:g}_w{point.first_guided_step}_{end}"


def kontext_arms(config: KontextConfig) -> tuple[KontextArmSpec, ...]:
    """Baseline first, the primary guided arm, then one arm per grid point."""
    guidance = config.guidance
    specs = [
        KontextArmSpec(BASELINE_ARM, False, 0.0, 0, None, True),
        KontextArmSpec(
            GUIDED_ARM,
            True,
            guidance.eta,
            guidance.first_guided_step,
            guidance.guided_steps,
            True,
        ),
    ]
    for point in config.grid:
        specs.append(
            KontextArmSpec(
                grid_arm_id(point),
                True,
                point.eta,
                point.first_guided_step,
                point.guided_steps,
                False,
            )
        )
    names = [spec.arm for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(
            f"kontext arms collide: {names} — a grid point duplicates another "
            f"or shadows a named arm"
        )
    return tuple(specs)


@dataclass(frozen=True)
class StepTraceRow:
    """One step's instrumentation: what the reward read and what steering applied.

    Args:
        energy: the reward's mean energy at the step's own state (the first
            evaluation, before any inner-loop displacement); `None` on a step
            whose reward was never evaluated.
        gradient_rms: RMS of the raw energy gradient at that same state.
        displacement_rms: RMS of the displacement the tilt actually applied —
            zero-signal steps are absent rather than zeroed.
    """

    step: int
    sigma: float
    energy: float | None
    gradient_rms: float | None
    displacement_rms: float | None

    def as_row(self) -> dict[str, object]:
        return {
            "step": self.step,
            "sigma": self.sigma,
            "energy": self.energy,
            "gradient_rms": self.gradient_rms,
            "displacement_rms": self.displacement_rms,
        }


@dataclass(frozen=True)
class KontextArmResult:
    """One arm's render, its depth readings against both references, and its scores.

    Args:
        depth_error: against the stripped empty room's shell on the common
            empty mask — the primary reading and what the guided arms steered
            toward.
        depth_error_photo: against the furnished photo's shell on the common
            photo mask — the secondary column that quantifies the reference
            switch.
        trace: the walk's per-step instrumentation, empty for a walk that
            evaluated the reward on no step.
    """

    arm: str
    guided: bool
    eta: float
    first_guided_step: int
    guided_steps: int | None
    prompt: str
    depth_error: float
    depth_error_photo: float
    shell_coverage: float
    adherence: float
    image_similarity: float | None
    preference: float
    render: str
    depth_map: str
    residual_map: str
    seconds: float
    peak_vram_bytes: int | None
    trace: tuple[StepTraceRow, ...]

    def as_row(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "guided": self.guided,
            "eta": self.eta,
            "first_guided_step": self.first_guided_step,
            "guided_steps": self.guided_steps,
            "prompt": self.prompt,
            "depth_error": self.depth_error,
            "depth_error_photo": self.depth_error_photo,
            "shell_coverage": self.shell_coverage,
            "adherence": self.adherence,
            "image_similarity": self.image_similarity,
            "preference": self.preference,
            "render": self.render,
            "depth_map": self.depth_map,
            "residual_map": self.residual_map,
            "seconds": self.seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
            "trace": [row.as_row() for row in self.trace],
        }


@dataclass(frozen=True)
class EndpointSnapshot:
    """One decoded one-step endpoint estimate of a walk, with everything read off it.

    Args:
        depth_error: the estimate's aligned depth error against the empty-room
            reference, scored on the full reference shell — never a
            resegmentation of a half-formed estimate — so the curve is one mask
            across all steps and both filmed arms.
        segmentation_map: the estimate's EoMT segmentation, painted class by
            class — what the per-step resegmented reward mask was built from.
        depth_map: the estimate's masked disparity over the reference shell —
            what the depth network read off the estimate.
    """

    step: int
    sigma: float
    image: str
    segmentation_map: str
    depth_map: str
    depth_error: float

    def as_row(self) -> dict[str, object]:
        return {
            "step": self.step,
            "sigma": self.sigma,
            "image": self.image,
            "segmentation_map": self.segmentation_map,
            "depth_map": self.depth_map,
            "depth_error": self.depth_error,
        }


@dataclass(frozen=True)
class KontextSceneResult:
    """Every arm of one (scene, seed), scored on one common pixel set per reference.

    Args:
        common_coverage: share of the empty room's shell the common mask still
            scores; `photo_common_coverage` is the same reading for the
            secondary photo-shell column.
        common_frame_fraction / photo_common_frame_fraction: the two common
            masks as shares of the whole frame — the enlarged valid area the
            empty-room reference was chosen for, quantified.
        strip: the stage-zero strip's own numbers row, verbatim — including
            `residual_contents_fraction`, the flag that names a scene whose
            "empty" reference still holds furniture the strip failed to cut.
        endpoint_strips: each filmed arm's decoded endpoint estimates in step
            order — the sharpness evidence, and through each snapshot's depth
            error the convergence curve.
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
    common_frame_fraction: float
    photo_common_coverage: float
    photo_common_frame_fraction: float
    photo_shell_frame_fraction: float
    empty_shell_frame_fraction: float
    reconstruction_depth_error: float
    reconstruction_depth_error_photo: float
    mean_abs_baseline_residual: float
    photo_image: str
    empty_room_image: str
    contents_overlay: str
    photo_segmentation_map: str
    empty_room_segmentation_map: str
    photo_depth_map: str
    empty_room_depth_map: str
    photo_depth_map_masked: str
    empty_depth_map_masked: str
    empty_shell_mask: str
    photo_shell_mask: str
    eval_mask_common: str
    eval_mask_common_photo: str
    strip: dict[str, object]
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

    def photo_depth_delta_vs_baseline(self, arm_id: str) -> float | None:
        """The same delta under the secondary photo-shell reference."""
        baseline = self.arm(BASELINE_ARM).depth_error_photo
        if baseline == 0.0:
            return None
        return self.arm(arm_id).depth_error_photo / baseline - 1.0

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
            "common_frame_fraction": self.common_frame_fraction,
            "photo_common_coverage": self.photo_common_coverage,
            "photo_common_frame_fraction": self.photo_common_frame_fraction,
            "photo_shell_frame_fraction": self.photo_shell_frame_fraction,
            "empty_shell_frame_fraction": self.empty_shell_frame_fraction,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "reconstruction_depth_error_photo": self.reconstruction_depth_error_photo,
            "mean_abs_baseline_residual": self.mean_abs_baseline_residual,
            "photo_image": self.photo_image,
            "empty_room_image": self.empty_room_image,
            "contents_overlay": self.contents_overlay,
            "photo_segmentation_map": self.photo_segmentation_map,
            "empty_room_segmentation_map": self.empty_room_segmentation_map,
            "photo_depth_map": self.photo_depth_map,
            "empty_room_depth_map": self.empty_room_depth_map,
            "photo_depth_map_masked": self.photo_depth_map_masked,
            "empty_depth_map_masked": self.empty_depth_map_masked,
            "empty_shell_mask": self.empty_shell_mask,
            "photo_shell_mask": self.photo_shell_mask,
            "eval_mask_common": self.eval_mask_common,
            "eval_mask_common_photo": self.eval_mask_common_photo,
            "strip": self.strip,
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
                    "photo_depth_delta_vs_baseline": self.photo_depth_delta_vs_baseline(
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


@dataclass(frozen=True)
class _StrippedScene:
    """Stage zero's product for one scene: the empty room and the reference read off it."""

    empty_room: Float[Tensor, "1 3 height width"]
    reference: GeometricReference
    photo_shell_frame_fraction: float
    empty_shell_frame_fraction: float
    strip_row: dict[str, object]
    images: dict[str, str]


def _strip_scenes(
    config: KontextConfig,
    *,
    photo_paths: dict[str, Path],
    image_dir: Path,
    settings: RoomgenSettings,
) -> dict[str, _StrippedScene]:
    """Stage zero: every photo stripped bare and its empty-room reference built.

    The whole SD strip stack is released before this returns — nothing later
    shares a card with it. The reference tensors survive the release; they are
    constants of the empty room.
    """
    _ensure_strip_study_on_path()
    # Importable only once the strip study is on the path, exactly as the fake
    # stacks are reached.
    from strip_and_furnish.backends import build_backends  # noqa: PLC0415
    from strip_and_furnish.config import (  # noqa: PLC0415
        BackendConfig as StripBackendConfig,
    )
    from strip_and_furnish.config import (  # noqa: PLC0415
        SceneConfig as StripSceneConfig,
    )
    from strip_and_furnish.config import (  # noqa: PLC0415
        StripConfig,
    )
    from strip_and_furnish.strip import strip_scene  # noqa: PLC0415

    strip = config.strip
    backends = build_backends(
        StripBackendConfig(
            kind=config.backends.kind,
            device=config.backends.device,
            deterministic=config.backends.deterministic,
            dtype=strip.dtype,
            depth_dtype=config.backends.depth_dtype,
            segmentation_dtype=config.backends.segmentation_dtype,
            # The fake stack ships its own scheduler and refuses any other.
            scheduler="ddim" if config.backends.kind == "real" else "pipeline",
            depth_checkpoint=config.backends.depth_checkpoint,
            segmentation_checkpoint=config.backends.segmentation_checkpoint,
        ),
        cache_dir=settings.hub_cache,
    )
    strip_config = StripConfig(
        prompt=strip.prompt,
        negative_prompt=strip.negative_prompt,
        dilation_radius=strip.dilation_radius,
        passes=strip.passes,
        steps=strip.steps,
        fill_strength=strip.fill_strength,
        guidance_scale=strip.guidance_scale,
    )
    inpainter = backends.build_inpainter()
    device = backends.device
    print(f"\n[strip] {len(config.scenes)} scenes bare before anything samples")
    stripped: dict[str, _StrippedScene] = {}
    for scene in config.scenes:
        photo = load_photo(
            photo_paths[scene.photo],
            resolution=config.resolution,
            device=device,
            dtype=backends.dtype,
        )
        result = strip_scene(
            StripSceneConfig(photo=scene.photo, prompt=scene.instruction_prompt),
            photo,
            strip=strip_config,
            erosion_radius=config.guidance.erosion_radius,
            backends=backends,
            inpainter=inpainter,
            seed=config.seeds[0],
            gallery=image_dir,
        )
        empty_room = result.empty_room.to(torch.float32)
        reference = GeometricReference.from_photo(
            empty_room,
            depth_estimator=backends.depth_estimator,
            segmenter=backends.segmenter,
            erosion_radius=config.guidance.erosion_radius,
        )
        entry = _StrippedScene(
            empty_room=empty_room,
            reference=reference,
            photo_shell_frame_fraction=float(
                result.reference.mask.to(torch.float32).mean()
            ),
            empty_shell_frame_fraction=float(reference.mask.to(torch.float32).mean()),
            strip_row=result.as_row(),
            images=dict(result.images),
        )
        stripped[scene.photo] = entry
        print(
            f"  {scene.photo}: cut {result.numbers.contents_fraction:.0%} of the "
            f"frame, hole still recognisable contents "
            f"{result.numbers.residual_contents_fraction:.1%}, empty shell "
            f"holds {entry.empty_shell_frame_fraction:.0%} of the frame vs the "
            f"photo shell's {entry.photo_shell_frame_fraction:.0%}, "
            f"{result.numbers.seconds:.1f}s"
        )
    del inpainter, backends
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stripped


def run_kontext_study(
    config: KontextConfig, settings: RoomgenSettings
) -> KontextSummary:
    """Strip every scene, run every arm of every (scene, seed), write the artifacts."""
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

    # Stage zero on an otherwise empty GPU: the SD strip stack in, the empty
    # rooms and their references out, the stack freed before Kontext loads.
    stripped = _strip_scenes(
        config, photo_paths=photo_paths, image_dir=image_dir, settings=settings
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
                stripped=stripped[scene.photo],
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
    """FMRG's step rule, told which step it is and gated by the step/sigma window.

    The wrapper is also what stamps the current step index into the shared
    cell *before* the reward is evaluated, which is how the endpoint recorder
    and the energy tap know which step an evaluation belongs to — and, when a
    `trace` is carried, what records each steered step's gradient RMS and
    applied displacement RMS. The recording reads detached copies only; the
    gradient handed to the inner rule is the reward's own tensor.
    """

    inner: AnnealedSignTilt
    sigmas: tuple[float, ...]
    max_sigma: float | None
    first_step: int
    cell: _StepCell
    trace: dict[int, dict[str, float]] | None = None

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, batch * state],
        step: StepContext,
    ) -> Float[Tensor, batch * state] | None:
        self.cell.index = step.index
        if step.index < self.first_step:
            return None
        if self.max_sigma is not None and self.sigmas[step.index] > self.max_sigma:
            return None
        if self.trace is None:
            return self.inner(energy_gradient, state=state, step=step)
        captured: dict[str, float] = {}

        def tapped(state: Float[Tensor, batch2 * rest]) -> Float[Tensor, batch2 * rest]:
            gradient = energy_gradient(state)
            # Only the first evaluation — the step's own state — is the trace's.
            captured.setdefault("gradient_rms", _rms(gradient))
            return gradient

        steering = self.inner(tapped, state=state, step=step)
        if steering is not None:
            self.trace[step.index] = {
                "gradient_rms": captured.get("gradient_rms", 0.0),
                "displacement_rms": _rms(steering),
            }
        return steering


def _rms(tensor: Float[Tensor, batch * state]) -> float:
    return float(tensor.detach().to(torch.float32).square().mean().sqrt())


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


@dataclass
class _EnergyTap:
    """The reward with a meter on it: each step's first energy, kept as a float.

    Observation only, like the recorder — the returned energy is the inner
    reward's own attached tensor.
    """

    inner: Guidance
    cell: _StepCell
    energies: dict[int, float] = field(default_factory=dict)

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, batch]:
        energy = self.inner(predicted_image)
        self.energies.setdefault(self.cell.index, float(energy.detach().mean()))
        return energy


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
    first_guided_step: int = 0,
    guided_steps: int | None = None,
    seed: int,
    steps: int | None = None,
    recorder_every: int | None = None,
) -> tuple[
    Float[Tensor, "1 3 height width"],
    float,
    int | None,
    list[EndpointFrame],
    tuple[StepTraceRow, ...],
]:
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
    tilt_trace: dict[int, dict[str, float]] = {}
    tilt: TiltPolicy
    if eta == 0.0 and recorder_every is not None:
        tilt = _ObservationTilt(cell=cell, every=recorder_every)
    else:
        tilt = _SigmaWindowedTilt(
            inner=AnnealedSignTilt(
                strength=eta,
                guided_steps=guided_steps,
                anneal_steps=process.steps,
                iterations=config.guidance.iterations,
            ),
            sigmas=sigmas,
            max_sigma=config.guidance.max_sigma,
            first_step=first_guided_step,
            cell=cell,
            trace=tilt_trace,
        )
    reward: Guidance = GeometricGuidance(
        depth_estimator=stacks.depth_estimator,
        reference=reference,
        segmenter=stacks.segmenter if config.guidance.resegment else None,
        erosion_radius=(
            config.guidance.erosion_radius if config.guidance.resegment else None
        ),
    )
    tap = _EnergyTap(inner=reward, cell=cell)
    reward = tap
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
    trace = tuple(
        StepTraceRow(
            step=index,
            sigma=sigmas[index],
            energy=tap.energies.get(index),
            gradient_rms=(tilt_trace.get(index) or {}).get("gradient_rms"),
            displacement_rms=(tilt_trace.get(index) or {}).get("displacement_rms"),
        )
        for index in sorted(set(tap.energies) | set(tilt_trace))
    )
    return (
        image.to(torch.float32),
        seconds,
        peak,
        [EndpointFrame(index, sigmas[index], frame) for index, frame in frames],
        trace,
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
    stripped: _StrippedScene,
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
    # The stripped empty room's shell: the reference the reward defends and the
    # primary depth reading — the owner's setting, as the Redux sweep took it.
    reference = stripped.reference
    empty_room = stripped.empty_room.to(stacks.device)
    # The furnished photo's shell stays as the secondary depth column, rebuilt
    # here from the float32 photo so the numbers stay comparable to the
    # photo-reference run this study replaces (results_photoref.json).
    photo_reference = GeometricReference.from_photo(
        photo,
        depth_estimator=stacks.depth_estimator,
        segmenter=stacks.segmenter,
        erosion_radius=config.guidance.erosion_radius,
    )
    empty_shell_mask = f"{scene.photo}_empty_shell_mask.png"
    photo_shell_mask = f"{scene.photo}_shell_mask.png"
    empty_depth_map_masked = f"{scene.photo}_empty_depth_masked.png"
    photo_depth_map_masked = f"{scene.photo}_photo_depth_masked.png"
    _write_mask(reference.mask, image_dir / empty_shell_mask)
    _write_mask(photo_reference.mask, image_dir / photo_shell_mask)
    write_masked_disparity_map(
        reference.disparity,
        mask=reference.mask,
        path=image_dir / empty_depth_map_masked,
    )
    write_masked_disparity_map(
        photo_reference.disparity,
        mask=photo_reference.mask,
        path=image_dir / photo_depth_map_masked,
    )

    backend = _scene_backend(scene, config=config, stacks=stacks, photo=photo)
    # The floor the primary depth column is read against: what a perfect
    # geometry-preserving renovation still pays for FLUX's VAE round trip of
    # the reference geometry alone. The photo reference's floor rides beside it
    # for the secondary column.
    reconstruction = depth_error(
        _vae_roundtrip(backend, empty_room, device=stacks.device),
        reference=reference,
        depth_estimator=stacks.depth_estimator,
    )
    reconstruction_photo = depth_error(
        _vae_roundtrip(backend, photo, device=stacks.device),
        reference=photo_reference,
        depth_estimator=stacks.depth_estimator,
    )
    sigmas = _walk_sigmas(
        photo, backend, steps=config.sampler.steps, device=stacks.device
    )
    print(
        f"[scene] {scene.photo}: empty shell covers "
        f"{stripped.empty_shell_frame_fraction:.1%} of the frame (photo shell "
        f"{stripped.photo_shell_frame_fraction:.1%}), reconstruction floor "
        f"{reconstruction:.6f} (photo {reconstruction_photo:.6f}), first sigma "
        f"{sigmas[0]:.4f}"
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
        scene,
        config=config,
        stacks=stacks,
        settings=settings,
        reference=reference,
        photo_reference=photo_reference,
    )

    specs = kontext_arms(config)
    results: list[_PendingScene] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        peaks: dict[str, int | None] = {}
        traces: dict[str, tuple[StepTraceRow, ...]] = {}
        strips: dict[str, tuple[EndpointSnapshot, ...]] = {}
        for spec in specs:
            image, elapsed, peak, frames, trace = _sample_walk(
                photo,
                backend,
                config=config,
                stacks=stacks,
                reference=reference,
                eta=spec.eta,
                first_guided_step=spec.first_guided_step,
                guided_steps=spec.guided_steps,
                seed=seed,
                recorder_every=(
                    config.endpoint_strip_every if spec.records_strip else None
                ),
            )
            renders[spec.arm] = image
            seconds[spec.arm] = elapsed
            peaks[spec.arm] = peak
            traces[spec.arm] = trace
            if spec.records_strip:
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

        # One segmentation per finished render serves both references' masks.
        # One pixel set per reference for every arm: that reference's shell
        # narrowed by each arm's own segmentation, intersected across all of
        # them, so the depth errors are commensurable and covering shell up
        # buys nothing.
        arm_masks: dict[str, Bool[Tensor, "1 height width"]] = {}
        arm_masks_photo: dict[str, Bool[Tensor, "1 height width"]] = {}
        for spec in specs:
            if config.guidance.resegment:
                with torch.no_grad():
                    segmentation = stacks.segmenter(renders[spec.arm])
                scorable = scorable_mask(
                    segmentation, erosion_radius=config.guidance.erosion_radius
                )
                arm_masks[spec.arm] = reference.mask & scorable
                arm_masks_photo[spec.arm] = photo_reference.mask & scorable
            else:
                arm_masks[spec.arm] = reference.mask
                arm_masks_photo[spec.arm] = photo_reference.mask
        common = reference.mask.clone()
        common_photo = photo_reference.mask.clone()
        for spec in specs:
            common &= arm_masks[spec.arm]
            common_photo &= arm_masks_photo[spec.arm]
        eval_mask_common = f"{scene.photo}_seed{seed}_evalmask_common.png"
        eval_mask_common_photo = f"{scene.photo}_seed{seed}_evalmask_common_photo.png"
        _write_mask(common, image_dir / eval_mask_common)
        _write_mask(common_photo, image_dir / eval_mask_common_photo)

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
                    eta=spec.eta,
                    first_guided_step=spec.first_guided_step,
                    guided_steps=spec.guided_steps,
                    prompt=scene.instruction_prompt,
                    depth_error=depth_error(
                        render,
                        reference=reference,
                        depth_estimator=stacks.depth_estimator,
                        mask=common,
                    ),
                    depth_error_photo=depth_error(
                        render,
                        reference=photo_reference,
                        depth_estimator=stacks.depth_estimator,
                        mask=common_photo,
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
                    trace=traces[spec.arm],
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
            common_frame_fraction=float(common.to(torch.float32).mean()),
            photo_common_coverage=shell_coverage(
                common_photo, reference=photo_reference
            ),
            photo_common_frame_fraction=float(common_photo.to(torch.float32).mean()),
            photo_shell_frame_fraction=stripped.photo_shell_frame_fraction,
            empty_shell_frame_fraction=stripped.empty_shell_frame_fraction,
            reconstruction_depth_error=reconstruction,
            reconstruction_depth_error_photo=reconstruction_photo,
            mean_abs_baseline_residual=baseline_scale,
            photo_image=stripped.images["photo"],
            empty_room_image=stripped.images["empty_room"],
            contents_overlay=stripped.images["contents_overlay"],
            photo_segmentation_map=stripped.images["photo_segmentation"],
            empty_room_segmentation_map=stripped.images["empty_room_segmentation"],
            photo_depth_map=stripped.images["photo_depth"],
            empty_room_depth_map=stripped.images["empty_room_depth"],
            photo_depth_map_masked=photo_depth_map_masked,
            empty_depth_map_masked=empty_depth_map_masked,
            empty_shell_mask=empty_shell_mask,
            photo_shell_mask=photo_shell_mask,
            eval_mask_common=eval_mask_common,
            eval_mask_common_photo=eval_mask_common_photo,
            strip=stripped.strip_row,
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
    """Each frame written small with what the judges read off it, at recording time only.

    Beside the RGB decode: the frame's EoMT segmentation (the per-step reward
    mask's raw material) and its masked disparity over the reference shell —
    both computed under `no_grad` on the already-detached frame, so nothing
    here can touch the guidance path. The depth error is read on the full
    reference shell: one mask across all steps and both filmed arms.
    """
    _ensure_strip_study_on_path()
    from strip_and_furnish.images import write_segmentation_map  # noqa: PLC0415

    snapshots = []
    size = (_STRIP_HEIGHT, _STRIP_HEIGHT)
    small_mask = (
        torch.nn.functional.interpolate(
            reference.mask.to(torch.float32).unsqueeze(1), size=size, mode="nearest"
        ).squeeze(1)
        > 0.5
    )
    for frame in frames:
        name = f"{scene}_seed{seed}_{arm}_endpoint_step{frame.step:02d}.png"
        segmentation_name = (
            f"{scene}_seed{seed}_{arm}_endpoint_step{frame.step:02d}_seg.png"
        )
        depth_name = f"{scene}_seed{seed}_{arm}_endpoint_step{frame.step:02d}_depth.png"
        image = frame.image.to(stacks.device)
        error = depth_error(
            image, reference=reference, depth_estimator=stacks.depth_estimator
        )
        with torch.no_grad():
            disparity = stacks.depth_estimator(image)
            segmentation = stacks.segmenter(image)
        small = torch.nn.functional.interpolate(frame.image, size=size, mode="area")
        _write_image(small, image_dir / name)
        # Class ids survive a nearest resample; anything else would blend them.
        small_segmentation = (
            torch.nn.functional.interpolate(
                segmentation.to(torch.float32).unsqueeze(1), size=size, mode="nearest"
            )
            .squeeze(1)
            .to(torch.long)
        )
        write_segmentation_map(small_segmentation, image_dir / segmentation_name)
        small_disparity = torch.nn.functional.interpolate(
            disparity.to(torch.float32).unsqueeze(1), size=size, mode="area"
        ).squeeze(1)
        write_masked_disparity_map(
            small_disparity, mask=small_mask, path=image_dir / depth_name
        )
        snapshots.append(
            EndpointSnapshot(
                frame.step, frame.sigma, name, segmentation_name, depth_name, error
            )
        )
    return tuple(snapshots)


def _fmrg_comparison(
    scene: KontextSceneConfig,
    *,
    config: KontextConfig,
    stacks: KontextStacks,
    settings: RoomgenSettings,
    reference: GeometricReference,
    photo_reference: GeometricReference,
) -> dict[str, object] | None:
    """The finished FMRG run's text-only pair, made comparable to this study's arms.

    That run steered and scored against its own strip's empty room over its
    own masks, so its recorded depth errors do not compare directly. The
    renders it saved are re-scored here against both of this study's
    references (each reference's shell AND the render's own shell), which are
    the numbers the comparison column quotes; the original empty-reference
    error rides along, labelled as what it is.
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
        photo_mask = _scorable(
            render,
            stacks=stacks,
            reference=photo_reference,
            resegment=config.guidance.resegment,
            erosion_radius=config.guidance.erosion_radius,
        )
        rows.append(
            {
                "arm": arm_id,
                "guided": bool(arm.get("guided")),
                "depth_error_vs_empty_shell": depth_error(
                    render,
                    reference=reference,
                    depth_estimator=stacks.depth_estimator,
                    mask=mask,
                ),
                "shell_coverage_vs_empty_shell": shell_coverage(
                    mask, reference=reference
                ),
                "depth_error_vs_photo_shell": depth_error(
                    render,
                    reference=photo_reference,
                    depth_estimator=stacks.depth_estimator,
                    mask=photo_mask,
                ),
                "shell_coverage_vs_photo_shell": shell_coverage(
                    photo_mask, reference=photo_reference
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
            "depth_error_vs_empty_shell and depth_error_vs_photo_shell are "
            "recomputed here on the run's saved renders against this study's "
            "empty-room and furnished-photo references, each over the render's "
            "own scorable mask; neither is the common-mask number of either "
            "study. depth_error_vs_empty_reference is the source run's own "
            "reading against its own strip's empty room."
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
        f"{cell.arm} E={cell.depth_error:.4f} Ep={cell.depth_error_photo:.4f} "
        f"adh={cell.adherence:.4f}"
        for cell in result.arms
    )
    return (
        f"seed {result.seed}: {cells}, empty shell scored "
        f"{result.common_coverage:.0%} in common "
        f"({result.common_frame_fraction:.0%} of frame)"
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
            "depth_error_photo": _mean([cell.depth_error_photo for cell in cells]),
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
                # What steered every guided arm and what the primary depth
                # column is read against: the stripped empty room's shell
                # depth. The furnished photo's shell rides along as the
                # secondary `depth_error_photo` column.
                "guidance_reference": "empty_room",
                # The strip's depth errors are read on the full empty
                # reference shell (one mask across all steps and both filmed
                # arms), never a resegmentation of a half-formed estimate.
                "endpoint_strip_mask": "empty_reference_shell",
                "eta": summary.config.guidance.eta,
                "grid": [
                    point.model_dump(mode="json") for point in summary.config.grid
                ],
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
