"""The image-reward study: the user's photo injected at the reward point, per helper model.

Per scene at one seed, all arms off one transformer load: an unguided control
and the depth-only anchor under the scene's style prompt, the same pair again
under an explicit renovation instruction (keep the shell, remove one named
object), one image-only guided arm per frozen helper model (the render pulled
toward the reference photo in that model's feature space) and an
image-plus-depth composite per configured model at a measured weight — the
helper arms all steering under the instruction prompt, so instruction-following
is read against prompt-matched baselines. The photo enters only at the reward
point and in the depth reference — the prior sees the one SDEdit init every arm
shares. Every arm is scored on the one pixel set all arms left visible, judged
by an adherence CLIP (always against the scene's one neutral style text), an
image-similarity CLIP and a preference model that are all independent of every
feature space any arm descended, plus two instruction readings: the share of
frame still segmented as the removed object (the photo's share as baseline) and
the aligned depth error restricted to the photo's windowpane pixels. Beside the
numbers, every arm leaves a masked depth map and a signed depth-error map on
the control's colour scale.

The helper models are parked on the CPU between walks: FMRG's guided walk peaks
near a 32GB card's ceiling, and only the arm being walked needs its tower
resident.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import torch
from jaxtyping import Float
from torch import Tensor

from framework.config import (
    TORCH_DTYPES,
    FmrgImageConfig,
    ImageModelName,
    ImageSceneConfig,
)
from framework.depth_maps import write_disparity_map, write_masked_disparity_map
from framework.fmrg_alignment_study import shell_class_table
from framework.fmrg_study import (
    _FAKE_PACKAGES,
    PHOTO_SUBDIRECTORY,
    Stacks,
    _git_commit,
    _vae_roundtrip,
    _write_image,
    _write_mask,
    build_stacks,
    sample_walk,
    walked_steps,
    warm_up_walk,
)
from framework.metrics import build_adherence, class_area_fraction, masked_mean_square
from framework.photos import ensure_photos, load_photo
from framework.preference import build_preference
from framework.residual_maps import mean_absolute_residual, write_signed_residual_map
from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.segmentation.ade20k import ADE20K_LABELS
from roomgen_guide.adapters.flow_map import FlowMapProcess, GuidanceWindow
from roomgen_guide.adapters.fmrg_flux import select_cached_prompt
from roomgen_guide.eval import (
    ClipImageSimilarity,
    PromptAdherence,
    depth_error,
    shell_coverage,
    signed_depth_residual_map,
)
from roomgen_guide.guidance.base import Guidance
from roomgen_guide.guidance.composite import WeightedGuidance, WeightedTerm
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.guidance.image_reference import (
    ImageReferenceEnergyFactory,
    load_clip_image_reference_energies,
    load_dinov2_image_reference_energies,
    load_siglip_image_reference_energies,
    load_vgg_perceptual_energies,
)
from roomgen_guide.guidance.preference import ImagePreferenceModel
from roomgen_guide.process.endpoint import Endpoint, StepContext
from roomgen_guide.settings import RoomgenSettings

# The baseline arm's id: unguided at the sampler's operating point. Every other
# arm's depth delta and adherence ratio divide by it, and its mean absolute
# residual sets the scene's residual-map colour scale.
CONTROL_ARM: Final = "control"
DEPTH_ARM: Final = "depth"

# The prompt-matched baselines the helper arms' instruction-following is read
# against: the same control and depth anchor, steering under the scene's
# instruction prompt — without them, "the table went away" cannot be attributed
# to the helper model rather than to the prompt itself.
CONTROL_INSTRUCTION_ARM: Final = "control_instr"
DEPTH_INSTRUCTION_ARM: Final = "depth_instr"

# The ADE20K id whose photo pixels the windows-kept reading is restricted to.
WINDOWPANE_CLASS: Final = ADE20K_LABELS.index("windowpane")

ArmKind = Literal["control", "depth", "image", "composite"]


@dataclass(frozen=True)
class ImageArmSpec:
    """One arm of the study; `model` names the helper for the image-steered kinds.

    `instruction` is the prompt the walk steers under: false is the scene's
    plain style description, true its renovation-instruction rephrasing. The
    eval prompt never varies with it.
    """

    arm: str
    kind: ArmKind
    model: ImageModelName | None
    instruction: bool

    @property
    def guided(self) -> bool:
        return self.kind != "control"

    def steering_prompt(self, scene: ImageSceneConfig) -> str:
        return scene.instruction_prompt if self.instruction else scene.style_prompt


def image_arms(config: FmrgImageConfig) -> tuple[ImageArmSpec, ...]:
    """The arms: both prompt-matched baseline pairs, then the model sweep, the composites.

    Style-prompt control and depth anchor first (continuity with the study's
    first run), then the same pair under the instruction prompt, then every
    helper arm — image-only and composite — under the instruction prompt.
    """
    arms = [
        ImageArmSpec(CONTROL_ARM, "control", None, instruction=False),
        ImageArmSpec(DEPTH_ARM, "depth", None, instruction=False),
        ImageArmSpec(CONTROL_INSTRUCTION_ARM, "control", None, instruction=True),
        ImageArmSpec(DEPTH_INSTRUCTION_ARM, "depth", None, instruction=True),
    ]
    for model in config.image.models:
        arms.append(ImageArmSpec(f"img_{model}", "image", model, instruction=True))
    for model in config.image.composite_models:
        arms.append(
            ImageArmSpec(f"imgdepth_{model}", "composite", model, instruction=True)
        )
    return tuple(arms)


@dataclass(frozen=True)
class WeightSizing:
    """Each term's pull at one real midpoint sample, and the weights read off it.

    `weights[model]` is that model's image-gradient RMS over the depth term's,
    so the composite's two terms pull equally hard in aggregate at that step.
    Equal RMS is not equal influence under FMRG's sign step, which spends the
    same magnitude on every element: a dense image gradient reaches every pixel
    while the masked depth gradient concentrates on the shell — recorded so the
    report can say it, not corrected here.
    """

    step_index: int
    total_steps: int
    depth_gradient_rms: float
    image_gradient_rms: dict[str, float]
    weights: dict[str, float]

    def as_row(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "depth_gradient_rms": self.depth_gradient_rms,
            "image_gradient_rms": self.image_gradient_rms,
            "weights": self.weights,
        }


@dataclass(frozen=True)
class ImageArmResult:
    """One arm's render, its depth reading, and its scores against the control's.

    `steering_prompt` is the exact text the walk conditioned on; `adherence` is
    still read against the scene's one neutral eval text. `removal_fraction` is
    the share of frame the segmenter still calls the removed object (the scene
    row carries the photo's share as baseline); `window_depth_error` is the
    common-mask-aligned squared depth residual over the photo's windowpane
    pixels, `None` for a scene the segmenter found no windowpane in.
    """

    arm: str
    kind: ArmKind
    model: str | None
    steering_prompt: str
    depth_weight: float | None
    depth_error: float
    shell_coverage: float
    removal_fraction: float
    window_depth_error: float | None
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
            "kind": self.kind,
            "model": self.model,
            "steering_prompt": self.steering_prompt,
            "depth_weight": self.depth_weight,
            "depth_error": self.depth_error,
            "shell_coverage": self.shell_coverage,
            "removal_fraction": self.removal_fraction,
            "window_depth_error": self.window_depth_error,
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
class ImageSceneResult:
    """Every arm of one (scene, seed), scored on one common pixel set."""

    scene: str
    seed: int
    style_prompt: str
    instruction_prompt: str
    eval_prompt: str
    removal_object: str
    removal_classes: tuple[int, ...]
    removal_substituted: bool
    photo_removal_fraction: float
    window_fraction: float
    start_sigma: float
    effective_first_sigma: float
    arms: tuple[ImageArmResult, ...]
    sizing: WeightSizing | None
    common_coverage: float
    reconstruction_depth_error: float
    mean_abs_control_residual: float
    photo_image: str
    photo_depth_map: str
    photo_depth_map_masked: str
    shell_mask: str
    window_mask: str
    eval_mask_common: str

    def arm(self, arm_id: str) -> ImageArmResult:
        for measurement in self.arms:
            if measurement.arm == arm_id:
                return measurement
        raise KeyError(arm_id)

    def depth_delta_vs_control(self, arm_id: str) -> float | None:
        """The arm's depth error relative to the control's, signed; `None` on a zero control."""
        control = self.arm(CONTROL_ARM).depth_error
        if control == 0.0:
            return None
        return self.arm(arm_id).depth_error / control - 1.0

    def adherence_ratio_vs_control(self, arm_id: str) -> float | None:
        """The arm's adherence as a share of the control's; `None` on a zero control."""
        control = self.arm(CONTROL_ARM).adherence
        if control == 0.0:
            return None
        return self.arm(arm_id).adherence / control

    def as_row(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "seed": self.seed,
            "style_prompt": self.style_prompt,
            "instruction_prompt": self.instruction_prompt,
            "eval_prompt": self.eval_prompt,
            "removal_object": self.removal_object,
            "removal_classes": list(self.removal_classes),
            "removal_substituted": self.removal_substituted,
            "photo_removal_fraction": self.photo_removal_fraction,
            "window_fraction": self.window_fraction,
            "start_sigma": self.start_sigma,
            "effective_first_sigma": self.effective_first_sigma,
            "sizing": None if self.sizing is None else self.sizing.as_row(),
            "common_coverage": self.common_coverage,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "mean_abs_control_residual": self.mean_abs_control_residual,
            "photo_image": self.photo_image,
            "photo_depth_map": self.photo_depth_map,
            "photo_depth_map_masked": self.photo_depth_map_masked,
            "shell_mask": self.shell_mask,
            "window_mask": self.window_mask,
            "eval_mask_common": self.eval_mask_common,
            "arms": [
                {
                    **measurement.as_row(),
                    "depth_delta_vs_control": self.depth_delta_vs_control(
                        measurement.arm
                    ),
                    "adherence_ratio_vs_control": self.adherence_ratio_vs_control(
                        measurement.arm
                    ),
                }
                for measurement in self.arms
            ],
        }


@dataclass(frozen=True)
class ImageSummary:
    """What the study measured, where its artifacts landed, and the code behind them."""

    config: FmrgImageConfig
    results: tuple[ImageSceneResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


@dataclass(frozen=True)
class _EnergyBank:
    """The loaded helper models, parked on the CPU until an arm needs one."""

    factories: dict[ImageModelName, ImageReferenceEnergyFactory]
    device: torch.device

    def energy(
        self, model: ImageModelName, photo: Float[Tensor, "1 3 height width"]
    ) -> Guidance:
        factory = self.factories[model]
        factory.module.to(self.device)
        return factory(photo)

    def park(self, model: ImageModelName) -> None:
        self.factories[model].module.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def build_energy_bank(
    config: FmrgImageConfig, *, device: torch.device, settings: RoomgenSettings
) -> _EnergyBank:
    """Every helper model any arm will steer by, loaded once and parked on the CPU."""
    needed = tuple(
        dict.fromkeys((*config.image.models, *config.image.composite_models))
    )
    factories: dict[ImageModelName, ImageReferenceEnergyFactory] = {}
    if config.backends.kind == "fake":
        for package in _FAKE_PACKAGES:
            if str(package) not in sys.path:
                sys.path.insert(0, str(package))
        # Importable only once the fakes are on the path, exactly as the fake
        # sampler stack is reached.
        from roomgen_guide_fakes import (  # noqa: PLC0415
            fake_image_reference_energies,
            fake_perceptual_energies,
        )

        for index, model in enumerate(needed):
            factories[model] = (
                fake_perceptual_energies()
                if model == "vgg"
                else fake_image_reference_energies(seed=index)
            )
        return _EnergyBank(factories=factories, device=device)

    dtype = TORCH_DTYPES[config.backends.depth_dtype]
    for model in needed:
        match model:
            case "clip":
                factories[model] = load_clip_image_reference_energies(
                    config.image.clip_checkpoint,
                    device=device,
                    dtype=dtype,
                    cache_dir=settings.hub_cache,
                )
            case "dinov2":
                factories[model] = load_dinov2_image_reference_energies(
                    config.image.dinov2_checkpoint,
                    device=device,
                    dtype=dtype,
                    cache_dir=settings.hub_cache,
                )
            case "siglip":
                factories[model] = load_siglip_image_reference_energies(
                    config.image.siglip_checkpoint,
                    device=device,
                    dtype=dtype,
                    cache_dir=settings.hub_cache,
                )
            case "vgg":
                factories[model] = load_vgg_perceptual_energies(
                    device=device,
                    dtype=dtype,
                    cache_dir=settings.hf_home,
                    input_size=config.image.vgg_input_size,
                )
        factories[model].module.to("cpu")
        parameters = sum(
            parameter.numel() for parameter in factories[model].module.parameters()
        )
        print(f"[bank] {model}: {parameters / 1e6:.0f}M parameters, parked on CPU")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _EnergyBank(factories=factories, device=device)


def run_image_study(config: FmrgImageConfig, settings: RoomgenSettings) -> ImageSummary:
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
    # Every prompt any arm will steer under, cached before the text encoders
    # are freed: each scene's style description and its instruction rephrasing.
    stacks = build_stacks(
        config.backends,
        prompts=list(
            dict.fromkeys(
                prompt
                for scene in config.scenes
                for prompt in (scene.style_prompt, scene.instruction_prompt)
            )
        ),
        text_guidance_scale=config.sampler.text_guidance_scale,
        settings=settings,
    )
    bank = build_energy_bank(config, device=stacks.device, settings=settings)
    if stacks.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(stacks.device)

    pending: list[_PendingScene] = []
    for index, scene in enumerate(config.scenes):
        pending.extend(
            _run_scene(
                scene,
                config=config,
                stacks=stacks,
                bank=bank,
                photo_path=photo_paths[scene.photo],
                image_dir=image_dir,
                warm_up=index == 0,
            )
        )

    # Loaded only after the last walk, one judge resident at a time: beside the
    # transformer, a resident eval tower is exactly what tipped a 32GB card
    # into out-of-memory. Renders wait on the CPU; scoring one at a time
    # through a frozen encoder is cheap.
    results = _score_pending(pending, config=config, stacks=stacks, settings=settings)
    for result in results:
        print(f"  {_format_result(result)}")

    on_cuda = stacks.device.type == "cuda"
    summary = ImageSummary(
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
    """One scene measured on everything but the deferred judges, renders parked on CPU."""

    result: ImageSceneResult
    renders: dict[str, Float[Tensor, "1 3 height width"]]
    photo: Float[Tensor, "1 3 height width"]
    reference: GeometricReference


def _score_pending(
    pending: list[_PendingScene],
    *,
    config: FmrgImageConfig,
    stacks: Stacks,
    settings: RoomgenSettings,
) -> list[ImageSceneResult]:
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
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_scene(
    scene: ImageSceneConfig,
    *,
    config: FmrgImageConfig,
    stacks: Stacks,
    bank: _EnergyBank,
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
    # The photo's own class map, for the two instruction baselines: how much of
    # the frame is the object the instruction removes, and where its windowpanes
    # are. Restricted to the eroded shell so the window reading shares the
    # reward's trust boundary at segment edges.
    with torch.no_grad():
        photo_segmentation = stacks.segmenter(photo)
    window_mask = (photo_segmentation == WINDOWPANE_CLASS) & reference.mask
    photo_removal_fraction = class_area_fraction(
        photo_segmentation, scene.removal_classes
    )
    photo_image = f"{scene.photo}_photo.png"
    shell_mask = f"{scene.photo}_shell_mask.png"
    window_mask_name = f"{scene.photo}_window_mask.png"
    photo_depth_map = f"{scene.photo}_photo_depth.png"
    photo_depth_map_masked = f"{scene.photo}_photo_depth_masked.png"
    _write_image(photo, image_dir / photo_image)
    _write_mask(reference.mask, image_dir / shell_mask)
    _write_mask(window_mask, image_dir / window_mask_name)
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
    window_fraction = float(window_mask.to(torch.float32).mean())
    print(
        f"[scene] {scene.photo}: shell covers "
        f"{float(reference.mask.to(torch.float32).mean()):.1%} of the frame, "
        f"reconstruction floor {reconstruction:.6f}, effective first sigma "
        f"{config.sampler.start_sigma:g}->{effective_sigma:.4f}"
    )
    print(
        f"[instr] remove the {scene.removal_object} "
        f"(classes {sorted(scene.removal_classes)}"
        f"{', substituted' if scene.removal_substituted else ''}): photo share "
        f"{photo_removal_fraction:.1%}; windowpane covers {window_fraction:.1%} "
        f"of the frame"
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

    specs = image_arms(config)
    # The composites steer under the instruction prompt, so the weights are
    # measured on the trajectory they will actually walk.
    if stacks.sampler is not None:
        select_cached_prompt(stacks.sampler, scene.instruction_prompt)
    sizing = (
        _measure_sizing(
            photo,
            config=config,
            stacks=stacks,
            bank=bank,
            reference=reference,
            seed=config.seeds[0],
        )
        if config.image.composite_models
        else None
    )
    if sizing is not None:
        rms = ", ".join(
            f"{model} {value:.3e} (w {sizing.weights[model]:.3g})"
            for model, value in sizing.image_gradient_rms.items()
        )
        print(
            f"[sizing] step {sizing.step_index}/{sizing.total_steps}: depth rms "
            f"{sizing.depth_gradient_rms:.3e}, {rms}"
        )

    results: list[_PendingScene] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        peaks: dict[str, int | None] = {}
        weights: dict[str, float | None] = {}
        for spec in specs:
            if stacks.sampler is not None:
                select_cached_prompt(stacks.sampler, spec.steering_prompt(scene))
            reward, weight = _arm_reward(
                spec,
                photo=photo,
                config=config,
                stacks=stacks,
                bank=bank,
                reference=reference,
                sizing=sizing,
            )
            weights[spec.arm] = weight
            image, elapsed, peak = sample_walk(
                photo,
                stacks=stacks,
                sampler=config.sampler,
                guidance=config.guidance,
                reference=reference,
                eta=config.guidance.eta if spec.guided else 0.0,
                seed=seed,
                reward=reward,
            )
            if spec.model is not None:
                bank.park(spec.model)
            renders[spec.arm] = image
            seconds[spec.arm] = elapsed
            peaks[spec.arm] = peak
            _write_image(image, image_dir / f"{scene.photo}_seed{seed}_{spec.arm}.png")

        # Each finished render is segmented once, and that one class map serves
        # both readings taken off it: the arm's evaluation mask and the share of
        # frame still labelled as the object the instruction removed.
        with torch.no_grad():
            segmentations = {
                spec.arm: stacks.segmenter(renders[spec.arm]) for spec in specs
            }
        removal_fractions = {
            spec.arm: class_area_fraction(
                segmentations[spec.arm], scene.removal_classes
            )
            for spec in specs
        }
        # One pixel set for every arm: the reference shell narrowed by each
        # arm's own segmentation, intersected across all of them, so the depth
        # errors are commensurable and covering shell up buys nothing.
        arm_masks = {
            spec.arm: (
                reference.mask
                & scorable_mask(
                    segmentations[spec.arm],
                    erosion_radius=config.guidance.erosion_radius,
                )
                if config.guidance.resegment
                else reference.mask
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
        control_scale = mean_absolute_residual(residuals[CONTROL_ARM], common)

        measurements: list[ImageArmResult] = []
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
                normalizer=control_scale,
                path=image_dir / residual_map,
            )
            measurements.append(
                ImageArmResult(
                    arm=spec.arm,
                    kind=spec.kind,
                    model=spec.model,
                    steering_prompt=spec.steering_prompt(scene),
                    depth_weight=weights[spec.arm],
                    depth_error=depth_error(
                        render,
                        reference=reference,
                        depth_estimator=stacks.depth_estimator,
                        mask=common,
                    ),
                    shell_coverage=shell_coverage(
                        arm_masks[spec.arm], reference=reference
                    ),
                    removal_fraction=removal_fractions[spec.arm],
                    # The residual map is aligned over the whole common shell;
                    # the window reading is its mean square over the photo's
                    # windowpane pixels that survived every arm's mask.
                    window_depth_error=masked_mean_square(
                        residuals[spec.arm], window_mask & common
                    ),
                    # Filled in by the deferred judges once sampling is over
                    # and an eval tower can afford to be resident.
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

        result = ImageSceneResult(
            scene=scene.photo,
            seed=seed,
            style_prompt=scene.style_prompt,
            instruction_prompt=scene.instruction_prompt,
            eval_prompt=scene.eval_prompt,
            removal_object=scene.removal_object,
            removal_classes=scene.removal_classes,
            removal_substituted=scene.removal_substituted,
            photo_removal_fraction=photo_removal_fraction,
            window_fraction=window_fraction,
            start_sigma=config.sampler.start_sigma,
            effective_first_sigma=effective_sigma,
            arms=tuple(measurements),
            sizing=sizing,
            common_coverage=shell_coverage(common, reference=reference),
            reconstruction_depth_error=reconstruction,
            mean_abs_control_residual=control_scale,
            photo_image=photo_image,
            photo_depth_map=photo_depth_map,
            photo_depth_map_masked=photo_depth_map_masked,
            shell_mask=shell_mask,
            window_mask=window_mask_name,
            eval_mask_common=eval_mask_common,
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


def _arm_reward(
    spec: ImageArmSpec,
    *,
    photo: Float[Tensor, "1 3 height width"],
    config: FmrgImageConfig,
    stacks: Stacks,
    bank: _EnergyBank,
    reference: GeometricReference,
    sizing: WeightSizing | None,
) -> tuple[Guidance | None, float | None]:
    """The energy one arm descends, and the depth weight if it carries one.

    `None` hands the walk to `sample_walk`'s own geometric energy (the depth
    anchor) or to nothing at all (the eta-0 control, which never evaluates it).
    """
    match spec.kind:
        case "control" | "depth":
            return None, None
        case "image":
            assert spec.model is not None
            return bank.energy(spec.model, photo), None
        case "composite":
            assert spec.model is not None
            weight = _composite_weight(spec.model, config=config, sizing=sizing)
            image_term = bank.energy(spec.model, photo)
            depth_term = GeometricGuidance(
                depth_estimator=stacks.depth_estimator,
                reference=reference,
                segmenter=stacks.segmenter if config.guidance.resegment else None,
                erosion_radius=(
                    config.guidance.erosion_radius
                    if config.guidance.resegment
                    else None
                ),
            )
            return (
                WeightedGuidance(
                    (WeightedTerm(image_term, 1.0), WeightedTerm(depth_term, weight))
                ),
                weight,
            )


def _composite_weight(
    model: ImageModelName, *, config: FmrgImageConfig, sizing: WeightSizing | None
) -> float:
    if config.image.depth_weight is not None:
        return config.image.depth_weight
    if sizing is None or model not in sizing.weights:
        raise RuntimeError(
            f"no measured weight for {model}: the sizing pass covers every "
            f"composite model, so a composite arm without one is a bug"
        )
    return sizing.weights[model]


def _measure_sizing(
    photo: Float[Tensor, "1 3 height width"],
    *,
    config: FmrgImageConfig,
    stacks: Stacks,
    bank: _EnergyBank,
    reference: GeometricReference,
    seed: int,
) -> WeightSizing:
    """Each term's gradient RMS at one real, partially denoised sample.

    Guessing a weight between an embedding cosine and a masked squared
    disparity is guessing a unit conversion; this measures it instead, on the
    very walk the arms run, at a step `image.sizing_fraction` of the way into
    the guided window. One endpoint decode serves every term's backward.
    """
    generator = torch.Generator(device=stacks.device).manual_seed(seed)
    window = GuidanceWindow(
        early_stop=config.sampler.early_stop,
        unguided_steps=config.sampler.unguided_steps,
    )
    process = FlowMapProcess(
        stacks.flow_backend,
        photo,
        steps=config.sampler.grid_steps,
        start_sigma=config.sampler.start_sigma,
        generator=generator,
        window=window,
        stepping=config.sampler.stepping,
        differentiate_through_velocity=config.sampler.differentiate_through_velocity,
    )
    index = min(
        max(1, int(window.guided_steps * config.image.sizing_fraction)),
        process.steps - 1,
    )
    state = process.initial_state()
    with torch.no_grad():
        for walked in range(index):
            state = process.advance(
                state, _context(process, walked), prediction=None, steering=None
            )

    depth_term = GeometricGuidance(
        depth_estimator=stacks.depth_estimator,
        reference=reference,
        segmenter=stacks.segmenter if config.guidance.resegment else None,
        erosion_radius=(
            config.guidance.erosion_radius if config.guidance.resegment else None
        ),
    )
    # `enable_grad` rather than an ambient mode, exactly as the driver takes
    # its own gradient; the decode graph is retained so every term's backward
    # shares the one endpoint.
    with torch.enable_grad():
        endpoint = process.endpoint(state, _context(process, index))
        depth_rms = _gradient_rms(endpoint, depth_term)
        image_rms: dict[str, float] = {}
        weights: dict[str, float] = {}
        for model in config.image.composite_models:
            energy = bank.energy(model, photo)
            image_rms[model] = _gradient_rms(endpoint, energy)
            bank.park(model)
            if depth_rms <= 0.0:
                raise RuntimeError(
                    "the depth term measured a zero gradient at the sizing "
                    "step, so there is no weight to read off the ratio — the "
                    "gradient path is detached somewhere"
                )
            weights[model] = image_rms[model] / depth_rms
    return WeightSizing(
        step_index=index,
        total_steps=process.steps,
        depth_gradient_rms=depth_rms,
        image_gradient_rms=image_rms,
        weights=weights,
    )


def _context(process: FlowMapProcess, index: int) -> StepContext:
    return StepContext(index=index, total=process.steps, span=process.step_span(index))


def _gradient_rms(endpoint: Endpoint[Tensor], energy: Guidance) -> float:
    (gradient,) = torch.autograd.grad(
        energy(endpoint.image).mean(), endpoint.state, retain_graph=True
    )
    return float(gradient.to(torch.float32).square().mean().sqrt())


def _effective_first_sigma(
    photo: Float[Tensor, "1 3 height width"],
    *,
    config: FmrgImageConfig,
    stacks: Stacks,
) -> float:
    """The sigma every walk actually starts at, read off the backend's own schedule.

    FLUX's mu-shift warps the grid by resolution, so the configured
    `start_sigma` is a request, not the truth. One value serves the whole
    scene, since every arm shares the one init.
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


def _format_result(result: ImageSceneResult) -> str:
    cells = ", ".join(
        f"{cell.arm} E={cell.depth_error:.4f} adh={cell.adherence:.4f}"
        for cell in result.arms
    )
    return (
        f"{result.scene} seed {result.seed}: {cells}, shell scored "
        f"{result.common_coverage:.0%} in common"
    )


def arm_means(
    results: tuple[ImageSceneResult, ...], specs: tuple[ImageArmSpec, ...]
) -> dict[str, dict[str, float | None]]:
    """Each arm's mean scores over every (scene, seed)."""
    means: dict[str, dict[str, float | None]] = {}
    for spec in specs:
        cells = [result.arm(spec.arm) for result in results]
        similarities = [
            cell.image_similarity for cell in cells if cell.image_similarity is not None
        ]
        windows = [
            cell.window_depth_error
            for cell in cells
            if cell.window_depth_error is not None
        ]
        means[spec.arm] = {
            "depth_error": _mean([cell.depth_error for cell in cells]),
            "shell_coverage": _mean([cell.shell_coverage for cell in cells]),
            "removal_fraction": _mean([cell.removal_fraction for cell in cells]),
            "window_depth_error": _mean(windows) if windows else None,
            "adherence": _mean([cell.adherence for cell in cells]),
            "image_similarity": _mean(similarities) if similarities else None,
            "preference": _mean([cell.preference for cell in cells]),
            "seconds": _mean([cell.seconds for cell in cells]),
        }
    return means


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_results(summary: ImageSummary, settings: RoomgenSettings) -> None:
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
                # The shell definition is under revision: the results must say
                # which one scored them, not point at whatever the package says
                # later.
                "shell_classes": shell_class_table(),
                "eta": summary.config.guidance.eta,
                "torch_version": torch.__version__,
                "device_name": summary.device_name,
                "walked_steps": summary.walked_steps,
                "peak_vram_bytes": summary.peak_vram_bytes,
                "arm_means": arm_means(summary.results, image_arms(summary.config)),
                "runs": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
