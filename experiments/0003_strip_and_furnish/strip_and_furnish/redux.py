"""The Redux x depth x image-reward grid over the two-stage FMRG furnish.

Stage 1 is the unchanged strip. Stage 2 walks the stripped empty room seven
times per (scene, seed): an unguided control, then every owner-named
combination of three channels that all point back at the same empty room —
the room injected into the *prior* as FLUX.1-Redux image context (appended to
the instruction prompt at a scale that keeps the text audible), the depth
reward against the empty room's shell depth, and a DINOv2 patch-cosine image
reward toward the empty room restricted to shell patch positions. The image
reward is masked because its reference contains no furniture: a full-frame
similarity would punish furnishing itself, so only patches the empty room's
shell covers are scored, intersected per step with the prediction's own
resegmented shell exactly as the depth reward's pixel mask is.

Every furnish arm steers under one instruction prompt built from the scene's
style descriptors; the scene's neutral style text is what every judge reads.
The judges — adherence CLIP, an image-similarity CLIP scored against both the
empty room and the original photo, and PickScore — load only after the last
walk, and are all independent of every feature space any arm descends.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.segmentation.ade20k import (
    ADE20K_LABELS,
    ARCHITECTURAL_CLASSES,
    REMOVABLE_CLASSES,
)
from roomgen_guide.adapters.flow_map import FlowMapProcess, GuidanceWindow
from roomgen_guide.adapters.fmrg_flux import (
    encode_redux_image_tokens,
    register_image_context_prompt,
    select_cached_prompt,
)
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
    MaskedPatchReferenceEnergies,
    load_dinov2_masked_patch_energies,
)
from roomgen_guide.guidance.preference import ImagePreferenceModel
from roomgen_guide.process.endpoint import Endpoint, StepContext
from roomgen_guide.settings import RoomgenSettings
from strip_and_furnish.backends import build_adherence, build_backends
from strip_and_furnish.config import TORCH_DTYPES, FmrgReduxConfig, SceneConfig
from strip_and_furnish.fmrg import (
    _FAKE_PACKAGES,
    CONTROL_ARM,
    FurnishStacks,
    _effective_first_sigma,
    _format_strip,
    _git_commit,
    _photo_path,
    _sample_walk,
    _vae_roundtrip,
    _walked_steps,
    _warm_up_walk,
    build_flux_stack,
    furnish_reference,
)
from strip_and_furnish.images import (
    REWARD_TINT,
    SEGMENTATION_COLOR_KEY,
    DisparityScale,
    mean_absolute_residual,
    write_disparity_map,
    write_image,
    write_mask,
    write_mask_overlay,
    write_masked_disparity_map,
    write_segmentation_map,
    write_signed_residual_map,
)
from strip_and_furnish.photos import ensure_photos, load_photo
from strip_and_furnish.preference import build_preference
from strip_and_furnish.strip import StripResult, strip_scene
from strip_and_furnish.study import GALLERY_A, GALLERY_B, PHOTO_SUBDIRECTORY


@dataclass(frozen=True)
class ReduxArmSpec:
    """One furnish walk, named by which of the three channels it turns on."""

    arm: str
    redux: bool
    depth: bool
    image: bool

    @property
    def guided(self) -> bool:
        """Whether the walk descends any reward; Redux is conditioning, not steering."""
        return self.depth or self.image


# The owner's grid: a control for the deltas, then every named combination.
REDUX_ARMS: Final[tuple[ReduxArmSpec, ...]] = (
    ReduxArmSpec(CONTROL_ARM, redux=False, depth=False, image=False),
    ReduxArmSpec("redux", redux=True, depth=False, image=False),
    ReduxArmSpec("redux_depth", redux=True, depth=True, image=False),
    ReduxArmSpec("depth", redux=False, depth=True, image=False),
    ReduxArmSpec("img", redux=False, depth=False, image=True),
    ReduxArmSpec("redux_img", redux=True, depth=False, image=True),
    ReduxArmSpec("redux_img_depth", redux=True, depth=True, image=True),
)


def instruction_prompt(scene: SceneConfig) -> str:
    """The owner's phrasing around the scene's style descriptors, verbatim."""
    return (
        f"renovate this room into {scene.prompt}, keep the windows, doors, "
        f"walls, floor and ceiling's positions"
    )


def redux_context_name(scene: SceneConfig) -> str:
    """The prompt-cache key of `scene`'s instruction prompt with Redux tokens behind it."""
    return f"redux::{scene.photo}"


@dataclass(frozen=True)
class ReduxWeightSizing:
    """The two terms' pull at one real midpoint sample, and the weight read off it.

    `depth_weight` is the image term's gradient RMS over the depth term's, so
    the composite's two terms pull equally hard in aggregate at that step —
    the image term rides at 1.0 and the depth term at this weight.
    """

    step_index: int
    total_steps: int
    depth_gradient_rms: float
    image_gradient_rms: float
    depth_weight: float

    def as_row(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "depth_gradient_rms": self.depth_gradient_rms,
            "image_gradient_rms": self.image_gradient_rms,
            "depth_weight": self.depth_weight,
        }


@dataclass(frozen=True)
class ReduxArmResult:
    """One arm's render and every reading taken off it.

    The deferred judges (`adherence`, the two similarities, `preference`) are
    NaN/None until sampling is over and an eval tower can afford to be
    resident; `depth_weight` is set only on the composite arm.
    """

    arm: str
    redux: bool
    depth_reward: bool
    image_reward: bool
    eta: float
    steering_context: str
    depth_weight: float | None
    depth_error: float
    shell_coverage: float
    adherence: float
    similarity_empty_room: float | None
    similarity_photo: float | None
    preference: float
    render: str
    segmentation_map: str
    depth_map: str
    residual_map: str
    scorable_overlay: str
    seconds: float
    peak_vram_bytes: int | None

    def as_row(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "redux": self.redux,
            "depth_reward": self.depth_reward,
            "image_reward": self.image_reward,
            "eta": self.eta,
            "steering_context": self.steering_context,
            "depth_weight": self.depth_weight,
            "depth_error": self.depth_error,
            "shell_coverage": self.shell_coverage,
            "adherence": self.adherence,
            "similarity_empty_room": self.similarity_empty_room,
            "similarity_photo": self.similarity_photo,
            "preference": self.preference,
            "render": self.render,
            "segmentation_map": self.segmentation_map,
            "depth_map": self.depth_map,
            "residual_map": self.residual_map,
            "scorable_overlay": self.scorable_overlay,
            "seconds": self.seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


@dataclass(frozen=True)
class ReduxSceneResult:
    """Every arm of one (scene, seed), scored on one common pixel set vs the empty room."""

    scene: str
    seed: int
    style_prompt: str
    instruction_prompt: str
    redux_context: str | None
    redux_scale: float
    effective_first_sigma: float
    arms: tuple[ReduxArmResult, ...]
    sizing: ReduxWeightSizing | None
    common_coverage: float
    common_frame_fraction: float
    empty_shell_frame_fraction: float
    photo_shell_frame_fraction: float
    reconstruction_depth_error: float
    mean_abs_control_residual: float
    empty_depth_map: str
    empty_shell_mask: str
    empty_shell_overlay: str
    empty_segmentation_map: str
    common_mask_overlay: str

    def arm(self, arm_id: str) -> ReduxArmResult:
        for measurement in self.arms:
            if measurement.arm == arm_id:
                return measurement
        raise KeyError(arm_id)

    def depth_delta_vs_control(self, arm_id: str) -> float | None:
        baseline = self.arm(CONTROL_ARM).depth_error
        if baseline == 0.0:
            return None
        return self.arm(arm_id).depth_error / baseline - 1.0

    def adherence_ratio_vs_control(self, arm_id: str) -> float | None:
        baseline = self.arm(CONTROL_ARM).adherence
        if baseline == 0.0:
            return None
        return self.arm(arm_id).adherence / baseline

    def as_row(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "seed": self.seed,
            "style_prompt": self.style_prompt,
            "instruction_prompt": self.instruction_prompt,
            "redux_context": self.redux_context,
            "redux_scale": self.redux_scale,
            "effective_first_sigma": self.effective_first_sigma,
            "sizing": None if self.sizing is None else self.sizing.as_row(),
            "common_coverage": self.common_coverage,
            "common_frame_fraction": self.common_frame_fraction,
            "empty_shell_frame_fraction": self.empty_shell_frame_fraction,
            "photo_shell_frame_fraction": self.photo_shell_frame_fraction,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "mean_abs_control_residual": self.mean_abs_control_residual,
            "empty_depth_map": self.empty_depth_map,
            "empty_shell_mask": self.empty_shell_mask,
            "empty_shell_overlay": self.empty_shell_overlay,
            "empty_segmentation_map": self.empty_segmentation_map,
            "common_mask_overlay": self.common_mask_overlay,
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
class ReduxSummary:
    """What the run measured, where its artifacts landed, and the code behind them."""

    config: FmrgReduxConfig
    strips: tuple[StripResult, ...]
    results: tuple[ReduxSceneResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


@dataclass(frozen=True)
class _ImageEnergyBank:
    """The one DINOv2 tower, parked on the CPU except while an arm steers by it."""

    factory: MaskedPatchReferenceEnergies
    device: torch.device

    def energy(
        self,
        photo: Float[Tensor, "1 3 height width"],
        *,
        mask: Bool[Tensor, "1 height width"],
        config: FmrgReduxConfig,
        stacks: FurnishStacks,
    ) -> Guidance:
        self.factory.module.to(self.device)
        resegment = config.furnish.resegment
        return self.factory(
            photo,
            mask=mask,
            segmenter=stacks.segmenter if resegment else None,
            erosion_radius=(config.furnish.erosion_radius if resegment else None),
            min_coverage=config.image.min_patch_coverage,
        )

    def park(self) -> None:
        self.factory.module.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def build_image_energy_bank(
    config: FmrgReduxConfig, *, device: torch.device, settings: RoomgenSettings
) -> _ImageEnergyBank:
    """The shell-masked patch-cosine factory, real or weightless by the backend kind."""
    if config.backends.kind == "fake":
        for package in _FAKE_PACKAGES:
            if str(package) not in sys.path:
                sys.path.insert(0, str(package))
        from roomgen_guide_fakes import fake_masked_patch_energies  # noqa: PLC0415

        factory: MaskedPatchReferenceEnergies = fake_masked_patch_energies()
        return _ImageEnergyBank(factory=factory, device=device)
    factory = load_dinov2_masked_patch_energies(
        config.image.dinov2_checkpoint,
        device=device,
        dtype=TORCH_DTYPES[config.backends.depth_dtype],
        cache_dir=settings.hub_cache,
    )
    factory.module.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _ImageEnergyBank(factory=factory, device=device)


def run_redux_study(config: FmrgReduxConfig, settings: RoomgenSettings) -> ReduxSummary:
    """Strip every scene, then walk the seven-arm grid and score it with deferred judges."""
    started_at = datetime.now(UTC).isoformat()
    output_dir = settings.wdir / config.name
    gallery_a = output_dir / GALLERY_A
    gallery_b = output_dir / GALLERY_B
    for directory in (gallery_a, gallery_b):
        directory.mkdir(parents=True, exist_ok=True)

    photo_paths = ensure_photos(settings.wdir / PHOTO_SUBDIRECTORY)
    backends = build_backends(config.backends, cache_dir=settings.hub_cache)
    if backends.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backends.device)

    print(f"\n[stage 1] strip: {len(config.scenes)} scenes, unchanged protocol")
    inpainter = backends.build_inpainter()
    strips = tuple(
        strip_scene(
            scene,
            load_photo(
                _photo_path(scene, photo_paths),
                resolution=config.resolution,
                device=backends.device,
                dtype=backends.dtype,
            ),
            strip=config.strip,
            erosion_radius=config.furnish.erosion_radius,
            backends=backends,
            inpainter=inpainter,
            seed=config.seeds[0],
            gallery=gallery_a,
        )
        for scene in config.scenes
    )
    for strip in strips:
        print(f"  {_format_strip(strip)}")

    references = {
        strip.scene: furnish_reference(
            strip,
            depth_estimator=backends.depth_estimator,
            segmenter=backends.segmenter,
            erosion_radius=config.furnish.erosion_radius,
        )
        for strip in strips
    }
    shell_fractions = {
        strip.scene: {
            "photo": float(strip.reference.mask.to(torch.float32).mean()),
            "empty_room": float(references[strip.scene].mask.to(torch.float32).mean()),
        }
        for strip in strips
    }

    depth_estimator = backends.depth_estimator
    segmenter = backends.segmenter
    device = backends.device
    # Stage 1's stack is dropped before anything FLUX-sized loads.
    del inpainter, backends
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # The Redux encoders are a ~1 GiB transient; they load, encode each
    # stripped room, and are freed per call — all strictly before the FLUX
    # transformer claims the card.
    redux_tokens: dict[str, Float[Tensor, "1 tokens embedding"]] = {}
    if config.backends.kind == "real":
        for strip in strips:
            redux_tokens[strip.scene] = encode_redux_image_tokens(
                strip.empty_room.to(torch.float32),
                device=device,
                dtype=TORCH_DTYPES[config.flux.dtype],
                cache_dir=settings.hub_cache,
                checkpoint=config.redux.checkpoint,
            )
            print(
                f"[redux] {strip.scene}: empty room encoded to "
                f"{redux_tokens[strip.scene].shape[1]} context tokens "
                f"(scale {config.redux.scale:g})"
            )

    bank = build_image_energy_bank(config, device=device, settings=settings)
    prompts = list(dict.fromkeys(instruction_prompt(scene) for scene in config.scenes))
    stacks = build_flux_stack(
        config,
        prompts=prompts,
        depth_estimator=depth_estimator,
        segmenter=segmenter,
        device=device,
        settings=settings,
    )
    if stacks.sampler is not None:
        for scene in config.scenes:
            register_image_context_prompt(
                stacks.sampler,
                name=redux_context_name(scene),
                prompt=instruction_prompt(scene),
                image_tokens=redux_tokens[scene.photo],
                scale=config.redux.scale,
            )

    print(
        f"\n[stage 2] furnish grid on FMRG: {len(config.scenes)} scenes x "
        f"{len(config.seeds)} seeds x {len(REDUX_ARMS)} arms; reference = the "
        f"stripped empty room (depth on its shell depth, image reward on its "
        f"shell patches, Redux context on its pixels)"
    )
    pending: list[_PendingScene] = []
    for index, (scene, strip) in enumerate(zip(config.scenes, strips, strict=True)):
        pending.extend(
            _run_scene(
                scene,
                strip,
                config=config,
                stacks=stacks,
                bank=bank,
                reference=references[scene.photo],
                shell_fractions=shell_fractions[scene.photo],
                gallery=gallery_b,
                warm_up=index == 0,
            )
        )

    results = _score_pending(pending, config=config, device=device, settings=settings)
    for result in results:
        print(f"  {_format_result(result)}")

    on_cuda = device.type == "cuda"
    summary = ReduxSummary(
        config=config,
        strips=strips,
        results=tuple(results),
        output_dir=output_dir,
        peak_vram_bytes=(torch.cuda.max_memory_allocated(device) if on_cuda else None),
        started_at=started_at,
        git_commit=_git_commit(),
        walked_steps=_walked_steps(config),
        device_name=(torch.cuda.get_device_name(device) if on_cuda else None),
    )
    _write_results(summary, settings)
    return summary


@dataclass(frozen=True)
class _PendingScene:
    """One (scene, seed) measured on everything but the deferred judges."""

    result: ReduxSceneResult
    renders: dict[str, Float[Tensor, "1 3 height width"]]
    empty_room: Float[Tensor, "1 3 height width"]
    photo: Float[Tensor, "1 3 height width"]
    reference: GeometricReference


def _score_pending(
    pending: list[_PendingScene],
    *,
    config: FmrgReduxConfig,
    device: torch.device,
    settings: RoomgenSettings,
) -> list[ReduxSceneResult]:
    """The deferred judges, one resident at a time, after every walk is over."""
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
            _with_similarity(entry, similarity=similarity, device=device)
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
            # Always the scene's neutral style text — never the instruction the
            # walk steered under, and never the Redux-extended context.
            adherence=adherence(
                entry.renders[cell.arm].to(device),
                prompt=result.style_prompt,
                photo=entry.empty_room,
                reference=entry.reference,
            ),
        )
        for cell in result.arms
    )
    return replace(entry, result=replace(result, arms=arms))


def _with_similarity(
    entry: _PendingScene, *, similarity: ClipImageSimilarity, device: torch.device
) -> _PendingScene:
    result = entry.result
    arms = tuple(
        replace(
            cell,
            similarity_empty_room=similarity(
                entry.renders[cell.arm].to(device),
                photo=entry.empty_room.to(device),
            ),
            similarity_photo=similarity(
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
                entry.renders[cell.arm].to(device), [result.style_prompt]
            )
        arms.append(replace(cell, preference=float(score.mean())))
    return replace(entry, result=replace(result, arms=tuple(arms)))


def _release(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_scene(
    scene: SceneConfig,
    strip: StripResult,
    *,
    config: FmrgReduxConfig,
    stacks: FurnishStacks,
    bank: _ImageEnergyBank,
    reference: GeometricReference,
    shell_fractions: dict[str, float],
    gallery: Path,
    warm_up: bool,
) -> list[_PendingScene]:
    empty_room = strip.empty_room.to(torch.float32)
    instruction = instruction_prompt(scene)
    context_available = stacks.sampler is not None
    scale = DisparityScale.of(reference.disparity)

    empty_depth_map = f"{scene.photo}_empty_depth.png"
    empty_shell_mask = f"{scene.photo}_empty_shell_mask.png"
    empty_shell_overlay = f"{scene.photo}_empty_shell_overlay.png"
    empty_segmentation_map = f"{scene.photo}_empty_segmentation.png"
    write_disparity_map(
        reference.disparity,
        scale=scale,
        mask=reference.mask,
        path=gallery / empty_depth_map,
    )
    write_mask(reference.mask, gallery / empty_shell_mask)
    write_mask_overlay(
        empty_room, reference.mask, gallery / empty_shell_overlay, tint=REWARD_TINT
    )
    with torch.no_grad():
        write_segmentation_map(
            stacks.segmenter(empty_room), gallery / empty_segmentation_map
        )

    reconstruction = depth_error(
        _vae_roundtrip(stacks.flow_backend, empty_room, device=stacks.device),
        reference=reference,
        depth_estimator=stacks.depth_estimator,
    )
    effective_sigma = _effective_first_sigma(
        empty_room, start_sigma=config.furnish.start_sigma, config=config, stacks=stacks
    )
    print(
        f"[scene] {scene.photo}: empty shell covers "
        f"{shell_fractions['empty_room']:.1%} of the frame (photo shell "
        f"{shell_fractions['photo']:.1%}), reconstruction floor "
        f"{reconstruction:.6f}, effective first sigma "
        f"{config.furnish.start_sigma:g}->{effective_sigma:.4f}"
    )

    if warm_up:
        _warm_up_walk(empty_room, config=config, stacks=stacks, reference=reference)

    # The composite's weight is measured on the trajectory that arm will walk:
    # the Redux-extended context when the backend carries one.
    if context_available:
        assert stacks.sampler is not None
        select_cached_prompt(stacks.sampler, redux_context_name(scene))
    sizing = (
        _measure_sizing(
            empty_room,
            config=config,
            stacks=stacks,
            bank=bank,
            reference=reference,
            seed=config.seeds[0],
        )
        if config.image.depth_weight is None
        else None
    )
    if sizing is not None:
        print(
            f"[sizing] step {sizing.step_index}/{sizing.total_steps}: depth rms "
            f"{sizing.depth_gradient_rms:.3e}, image rms "
            f"{sizing.image_gradient_rms:.3e} -> depth weight "
            f"{sizing.depth_weight:.3g}"
        )
        _release(stacks.device)
    composite_weight = (
        config.image.depth_weight
        if config.image.depth_weight is not None
        else sizing.depth_weight
        if sizing is not None
        else None
    )

    results: list[_PendingScene] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        peaks: dict[str, int | None] = {}
        for spec in REDUX_ARMS:
            if context_available:
                assert stacks.sampler is not None
                select_cached_prompt(
                    stacks.sampler,
                    redux_context_name(scene) if spec.redux else instruction,
                )
            reward = _arm_reward(
                spec,
                empty_room=empty_room,
                config=config,
                stacks=stacks,
                bank=bank,
                reference=reference,
                composite_weight=composite_weight,
            )
            image, elapsed, peak = _sample_walk(
                empty_room,
                config=config,
                stacks=stacks,
                reference=reference,
                eta=config.furnish.eta if spec.guided else 0.0,
                start_sigma=config.furnish.start_sigma,
                seed=seed,
                reward=reward,
            )
            # The arm's energy — and any reference features it froze on the
            # device — is dead the moment its walk finishes; the allocator is
            # swept at every arm boundary, not only where a tower parks.
            del reward
            if spec.image:
                bank.park()
            else:
                _release(stacks.device)
            renders[spec.arm] = image
            seconds[spec.arm] = elapsed
            peaks[spec.arm] = peak
            write_image(image, gallery / f"{scene.photo}_seed{seed}_{spec.arm}.png")

        # One segmentation per finished render serves both readings taken off
        # it: the arm's evaluation mask and its class-coloured map.
        with torch.no_grad():
            segmentations: dict[str, Int[Tensor, "1 height width"]] = {
                spec.arm: stacks.segmenter(renders[spec.arm]) for spec in REDUX_ARMS
            }
        arm_masks = {
            spec.arm: (
                reference.mask
                & scorable_mask(
                    segmentations[spec.arm],
                    erosion_radius=config.furnish.erosion_radius,
                )
                if config.furnish.resegment
                else reference.mask
            )
            for spec in REDUX_ARMS
        }
        common = reference.mask.clone()
        for mask in arm_masks.values():
            common &= mask
        common_mask_overlay = f"{scene.photo}_seed{seed}_common_mask.png"
        write_mask_overlay(
            empty_room, common, gallery / common_mask_overlay, tint=REWARD_TINT
        )

        residuals = {
            spec.arm: signed_depth_residual_map(
                renders[spec.arm],
                reference=reference,
                depth_estimator=stacks.depth_estimator,
                mask=common,
            )
            for spec in REDUX_ARMS
        }
        control_scale = mean_absolute_residual(residuals[CONTROL_ARM], common)

        measurements: list[ReduxArmResult] = []
        for spec in REDUX_ARMS:
            render = renders[spec.arm]
            segmentation_map = f"{scene.photo}_seed{seed}_{spec.arm}_segmentation.png"
            depth_map = f"{scene.photo}_seed{seed}_{spec.arm}_depth.png"
            residual_map = f"{scene.photo}_seed{seed}_{spec.arm}_residual.png"
            scorable_overlay = f"{scene.photo}_seed{seed}_{spec.arm}_scorable.png"
            write_segmentation_map(segmentations[spec.arm], gallery / segmentation_map)
            with torch.no_grad():
                write_masked_disparity_map(
                    stacks.depth_estimator(render),
                    scale=scale,
                    mask=common,
                    path=gallery / depth_map,
                )
            write_signed_residual_map(
                residuals[spec.arm],
                mask=common,
                normalizer=control_scale,
                path=gallery / residual_map,
            )
            write_mask_overlay(
                render,
                arm_masks[spec.arm],
                gallery / scorable_overlay,
                tint=REWARD_TINT,
            )
            measurements.append(
                ReduxArmResult(
                    arm=spec.arm,
                    redux=spec.redux,
                    depth_reward=spec.depth,
                    image_reward=spec.image,
                    eta=config.furnish.eta if spec.guided else 0.0,
                    steering_context=(
                        "instruction+redux" if spec.redux else "instruction"
                    ),
                    depth_weight=(
                        composite_weight if spec.depth and spec.image else None
                    ),
                    depth_error=depth_error(
                        render,
                        reference=reference,
                        depth_estimator=stacks.depth_estimator,
                        mask=common,
                    ),
                    shell_coverage=shell_coverage(
                        arm_masks[spec.arm], reference=reference
                    ),
                    # Filled in by the deferred judges.
                    adherence=math.nan,
                    similarity_empty_room=None,
                    similarity_photo=None,
                    preference=math.nan,
                    render=f"{scene.photo}_seed{seed}_{spec.arm}.png",
                    segmentation_map=segmentation_map,
                    depth_map=depth_map,
                    residual_map=residual_map,
                    scorable_overlay=scorable_overlay,
                    seconds=seconds[spec.arm],
                    peak_vram_bytes=peaks[spec.arm],
                )
            )

        result = ReduxSceneResult(
            scene=scene.photo,
            seed=seed,
            style_prompt=scene.prompt,
            instruction_prompt=instruction,
            redux_context=(redux_context_name(scene) if context_available else None),
            redux_scale=config.redux.scale,
            effective_first_sigma=effective_sigma,
            arms=tuple(measurements),
            sizing=sizing,
            common_coverage=shell_coverage(common, reference=reference),
            common_frame_fraction=float(common.to(torch.float32).mean()),
            empty_shell_frame_fraction=shell_fractions["empty_room"],
            photo_shell_frame_fraction=shell_fractions["photo"],
            reconstruction_depth_error=reconstruction,
            mean_abs_control_residual=control_scale,
            empty_depth_map=empty_depth_map,
            empty_shell_mask=empty_shell_mask,
            empty_shell_overlay=empty_shell_overlay,
            empty_segmentation_map=empty_segmentation_map,
            common_mask_overlay=common_mask_overlay,
        )
        print(f"  [sampled] {result.scene} seed {result.seed}, judges pending")
        results.append(
            _PendingScene(
                result=result,
                renders={
                    arm: image.detach().to(device="cpu", dtype=torch.float32)
                    for arm, image in renders.items()
                },
                empty_room=empty_room,
                photo=strip.photo.to(torch.float32),
                reference=reference,
            )
        )
    return results


def _arm_reward(
    spec: ReduxArmSpec,
    *,
    empty_room: Float[Tensor, "1 3 height width"],
    config: FmrgReduxConfig,
    stacks: FurnishStacks,
    bank: _ImageEnergyBank,
    reference: GeometricReference,
    composite_weight: float | None,
) -> Guidance | None:
    """The energy one arm descends; `None` is `_sample_walk`'s own depth default.

    The unguided arms never evaluate it (`eta` 0), so handing them the default
    costs nothing and keeps the walk construction identical across arms.
    """
    if not spec.image:
        return None
    image_term = bank.energy(
        empty_room, mask=reference.mask, config=config, stacks=stacks
    )
    if not spec.depth:
        return image_term
    if composite_weight is None:
        raise RuntimeError(
            "the composite arm needs a depth weight: the sizing pass runs "
            "whenever image.depth_weight is null, so a missing weight is a bug"
        )
    depth_term = GeometricGuidance(
        depth_estimator=stacks.depth_estimator,
        reference=reference,
        segmenter=stacks.segmenter if config.furnish.resegment else None,
        erosion_radius=(
            config.furnish.erosion_radius if config.furnish.resegment else None
        ),
    )
    return WeightedGuidance(
        (WeightedTerm(image_term, 1.0), WeightedTerm(depth_term, composite_weight))
    )


def _measure_sizing(
    empty_room: Float[Tensor, "1 3 height width"],
    *,
    config: FmrgReduxConfig,
    stacks: FurnishStacks,
    bank: _ImageEnergyBank,
    reference: GeometricReference,
    seed: int,
) -> ReduxWeightSizing:
    """Each term's gradient RMS at one real, partially denoised sample.

    Guessing a weight between a patch cosine and a masked squared disparity is
    guessing a unit conversion; this measures it instead, on the walk the
    composite arm runs, a `sizing_fraction` of the way into the guided window.
    One endpoint decode serves both terms' backwards.
    """
    furnish = config.furnish
    generator = torch.Generator(device=stacks.device).manual_seed(seed)
    window = GuidanceWindow(
        early_stop=furnish.early_stop, unguided_steps=furnish.unguided_steps
    )
    process = FlowMapProcess(
        stacks.flow_backend,
        empty_room,
        steps=furnish.grid_steps,
        start_sigma=furnish.start_sigma,
        generator=generator,
        window=window,
        stepping=furnish.stepping,
        differentiate_through_velocity=furnish.differentiate_through_velocity,
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
        segmenter=stacks.segmenter if furnish.resegment else None,
        erosion_radius=(furnish.erosion_radius if furnish.resegment else None),
    )
    with torch.enable_grad():
        endpoint = process.endpoint(state, _context(process, index))
        depth_rms = _gradient_rms(endpoint, depth_term)
        image_term = bank.energy(
            empty_room, mask=reference.mask, config=config, stacks=stacks
        )
        image_rms = _gradient_rms(endpoint, image_term)
        del image_term
        bank.park()
    if depth_rms <= 0.0:
        raise RuntimeError(
            "the depth term measured a zero gradient at the sizing step, so "
            "there is no weight to read off the ratio — the gradient path is "
            "detached somewhere"
        )
    return ReduxWeightSizing(
        step_index=index,
        total_steps=process.steps,
        depth_gradient_rms=depth_rms,
        image_gradient_rms=image_rms,
        depth_weight=image_rms / depth_rms,
    )


def _context(process: FlowMapProcess, index: int) -> StepContext:
    return StepContext(index=index, total=process.steps, span=process.step_span(index))


def _gradient_rms(endpoint: Endpoint[Tensor], energy: Guidance) -> float:
    (gradient,) = torch.autograd.grad(
        energy(endpoint.image).mean(), endpoint.state, retain_graph=True
    )
    return float(gradient.to(torch.float32).square().mean().sqrt())


def _format_result(result: ReduxSceneResult) -> str:
    cells = ", ".join(
        f"{cell.arm} E={cell.depth_error:.4f} adh={cell.adherence:.4f}"
        for cell in result.arms
    )
    return (
        f"{result.scene} seed {result.seed}: {cells}, common mask "
        f"{result.common_frame_fraction:.0%} of the frame "
        f"({result.common_coverage:.0%} of the empty shell)"
    )


def arm_means(
    results: tuple[ReduxSceneResult, ...],
) -> dict[str, dict[str, float | None]]:
    """Each arm's mean scores over every (scene, seed)."""
    means: dict[str, dict[str, float | None]] = {}
    for spec in REDUX_ARMS:
        cells = [result.arm(spec.arm) for result in results]
        empties = [
            cell.similarity_empty_room
            for cell in cells
            if cell.similarity_empty_room is not None
        ]
        photos = [
            cell.similarity_photo for cell in cells if cell.similarity_photo is not None
        ]
        means[spec.arm] = {
            "depth_error": _mean([cell.depth_error for cell in cells]),
            "shell_coverage": _mean([cell.shell_coverage for cell in cells]),
            "adherence": _mean([cell.adherence for cell in cells]),
            "similarity_empty_room": _mean(empties) if empties else None,
            "similarity_photo": _mean(photos) if photos else None,
            "preference": _mean([cell.preference for cell in cells]),
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
                "torch_version": torch.__version__,
                "device_name": summary.device_name,
                "walked_steps": summary.walked_steps,
                "peak_vram_bytes": summary.peak_vram_bytes,
                "eta": summary.config.furnish.eta,
                "redux_scale": summary.config.redux.scale,
                "guidance_reference": "empty_room",
                "shell_classes": [
                    ADE20K_LABELS[index] for index in sorted(ARCHITECTURAL_CLASSES)
                ],
                "removable_classes": [
                    ADE20K_LABELS[index] for index in sorted(REMOVABLE_CLASSES)
                ],
                "galleries": {
                    "strip": str(summary.output_dir / GALLERY_A),
                    "furnish": str(summary.output_dir / GALLERY_B),
                },
                "segmentation_color_key": {
                    label: list(color)
                    for label, color in SEGMENTATION_COLOR_KEY.items()
                },
                "arm_means": arm_means(summary.results),
                "strip": [strip.as_row() for strip in summary.strips],
                "furnish": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
