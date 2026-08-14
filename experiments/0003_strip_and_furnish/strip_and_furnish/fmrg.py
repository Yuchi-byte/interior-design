"""Stage 2 on FMRG: the stripped room refurnished by the FLUX flow map, held by its own depth.

Stage 1 is the unchanged strip — EoMT decides, SD-inpainting fills, the
inverted shell acceptance re-checks — and stage 2 swaps SD1.5's guided img2img
for the FMRG walk: the empty room renoised to `start_sigma` and steered by the
masked depth reward alone, FLUX's native text conditioning carrying the prompt.

The guidance reference is the *stripped empty room's* shell depth, not the
photograph's retained one: the empty room's shell is nearly the whole frame —
windows and doors included, since they joined the shell — so the valid
comparison area is much larger than photo-shell ∩ prediction-shell. The
per-step resegmentation keeps the prediction side honest: the effective mask is
empty-room shell ∩ prediction shell (`GeometricGuidance._scored_pixels`), so
furniture the furnishing adds is excused rather than scored against empty-room
depth, which would punish furnishing itself. Eval mirrors that intersection.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.segmentation.base import Segmenter
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
from roomgen_guide.eval import (
    PromptAdherence,
    depth_error,
    shell_coverage,
    signed_depth_residual_map,
)
from roomgen_guide.guidance.geometric import GeometricGuidance
from roomgen_guide.process.driver import sample_endpoint_process
from roomgen_guide.process.tilt import AnnealedSignTilt
from roomgen_guide.settings import RoomgenSettings
from strip_and_furnish.backends import build_adherence, build_backends
from strip_and_furnish.config import TORCH_DTYPES, FmrgStudyConfig, SceneConfig
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
    write_signed_residual_map,
)
from strip_and_furnish.photos import ensure_photos, load_photo
from strip_and_furnish.strip import StripResult, strip_scene
from strip_and_furnish.study import GALLERY_A, GALLERY_B, PHOTO_SUBDIRECTORY

# The unguided arm every guided arm's depth delta and adherence ratio divide by,
# run at the primary start sigma.
CONTROL_ARM: Final = "control"

# Where the repo's download instructions put the 512-res FlowMap LoRA, relative
# to `RoomgenSettings.hf_home`.
_DEFAULT_LORA_SUBPATH: Final = Path("fmrg") / "flux-flowmap-lora-512"
_LORA_WEIGHT_NAME: Final = "pytorch_lora_weights.safetensors"

# The diffusers-format FLUX load reads none of these, and the two root files
# alone are 24GB: the offline cache check must not fail over files that were
# deliberately never fetched.
_FLUX_IGNORE: Final = (
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


@dataclass(frozen=True)
class FmrgArmSpec:
    """One furnish walk: a guidance strength and an init noise level, all else held."""

    arm: str
    eta: float
    start_sigma: float
    probe: bool


def furnish_arms(config: FmrgStudyConfig) -> tuple[FmrgArmSpec, ...]:
    """Control first, then the guided arm, then one clearly-labelled probe per sigma."""
    furnish = config.furnish
    return (
        FmrgArmSpec(CONTROL_ARM, 0.0, furnish.start_sigma, probe=False),
        FmrgArmSpec("guided", furnish.eta, furnish.start_sigma, probe=False),
        *(
            FmrgArmSpec(f"guided_s{sigma:g}", furnish.eta, sigma, probe=True)
            for sigma in furnish.probe_sigmas
        ),
    )


def furnish_reference(
    strip: StripResult,
    *,
    depth_estimator: DepthEstimator,
    segmenter: Segmenter,
    erosion_radius: int,
) -> GeometricReference:
    """The geometry the furnishing must keep: the *stripped empty room's*, by owner decision.

    Deliberately the opposite of the retained-photo protocol: the empty room's
    shell is nearly the whole frame — the furniture is gone and, since windows
    and doors joined the shell, the glazing counts too — so the valid comparison
    area is far larger than photo-shell ∩ prediction-shell. Any residual
    contents the strip left are still excluded, because the mask is read off a
    fresh segmentation of the empty room rather than assumed full-frame. The
    trade is that guidance no longer knows the original furniture's occlusions;
    what it inherits instead is whatever geometry the inpainting got wrong,
    which `reconstruction_depth_error` and the strip numbers bound.
    """
    return GeometricReference.from_photo(
        strip.empty_room.to(torch.float32),
        depth_estimator=depth_estimator,
        segmenter=segmenter,
        erosion_radius=erosion_radius,
    )


@dataclass(frozen=True)
class FurnishStacks:
    """Everything the FMRG furnish stage samples and scores with."""

    flow_backend: FlowMapBackend
    sampler: TwoTimeFlowMapSampler | None
    depth_estimator: DepthEstimator
    segmenter: Segmenter
    device: torch.device


def build_flux_stack(
    config: FmrgStudyConfig,
    *,
    prompts: list[str],
    depth_estimator: DepthEstimator,
    segmenter: Segmenter,
    device: torch.device,
    settings: RoomgenSettings,
) -> FurnishStacks:
    """The FLUX flow-map stack, every prompt encoded up front, judges carried over.

    The sampler's text encoders are freed the moment it finishes loading, so
    every prompt any walk will condition on must be named here; scenes switch
    between them with `select_cached_prompt`. The depth estimator and segmenter
    are stage 1's, handed through rather than reloaded.
    """
    match config.backends.kind:
        case "real":
            sampler = load_flow_map_sampler(
                fmrg_path=config.flux.fmrg_path or _REPO_ROOT / "packages" / "fmrg",
                lora_path=_lora_dir(config, settings),
                prompt=prompts[0],
                additional_prompts=prompts[1:],
                device=device,
                model_id=_flux_path(config, settings),
                dtype=TORCH_DTYPES[config.flux.dtype],
            )
            return FurnishStacks(
                flow_backend=FluxFlowMapBackend(
                    sampler=sampler,
                    text_guidance_scale=config.flux.text_guidance_scale,
                ),
                sampler=sampler,
                depth_estimator=depth_estimator,
                segmenter=segmenter,
                device=device,
            )
        case "fake":
            for package in _FAKE_PACKAGES:
                if str(package) not in sys.path:
                    sys.path.insert(0, str(package))
            from roomgen_guide_fakes import ToyFlowMapBackend  # noqa: PLC0415

            return FurnishStacks(
                flow_backend=ToyFlowMapBackend(dtype=TORCH_DTYPES[config.flux.dtype]),
                sampler=None,
                depth_estimator=depth_estimator,
                segmenter=segmenter,
                device=device,
            )


@dataclass(frozen=True)
class FmrgArmResult:
    """One arm's render, its depth reading against the empty room, and its cost."""

    arm: str
    eta: float
    start_sigma: float
    effective_first_sigma: float
    probe: bool
    depth_error: float
    shell_coverage: float
    adherence: float
    render: str
    depth_map: str
    residual_map: str
    scorable_overlay: str
    seconds: float
    peak_vram_bytes: int | None

    def as_row(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "eta": self.eta,
            "start_sigma": self.start_sigma,
            "effective_first_sigma": self.effective_first_sigma,
            "probe": self.probe,
            "depth_error": self.depth_error,
            "shell_coverage": self.shell_coverage,
            "adherence": self.adherence,
            "render": self.render,
            "depth_map": self.depth_map,
            "residual_map": self.residual_map,
            "scorable_overlay": self.scorable_overlay,
            "seconds": self.seconds,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


@dataclass(frozen=True)
class FmrgSceneResult:
    """Every arm of one (scene, seed), scored on one common pixel set vs the empty room.

    Args:
        common_coverage: share of the empty room's shell the common mask still
            scores.
        common_frame_fraction: the same mask as a share of the whole frame — the
            enlarged valid area the empty-room reference was chosen for, read
            beside `photo_shell_frame_fraction`, what the old protocol's
            reference-side mask held before any intersection.
    """

    scene: str
    prompt: str
    seed: int
    arms: tuple[FmrgArmResult, ...]
    common_coverage: float
    common_frame_fraction: float
    empty_shell_frame_fraction: float
    photo_shell_frame_fraction: float
    reconstruction_depth_error: float
    mean_abs_control_residual: float
    empty_depth_map: str
    empty_shell_mask: str
    empty_shell_overlay: str
    common_mask_overlay: str

    def arm(self, arm_id: str) -> FmrgArmResult:
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
            "prompt": self.prompt,
            "seed": self.seed,
            "common_coverage": self.common_coverage,
            "common_frame_fraction": self.common_frame_fraction,
            "empty_shell_frame_fraction": self.empty_shell_frame_fraction,
            "photo_shell_frame_fraction": self.photo_shell_frame_fraction,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "mean_abs_control_residual": self.mean_abs_control_residual,
            "empty_depth_map": self.empty_depth_map,
            "empty_shell_mask": self.empty_shell_mask,
            "empty_shell_overlay": self.empty_shell_overlay,
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
class FmrgSummary:
    """What the run measured, where its artifacts landed, and the code behind them."""

    config: FmrgStudyConfig
    strips: tuple[StripResult, ...]
    results: tuple[FmrgSceneResult, ...]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


def run_fmrg_study(config: FmrgStudyConfig, settings: RoomgenSettings) -> FmrgSummary:
    """Strip every scene, then furnish it on FMRG against its own empty-room depth."""
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

    print(f"\n[stage 1] strip: {len(config.scenes)} scenes, no depth guidance")
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

    # The guidance reference is the stripped room's own shell depth, computed
    # while stage 1's judges are still resident; the photo's shell fraction is
    # kept beside it as the coverage the old protocol's reference-side mask had.
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
    for scene_id, fractions in shell_fractions.items():
        print(
            f"  {scene_id}: empty room's shell holds "
            f"{fractions['empty_room']:.0%} of the frame vs the photo shell's "
            f"{fractions['photo']:.0%}"
        )

    depth_estimator = backends.depth_estimator
    segmenter = backends.segmenter
    device = backends.device
    # Everything stage 2 does not use is dropped before FLUX loads: the
    # inpainter, the SD1.5 pipeline and the guidance CLIP together are what
    # turns a 24GB transformer load into an out-of-memory error.
    del inpainter, backends
    if device.type == "cuda":
        torch.cuda.empty_cache()

    prompts = list(dict.fromkeys(scene.prompt for scene in config.scenes))
    stacks = build_flux_stack(
        config,
        prompts=prompts,
        depth_estimator=depth_estimator,
        segmenter=segmenter,
        device=device,
        settings=settings,
    )

    specs = furnish_arms(config)
    print(
        f"\n[stage 2] furnish on FMRG: {len(config.scenes)} scenes x "
        f"{len(config.seeds)} seeds x {len(specs)} arms, reference = the "
        f"stripped empty room's shell depth (depth reward only; FLUX's text "
        f"conditioning is native, the SD1.5 CLIP text energy has no counterpart)"
    )
    pending: list[_PendingScene] = []
    for index, (scene, strip) in enumerate(zip(config.scenes, strips, strict=True)):
        pending.extend(
            _run_scene(
                scene,
                strip,
                config=config,
                stacks=stacks,
                reference=references[scene.photo],
                shell_fractions=shell_fractions[scene.photo],
                specs=specs,
                gallery=gallery_b,
                warm_up=index == 0,
            )
        )

    # Loaded only after the last walk: beside the transformer and the guided
    # decode's backward transient, a resident ViT-L/14 is exactly what tipped a
    # 32GB card into out-of-memory. Renders wait on the CPU.
    adherence = build_adherence(
        config.adherence, device=device, cache_dir=settings.hub_cache
    )
    results = [_scored(entry, adherence=adherence, device=device) for entry in pending]
    for result in results:
        print(f"  {_format_result(result)}")

    on_cuda = device.type == "cuda"
    summary = FmrgSummary(
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
    """One (scene, seed) measured on everything but adherence, renders parked on CPU."""

    result: FmrgSceneResult
    renders: dict[str, Float[Tensor, "1 3 height width"]]
    empty_room: Float[Tensor, "1 3 height width"]
    reference: GeometricReference


def _scored(
    entry: _PendingScene, *, adherence: PromptAdherence, device: torch.device
) -> FmrgSceneResult:
    result = entry.result
    arms = tuple(
        replace(
            cell,
            adherence=adherence(
                entry.renders[cell.arm].to(device),
                prompt=result.prompt,
                photo=entry.empty_room,
                reference=entry.reference,
            ),
        )
        for cell in result.arms
    )
    return replace(result, arms=arms)


def _run_scene(
    scene: SceneConfig,
    strip: StripResult,
    *,
    config: FmrgStudyConfig,
    stacks: FurnishStacks,
    reference: GeometricReference,
    shell_fractions: dict[str, float],
    specs: tuple[FmrgArmSpec, ...],
    gallery: Path,
    warm_up: bool,
) -> list[_PendingScene]:
    empty_room = strip.empty_room.to(torch.float32)
    if stacks.sampler is not None:
        select_cached_prompt(stacks.sampler, scene.prompt)

    scale = DisparityScale.of(reference.disparity)
    empty_depth_map = f"{scene.photo}_empty_depth.png"
    empty_shell_mask = f"{scene.photo}_empty_shell_mask.png"
    empty_shell_overlay = f"{scene.photo}_empty_shell_overlay.png"
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

    # The floor every arm's depth error is read against: what a perfect
    # geometry-preserving furnish still pays for FLUX's VAE round trip alone.
    reconstruction = depth_error(
        _vae_roundtrip(stacks.flow_backend, empty_room, device=stacks.device),
        reference=reference,
        depth_estimator=stacks.depth_estimator,
    )
    effective_sigmas = {
        sigma: _effective_first_sigma(
            empty_room, start_sigma=sigma, config=config, stacks=stacks
        )
        for sigma in dict.fromkeys(spec.start_sigma for spec in specs)
    }
    print(
        f"[scene] {scene.photo}: empty shell covers "
        f"{shell_fractions['empty_room']:.1%} of the frame (photo shell "
        f"{shell_fractions['photo']:.1%}), reconstruction floor "
        f"{reconstruction:.6f}, effective first sigma "
        + ", ".join(
            f"{sigma:g}->{effective:.4f}"
            for sigma, effective in effective_sigmas.items()
        )
    )

    if warm_up:
        _warm_up_walk(empty_room, config=config, stacks=stacks, reference=reference)

    results: list[_PendingScene] = []
    for seed in config.seeds:
        renders: dict[str, Float[Tensor, "1 3 height width"]] = {}
        seconds: dict[str, float] = {}
        peaks: dict[str, int | None] = {}
        for spec in specs:
            image, elapsed, peak = _sample_walk(
                empty_room,
                config=config,
                stacks=stacks,
                reference=reference,
                eta=spec.eta,
                start_sigma=spec.start_sigma,
                seed=seed,
            )
            renders[spec.arm] = image
            seconds[spec.arm] = elapsed
            peaks[spec.arm] = peak
            write_image(image, gallery / f"{scene.photo}_seed{seed}_{spec.arm}.png")

        # One pixel set for every arm: the empty room's shell narrowed by each
        # arm's own segmentation — furniture the furnishing added is excused,
        # never scored against empty-room depth — intersected across arms so
        # the errors are commensurable and covering shell up buys nothing.
        arm_masks = {
            spec.arm: _scorable(
                renders[spec.arm], stacks=stacks, reference=reference, config=config
            )
            for spec in specs
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
            for spec in specs
        }
        control_scale = mean_absolute_residual(residuals[CONTROL_ARM], common)

        measurements: list[FmrgArmResult] = []
        for spec in specs:
            render = renders[spec.arm]
            depth_map = f"{scene.photo}_seed{seed}_{spec.arm}_depth.png"
            residual_map = f"{scene.photo}_seed{seed}_{spec.arm}_residual.png"
            scorable_overlay = f"{scene.photo}_seed{seed}_{spec.arm}_scorable.png"
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
                FmrgArmResult(
                    arm=spec.arm,
                    eta=spec.eta,
                    start_sigma=spec.start_sigma,
                    effective_first_sigma=effective_sigmas[spec.start_sigma],
                    probe=spec.probe,
                    depth_error=depth_error(
                        render,
                        reference=reference,
                        depth_estimator=stacks.depth_estimator,
                        mask=common,
                    ),
                    shell_coverage=shell_coverage(
                        arm_masks[spec.arm], reference=reference
                    ),
                    # Filled in by `_scored` once sampling is over and the eval
                    # CLIP can afford to be resident.
                    adherence=math.nan,
                    render=f"{scene.photo}_seed{seed}_{spec.arm}.png",
                    depth_map=depth_map,
                    residual_map=residual_map,
                    scorable_overlay=scorable_overlay,
                    seconds=seconds[spec.arm],
                    peak_vram_bytes=peaks[spec.arm],
                )
            )

        result = FmrgSceneResult(
            scene=scene.photo,
            prompt=scene.prompt,
            seed=seed,
            arms=tuple(measurements),
            common_coverage=shell_coverage(common, reference=reference),
            common_frame_fraction=float(common.to(torch.float32).mean()),
            empty_shell_frame_fraction=shell_fractions["empty_room"],
            photo_shell_frame_fraction=shell_fractions["photo"],
            reconstruction_depth_error=reconstruction,
            mean_abs_control_residual=control_scale,
            empty_depth_map=empty_depth_map,
            empty_shell_mask=empty_shell_mask,
            empty_shell_overlay=empty_shell_overlay,
            common_mask_overlay=common_mask_overlay,
        )
        print(f"  [sampled] {result.scene} seed {result.seed}, adherence pending")
        results.append(
            _PendingScene(
                result=result,
                renders={
                    arm: image.detach().to(device="cpu", dtype=torch.float32)
                    for arm, image in renders.items()
                },
                empty_room=empty_room,
                reference=reference,
            )
        )
    return results


def _sample_walk(
    empty_room: Float[Tensor, "1 3 height width"],
    *,
    config: FmrgStudyConfig,
    stacks: FurnishStacks,
    reference: GeometricReference,
    eta: float,
    start_sigma: float,
    seed: int,
    grid_steps: int | None = None,
    window: GuidanceWindow | bool | None = True,
) -> tuple[Float[Tensor, "1 3 height width"], float, int | None]:
    """One walk at `seed`; `eta` 0 is the control on the identical trajectory."""
    furnish = config.furnish
    generator = torch.Generator(device=stacks.device).manual_seed(seed)
    resolved_window = (
        GuidanceWindow(
            early_stop=furnish.early_stop, unguided_steps=furnish.unguided_steps
        )
        if window is True
        else (window or None)
    )
    process = FlowMapProcess(
        stacks.flow_backend,
        empty_room,
        steps=grid_steps if grid_steps is not None else furnish.grid_steps,
        start_sigma=start_sigma,
        generator=generator,
        window=resolved_window,
        stepping=furnish.stepping,
        differentiate_through_velocity=furnish.differentiate_through_velocity,
    )
    tilt = AnnealedSignTilt(
        strength=eta,
        guided_steps=(
            resolved_window.guided_steps if resolved_window is not None else None
        ),
        # Upstream anneals over the whole truncated grid — the guided steps plus
        # the terminal jump it never guides — not the guided window alone.
        anneal_steps=process.steps,
        iterations=furnish.iterations,
    )
    reward = GeometricGuidance(
        depth_estimator=stacks.depth_estimator,
        reference=reference,
        segmenter=stacks.segmenter if furnish.resegment else None,
        erosion_radius=(furnish.erosion_radius if furnish.resegment else None),
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


def _warm_up_walk(
    empty_room: Float[Tensor, "1 3 height width"],
    *,
    config: FmrgStudyConfig,
    stacks: FurnishStacks,
    reference: GeometricReference,
) -> None:
    """One tiny guided walk before the first timed arm; its output is discarded."""
    print("[warmup] 2 throwaway steps (1 guided), untimed")
    _sample_walk(
        empty_room,
        config=config,
        stacks=stacks,
        reference=reference,
        eta=max(config.furnish.eta, 1.0),
        start_sigma=config.furnish.start_sigma,
        seed=config.seeds[0],
        grid_steps=2,
        window=None,
    )


def _scorable(
    image: Float[Tensor, "1 3 height width"],
    *,
    stacks: FurnishStacks,
    reference: GeometricReference,
    config: FmrgStudyConfig,
) -> Bool[Tensor, "1 height width"]:
    """The empty room's shell narrowed by one finished render's own segmentation."""
    if not config.furnish.resegment:
        return reference.mask
    with torch.no_grad():
        segmentation = stacks.segmenter(image)
    return reference.mask & scorable_mask(
        segmentation, erosion_radius=config.furnish.erosion_radius
    )


def _effective_first_sigma(
    empty_room: Float[Tensor, "1 3 height width"],
    *,
    start_sigma: float,
    config: FmrgStudyConfig,
    stacks: FurnishStacks,
) -> float:
    """The sigma the walk actually starts at, read off the backend's own schedule.

    FLUX's mu-shift warps the grid by resolution, so the configured
    `start_sigma` is a request, not the truth — 0.9 at 512² begins near 0.944.
    """
    generator = torch.Generator(device=stacks.device).manual_seed(0)
    with torch.no_grad():
        encoded = stacks.flow_backend.encode(empty_room, generator=generator)
    return float(
        stacks.flow_backend.schedule(
            steps=config.furnish.grid_steps, start_sigma=start_sigma, latent=encoded
        )[0]
    )


def _vae_roundtrip(
    backend: FlowMapBackend,
    image: Float[Tensor, "1 3 height width"],
    *,
    device: torch.device,
) -> Float[Tensor, "1 3 height width"]:
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        return backend.decode(backend.encode(image, generator=generator)).to(
            torch.float32
        )


def _lora_dir(config: FmrgStudyConfig, settings: RoomgenSettings) -> Path:
    directory = config.flux.lora_dir or settings.hf_home / _DEFAULT_LORA_SUBPATH
    if not (directory / _LORA_WEIGHT_NAME).exists():
        raise RuntimeError(
            f"no {_LORA_WEIGHT_NAME} under {directory}. Fetch the FlowMap LoRA "
            f"(see packages/fmrg/checkpoints_download.sh for the repo and file "
            f"path) into that directory, or point flux.lora_dir at where it "
            f"lives."
        )
    return directory


def _flux_path(config: FmrgStudyConfig, settings: RoomgenSettings) -> str:
    """The cached snapshot directory, resolved offline — never the repo id."""
    checkpoint = config.flux.flux_checkpoint
    if Path(checkpoint).exists():
        return checkpoint
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    try:
        return str(
            snapshot_download(
                checkpoint,
                cache_dir=str(settings.hub_cache),
                local_files_only=True,
                ignore_patterns=list(_FLUX_IGNORE),
            )
        )
    except Exception as error:
        raise RuntimeError(
            f"{checkpoint} is not in the cache at {settings.hub_cache}. Download "
            f"its diffusers-format weights first (skip the single-file roots)."
        ) from error


def _photo_path(scene: SceneConfig, photo_paths: dict[str, Path]) -> Path:
    if scene.photo not in photo_paths:
        raise ValueError(
            f"scene photo {scene.photo!r} is not in experiments/assets/photos.json; "
            f"known ids: {', '.join(sorted(photo_paths))}"
        )
    return photo_paths[scene.photo]


def _format_strip(strip: StripResult) -> str:
    numbers = strip.numbers
    return (
        f"{strip.scene}: cut {numbers.contents_fraction:.0%} of the frame, "
        f"hole not shell {numbers.hole_not_shell_fraction:.1%} (recognisable "
        f"furniture {numbers.residual_contents_fraction:.1%}), depth error vs "
        f"photo {numbers.depth_error:.6f} "
        f"({numbers.unrepainted_shell_depth_error:.6f} on the "
        f"{numbers.unrepainted_shell_coverage:.0%} never repainted), "
        f"{numbers.seconds:.1f}s"
    )


def _format_result(result: FmrgSceneResult) -> str:
    cells = ", ".join(
        f"{cell.arm} E={cell.depth_error:.4f} adh={cell.adherence:.4f}"
        for cell in result.arms
    )
    return (
        f"{result.scene} seed {result.seed}: {cells}, common mask "
        f"{result.common_frame_fraction:.0%} of the frame "
        f"({result.common_coverage:.0%} of the empty shell; photo shell held "
        f"{result.photo_shell_frame_fraction:.0%})"
    )


def arm_means(
    results: tuple[FmrgSceneResult, ...], specs: tuple[FmrgArmSpec, ...]
) -> dict[str, dict[str, float]]:
    """Each arm's mean depth error, coverage and adherence over every (scene, seed)."""
    means: dict[str, dict[str, float]] = {}
    for spec in specs:
        cells = [result.arm(spec.arm) for result in results]
        means[spec.arm] = {
            "eta": spec.eta,
            "start_sigma": spec.start_sigma,
            "depth_error": _mean([cell.depth_error for cell in cells]),
            "shell_coverage": _mean([cell.shell_coverage for cell in cells]),
            "adherence": _mean([cell.adherence for cell in cells]),
            "seconds": _mean([cell.seconds for cell in cells]),
        }
    return means


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _walked_steps(config: FmrgStudyConfig) -> int:
    """The steps the truncated walk actually takes, by the window's own arithmetic."""
    window = GuidanceWindow(
        early_stop=config.furnish.early_stop,
        unguided_steps=config.furnish.unguided_steps,
    )
    return window.guided_steps + config.furnish.unguided_steps


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
                # What steered stage 2 and what every number is read against:
                # the stripped empty room's shell depth, intersected per step
                # and per eval with the prediction's own shell.
                "guidance_reference": "empty_room",
                "galleries": {
                    "strip": str(summary.output_dir / GALLERY_A),
                    "furnish": str(summary.output_dir / GALLERY_B),
                },
                "segmentation_color_key": {
                    label: list(color)
                    for label, color in SEGMENTATION_COLOR_KEY.items()
                },
                "arm_means": arm_means(summary.results, furnish_arms(summary.config)),
                "strip": [strip.as_row() for strip in summary.strips],
                "furnish": [result.as_row() for result in summary.results],
            },
            indent=2,
        )
        + "\n"
    )
