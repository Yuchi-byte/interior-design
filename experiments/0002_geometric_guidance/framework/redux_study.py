"""The Redux-scale sweep: the furnished photo's image tokens dialled against the text.

Per scene at one seed, off one transformer load: one unguided and one
depth-guided arm per configured Redux scale, every arm conditioned on the
scene's one renovation prompt — style descriptors, a keep-the-structure
clause, and an explicit remove-the-object instruction, so the scale dial
directly measures where the image tokens start to drown the text. Scale 0 is
the text-only pair (the plain prompt entry, never a zero-scaled token copy)
and its unguided arm is the baseline every delta divides by.

The depth reference is the *stripped empty room's* shell depth, by owner
decision: stage zero runs 0003's strip (EoMT decides, SD-inpainting fills) on
every photo first, and the reference is read off the empty room — whose shell
is nearly the whole frame, a far larger comparison area than the furnished
shell — while the Redux tokens still encode the furnished photo, which is the
axis under test. A consequence worth naming: the empty room contains none of
the furniture the removal instruction targets, so the depth reward never
defends the removed object's mass and the removal-compliance curve reads the
text-versus-image-tokens contest cleanly.

Every arm is scored on the one pixel set all arms left visible (empty shell ∩
every arm's own shell), judged against the scene's one neutral style text by
an adherence CLIP, a render-to-photo image similarity, a preference model —
all loaded only after the last walk — and the removal-compliance fraction the
segmenter reads off each render. Beside the numbers, every arm leaves a masked
depth map and a signed depth-error map against the empty-room reference, all
on the baseline arm's colour scale.

Model residency, in order: the SD strip stack (freed), the Redux encoders
(freed), the FLUX transformer with the walks, then the judges one at a time.
"""

from __future__ import annotations

import gc
import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from framework.config import TORCH_DTYPES, FmrgReduxConfig, ReduxSceneConfig
from framework.depth_maps import write_masked_disparity_map
from framework.fmrg_alignment_study import shell_class_table
from framework.fmrg_study import (
    _REPO_ROOT,
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
from framework.metrics import build_adherence, class_area_fraction
from framework.photos import ensure_photos, load_photo
from framework.preference import build_preference
from framework.residual_maps import mean_absolute_residual, write_signed_residual_map
from roomgen_geometry.masking import scorable_mask
from roomgen_geometry.reference import GeometricReference
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
from roomgen_guide.guidance.preference import ImagePreferenceModel
from roomgen_guide.settings import RoomgenSettings

# 0003's strip machinery is reused, not reimplemented; the frameworks are
# separate top-level directories, so it is reached the way the fakes are.
_STRIP_STUDY_DIR: Final = _REPO_ROOT / "experiments" / "0003_strip_and_furnish"


def arm_id(scale: float, guided: bool) -> str:
    """The arm's name, read straight off the dial: scale then u(nguided)/g(uided)."""
    return f"s{scale:g}{'g' if guided else 'u'}"


# The baseline arm's id: scale 0 (text only), unguided. Every other arm's depth
# delta and adherence ratio divide by it, and its mean absolute residual sets
# the scene's residual-map colour scale. Scale 0 is mandatory in the config.
BASELINE_ARM: Final = arm_id(0.0, guided=False)


def redux_prompt_key(prompt: str, scale: float) -> str:
    """The cache name of `prompt` with the scene's image tokens appended at `scale`."""
    return f"{prompt} || redux tokens x{scale:g}"


@dataclass(frozen=True)
class ReduxArmSpec:
    """One arm of the sweep; `redux_scale` 0 is a text-only arm with no image tokens."""

    arm: str
    guided: bool
    redux_scale: float


def redux_arms(config: FmrgReduxConfig) -> tuple[ReduxArmSpec, ...]:
    """The arms in dial order, an unguided and a guided arm per scale, baseline first."""
    return tuple(
        ReduxArmSpec(arm_id(scale, guided), guided, scale)
        for scale in config.redux_scales
        for guided in (False, True)
    )


@dataclass(frozen=True)
class ReduxArmResult:
    """One arm's render, its depth and removal readings, and its judged scores."""

    arm: str
    guided: bool
    redux_scale: float
    prompt: str
    depth_error: float
    shell_coverage: float
    removal_fraction: float
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
            "redux_scale": self.redux_scale,
            "prompt": self.prompt,
            "depth_error": self.depth_error,
            "shell_coverage": self.shell_coverage,
            "removal_fraction": self.removal_fraction,
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
class ReduxSceneResult:
    """Every arm of one (scene, seed), scored on one common pixel set vs the empty room.

    Args:
        common_coverage: share of the empty room's shell the common mask still
            scores.
        common_frame_fraction: the same mask as a share of the whole frame,
            read beside `photo_shell_frame_fraction` — the enlarged valid area
            the empty-room reference was chosen for, quantified.
        strip: the stage-zero strip's own numbers row, verbatim — including
            `residual_contents_fraction`, the flag that names a scene whose
            "empty" reference still holds furniture the strip failed to cut.
    """

    scene: str
    seed: int
    style_prompt: str
    renovation_prompt: str
    eval_prompt: str
    removal_object: str
    removal_classes: tuple[int, ...]
    removal_substituted: bool
    photo_removal_fraction: float
    effective_first_sigma: float
    arms: tuple[ReduxArmResult, ...]
    common_coverage: float
    common_frame_fraction: float
    photo_shell_frame_fraction: float
    empty_shell_frame_fraction: float
    reconstruction_depth_error: float
    mean_abs_baseline_residual: float
    photo_image: str
    empty_room_image: str
    contents_overlay: str
    photo_depth_map: str
    empty_room_depth_map: str
    empty_shell_mask: str
    eval_mask_common: str
    strip: dict[str, object]

    def arm(self, arm_id: str) -> ReduxArmResult:
        for measurement in self.arms:
            if measurement.arm == arm_id:
                return measurement
        raise KeyError(arm_id)

    def depth_delta_vs_baseline(self, arm_id: str) -> float | None:
        """The arm's depth error relative to the text-only unguided arm's, signed."""
        baseline = self.arm(BASELINE_ARM).depth_error
        if baseline == 0.0:
            return None
        return self.arm(arm_id).depth_error / baseline - 1.0

    def adherence_ratio_vs_baseline(self, arm_id: str) -> float | None:
        """The arm's adherence as a share of the text-only unguided arm's."""
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
            "removal_object": self.removal_object,
            "removal_classes": list(self.removal_classes),
            "removal_substituted": self.removal_substituted,
            "photo_removal_fraction": self.photo_removal_fraction,
            "effective_first_sigma": self.effective_first_sigma,
            "common_coverage": self.common_coverage,
            "common_frame_fraction": self.common_frame_fraction,
            "photo_shell_frame_fraction": self.photo_shell_frame_fraction,
            "empty_shell_frame_fraction": self.empty_shell_frame_fraction,
            "reconstruction_depth_error": self.reconstruction_depth_error,
            "mean_abs_baseline_residual": self.mean_abs_baseline_residual,
            "photo_image": self.photo_image,
            "empty_room_image": self.empty_room_image,
            "contents_overlay": self.contents_overlay,
            "photo_depth_map": self.photo_depth_map,
            "empty_room_depth_map": self.empty_room_depth_map,
            "empty_shell_mask": self.empty_shell_mask,
            "eval_mask_common": self.eval_mask_common,
            "strip": self.strip,
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


@dataclass(frozen=True)
class _StrippedScene:
    """Stage zero's product for one scene: the empty room and the reference read off it."""

    empty_room: Float[Tensor, "1 3 height width"]
    reference: GeometricReference
    photo_removal_fraction: float
    photo_shell_frame_fraction: float
    empty_shell_frame_fraction: float
    strip_row: dict[str, object]
    images: dict[str, str]


def _strip_scenes(
    config: FmrgReduxConfig,
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
    if str(_STRIP_STUDY_DIR) not in sys.path:
        sys.path.insert(0, str(_STRIP_STUDY_DIR))
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
            StripSceneConfig(photo=scene.photo, prompt=scene.renovation_prompt),
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
        with torch.no_grad():
            photo_segmentation = backends.segmenter(photo.to(torch.float32))
        entry = _StrippedScene(
            empty_room=empty_room,
            reference=reference,
            photo_removal_fraction=class_area_fraction(
                photo_segmentation, scene.removal_classes
            ),
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


def run_redux_study(config: FmrgReduxConfig, settings: RoomgenSettings) -> ReduxSummary:
    """Strip every scene, then run and measure every arm of every (scene, seed)."""
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

    # Stage zero on an otherwise empty GPU: the SD strip stack in, five empty
    # rooms and their references out, the stack freed.
    stripped = _strip_scenes(
        config, photo_paths=photo_paths, image_dir=image_dir, settings=settings
    )

    # Phase one: each *furnished* reference photo through SigLIP and the Redux
    # adapter, tokens parked on the CPU, encoders freed — all before the ~24GB
    # transformer exists. The empty room never meets Redux: image-context on
    # the furniture is the axis under test.
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

    prompts = list(dict.fromkeys(scene.renovation_prompt for scene in config.scenes))
    stacks = build_stacks(
        config.backends,
        prompts=prompts,
        text_guidance_scale=config.sampler.text_guidance_scale,
        settings=settings,
    )
    if stacks.sampler is not None:
        for scene in config.scenes:
            for scale in config.redux_scales:
                # Scale 0 is the plain text entry, already cached; a zero-scaled
                # token copy would be the same conditioning under a second name.
                if scale == 0.0:
                    continue
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
                stripped=stripped[scene.photo],
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
    """One scene measured on everything but the deferred judges, renders parked on CPU."""

    result: ReduxSceneResult
    renders: dict[str, Float[Tensor, "1 3 height width"]]
    photo: Float[Tensor, "1 3 height width"]
    reference: GeometricReference


def _score_pending(
    pending: list[_PendingScene],
    *,
    config: FmrgReduxConfig,
    stacks: Stacks,
    settings: RoomgenSettings,
) -> list[ReduxSceneResult]:
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


def _run_scene(
    scene: ReduxSceneConfig,
    *,
    config: FmrgReduxConfig,
    stacks: Stacks,
    stripped: _StrippedScene,
    photo_path: Path,
    image_dir: Path,
    warm_up: bool,
) -> list[_PendingScene]:
    # The furnished photo: the walks' init, the Redux tokens' source, and what
    # image similarity is read against. The depth reference is the empty room's.
    photo = load_photo(
        photo_path,
        resolution=config.resolution,
        device=stacks.device,
        dtype=torch.float32,
    )
    reference = stripped.reference
    empty_room = stripped.empty_room.to(stacks.device)
    empty_shell_mask = f"{scene.photo}_empty_shell_mask.png"
    _write_mask(reference.mask, image_dir / empty_shell_mask)

    # The floor every arm's depth error is read against: what a perfect
    # geometry-preserving renovation still pays for FLUX's VAE round trip of
    # the reference geometry alone.
    reconstruction = depth_error(
        _vae_roundtrip(stacks.flow_backend, empty_room, device=stacks.device),
        reference=reference,
        depth_estimator=stacks.depth_estimator,
    )
    effective_sigma = _effective_first_sigma(photo, config=config, stacks=stacks)
    print(
        f"[scene] {scene.photo}: empty shell covers "
        f"{stripped.empty_shell_frame_fraction:.1%} of the frame (photo shell "
        f"{stripped.photo_shell_frame_fraction:.1%}), photo "
        f"{scene.removal_object} share {stripped.photo_removal_fraction:.1%}, "
        f"reconstruction floor {reconstruction:.6f}, effective first sigma "
        f"{config.sampler.start_sigma:g}->{effective_sigma:.4f}"
    )

    if warm_up:
        if stacks.sampler is not None:
            select_cached_prompt(stacks.sampler, scene.renovation_prompt)
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

        # One segmentation per finished render serves both readings: the arm's
        # own scorable mask (empty shell ∩ its shell, so furniture the arm adds
        # is excused rather than scored against empty-room depth) and the share
        # of frame still reading as the object the prompt removed.
        arm_masks: dict[str, Bool[Tensor, "1 height width"]] = {}
        removal_fractions: dict[str, float] = {}
        for spec in specs:
            with torch.no_grad():
                segmentation = stacks.segmenter(renders[spec.arm])
            arm_masks[spec.arm] = (
                reference.mask
                & scorable_mask(
                    segmentation, erosion_radius=config.guidance.erosion_radius
                )
                if config.guidance.resegment
                else reference.mask
            )
            removal_fractions[spec.arm] = class_area_fraction(
                segmentation, scene.removal_classes
            )

        # One pixel set for every arm: the empty room's shell narrowed by each
        # arm's own segmentation, intersected across all of them, so the depth
        # errors are commensurable and covering shell up buys nothing.
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
                    guided=spec.guided,
                    redux_scale=spec.redux_scale,
                    prompt=scene.renovation_prompt,
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

        result = ReduxSceneResult(
            scene=scene.photo,
            seed=seed,
            style_prompt=scene.style_prompt,
            renovation_prompt=scene.renovation_prompt,
            eval_prompt=scene.eval_prompt,
            removal_object=scene.removal_object,
            removal_classes=scene.removal_classes,
            removal_substituted=scene.removal_substituted,
            photo_removal_fraction=stripped.photo_removal_fraction,
            effective_first_sigma=effective_sigma,
            arms=tuple(measurements),
            common_coverage=shell_coverage(common, reference=reference),
            common_frame_fraction=float(common.to(torch.float32).mean()),
            photo_shell_frame_fraction=stripped.photo_shell_frame_fraction,
            empty_shell_frame_fraction=stripped.empty_shell_frame_fraction,
            reconstruction_depth_error=reconstruction,
            mean_abs_baseline_residual=baseline_scale,
            photo_image=stripped.images["photo"],
            empty_room_image=stripped.images["empty_room"],
            contents_overlay=stripped.images["contents_overlay"],
            photo_depth_map=stripped.images["photo_depth"],
            empty_room_depth_map=stripped.images["empty_room_depth"],
            empty_shell_mask=empty_shell_mask,
            eval_mask_common=eval_mask_common,
            strip=stripped.strip_row,
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


def _prompt_key(scene: ReduxSceneConfig, spec: ReduxArmSpec) -> str:
    """The prompt-cache entry the arm conditions on; scale 0 is the plain text entry."""
    if spec.redux_scale == 0.0:
        return scene.renovation_prompt
    return redux_prompt_key(scene.renovation_prompt, spec.redux_scale)


def _format_result(result: ReduxSceneResult) -> str:
    cells = ", ".join(
        f"{cell.arm} E={cell.depth_error:.4f} adh={cell.adherence:.4f} "
        f"rem={cell.removal_fraction:.1%}"
        for cell in result.arms
    )
    return (
        f"seed {result.seed}: {cells}, common mask "
        f"{result.common_frame_fraction:.0%} of the frame "
        f"({result.common_coverage:.0%} of the empty shell)"
    )


def arm_means(
    results: tuple[ReduxSceneResult, ...], specs: tuple[ReduxArmSpec, ...]
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
            "removal_fraction": _mean([cell.removal_fraction for cell in cells]),
            "adherence": _mean([cell.adherence for cell in cells]),
            "image_similarity": _mean(similarities) if similarities else None,
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
                # The shell definition is under revision (windows, doors, soft
                # furnishings joining): the results must say which one scored
                # them, not point at whatever the package says later.
                "shell_classes": shell_class_table(),
                # What steered every guided arm and what every depth number is
                # read against: the stripped empty room's shell depth,
                # intersected per step and per eval with the render's own shell.
                "guidance_reference": "empty_room",
                "eta": summary.config.guidance.eta,
                "redux_scales": list(summary.config.redux_scales),
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
