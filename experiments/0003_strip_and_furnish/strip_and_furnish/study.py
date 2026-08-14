"""The study: strip every scene, size the composite's weight, furnish under both energies.

Two stages, two galleries, one `results.json`. Stage 1 never sees the depth
energy. Stage 2 is steered by the *photograph's* shell depth, retained from the
reference stage 1 froze to score itself — never re-estimated from the inpainted
room, so guidance inherits nothing the inpainting got wrong. The stripped
room's own reference is still computed and every arm scored against it too, as
the diagnostic that keeps the run comparable with the previous protocol.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from roomgen_geometry.reference import GeometricReference

# `_renoised_timesteps` is imported rather than restated so the number of steps
# a run records having walked cannot drift from the number the sampler walks.
from roomgen_guide.sampling.guided import _renoised_timesteps
from roomgen_guide.settings import RoomgenSettings
from strip_and_furnish.backends import build_adherence, build_backends
from strip_and_furnish.config import SceneConfig, StudyConfig
from strip_and_furnish.furnish import (
    ARMS,
    FurnishResult,
    GradientSizing,
    build_energies,
    empty_room_reference,
    furnish_scene,
    guidance_reference,
    measure_gradient_sizing,
    write_scene_gallery,
)
from strip_and_furnish.images import SEGMENTATION_COLOR_KEY
from strip_and_furnish.photos import ensure_photos, load_photo
from strip_and_furnish.strip import StripResult, strip_scene

PHOTO_SUBDIRECTORY = "benchmark_photos"
GALLERY_A = "gallery_a_strip"
GALLERY_B = "gallery_b_furnish"


@dataclass(frozen=True)
class Summary:
    """What the study measured, where its artifacts landed, and the code that produced them.

    Args:
        shell_fractions: per scene, the share of the frame each reference's
            shell mask covers, keyed `photo` and `empty_room`. The coverages in
            the furnish rows are fractions of each reference's *own* shell, so
            without these two denominators the retained reference's smaller
            shell — furniture occluded part of it — would be invisible.
        scene_images: per scene, gallery B's seed-independent frames — the
            empty room going in, and the guidance reward mask painted on the
            photo and on the empty room.
    """

    config: StudyConfig
    strips: tuple[StripResult, ...]
    sizings: dict[str, GradientSizing]
    depth_weight: float
    furnishes: tuple[FurnishResult, ...]
    shell_fractions: dict[str, dict[str, float]]
    scene_images: dict[str, dict[str, str]]
    output_dir: Path
    peak_vram_bytes: int | None
    started_at: str
    git_commit: str
    walked_steps: int
    device_name: str | None


def run_study(config: StudyConfig, settings: RoomgenSettings) -> Summary:
    """Run both stages over every scene and seed of `config` and write every artifact."""
    started_at = datetime.now(UTC).isoformat()
    output_dir = settings.wdir / config.name
    gallery_a = output_dir / GALLERY_A
    gallery_b = output_dir / GALLERY_B
    for directory in (gallery_a, gallery_b):
        directory.mkdir(parents=True, exist_ok=True)

    photo_paths = ensure_photos(settings.wdir / PHOTO_SUBDIRECTORY)
    backends = build_backends(config.backends, cache_dir=settings.hub_cache)
    adherence = build_adherence(
        config.adherence, device=backends.device, cache_dir=settings.hub_cache
    )
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
    # Stage 2 never inpaints, and a guided step's VAE backward is the largest
    # allocation in the run: holding a second pipeline through it is what turns
    # a 512px study into an out-of-memory error.
    del inpainter
    if backends.device.type == "cuda":
        torch.cuda.empty_cache()

    # The guidance target is the photograph's shell depth, retained from the
    # reference each strip froze — not re-estimated from the inpainted room.
    # The stripped room's reference is computed beside it purely for scoring.
    references = {strip.scene: guidance_reference(strip) for strip in strips}
    empty_references = {
        strip.scene: empty_room_reference(strip, config=config, backends=backends)
        for strip in strips
    }
    shell_fractions = {
        scene_id: {
            "photo": _mask_fraction(references[scene_id]),
            "empty_room": _mask_fraction(empty_references[scene_id]),
        }
        for scene_id in references
    }
    for scene_id, fractions in shell_fractions.items():
        # The retained reference's inherent cost, stated per scene: shell the
        # furniture occluded is outside the photo's mask, so it holds less of
        # the frame than the empty room's.
        print(
            f"  {scene_id}: photo shell {fractions['photo']:.0%} of the frame, "
            f"empty room's {fractions['empty_room']:.0%}"
        )
    texts = {scene.photo: backends.text_energy(scene.prompt) for scene in config.scenes}

    print("\n[sizing]  gradient RMS of each term on a real sample, per scene")
    # Each term measured on its own first: the weight is read off that ratio, so
    # it cannot be measured through a composite that already carries it.
    terms = {
        scene.photo: build_energies(
            texts[scene.photo],
            reference=references[scene.photo],
            config=config,
            backends=backends,
            depth_weight=1.0,
        )
        for scene in config.scenes
    }
    unweighted = {
        scene.photo: measure_gradient_sizing(
            strip.empty_room,
            scene.prompt,
            text=terms[scene.photo].text_only,
            depth=terms[scene.photo].depth_only,
            config=config,
            backends=backends,
            seed=config.seeds[0],
        )
        for scene, strip in zip(config.scenes, strips, strict=True)
    }
    depth_weight = _chosen_weight(config, unweighted)
    energies_by_scene = {
        scene.photo: build_energies(
            texts[scene.photo],
            reference=references[scene.photo],
            config=config,
            backends=backends,
            depth_weight=depth_weight,
        )
        for scene in config.scenes
    }
    sizings = {
        scene.photo: measure_gradient_sizing(
            strip.empty_room,
            scene.prompt,
            text=energies_by_scene[scene.photo].text_only,
            depth=energies_by_scene[scene.photo].depth_only,
            composite=energies_by_scene[scene.photo].text_and_depth,
            weight=depth_weight,
            config=config,
            backends=backends,
            seed=config.seeds[0],
        )
        for scene, strip in zip(config.scenes, strips, strict=True)
    }
    for scene_id, sizing in sizings.items():
        print(f"  {_format_sizing(scene_id, sizing)}")
    print(f"  depth weight {depth_weight:.6g} ({_weight_source(config)})")

    print(
        f"\n[stage 2] furnish: {len(config.scenes)} scenes x {len(config.seeds)} seeds "
        f"x {len(ARMS)} arms, reference = the photograph's shell (retained from "
        f"stage 1; the stripped room's is scored as a diagnostic)"
    )
    furnishes: list[FurnishResult] = []
    scene_images: dict[str, dict[str, str]] = {}
    for scene, strip in zip(config.scenes, strips, strict=True):
        # Gallery B stands alone: the empty room going in is beside the
        # furnished rooms coming out, without a strip arm in sight — and the
        # guidance reward mask beside both, so where the depth loss is computed
        # is visible rather than implied.
        scene_images[scene.photo] = write_scene_gallery(
            strip, reference=references[scene.photo], gallery=gallery_b
        )
        results = furnish_scene(
            scene,
            strip,
            config=config,
            backends=backends,
            adherence=adherence,
            reference=references[scene.photo],
            empty_reference=empty_references[scene.photo],
            energies=energies_by_scene[scene.photo],
            gallery=gallery_b,
        )
        for result in results:
            print(f"  {_format_furnish(result)}")
        furnishes.extend(results)

    on_cuda = backends.device.type == "cuda"
    summary = Summary(
        config=config,
        strips=strips,
        sizings=sizings,
        depth_weight=depth_weight,
        furnishes=tuple(furnishes),
        shell_fractions=shell_fractions,
        scene_images=scene_images,
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


def _mask_fraction(reference: GeometricReference) -> float:
    """The share of the frame a reference's shell mask covers."""
    return float(reference.mask.to(torch.float32).mean())


def _chosen_weight(config: StudyConfig, sizings: dict[str, GradientSizing]) -> float:
    """The configured weight, or the one the measurement asks for: RMS-matched, averaged.

    Averaged over scenes rather than taken per scene, so every arm-B render in
    the study is steered by one composite and the comparison between scenes is
    a comparison of scenes.
    """
    if config.furnish.depth_weight is not None:
        return config.furnish.depth_weight
    ratios = [
        sizing.text_gradient_rms / sizing.depth_gradient_rms
        for sizing in sizings.values()
        if sizing.depth_gradient_rms > 0.0
    ]
    if not ratios:
        raise ValueError(
            "every scene measured a zero depth gradient, so there is no weight to "
            "read off it: the depth term would be steering nothing"
        )
    return sum(ratios) / len(ratios)


def _weight_source(config: StudyConfig) -> str:
    return "configured" if config.furnish.depth_weight is not None else "measured"


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
        f"{strip.scene}: cut {numbers.contents_fraction:.0%} of the frame, shell "
        f"{numbers.shell_fraction:.0%}, hole not whitelisted fill "
        f"{numbers.hole_not_fill_fraction:.1%} (not shell "
        f"{numbers.hole_not_shell_fraction:.1%}, recognisable furniture "
        f"{numbers.residual_contents_fraction:.1%}), "
        f"depth error vs photo {numbers.depth_error:.6f} "
        f"({numbers.unrepainted_shell_depth_error:.6f} on the "
        f"{numbers.unrepainted_shell_coverage:.0%} of shell never repainted) "
        f"(floor {numbers.reconstruction_depth_error:.6f}), outside the mask "
        f"max |Δ| {numbers.unmasked_max_absolute_difference:.2e} composited vs "
        f"{numbers.generated_unmasked_max_absolute_difference:.2e} raw, "
        f"{numbers.seconds:.1f}s"
    )


def _format_sizing(scene: str, sizing: GradientSizing) -> str:
    composite = (
        "n/a"
        if sizing.composite_gradient_rms is None
        else f"{sizing.composite_gradient_rms:.3e}"
    )
    return (
        f"{scene}: step {sizing.step_index}/{sizing.total_steps}, text "
        f"{sizing.text_gradient_rms:.3e}, depth {sizing.depth_gradient_rms:.3e}, "
        f"composite {composite}"
    )


def _format_furnish(result: FurnishResult) -> str:
    against_control = result.metrics("text_and_depth", "control")
    against_text = result.metrics("text_and_depth", "text_only")
    return (
        f"{result.scene} seed {result.seed}: depth vs photo "
        f"A {result.depth_errors['text_only']:.6f} / "
        f"B {result.depth_errors['text_and_depth']:.6f} / "
        f"ctl {result.depth_errors['control']:.6f} "
        f"(B vs ctl {_signed(against_control.depth_error_reduction)}, "
        f"B vs A {_signed(against_text.depth_error_reduction)}), vs empty room "
        f"A {result.empty_room_depth_errors['text_only']:.6f} / "
        f"B {result.empty_room_depth_errors['text_and_depth']:.6f} / "
        f"ctl {result.empty_room_depth_errors['control']:.6f}, adherence "
        f"A {result.adherences['text_only']:.4f} / "
        f"B {result.adherences['text_and_depth']:.4f} / "
        f"ctl {result.adherences['control']:.4f}, shell scored "
        f"{result.common_coverage:.0%} in common vs photo, "
        f"{result.empty_room_common_coverage:.0%} vs empty room"
    )


def _signed(value: float | None) -> str:
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


def _walked_steps(config: StudyConfig) -> int:
    """How many of the scheduler's steps SDEdit actually walks, by the sampler's own arithmetic."""
    return len(
        _renoised_timesteps(
            torch.arange(config.furnish.steps),
            noise_strength=config.furnish.noise_strength,
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
                "clip_checkpoints": {
                    "guidance": summary.config.backends.text_energy_checkpoint,
                    "evaluation": summary.config.adherence.checkpoint,
                },
                "galleries": {
                    "strip": str(summary.output_dir / GALLERY_A),
                    "furnish": str(summary.output_dir / GALLERY_B),
                },
                # What steered stage 2 and what the primary numbers are read
                # against; the `*_empty_room` fields are the previous
                # protocol's reference, scored as a diagnostic.
                "guidance_reference": "photo",
                "reference_shell_fractions": summary.shell_fractions,
                # The colour every segmentation map in gallery A is painted
                # under, and the seed-independent gallery B frames per scene:
                # the empty room and the guidance reward-mask overlays.
                "segmentation_color_key": {
                    label: list(color)
                    for label, color in SEGMENTATION_COLOR_KEY.items()
                },
                "furnish_scene_images": summary.scene_images,
                "depth_weight": summary.depth_weight,
                "weight_sizing": {
                    scene: sizing.as_row() for scene, sizing in summary.sizings.items()
                },
                "strip": [strip.as_row() for strip in summary.strips],
                "furnish": [result.as_row() for result in summary.furnishes],
            },
            indent=2,
        )
        + "\n"
    )
