"""Entry point of 0003_strip_and_furnish: strip the rooms, furnish them, report what it measured.

Glue only — the sampler, the inpainting and the eval are `roomgen_guide`'s, the
run loop is `strip_and_furnish/`'s. A variant with a top-level `study: fmrg`
key runs the furnish stage on FMRG's FLUX flow map against the stripped empty
room's shell depth; without it, the SD1.5-era `StudyConfig` shape is assumed.
The run reports numbers and renders the two galleries beside them; reading them
is the owner's job, so it exits non-zero only on an actual error.

    uv run --no-sync python experiments/0003_strip_and_furnish/run.py --config smoke
    uv run --no-sync python experiments/0003_strip_and_furnish/run.py --config fmrg
    ... --config fmrg --set furnish.eta=5
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from strip_and_furnish.config import (
    FmrgReduxConfig,
    FmrgStudyConfig,
    load_study_config,
)
from strip_and_furnish.fmrg import (
    CONTROL_ARM,
    FmrgSummary,
    furnish_arms,
    run_fmrg_study,
)
from strip_and_furnish.furnish import ARMS, Arm, FurnishResult
from strip_and_furnish.redux import REDUX_ARMS, ReduxSummary, run_redux_study
from strip_and_furnish.study import Summary, run_study

from roomgen_guide.settings import RoomgenSettings

CONF_DIR = Path(__file__).resolve().parent / "conf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="study",
        help=f"variant to run, by file stem in {CONF_DIR} (default: study)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist override, repeatable (e.g. --set furnish.strength=20)",
    )
    arguments = parser.parse_args()

    config = load_study_config(
        CONF_DIR / f"{arguments.config}.yaml", arguments.overrides
    )
    settings = RoomgenSettings()
    if isinstance(config, FmrgReduxConfig):
        print(
            f"[config] {config.name} ({arguments.config}.yaml, study fmrg_redux): "
            f"{len(config.scenes)} scenes x {len(config.seeds)} seeds x "
            f"{len(REDUX_ARMS)} arms on {config.backends.kind} backends, device "
            f"{config.backends.device}, eta {config.furnish.eta}, start sigma "
            f"{config.furnish.start_sigma}, redux scale {config.redux.scale}"
        )
        _report_redux(run_redux_study(config, settings))
        return 0
    if isinstance(config, FmrgStudyConfig):
        print(
            f"[config] {config.name} ({arguments.config}.yaml, study fmrg): "
            f"{len(config.scenes)} scenes x {len(config.seeds)} seeds on "
            f"{config.backends.kind} backends, device {config.backends.device}, "
            f"eta {config.furnish.eta}, start sigma {config.furnish.start_sigma}"
            + (
                f" (+ probes {list(config.furnish.probe_sigmas)})"
                if config.furnish.probe_sigmas
                else ""
            )
        )
        _report_fmrg(run_fmrg_study(config, settings))
        return 0
    print(
        f"[config] {config.name} ({arguments.config}.yaml): "
        f"{len(config.scenes)} scenes x {len(config.seeds)} seeds on "
        f"{config.backends.kind} backends, device {config.backends.device}, "
        f"{config.backends.dtype}, guidance strength {config.furnish.strength}\n"
        f"[clip]   guidance {config.backends.text_energy_checkpoint}, "
        f"adherence {config.adherence.checkpoint} ({config.adherence.kind})"
    )

    summary = run_study(config, settings)
    _report(summary)
    return 0


def _report_redux(summary: ReduxSummary) -> None:
    """The grid's numbers in the mean, per arm. No claim is made about them."""
    results = summary.results
    print(
        f"\n[stage 2 mean] over {len(results)} (scene, seed) pairs, one common "
        f"mask each vs the empty room\n"
        f"{'arm':<17}{'depth':>11}{'Δ vs ctl':>10}{'adherence':>11}{'vs ctl':>9}"
        f"{'sim(empty)':>11}{'sim(photo)':>11}{'pick':>8}{'cover':>7}{'sec':>7}"
    )
    for spec in REDUX_ARMS:
        cells = [result.arm(spec.arm) for result in results]
        delta = _optional_mean(
            result.depth_delta_vs_control(spec.arm) for result in results
        )
        ratio = _optional_mean(
            result.adherence_ratio_vs_control(spec.arm) for result in results
        )
        empty = _optional_mean(cell.similarity_empty_room for cell in cells)
        photo = _optional_mean(cell.similarity_photo for cell in cells)
        print(
            f"{spec.arm:<17}"
            f"{_mean(cell.depth_error for cell in cells):>11.6f}"
            f"{_signed(None if spec.arm == CONTROL_ARM else delta):>10}"
            f"{_mean(cell.adherence for cell in cells):>11.4f}"
            f"{_share(None if spec.arm == CONTROL_ARM else ratio):>9}"
            f"{_float(empty):>11}"
            f"{_float(photo):>11}"
            f"{_mean(cell.preference for cell in cells):>8.2f}"
            f"{_mean(cell.shell_coverage for cell in cells):>7.0%}"
            f"{_mean(cell.seconds for cell in cells):>7.1f}"
        )
    weights = [
        result.sizing.depth_weight for result in results if result.sizing is not None
    ]
    if weights:
        print(
            "[weights] measured depth weight per scene: "
            + ", ".join(f"{weight:.3g}" for weight in weights)
            + " (image term 1.0, gradient-RMS matched)"
        )
    print(
        f"[mask]    common mask holds "
        f"{_mean(result.common_frame_fraction for result in results):.0%} of the "
        f"frame at the mean"
    )
    print(
        f"[run]     commit {summary.git_commit[:8]}, "
        f"{summary.walked_steps} walked steps, "
        f"device {summary.device_name or summary.config.backends.device}"
    )
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")


def _float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _report_fmrg(summary: FmrgSummary) -> None:
    """The FMRG run's numbers in the mean, per arm. No claim is made about them."""
    results = summary.results
    specs = furnish_arms(summary.config)
    print(
        f"\n[stage 2 mean] over {len(results)} (scene, seed) pairs, one common "
        f"mask each vs the empty room\n"
        f"{'arm':<14}{'depth':>11}{'Δ vs ctl':>10}{'adherence':>11}{'vs ctl':>9}"
        f"{'coverage':>10}{'sec':>8}"
    )
    for spec in specs:
        cells = [result.arm(spec.arm) for result in results]
        delta = _optional_mean(
            result.depth_delta_vs_control(spec.arm) for result in results
        )
        ratio = _optional_mean(
            result.adherence_ratio_vs_control(spec.arm) for result in results
        )
        print(
            f"{spec.arm:<14}"
            f"{_mean(cell.depth_error for cell in cells):>11.6f}"
            f"{_signed(None if spec.arm == CONTROL_ARM else delta):>10}"
            f"{_mean(cell.adherence for cell in cells):>11.4f}"
            f"{_share(None if spec.arm == CONTROL_ARM else ratio):>9}"
            f"{_mean(cell.shell_coverage for cell in cells):>10.0%}"
            f"{_mean(cell.seconds for cell in cells):>8.1f}"
        )
    print(
        f"[mask]    common mask holds "
        f"{_mean(result.common_frame_fraction for result in results):.0%} of the "
        f"frame (photo shell would have held "
        f"{_mean(result.photo_shell_frame_fraction for result in results):.0%})"
    )
    print(
        f"[run]     commit {summary.git_commit[:8]}, "
        f"{summary.walked_steps} walked steps, "
        f"device {summary.device_name or summary.config.backends.device}"
    )
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")


_COLUMNS = (
    f"{'arm':<16}{'depth':>11}{'Δ vs ctl':>10}{'depth(empty)':>13}{'adherence':>11}"
    f"{'vs ctl':>9}{'coverage':>10}{'sec':>8}"
)


def _report(summary: Summary) -> None:
    """The study's numbers in the mean, per arm. No claim is made about them.

    `depth` is read against the photograph's retained reference — the one the
    guided arm descends — and `depth(empty)` against the stripped room's, kept
    as a diagnostic. Each reference scores
    all arms of a pair on its own common mask.
    """
    results = summary.furnishes
    print(
        f"\n[stage 2 mean] over {len(results)} (scene, seed) pairs, one common "
        f"mask each per reference\n{_COLUMNS}"
    )
    for arm in ARMS:
        reduction = _optional_mean(
            result.metrics(arm, "control").depth_error_reduction for result in results
        )
        ratio = _optional_mean(
            result.metrics(arm, "control").adherence_ratio for result in results
        )
        print(
            f"{arm:<16}{_mean(_depth(results, arm)):>11.6f}"
            f"{_signed(reduction):>10}"
            f"{_mean(result.empty_room_depth_errors[arm] for result in results):>13.6f}"
            f"{_mean(_adherence(results, arm)):>11.4f}{_share(ratio):>9}"
            f"{_mean(result.arm_coverages[arm] for result in results):>10.0%}"
            f"{_mean(result.seconds[arm] for result in results):>8.1f}"
        )
    print(
        f"{'common (photo)':<16}{'':>11}{'':>10}{'':>13}{'':>11}{'':>9}"
        f"{_mean(result.common_coverage for result in results):>10.0%}"
    )
    print(
        f"{'common (empty)':<16}{'':>11}{'':>10}{'':>13}{'':>11}{'':>9}"
        f"{_mean(result.empty_room_common_coverage for result in results):>10.0%}"
    )
    print(
        f"[weights] depth term {summary.depth_weight:.6g}, text term 1.0 "
        f"(gradient-RMS matched at the sizing step)"
    )
    print(
        f"[run]     commit {summary.git_commit[:8]}, "
        f"{summary.walked_steps} walked steps, "
        f"device {summary.device_name or summary.config.backends.device}"
    )
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")


def _depth(results: tuple[FurnishResult, ...], arm: Arm) -> Iterable[float]:
    return (result.depth_errors[arm] for result in results)


def _adherence(results: tuple[FurnishResult, ...], arm: Arm) -> Iterable[float]:
    return (result.adherences[arm] for result in results)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _optional_mean(values: Iterable[float | None]) -> float | None:
    """The mean over the pairs that could report the ratio at all; `None` if none could."""
    measured = [value for value in values if value is not None]
    return sum(measured) / len(measured) if measured else None


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


if __name__ == "__main__":
    raise SystemExit(main())
