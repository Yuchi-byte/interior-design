"""Entry point of 0003_strip_and_furnish: strip the rooms, furnish them, report what it measured.

Glue only — the sampler, the inpainting and the eval are `roomgen_guide`'s, the
run loop is `strip_and_furnish/`'s. The run reports numbers and renders the two
galleries beside them; reading them is the owner's job, so it exits non-zero only
on an actual error.

    uv run --no-sync python experiments/0003_strip_and_furnish/run.py --config smoke
    uv run --no-sync python experiments/0003_strip_and_furnish/run.py --config study
    ... --config study --set furnish.strength=20
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from strip_and_furnish.config import StudyConfig
from strip_and_furnish.furnish import ARMS, Arm, FurnishResult
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

    config = StudyConfig.from_file(
        CONF_DIR / f"{arguments.config}.yaml", arguments.overrides
    )
    settings = RoomgenSettings()
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


_COLUMNS = (
    f"{'arm':<16}{'depth':>11}{'Δ vs ctl':>10}{'adherence':>11}"
    f"{'vs ctl':>9}{'coverage':>10}{'sec':>8}"
)


def _report(summary: Summary) -> None:
    """The study's numbers in the mean, per arm. No claim is made about them."""
    results = summary.furnishes
    print(
        f"\n[stage 2 mean] over {len(results)} (scene, seed) pairs, one common "
        f"mask each\n{_COLUMNS}"
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
            f"{_mean(_adherence(results, arm)):>11.4f}{_share(ratio):>9}"
            f"{_mean(result.arm_coverages[arm] for result in results):>10.0%}"
            f"{_mean(result.seconds[arm] for result in results):>8.1f}"
        )
    print(
        f"{'common mask':<16}{'':>11}{'':>10}{'':>11}{'':>9}"
        f"{_mean(result.common_coverage for result in results):>10.0%}"
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
