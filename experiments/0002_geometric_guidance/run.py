"""Entry point of 0002_geometric_guidance: pick a `conf/` variant, run it, report what it measured.

Glue only — the sampler and the eval are `roomgen_guide`'s, the run loop is
`framework/`'s. The run reports numbers and renders the gallery beside them;
reading them is the owner's job, so it exits non-zero only on an actual error.

    uv run --no-sync python experiments/0002_geometric_guidance/run.py --config smoke
    uv run --no-sync python experiments/0002_geometric_guidance/run.py --config benchmark
    ... --config benchmark --set guidance.strength=150
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from framework.benchmark import Summary, run_benchmark
from framework.config import BenchmarkConfig
from framework.gallery import write_gallery

from roomgen_guide.settings import RoomgenSettings

CONF_DIR = Path(__file__).resolve().parent / "conf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="benchmark",
        help=f"variant to run, by file stem in {CONF_DIR} (default: benchmark)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist override, repeatable (e.g. --set guidance.strength=150)",
    )
    arguments = parser.parse_args()

    config = BenchmarkConfig.from_file(
        CONF_DIR / f"{arguments.config}.yaml", arguments.overrides
    )
    settings = RoomgenSettings()
    print(
        f"[config] {config.name} ({arguments.config}.yaml): "
        f"{len(config.scenes)} scenes x {len(config.seeds)} seeds on "
        f"{config.backends.kind} backends, device {config.backends.device}, "
        f"{config.backends.dtype}, guidance strength {config.guidance.strength}"
    )

    summary = run_benchmark(config, settings)
    _report(summary)
    gallery = write_gallery(summary.output_dir / "results.json")
    print(f"[gallery] {gallery}")
    return 0


_COLUMNS = (
    f"{'scene':<26}{'seed':>5}{'depth ctl':>11}{'depth gd':>11}{'Δdepth':>9}"
    f"{'adherence':>11}{'coverage':>10}{'sec gd':>9}{'sec ctl':>9}"
)


def _report(summary: Summary) -> None:
    """The run's numbers, per (scene, seed) and in the mean. No claim is made about them."""
    print(f"\n[metrics] {len(summary.results)} (scene, seed) pairs\n{_COLUMNS}")
    for result in summary.results:
        metrics = result.metrics
        print(
            f"{result.scene[:25]:<26}{result.seed:>5}"
            f"{metrics.control_depth_error:>11.5f}{metrics.guided_depth_error:>11.5f}"
            f"{_signed(metrics.depth_error_reduction):>9}"
            f"{_share(metrics.adherence_ratio):>11}"
            f"{metrics.common_coverage:>9.0%} "
            f"{result.guided_seconds:>8.1f}{result.control_seconds:>9.1f}"
        )
    print(
        f"{'mean':<26}{'':>5}"
        f"{_mean(r.metrics.control_depth_error for r in summary.results):>11.5f}"
        f"{_mean(r.metrics.guided_depth_error for r in summary.results):>11.5f}"
        f"{_signed(_optional_mean(r.metrics.depth_error_reduction for r in summary.results)):>9}"
        f"{_share(_optional_mean(r.metrics.adherence_ratio for r in summary.results)):>11}"
        f"{_mean(r.metrics.common_coverage for r in summary.results):>9.0%} "
        f"{_mean(r.guided_seconds for r in summary.results):>8.1f}"
        f"{_mean(r.control_seconds for r in summary.results):>9.1f}"
    )
    print(
        f"[run]     commit {summary.git_commit[:8]}, "
        f"{summary.walked_steps} walked steps, "
        f"device {summary.device_name or summary.config.backends.device}"
    )
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")


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
