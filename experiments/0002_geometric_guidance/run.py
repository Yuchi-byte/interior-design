"""Entry point of 0002_geometric_guidance: pick a `conf/` variant, run it, report what it measured.

Glue only — the sampler and the eval are `roomgen_guide`'s, the run loop is
`framework/`'s. A variant names its study shape in a top-level `study` key
(`benchmark` is guided-vs-control, `framing` is the prompt-framing 2x2). The
run reports numbers and renders the gallery beside them; reading them is the
owner's job, so it exits non-zero only on an actual error.

    uv run --no-sync python experiments/0002_geometric_guidance/run.py --config smoke
    uv run --no-sync python experiments/0002_geometric_guidance/run.py --config benchmark
    uv run --no-sync python experiments/0002_geometric_guidance/run.py --config framing
    ... --config benchmark --set guidance.strength=150
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from framework.alignment_study import (
    AlignmentSummary,
    arm_means,
    arm_specs,
    run_alignment_study,
)
from framework.benchmark import Summary, run_benchmark
from framework.config import (
    AlignmentStudyConfig,
    BenchmarkConfig,
    FramingConfig,
    load_study_config,
)
from framework.framing import ARMS, FramingSummary, framing_comparisons, run_framing
from framework.framing_gallery import write_framing_gallery
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

    config = load_study_config(
        CONF_DIR / f"{arguments.config}.yaml", arguments.overrides
    )
    settings = RoomgenSettings()
    print(
        f"[config] {config.name} ({arguments.config}.yaml, study {config.study}): "
        f"{len(config.scenes)} scenes x {len(config.seeds)} seeds on "
        f"{config.backends.kind} backends, device {config.backends.device}, "
        f"{config.backends.dtype}, guidance strength {config.guidance.strength}"
    )

    if isinstance(config, FramingConfig):
        framing_summary = run_framing(config, settings)
        _report_framing(framing_summary)
        gallery = write_framing_gallery(framing_summary.output_dir / "results.json")
    elif isinstance(config, AlignmentStudyConfig):
        # No HTML gallery of its own: the study's record is its results.json
        # and the renders and depth maps beside it.
        _report_alignment(run_alignment_study(config, settings))
        gallery = None
    elif isinstance(config, BenchmarkConfig):
        summary = run_benchmark(config, settings)
        _report(summary)
        gallery = write_gallery(summary.output_dir / "results.json")
    else:
        raise NotImplementedError(
            f"no runner is wired into run.py for study {config.study!r}"
        )
    if gallery is not None:
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


_FRAMING_COLUMNS = (
    f"{'scene':<26}{'seed':>5}{'arm':<22}{'depth':>9}{'adher':>8}{'pref':>8}"
    f"{'shell':>7}{'sec':>7}"
)


def _report_framing(summary: FramingSummary) -> None:
    """The 2x2's numbers, per (scene, seed, arm) and in the mean. No claim is made about them."""
    print(
        f"\n[metrics] {len(summary.results)} (scene, seed) grids x {len(ARMS)} arms"
        f"\n{_FRAMING_COLUMNS}"
    )
    for result in summary.results:
        for cell in result.arms:
            print(
                f"{result.scene[:25]:<26}{result.seed:>5}{cell.arm:<22}"
                f"{cell.depth_error:>9.4f}{cell.adherence:>8.4f}"
                f"{cell.preference:>8.4f}{cell.shell_coverage:>6.0%} {cell.seconds:>6.1f}"
            )
        print(
            f"{'':<31}{'common mask':<22}"
            f"{'':>9}{'':>8}{'':>8}{result.common_coverage:>6.0%}"
        )
    print("\n[means]   per arm, over every (scene, seed)")
    for label, comparison in framing_comparisons(summary.results).items():
        print(f"  {label}:")
        for metric, values in comparison.items():
            print(
                f"    {metric:<12} style {values['style']:.4f} -> "
                f"renovation {values['renovation']:.4f} (delta {values['delta']:+.4f})"
            )
    print(
        f"[run]     commit {summary.git_commit[:8]}, "
        f"{summary.walked_steps} walked steps, "
        f"device {summary.device_name or summary.config.backends.device}"
    )
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")


_ALIGNMENT_COLUMNS = (
    f"{'arm':<16}{'DA3 none':>11}{'DA3 shift':>11}{'DA3 ssi':>11}{'DA2 ssi':>11}"
    f"{'adherence':>11}{'coverage':>10}{'sec':>8}"
)


def _report_alignment(summary: AlignmentSummary) -> None:
    """The study's numbers per (scene, seed, arm) and in the mean. No claim is made about them."""
    specs = arm_specs(summary.config)
    print(
        f"\n[metrics] {len(summary.results)} (scene, seed) tuples x "
        f"{len(specs)} arms, one common mask each\n{_ALIGNMENT_COLUMNS}"
    )
    for result in summary.results:
        print(f"{result.scene} seed {result.seed}:")
        for cell in result.arms:
            print(
                f"{cell.arm:<16}{cell.metric_depth_errors['none']:>11.6f}"
                f"{cell.metric_depth_errors['shift']:>11.6f}"
                f"{cell.metric_depth_errors['scale_shift']:>11.6f}"
                f"{cell.judge_depth_error:>11.6f}"
                f"{cell.adherence:>11.4f}{cell.shell_coverage:>10.0%}"
                f"{cell.seconds:>8.1f}"
            )
        print(f"{'common mask':<16}{'':>44}{'':>11}{result.common_coverage:>10.0%}")
    means = arm_means(summary.results, specs)
    print("\n[means]   per arm, over every (scene, seed)")
    print(_ALIGNMENT_COLUMNS)
    for spec in specs:
        row = means[spec.arm]
        print(
            f"{spec.arm:<16}{row['metric_none']:>11.6f}{row['metric_shift']:>11.6f}"
            f"{row['metric_scale_shift']:>11.6f}{row['judge_scale_shift']:>11.6f}"
            f"{row['adherence']:>11.4f}{row['shell_coverage']:>10.0%}"
            f"{row['seconds']:>8.1f}"
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
