"""Entry point of 0002_geometric_guidance: pick a `conf/` variant, run it, report what it measured.

Glue only — the sampler and the eval are `roomgen_guide`'s, the run loop is
`framework/`'s. A variant names its study shape in a top-level `study` key
(`fmrg` is guided-vs-control on the FLUX flow map and the default, `benchmark`
is the same claim on SD1.5, `framing` is the prompt-framing 2x2). The run
reports numbers and renders the gallery beside them; reading them is the
owner's job, so it exits non-zero only on an actual error.

    uv run --no-sync python experiments/0002_geometric_guidance/run.py --config fmrg_smoke
    uv run --no-sync python experiments/0002_geometric_guidance/run.py                # fmrg
    uv run --no-sync python experiments/0002_geometric_guidance/run.py --config benchmark
    ... --config fmrg --set guidance.eta=2.5
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
    FmrgAlignmentConfig,
    FmrgConfig,
    FmrgContextConfig,
    FmrgEtaConfig,
    FmrgImageConfig,
    FmrgReduxConfig,
    FramingConfig,
    KontextConfig,
    load_study_config,
)
from framework.context_study import (
    ContextSummary,
    context_arms,
    run_context_study,
)
from framework.context_study import (
    arm_means as context_arm_means,
)
from framework.eta_study import EtaSummary, eta_arms, run_eta_study
from framework.eta_study import (
    arm_means as eta_arm_means,
)
from framework.fmrg_alignment_study import (
    AlignmentSummary as FmrgAlignmentSummary,
)
from framework.fmrg_alignment_study import (
    alignment_arms,
    run_fmrg_alignment_study,
)
from framework.fmrg_alignment_study import (
    arm_means as fmrg_alignment_arm_means,
)
from framework.fmrg_study import FmrgSummary, run_fmrg_study
from framework.framing import ARMS, FramingSummary, framing_comparisons, run_framing
from framework.framing_gallery import write_framing_gallery
from framework.gallery import write_gallery
from framework.image_study import ImageSummary, image_arms, run_image_study
from framework.image_study import arm_means as image_arm_means
from framework.kontext_study import (
    KontextSummary,
    kontext_arms,
    run_kontext_study,
)
from framework.kontext_study import (
    arm_means as kontext_arm_means,
)
from framework.redux_study import (
    ReduxSummary,
    redux_arms,
    run_redux_study,
)
from framework.redux_study import (
    arm_means as redux_arm_means,
)

from roomgen_guide.settings import RoomgenSettings

CONF_DIR = Path(__file__).resolve().parent / "conf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="fmrg",
        help=f"variant to run, by file stem in {CONF_DIR} (default: fmrg, the "
        "FLUX flow-map study — the SD1.5 claim config is `benchmark`)",
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
    pull = (
        f"etas {list(config.etas)}"
        if isinstance(config, FmrgEtaConfig)
        else f"eta {config.guidance.eta}"
        if isinstance(
            config,
            FmrgConfig
            | FmrgContextConfig
            | FmrgReduxConfig
            | FmrgAlignmentConfig
            | FmrgImageConfig
            | KontextConfig,
        )
        else f"strength {config.guidance.strength}"
    )
    print(
        f"[config] {config.name} ({arguments.config}.yaml, study {config.study}): "
        f"{len(config.scenes)} scenes x {len(config.seeds)} seeds on "
        f"{config.backends.kind} backends, device {config.backends.device}, "
        f"{config.backends.dtype}, guidance {pull}"
    )

    if isinstance(config, FmrgConfig):
        fmrg_summary = run_fmrg_study(config, settings)
        _report_fmrg(fmrg_summary)
        gallery = write_gallery(fmrg_summary.output_dir / "results.json")
    elif isinstance(config, FmrgContextConfig):
        # No HTML gallery of its own: the study's record is its results.json
        # and the renders, depth maps and residual maps beside it.
        _report_context(run_context_study(config, settings))
        gallery = None
    elif isinstance(config, FmrgReduxConfig):
        # No HTML gallery of its own: the study's record is its results.json
        # and the renders, depth maps and residual maps beside it.
        _report_redux(run_redux_study(config, settings))
        gallery = None
    elif isinstance(config, FmrgEtaConfig):
        # No HTML gallery of its own: the study's record is its results.json
        # and the renders, depth maps and residual maps beside it.
        _report_eta(run_eta_study(config, settings))
        gallery = None
    elif isinstance(config, FmrgAlignmentConfig):
        # No HTML gallery of its own: the study's record is its results.json
        # and the renders, depth maps and residual maps beside it.
        _report_fmrg_alignment(run_fmrg_alignment_study(config, settings))
        gallery = None
    elif isinstance(config, FmrgImageConfig):
        # No HTML gallery of its own: the study's record is its results.json
        # and the renders, depth maps and residual maps beside it.
        _report_image(run_image_study(config, settings))
        gallery = None
    elif isinstance(config, KontextConfig):
        # No HTML gallery of its own: the study's record is its results.json
        # and the renders, depth maps, residual maps and the endpoint strip.
        _report_kontext(run_kontext_study(config, settings))
        gallery = None
    elif isinstance(config, FramingConfig):
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


_FMRG_COLUMNS = (
    f"{'scene':<26}{'seed':>5}{'depth ctl':>11}{'depth gd':>11}{'Δdepth':>9}"
    f"{'adherence':>11}{'coverage':>10}{'sec gd':>9}{'sec ctl':>9}{'GiB gd':>8}"
)


def _report_fmrg(summary: FmrgSummary) -> None:
    """The FLUX study's numbers, per (scene, seed) and in the mean. No claim is made."""
    print(f"\n[metrics] {len(summary.results)} (scene, seed) pairs\n{_FMRG_COLUMNS}")
    for result in summary.results:
        metrics = result.metrics
        peak = result.guided_peak_vram_bytes
        print(
            f"{result.scene[:25]:<26}{result.seed:>5}"
            f"{metrics.control_depth_error:>11.5f}{metrics.guided_depth_error:>11.5f}"
            f"{_signed(metrics.depth_error_reduction):>9}"
            f"{_share(metrics.adherence_ratio):>11}"
            f"{metrics.common_coverage:>9.0%} "
            f"{result.guided_seconds:>8.1f}{result.control_seconds:>9.1f}"
            f"{'' if peak is None else f'{peak / 1024**3:>8.2f}'}"
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


_CONTEXT_COLUMNS = (
    f"{'arm':<9}{'sigma':>7}{'eff':>8}{'depth':>10}{'vs A':>9}{'adher':>9}"
    f"{'ratio A':>9}{'shell':>7}{'sec':>7}"
)


def _report_context(summary: ContextSummary) -> None:
    """The context study's numbers per (scene, seed, arm) and in the mean. No claim is made."""
    specs = context_arms(summary.config)
    print(
        f"\n[metrics] {len(summary.results)} (scene, seed) tuples x "
        f"{len(specs)} arms, one common mask each"
    )
    for result in summary.results:
        print(f"{result.scene} seed {result.seed}:\n{_CONTEXT_COLUMNS}")
        for cell in result.arms:
            print(
                f"{cell.arm:<9}{cell.start_sigma:>7.2f}"
                f"{cell.effective_first_sigma:>8.4f}{cell.depth_error:>10.5f}"
                f"{_signed(result.depth_delta_vs_baseline(cell.arm)):>9}"
                f"{cell.adherence:>9.4f}"
                f"{_share(result.adherence_ratio_vs_baseline(cell.arm)):>9}"
                f"{cell.shell_coverage:>6.0%} {cell.seconds:>6.1f}"
            )
        print(
            f"{'common mask':<24}{'':>10}{'':>9}{'':>9}{'':>9}"
            f"{result.common_coverage:>6.0%}"
        )
    means = context_arm_means(summary.results, specs)
    print("\n[means]   per arm, over every (scene, seed)")
    for spec in specs:
        row = means[spec.arm]
        print(
            f"{spec.arm:<9}{row['depth_error']:>10.5f}{row['adherence']:>9.4f}"
            f"{row['shell_coverage']:>7.0%}{row['seconds']:>7.1f}"
        )
    print(
        f"[run]     commit {summary.git_commit[:8]}, "
        f"{summary.walked_steps} walked steps, "
        f"device {summary.device_name or summary.config.backends.device}"
    )
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")


_REDUX_COLUMNS = (
    f"{'arm':<9}{'scale':>7}{'depth':>10}{'vs s0u':>9}{'adher':>9}"
    f"{'ratio':>8}{'imgsim':>9}{'pref':>8}{'remove':>8}{'shell':>7}{'sec':>7}"
)


def _report_redux(summary: ReduxSummary) -> None:
    """The Redux sweep's numbers per (scene, seed, arm) and in the mean. No claim is made."""
    specs = redux_arms(summary.config)
    print(
        f"\n[metrics] {len(summary.results)} (scene, seed) tuples x "
        f"{len(specs)} arms, one common mask each"
    )
    for result in summary.results:
        print(
            f"{result.scene} seed {result.seed} (photo {result.removal_object} "
            f"share {result.photo_removal_fraction:.1%}):\n{_REDUX_COLUMNS}"
        )
        for cell in result.arms:
            similarity = (
                "n/a"
                if cell.image_similarity is None
                else f"{cell.image_similarity:.4f}"
            )
            print(
                f"{cell.arm:<9}{cell.redux_scale:>7g}{cell.depth_error:>10.5f}"
                f"{_signed(result.depth_delta_vs_baseline(cell.arm)):>9}"
                f"{cell.adherence:>9.4f}"
                f"{_share(result.adherence_ratio_vs_baseline(cell.arm)):>8}"
                f"{similarity:>9}{cell.preference:>8.3f}"
                f"{cell.removal_fraction:>7.1%} "
                f"{cell.shell_coverage:>6.0%} {cell.seconds:>6.1f}"
            )
        print(
            f"{'common mask':<25}{'':>9}{'':>9}{'':>25}{result.common_coverage:>6.0%}"
        )
    means = redux_arm_means(summary.results, specs)
    print("\n[means]   per arm, over every (scene, seed)")
    for spec in specs:
        row = means[spec.arm]
        similarity = (
            "n/a"
            if row["image_similarity"] is None
            else f"{row['image_similarity']:.4f}"
        )
        print(
            f"{spec.arm:<9}{row['depth_error']:>10.5f}{row['adherence']:>9.4f}"
            f"{similarity:>9}{row['preference']:>8.3f}"
            f"{row['removal_fraction']:>7.1%} "
            f"{row['shell_coverage']:>7.0%}{row['seconds']:>7.1f}"
        )
    print(
        f"[run]     commit {summary.git_commit[:8]}, "
        f"{summary.walked_steps} walked steps, "
        f"device {summary.device_name or summary.config.backends.device}"
    )
    if summary.caching_peak_vram_bytes is not None:
        print(
            f"[vram]    redux caching peak "
            f"{summary.caching_peak_vram_bytes / 1024**3:.2f} GiB"
        )
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    walk peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")


_ETA_COLUMNS = (
    f"{'arm':<9}{'eta':>7}{'depth':>10}{'vs A':>9}{'adher':>9}"
    f"{'ratio A':>9}{'shell':>7}{'sec':>7}"
)


def _report_eta(summary: EtaSummary) -> None:
    """The eta sweep's numbers per (scene, seed, arm) and in the mean. No claim is made."""
    specs = eta_arms(summary.config)
    print(
        f"\n[metrics] {len(summary.results)} (scene, seed) tuples x "
        f"{len(specs)} arms, one common mask each"
    )
    for result in summary.results:
        print(f"{result.scene} seed {result.seed}:\n{_ETA_COLUMNS}")
        for cell in result.arms:
            print(
                f"{cell.arm:<9}{cell.eta:>7.1f}{cell.depth_error:>10.5f}"
                f"{_signed(result.depth_delta_vs_baseline(cell.arm)):>9}"
                f"{cell.adherence:>9.4f}"
                f"{_share(result.adherence_ratio_vs_baseline(cell.arm)):>9}"
                f"{cell.shell_coverage:>6.0%} {cell.seconds:>6.1f}"
            )
        print(f"{'common mask':<26}{'':>9}{'':>9}{result.common_coverage:>15.0%}")
    means = eta_arm_means(summary.results, specs)
    print("\n[means]   per arm, over every (scene, seed)")
    for spec in specs:
        row = means[spec.arm]
        print(
            f"{spec.arm:<9}{row['eta']:>7.1f}{row['depth_error']:>10.5f}"
            f"{row['adherence']:>9.4f}"
            f"{row['shell_coverage']:>7.0%}{row['seconds']:>7.1f}"
        )
    print(
        f"[run]     commit {summary.git_commit[:8]}, "
        f"{summary.walked_steps} walked steps, "
        f"device {summary.device_name or summary.config.backends.device}"
    )
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")


_IMAGE_COLUMNS = (
    f"{'arm':<16}{'depth':>10}{'vs ctl':>9}{'adher':>9}{'ratio':>8}"
    f"{'remove':>8}{'window':>10}{'imgsim':>9}{'pref':>9}{'shell':>7}{'sec':>7}"
)


def _report_image(summary: ImageSummary) -> None:
    """The image-reward study's numbers per (scene, seed, arm) and in the mean. No claim is made."""
    specs = image_arms(summary.config)
    print(
        f"\n[metrics] {len(summary.results)} (scene, seed) tuples x "
        f"{len(specs)} arms, one common mask each"
    )
    for result in summary.results:
        print(
            f"{result.scene} seed {result.seed} (photo {result.removal_object} "
            f"share {result.photo_removal_fraction:.1%}, windowpane "
            f"{result.window_fraction:.1%} of frame):\n{_IMAGE_COLUMNS}"
        )
        for cell in result.arms:
            similarity = (
                "n/a"
                if cell.image_similarity is None
                else f"{cell.image_similarity:.4f}"
            )
            window = (
                "n/a"
                if cell.window_depth_error is None
                else f"{cell.window_depth_error:.5f}"
            )
            print(
                f"{cell.arm:<16}{cell.depth_error:>10.5f}"
                f"{_signed(result.depth_delta_vs_control(cell.arm)):>9}"
                f"{cell.adherence:>9.4f}"
                f"{_share(result.adherence_ratio_vs_control(cell.arm)):>8}"
                f"{cell.removal_fraction:>7.1%} {window:>9}"
                f"{similarity:>9}{cell.preference:>9.3f}"
                f"{cell.shell_coverage:>6.0%} {cell.seconds:>6.1f}"
            )
        if result.sizing is not None:
            weights = ", ".join(
                f"{model} {weight:.3g}"
                for model, weight in result.sizing.weights.items()
            )
            print(f"{'measured depth weights':<25}{weights}")
        print(f"{'common mask':<25}{'':>50}{result.common_coverage:>16.0%}")
    means = image_arm_means(summary.results, specs)
    print("\n[means]   per arm, over every (scene, seed)")
    for spec in specs:
        row = means[spec.arm]
        similarity = (
            "n/a"
            if row["image_similarity"] is None
            else f"{row['image_similarity']:.4f}"
        )
        window = (
            "n/a"
            if row["window_depth_error"] is None
            else f"{row['window_depth_error']:.5f}"
        )
        print(
            f"{spec.arm:<16}{row['depth_error']:>10.5f}{row['adherence']:>9.4f}"
            f"{row['removal_fraction']:>7.1%} {window:>9}"
            f"{similarity:>9}{row['preference']:>9.3f}"
            f"{row['shell_coverage']:>7.0%}{row['seconds']:>7.1f}"
        )
    print(
        f"[run]     commit {summary.git_commit[:8]}, "
        f"{summary.walked_steps} walked steps, "
        f"device {summary.device_name or summary.config.backends.device}"
    )
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")


_KONTEXT_COLUMNS = (
    f"{'arm':<15}{'depth':>10}{'vs prior':>10}{'adher':>9}{'ratio':>8}"
    f"{'imgsim':>9}{'pref':>8}{'shell':>7}{'sec':>7}{'GiB':>7}"
)


def _report_kontext(summary: KontextSummary) -> None:
    """The pilot's numbers per (scene, seed, arm) and in the mean. No claim is made."""
    specs = kontext_arms(summary.config)
    print(
        f"\n[metrics] {len(summary.results)} (scene, seed) tuples x "
        f"{len(specs)} arms, one common mask each"
    )
    for result in summary.results:
        print(f"{result.scene} seed {result.seed}:\n{_KONTEXT_COLUMNS}")
        for cell in result.arms:
            similarity = (
                "n/a"
                if cell.image_similarity is None
                else f"{cell.image_similarity:.4f}"
            )
            peak = cell.peak_vram_bytes
            print(
                f"{cell.arm:<15}{cell.depth_error:>10.5f}"
                f"{_signed(result.depth_delta_vs_baseline(cell.arm)):>10}"
                f"{cell.adherence:>9.4f}"
                f"{_share(result.adherence_ratio_vs_baseline(cell.arm)):>8}"
                f"{similarity:>9}{cell.preference:>8.3f}"
                f"{cell.shell_coverage:>6.0%} {cell.seconds:>6.1f}"
                f"{'' if peak is None else f'{peak / 1024**3:>7.2f}'}"
            )
        print(f"{'common mask':<25}{'':>36}{result.common_coverage:>6.0%}")
        if result.fmrg_comparison is not None:
            rows = result.fmrg_comparison.get("rows")
            for row in rows if isinstance(rows, list) else []:
                assert isinstance(row, dict)
                print(
                    f"{'fmrg ' + str(row['arm']):<15}"
                    f"{float(row['depth_error_vs_photo_shell']):>10.5f}"
                    f"  (recomputed vs photo shell; source run steered vs empty room)"
                )
        if result.endpoint_strip:
            steps = ", ".join(
                f"{snapshot.step}@sigma{snapshot.sigma:.2f}"
                for snapshot in result.endpoint_strip
            )
            print(f"{'endpoint strip':<15}{steps}")
    means = kontext_arm_means(summary.results, specs)
    print("\n[means]   per arm, over every (scene, seed)")
    for spec in specs:
        row = means[spec.arm]
        similarity = (
            "n/a"
            if row["image_similarity"] is None
            else f"{row['image_similarity']:.4f}"
        )
        print(
            f"{spec.arm:<15}{row['depth_error']:>10.5f}{row['adherence']:>9.4f}"
            f"{similarity:>9}{row['preference']:>8.3f}"
            f"{row['shell_coverage']:>7.0%}{row['seconds']:>7.1f}"
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


def _report_fmrg_alignment(summary: FmrgAlignmentSummary) -> None:
    """The FMRG study's numbers per (scene, seed, arm) and in the mean. No claim is made."""
    specs = alignment_arms(summary.config)
    print(
        f"\n[metrics] {len(summary.results)} (scene, seed) tuples x "
        f"{len(specs)} arms, one common mask each\n{_ALIGNMENT_COLUMNS}"
    )
    for result in summary.results:
        print(
            f"{result.scene} seed {result.seed} "
            f"(effective first sigma {result.effective_first_sigma:.4f}):"
        )
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
    means = fmrg_alignment_arm_means(summary.results, specs)
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
    print(f"[shell]   {', '.join(summary.shell_classes.values())}")
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
