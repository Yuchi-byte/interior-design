"""Entry point of 0002_geometric_guidance: pick a `conf/` variant, run it, assert the gate.

Glue only — the sampler, eval and gate are `roomgen_guide`'s, the run loop is
`framework/`'s. Exits non-zero when any (scene, seed) fails either side of the
two-sided gate.

    uv run --no-sync python experiments/0002_geometric_guidance/run.py --config smoke
    uv run --no-sync python experiments/0002_geometric_guidance/run.py --config benchmark
    ... --config benchmark --set guidance.strength=150
"""

from __future__ import annotations

import argparse
from pathlib import Path

from framework.benchmark import Summary, run_benchmark
from framework.config import BenchmarkConfig

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
    return 0 if summary.passed else 1


def _report(summary: Summary) -> None:
    gate = summary.config.gate
    print(
        f"\n[verdict] {len(summary.results) - len(summary.failures)}/"
        f"{len(summary.results)} runs pass the two-sided gate "
        f"(depth error down >={gate.depth_error_margin:.0%} vs control, "
        f"adherence >={gate.prompt_adherence_floor:.0%} of control)"
    )
    for failure in summary.failures:
        sides = ", ".join(
            side
            for side, held in (
                ("geometry", failure.verdict.geometry_improved),
                ("adherence", failure.verdict.adherence_held),
            )
            if not held
        )
        print(f"  FAIL {failure.scene} seed {failure.seed}: {sides}")
    if summary.peak_vram_bytes is not None:
        print(f"[vram]    peak {summary.peak_vram_bytes / 1024**3:.2f} GiB")
    print(f"[outputs] {summary.output_dir}")
    print(f"[result]  {'PASSED' if summary.passed else 'FAILED'}")


if __name__ == "__main__":
    raise SystemExit(main())
