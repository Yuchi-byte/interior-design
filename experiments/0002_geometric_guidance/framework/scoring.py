"""The two-sided gate, applied.

A local stand-in for `roomgen_guide.verdict.score_run`, which is still a
`NotImplementedError` upstream (see `report.md`, "Gaps"). The `Verdict` it
returns is the package's own type, so the day the stub lands this module is
deleted and the import moves, not the semantics.
"""

from __future__ import annotations

from roomgen_guide.experiment import VerdictGateConfig
from roomgen_guide.verdict import Verdict


def score_run(
    *,
    guided_depth_error: float,
    control_depth_error: float,
    guided_prompt_adherence: float,
    control_prompt_adherence: float,
    gate: VerdictGateConfig,
) -> Verdict:
    """Judge one guided run against its unguided control at the same seed and prompt.

    Both thresholds are fractions of the control's own scores: neither masked
    depth error (arbitrary disparity units) nor adherence has a scale that
    carries from one scene to the next.
    """
    return Verdict(
        guided_depth_error=guided_depth_error,
        control_depth_error=control_depth_error,
        guided_prompt_adherence=guided_prompt_adherence,
        control_prompt_adherence=control_prompt_adherence,
        geometry_improved=(
            guided_depth_error <= (1.0 - gate.depth_error_margin) * control_depth_error
        ),
        adherence_held=(
            guided_prompt_adherence
            >= gate.prompt_adherence_floor * control_prompt_adherence
        ),
    )
