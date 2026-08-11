"""The two-sided pass/fail an experiment framework gates its claim on."""

from __future__ import annotations

from dataclasses import dataclass

from roomgen_guide.config import Config


class VerdictThresholds(Config):
    """The bar a guided run has to clear, stated per framework rather than assumed.

    Args:
        depth_error_margin: how much lower the guided run's masked depth error
            must be than the unguided control's at the same seed and prompt.
        prompt_adherence_floor: the adherence a guided run may not fall below.
    """

    depth_error_margin: float
    prompt_adherence_floor: float


@dataclass(frozen=True)
class Verdict:
    """One run's score against its control, on both sides of the gate.

    Depth error alone cannot decide anything: the way to win it outright is to
    return the input photo unchanged, which is the failure the adherence side
    exists to catch.
    """

    guided_depth_error: float
    control_depth_error: float
    prompt_adherence: float
    geometry_improved: bool
    adherence_held: bool

    @property
    def passed(self) -> bool:
        return self.geometry_improved and self.adherence_held


def score_run(
    guided_depth_error: float,
    control_depth_error: float,
    prompt_adherence: float,
    thresholds: VerdictThresholds,
) -> Verdict:
    """Judge one guided run against its unguided control.

    Whether `depth_error_margin` is read as an absolute or a relative reduction
    is fixed by the first framework that measures both errors on real scenes.
    """
    raise NotImplementedError
