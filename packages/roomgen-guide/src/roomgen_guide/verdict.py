"""The two-sided pass/fail an experiment framework gates its claim on."""

from __future__ import annotations

from dataclasses import dataclass

from roomgen_guide.experiment import VerdictGateConfig


@dataclass(frozen=True)
class Verdict:
    """One run's score against its control, on both sides of the gate.

    Depth error alone cannot decide anything: the way to win it outright is to
    return the input photo unchanged, which is the failure the adherence side
    exists to catch. Both raw scores of both runs are kept, so a report can say
    by how much a run passed rather than only that it did.
    """

    guided_depth_error: float
    control_depth_error: float
    guided_prompt_adherence: float
    control_prompt_adherence: float
    geometry_improved: bool
    adherence_held: bool

    @property
    def passed(self) -> bool:
        return self.geometry_improved and self.adherence_held


def score_run(
    guided_depth_error: float,
    control_depth_error: float,
    guided_prompt_adherence: float,
    control_prompt_adherence: float,
    gate: VerdictGateConfig,
) -> Verdict:
    """Judge one guided run against its unguided control at the same seed and prompt.

    Args:
        guided_depth_error: masked depth error of the guided run, in the
            arbitrary disparity units the reward returns.
        control_depth_error: the same measurement on the control, which is what
            makes the guided figure mean anything.
        guided_prompt_adherence: higher is better, on whatever scale the
            adherence metric uses.
        control_prompt_adherence: the same metric on the control.

    Both sides of `gate` are fractions of the control's scores, never absolute
    values: neither quantity has a scale that carries from one scene to the next.
    """
    raise NotImplementedError
