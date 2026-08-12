"""What one guided run and its control measured, and the two ratios read off them."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunMetrics:
    """One guided run's measurements beside its unguided control's, at one seed.

    Nothing here is a verdict: a framework reports these numbers and the reader
    judges them. The two derived ratios are the only interpretation, and each is
    `None` rather than a number when the control it divides by scored zero — a
    control with no depth error leaves nothing to improve, a control with no
    adherence makes the metric degenerate on that scene, and a 0.0 in either
    place would read as a measured result instead of an absent one.

    Args:
        common_coverage: share of the photographed shell that both arms were
            scored on. One pixel set for both, which is what makes the two depth
            errors commensurable; the per-arm coverages beside it say which
            render narrowed it.
        reconstruction_depth_error: what a perfect geometry-preserving
            renovation still pays for the VAE round trip alone, on the
            reference's own shell — the floor these depth errors are read
            against.
    """

    guided_depth_error: float
    control_depth_error: float
    guided_prompt_adherence: float
    control_prompt_adherence: float
    reconstruction_depth_error: float
    common_coverage: float
    guided_shell_coverage: float
    control_shell_coverage: float

    @property
    def depth_error_reduction(self) -> float | None:
        """Share of the control's depth error guidance removed; `None` if it had none to remove."""
        if self.control_depth_error == 0.0:
            return None
        return 1.0 - self.guided_depth_error / self.control_depth_error

    @property
    def adherence_ratio(self) -> float | None:
        """The guided run's adherence as a share of the control's; `None` if the control scored 0."""
        if self.control_prompt_adherence == 0.0:
            return None
        return self.guided_prompt_adherence / self.control_prompt_adherence
