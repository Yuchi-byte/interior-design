"""How hard to steer at each point of the denoising trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class GuidanceSchedule(Protocol):
    """Guidance strength as a function of position in the denoising trajectory.

    Args:
        step: index of the denoising step, 0 at the noisiest.
        total_steps: number of denoising steps in the run.

    The returned strength multiplies the energy gradient before it is applied to
    the latent, so 0 means the step is left unguided.
    """

    def __call__(self, step: int, total_steps: int) -> float: ...


@dataclass(frozen=True)
class ConstantStrength:
    """The same strength at every step; the control other schedules are read against."""

    strength: float

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")

    def __call__(self, step: int, total_steps: int) -> float:
        return self.strength


@dataclass(frozen=True)
class DecayingStrength:
    """Strong while the layout is still being decided, weak once it is only texture left.

    Args:
        initial: the strength at the first step.
        final: the strength at the last step.
    """

    initial: float
    final: float

    def __call__(self, step: int, total_steps: int) -> float:
        raise NotImplementedError
