"""How hard to steer at each point of the denoising trajectory."""

from __future__ import annotations

import math
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
        initial: the strength at the first step, the noisiest one.
        final: the strength at the last step, and the floor the run never goes
            below in between.
    """

    initial: float
    final: float = 0.0

    def __post_init__(self) -> None:
        if self.final < 0:
            raise ValueError(f"final must be non-negative, got {self.final}")
        if self.initial < self.final:
            raise ValueError(
                f"initial must be at least final, got {self.initial} and {self.final}"
            )

    def __call__(self, step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return self.initial
        progress = step / (total_steps - 1)
        # cos²(πu/2), written half-angle: the same curve, but it lands on
        # exactly `final` at u=1 instead of a float dust above it — which is
        # what lets a `final` of 0 skip the guidance backward on the last step.
        decay = (1.0 + math.cos(math.pi * progress)) / 2.0
        return self.final + (self.initial - self.final) * decay
