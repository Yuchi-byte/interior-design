"""The tilt contract: what a reward gradient is turned into before it moves the state."""

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor

from roomgen_guide.process.endpoint import StepContext


@runtime_checkable
class EnergyGradient(Protocol):
    """∂energy/∂state at a candidate state: one endpoint estimate, one reward, one backward.

    Args:
        state: where to evaluate. The first state it is called at is the one
            whose model output the step's `advance` reuses, so a policy that
            iterates over candidates must evaluate the step's own state first.
    """

    def __call__(
        self, state: Float[Tensor, "batch *state"]
    ) -> Float[Tensor, "batch *state"]: ...


@runtime_checkable
class TiltPolicy(Protocol):
    """Turns the reward's pull at the endpoint into the displacement the state takes.

    Args:
        energy_gradient: called as often as the policy needs — once for a single
            gradient step, repeatedly for an inner optimisation over candidate
            states.
        state: the step's own state.
        step: where this step sits in the walk, for a policy that guides inside
            a window or stops early.

    Returns the steering `advance` adds to the state, or `None` for a step left
    alone: a policy that returns `None` without calling `energy_gradient` costs
    that step no reward evaluation at all.

    A policy carries its own knobs and nothing about the walk, so the same
    instance drives every step of it.
    """

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, "batch *state"],
        step: StepContext,
    ) -> Float[Tensor, "batch *state"] | None: ...
