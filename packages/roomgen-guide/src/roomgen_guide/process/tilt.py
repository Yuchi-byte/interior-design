"""The tilt contract: what a reward gradient is turned into before it moves the state."""

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.process.endpoint import StepContext

# Small enough to be invisible against any gradient worth stepping along, and
# there only so that an exactly zero gradient rescales to zero rather than NaN.
_RMS_EPSILON: Final = 1e-12

# The step is sized in float32 whatever the state's dtype. A guidance gradient
# runs around 1e-5 per element, and squaring that in float16 lands under the
# 6e-8 subnormal floor: the RMS underflows to exactly zero, which turns
# `clip_step_rms` into a cap that never binds. Only the sizing arithmetic is
# lifted; the update is handed back in the state's own dtype.
_UPDATE_DTYPE: Final = torch.float32


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


@dataclass(frozen=True)
class ClippedGradientTilt:
    """The raw energy gradient as the step, capped at a fixed RMS.

    The raw gradient scales with the energy, so the step decays by itself on a
    sample the energy is already satisfied on; the cap is what stops a large
    energy running away into texture that satisfies the judge without being a
    room. Nothing here is diffusion-specific: the state is whatever the process
    walks.

    Args:
        strength: what the energy gradient is multiplied by; 0 leaves every step
            unsteered and evaluates the reward not at all.
        clip_step_rms: ceiling on the RMS of **each** applied update, in the
            state's own units, sized per batch element. Below it the update
            passes through bit-for-bit; above it the update is exactly this size
            and points where the raw one pointed.
        iterations: how many gradient steps are taken on the endpoint before the
            state moves — FMRG's `n_opt`, whose inner loop is otherwise
            unconstrained ascent against a frozen judge. Each iteration
            re-evaluates the reward at the state the previous ones displaced and
            is capped in its own right, so `iterations` bounds the step at
            `iterations * clip_step_rms` rather than uncapping it.
        guided_steps: guide only the first this many steps of the walk and leave
            the rest alone — the flow-map prescription of steering in the
            high-noise window and stopping early, which leaves the trailing
            steps free to re-project the sample onto the model's own manifold.
            `None` guides every step.
    """

    strength: float
    clip_step_rms: float | None = None
    iterations: int = 1
    guided_steps: int | None = None

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")
        if self.clip_step_rms is not None and self.clip_step_rms <= 0.0:
            raise ValueError(
                f"clip_step_rms must be positive, got {self.clip_step_rms}"
            )
        if self.iterations < 1:
            raise ValueError(f"iterations must be at least 1, got {self.iterations}")
        if self.guided_steps is not None and self.guided_steps < 0:
            raise ValueError(
                f"guided_steps must be non-negative, got {self.guided_steps}"
            )

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, "batch *state"],
        step: StepContext,
    ) -> Float[Tensor, "batch *state"] | None:
        if self.strength == 0.0:
            return None
        if self.guided_steps is not None and step.index >= self.guided_steps:
            return None
        steering: Tensor | None = None
        for _ in range(self.iterations):
            # The step's own state first, so the model output `advance` reuses is
            # the one this step actually predicted from.
            candidate = state if steering is None else state + steering
            # Negated because the energy is descended: the steering is the
            # displacement the state takes, not the direction the energy rises in.
            update = -clipped_update(
                energy_gradient(candidate),
                strength=self.strength,
                clip_step_rms=self.clip_step_rms,
            )
            steering = update if steering is None else steering + update
        return steering


@dataclass(frozen=True)
class AnnealedSignTilt:
    """FMRG-J's step rule: a fresh Adam step per guided step, annealed over the window.

    What FMRG's released reward-guided sampler actually runs, against what its
    paper describes. The paper (arXiv:2604.27147, §E.2) normalises the guidance
    signal to unit norm and rescales it to the flow map's velocity magnitude; the
    code that produced its published numbers builds a **fresh** `torch.optim.Adam`
    inside every guided step at PyTorch's default betas, and that branch never
    consults the normaliser. With `iterations` 1 — every headline FMRG-J
    configuration — a freshly-constructed Adam's first step is bias-corrected to
    exactly `rate * sign(gradient)` elementwise, so the applied displacement is
    `span * rate` per element **whatever the reward's scale is**. That is the
    invariance the paper attributes to its renormaliser, delivered elementwise
    instead of globally, and it is why `strength` here is in state units per unit
    time rather than in the reward's units: it is not comparable to
    `ClippedGradientTilt.strength`, which multiplies a raw gradient.

    The optimiser is constructed per step rather than carried, so no moment
    estimate crosses a step boundary; `iterations` above 1 is the only thing that
    lets Adam's moments matter at all, and upstream leaves it at 1 for the
    Jacobian variant because each iteration costs a model call.

    Args:
        strength: FMRG's `eta`, the per-unit-time displacement each guided step
            may apply; 0 leaves every step unsteered and evaluates the reward not
            at all.
        guided_steps: guide only the first this many steps and leave the rest
            alone. `None` guides every step.
        anneal_steps: the horizon the rate decays over, `rate = strength *
            (1 - index / anneal_steps)`. It is *not* `guided_steps`: upstream
            anneals over the whole truncated grid — the guided steps plus the
            terminal jump it never guides — so the last guided step still carries
            a rate above zero. `None` follows `guided_steps`, then the walk.
        iterations: FMRG's `n_opt`, gradient steps taken on the endpoint before
            the state moves. Each one re-evaluates the reward at the state the
            previous ones displaced, and so costs a model call.
    """

    strength: float
    guided_steps: int | None = None
    anneal_steps: int | None = None
    iterations: int = 1

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")
        if self.guided_steps is not None and self.guided_steps < 0:
            raise ValueError(
                f"guided_steps must be non-negative, got {self.guided_steps}"
            )
        if self.anneal_steps is not None and self.anneal_steps < 1:
            raise ValueError(
                f"anneal_steps must be at least 1, got {self.anneal_steps}"
            )
        if self.iterations < 1:
            raise ValueError(f"iterations must be at least 1, got {self.iterations}")

    def __call__(
        self,
        energy_gradient: EnergyGradient,
        *,
        state: Float[Tensor, "batch *state"],
        step: StepContext,
    ) -> Float[Tensor, "batch *state"] | None:
        if self.strength == 0.0:
            return None
        if self.guided_steps is not None and step.index >= self.guided_steps:
            return None
        rate = self.strength * (1.0 - step.index / self._horizon(step))
        # The sizing arithmetic is float32 whatever the state's dtype, exactly as
        # `clipped_update`'s is: Adam squares the gradient, and a guidance
        # gradient around 1e-5 squares to under float16's subnormal floor.
        #
        # Cloned, and not merely detached: `optimiser.step()` writes the
        # candidate **in place**, and `.detach().to(dtype)` on a state that is
        # already float32 hands back a view of the walk's own storage — so the
        # optimisation would rewrite the state it was supposed to be proposing a
        # displacement from, and the displacement would come back zero.
        candidate = state.detach().to(_UPDATE_DTYPE).clone().requires_grad_(True)
        optimiser = torch.optim.Adam([candidate], lr=rate)
        for _ in range(self.iterations):
            # The step's own state first — bit-for-bit, since the round trip
            # through float32 is exact — so the model output `advance` reuses is
            # the one this step actually predicted from.
            gradient = energy_gradient(candidate.detach().to(state.dtype))
            candidate.grad = gradient.to(_UPDATE_DTYPE)
            optimiser.step()
        # Folded into the velocity as upstream folds it: the displacement it
        # applies is `dt * (optimised - original)`, which is `lambda_t = eta * dt`
        # of §E.2 and not a second free scale. A step that crosses less of the
        # trajectory moves the state less.
        displacement = step.span * (candidate.detach() - state.to(_UPDATE_DTYPE))
        return displacement.to(state.dtype)

    def _horizon(self, step: StepContext) -> int:
        if self.anneal_steps is not None:
            return self.anneal_steps
        return self.guided_steps if self.guided_steps else step.total


def clipped_update(
    gradient: Float[Tensor, "batch *state"],
    *,
    strength: float,
    clip_step_rms: float | None,
) -> Float[Tensor, "batch *state"]:
    """The gradient scaled by `strength`, held to `clip_step_rms`, in its own dtype."""
    update = strength * gradient.to(_UPDATE_DTYPE)
    if clip_step_rms is not None:
        # min(1, cap/RMS): a step under the cap is multiplied by exactly 1.0 and
        # so passes through bit-for-bit, and one over it lands exactly on the cap.
        scale = clip_step_rms / (_root_mean_square(update) + _RMS_EPSILON)
        update = update * scale.clamp(max=1.0)
    return update.to(gradient.dtype)


def _root_mean_square(
    tensor: Float[Tensor, "batch *state"],
) -> Float[Tensor, "batch *ones"]:
    # Per batch element, keeping the reduction broadcastable back over it: the
    # cap is stated per image, so one loud element in a batch must not shrink
    # every other element's step, nor a quiet one let a loud element through
    # above the cap.
    dimensions = tuple(range(1, tensor.ndim))
    return tensor.square().mean(dim=dimensions, keepdim=True).sqrt()
