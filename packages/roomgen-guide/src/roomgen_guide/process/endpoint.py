"""The endpoint-process contract: what a generative process exposes to be steered."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@dataclass(frozen=True)
class StepContext:
    """Where a step sits in the walk, for a policy that schedules on it.

    Args:
        index: 0 on the first step of the walk, whatever noise level that is.
        total: how many steps the walk takes in all.
        span: how much of the trajectory this step crosses, in the process's own
            time units — the sigma gap for a flow map, the timestep gap over the
            training horizon for a diffusion schedule. A schedule is rarely
            uniform (FLUX warps its sigmas by sequence length), so a policy whose
            step size is a rate rather than a displacement — FMRG's
            ``lambda_t = eta * dt`` — cannot read it off `index` and `total`.
            Defaults to the whole trajectory, which is what a policy that never
            asks is implicitly using.
    """

    index: int
    total: int
    span: float = 1.0


@dataclass(frozen=True)
class Endpoint[Prediction]:
    """One step's differentiable view of the image the trajectory is heading for.

    Args:
        state: the leaf `image` is differentiable in — a detached copy of the
            state the endpoint was taken at, so a reward gradient taken against
            it is ∂energy/∂state at that step.
        image: the endpoint estimate in pixel space, NCHW, values in ``[0, 1]``,
            still attached to `state`.
        prediction: the model output `image` was formed from, opaque to
            everything but the process itself and handed back to
            :meth:`EndpointProcess.advance` so a step pays for one model call.
    """

    state: Float[Tensor, "batch *state"]
    image: Float[Tensor, "batch 3 height width"]
    prediction: Prediction


@runtime_checkable
class EndpointProcess[Prediction](Protocol):
    """A sampler as a steerable walk: a state, a differentiable endpoint estimate, a step.

    One instance drives one sample. Conditioning (a text prompt, a class id),
    the schedule, the photo a state is initialised from and the
    `torch.Generator` that is the walk's only source of randomness are all the
    implementation's own — the contract below never names them, so a backend
    conditioned on a class id and one conditioned on a caption are the same
    thing here.
    """

    @property
    def steps(self) -> int:
        """How many times the driver will advance the state."""
        ...

    def initial_state(self) -> Float[Tensor, "batch *state"]:
        """The state the walk starts from — a renoised photo, or pure noise.

        Drawn from the process's own generator, so calling it again draws a
        *different* start: that is what makes a restart a restart, and what makes
        the first of several draws the one a single-trajectory walk would have
        taken.
        """
        ...

    def step_span(self, index: int) -> float:
        """How much of the trajectory step `index` crosses — `StepContext.span`.

        In the process's own time units, and read by the tilt policy alone: the
        driver only carries it. A process whose steps are not scheduled in any
        meaningful unit returns ``1 / steps``.
        """
        ...

    def endpoint(
        self, state: Float[Tensor, "batch *state"], step: StepContext
    ) -> Endpoint[Prediction]:
        """The endpoint estimate at `state`, differentiable in it.

        Called once per reward evaluation, which is as many times per step as
        the tilt policy asks for and not at all on a step it leaves alone. The
        returned image must stay attached to `Endpoint.state`: an implementation
        that detaches anywhere returns a valid-looking image whose gradient is
        zero, and the walk proceeds looking perfectly normal.
        """
        ...

    def advance(
        self,
        state: Float[Tensor, "batch *state"],
        step: StepContext,
        *,
        prediction: Prediction | None,
        steering: Float[Tensor, "batch *state"] | None,
    ) -> Float[Tensor, "batch *state"]:
        """Take one step from `state`, after adding `steering` to it.

        Args:
            prediction: the model output of the first endpoint this step
                evaluated, so the step is not paid for twice; `None` when the
                policy evaluated no endpoint at all, which is what an unguided
                walk costs.
            steering: the displacement in state space the tilt policy asked for,
                added before the step is taken; `None` leaves the step unsteered.
        """
        ...

    def to_image(
        self, state: Float[Tensor, "batch *state"]
    ) -> Float[Tensor, "batch 3 height width"]:
        """The finished sample: the walk's final state in pixel space, values in ``[0, 1]``."""
        ...
