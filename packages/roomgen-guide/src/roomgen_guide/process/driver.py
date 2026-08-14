"""The one walk every backend is sampled by: endpoint, reward, tilt, advance."""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import (
    Guidance,
    require_descended_energy,
    require_per_element,
)
from roomgen_guide.process.endpoint import EndpointProcess, StepContext
from roomgen_guide.process.restart import WarmupRestarts
from roomgen_guide.process.tilt import TiltPolicy


def sample_endpoint_process[Prediction](
    process: EndpointProcess[Prediction],
    *,
    reward: Guidance,
    tilt: TiltPolicy,
    restarts: WarmupRestarts | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    """Walk `process` to a finished image, steering each step by the gradient of `reward`.

    Args:
        reward: the energy the walk descends, evaluated on the endpoint estimate
            in pixel space; a weighted sum of terms is one of these too. A reward
            declaring itself a score to ascend — MFM's adapters — is rejected
            here rather than quietly steered away from.
        tilt: what turns that energy's gradient into the step's steering, and
            what decides whether a step is steered at all.
        restarts: run the noisiest part of the walk from several draws and keep
            the one the reward likes best (FMRG's warmup particles); `None`
            walks a single trajectory, which is what one particle also does.

    No weights are updated: the only gradient taken is with respect to the state.
    """
    require_descended_energy(reward)
    total = process.steps
    if restarts is None:
        state, resume = process.initial_state(), 0
    else:
        restarts.check(total)
        state, resume = _best_particle(process, reward=reward, tilt=tilt, of=restarts)
    for index in range(resume, total):
        state = _advance(
            process,
            reward=reward,
            tilt=tilt,
            state=state,
            step=_context(process, index),
        )
    # The walk is over and the image is a product, not a variable: decoded on
    # the graph, the render would ride out of here dragging the final decode's
    # activations behind it — gigabytes at 512² — for as long as the caller
    # holds it, which is how a benchmark that keeps three arms' renders for one
    # comparison runs a 32GB card out of memory. Nothing downstream of a
    # finished render is differentiated, by the eval package's own contract.
    # `no_grad` spares building the decode's graph; the detach covers a backend
    # whose `to_image` returns the state itself rather than an op on it.
    with torch.no_grad():
        return process.to_image(state).detach()


def _context[Prediction](
    process: EndpointProcess[Prediction], index: int
) -> StepContext:
    return StepContext(index=index, total=process.steps, span=process.step_span(index))


def _advance[Prediction](
    process: EndpointProcess[Prediction],
    *,
    reward: Guidance,
    tilt: TiltPolicy,
    state: Float[Tensor, "batch *state"],
    step: StepContext,
) -> Float[Tensor, "batch *state"]:
    evaluated = _EvaluatedStep(process, reward, step, state)
    steering = tilt(evaluated, state=state, step=step)
    return process.advance(
        state, step, prediction=evaluated.prediction, steering=steering
    )


def _best_particle[Prediction](
    process: EndpointProcess[Prediction],
    *,
    reward: Guidance,
    tilt: TiltPolicy,
    of: WarmupRestarts,
) -> tuple[Float[Tensor, "batch *state"], int]:
    """The warmed-up state the reward scores lowest, and where the walk resumes.

    Each particle is an independent draw from the process — a fresh renoise of
    the photo, or fresh noise — walked through the same guided steps under the
    same policy, and scored on its endpoint estimate. Only the winner continues,
    so the cost is `particles` partial walks and one full one, and the selection
    is greedy on the very energy being steered by: it buys robustness against a
    trajectory that never took, not a better trajectory.
    """
    particles = [
        _warmed_up(process, reward=reward, tilt=tilt, steps=of.steps)
        for _ in range(of.particles)
    ]
    if len(particles) == 1:
        # Nothing to select between, and scoring it would cost an endpoint
        # evaluation that changes no decision: one particle is the single
        # trajectory, bit for bit.
        return particles[0], of.steps
    step = _context(process, of.steps)
    energies = [
        float(_endpoint_energy(process, reward=reward, state=state, step=step))
        for state in particles
    ]
    return particles[energies.index(min(energies))], of.steps


def _warmed_up[Prediction](
    process: EndpointProcess[Prediction],
    *,
    reward: Guidance,
    tilt: TiltPolicy,
    steps: int,
) -> Float[Tensor, "batch *state"]:
    state = process.initial_state()
    for index in range(steps):
        state = _advance(
            process,
            reward=reward,
            tilt=tilt,
            state=state,
            step=_context(process, index),
        )
    return state


def _endpoint_energy[Prediction](
    process: EndpointProcess[Prediction],
    *,
    reward: Guidance,
    state: Float[Tensor, "batch *state"],
    step: StepContext,
) -> Tensor:
    # Scoring only: nothing is differentiated here, so the endpoint's model call
    # is the whole cost of a particle's audition.
    with torch.no_grad():
        endpoint = process.endpoint(state, step)
        return require_per_element(reward(endpoint.image), reward=reward).mean()


@dataclass
class _EvaluatedStep[Prediction]:
    process: EndpointProcess[Prediction]
    reward: Guidance
    step: StepContext
    state: Float[Tensor, "batch *state"]
    prediction: Prediction | None = None

    def __call__(
        self, state: Float[Tensor, "batch *state"]
    ) -> Float[Tensor, "batch *state"]:
        if self.prediction is None and not torch.equal(state, self.state):
            raise ValueError(
                "a tilt policy must evaluate the step's own state before any "
                "candidate: the first evaluation's model output is what "
                "`advance` reuses, so probing a displaced state first steps the "
                "walk off a prediction taken at a state the walk was never at — "
                "with the steering itself perfectly correct and nothing to see"
            )
        # `enable_grad` rather than an ambient mode: the caller may be inside the
        # `no_grad` a diffusers pipeline runs under, and this region is the one
        # place in the walk where that would silently zero the steering.
        with torch.enable_grad():
            endpoint = self.process.endpoint(state, self.step)
            if self.prediction is None:
                self.prediction = endpoint.prediction
            # The reward is per batch element and one backward is taken over the
            # whole batch, so it is reduced here rather than by the reward. The
            # mean is what `strength` has always been calibrated against, and it
            # divides each element's pull by the batch size — batching therefore
            # weakens guidance unless `strength` is scaled with it, which is why
            # a batch is a throughput decision and not a free one. A backend that
            # weights batch members against each other by their own energies
            # (MFM-G's IWAE over posterior draws) reads the reward directly.
            energy = require_per_element(
                self.reward(endpoint.image), reward=self.reward
            ).mean()
            (gradient,) = torch.autograd.grad(energy, endpoint.state)
        return gradient
