"""The one walk every backend is sampled by: endpoint, reward, tilt, advance."""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.guidance.base import Guidance
from roomgen_guide.process.endpoint import EndpointProcess, StepContext
from roomgen_guide.process.tilt import TiltPolicy


def sample_endpoint_process[Prediction](
    process: EndpointProcess[Prediction],
    *,
    reward: Guidance,
    tilt: TiltPolicy,
) -> Float[Tensor, "batch 3 height width"]:
    """Walk `process` to a finished image, steering each step by the gradient of `reward`.

    Args:
        reward: the energy the walk descends, evaluated on the endpoint estimate
            in pixel space; a weighted sum of terms is one of these too.
        tilt: what turns that energy's gradient into the step's steering, and
            what decides whether a step is steered at all.

    No weights are updated: the only gradient taken is with respect to the state.
    """
    state = process.initial_state()
    total = process.steps
    for index in range(total):
        step = StepContext(index=index, total=total)
        evaluated = _EvaluatedStep(process, reward, step)
        steering = tilt(evaluated, state=state, step=step)
        state = process.advance(
            state, step, prediction=evaluated.prediction, steering=steering
        )
    return process.to_image(state)


@dataclass
class _EvaluatedStep[Prediction]:
    process: EndpointProcess[Prediction]
    reward: Guidance
    step: StepContext
    prediction: Prediction | None = None

    def __call__(
        self, state: Float[Tensor, "batch *state"]
    ) -> Float[Tensor, "batch *state"]:
        # `enable_grad` rather than an ambient mode: the caller may be inside the
        # `no_grad` a diffusers pipeline runs under, and this region is the one
        # place in the walk where that would silently zero the steering.
        with torch.enable_grad():
            endpoint = self.process.endpoint(state, self.step)
            if self.prediction is None:
                self.prediction = endpoint.prediction
            energy = self.reward(endpoint.image)
            (gradient,) = torch.autograd.grad(energy, endpoint.state)
        return gradient
