"""A weightless flow-map sampler: the second `EndpointProcess`, in pixel space.

Nothing here is Stable Diffusion — no latent space, no VAE, no epsilon
parameterisation, no scheduler. The state is the image itself, the endpoint is
the flow map's jump to t=1 rather than a Tweedie estimate, and the step is an
interpolation off that endpoint. What it shares with the real sampler is the
contract, which is the point.
"""

from typing import Final

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from roomgen_guide.process.endpoint import Endpoint, StepContext

# Where the walk starts on the interpolant: t=0 is noise, t=1 is data, as the
# flow-map samplers roomgen guides are parameterised.
START_TIME: Final = 0.2

# Divisor floor for the toy flow map's velocity, which is an endpoint jump
# divided by the time it spans.
_MINIMUM_SIGMA: Final = 1e-3


class ToyFlowMapProcess:
    """A few-step flow map over a photo, conditioned on a class id.

    Args:
        photo: NCHW, floating point, values in ``[0, 1]``; mixed with noise at
            `START_TIME` to start the walk.
        class_label: the conditioning, which the velocity field reads and the
            protocol never sees.
        generator: the only source of randomness in the walk.
    """

    def __init__(
        self,
        photo: Float[Tensor, "batch 3 height width"],
        *,
        class_label: int,
        steps: int,
        generator: torch.Generator,
    ) -> None:
        self._photo = photo
        self._class_label = class_label
        self._generator = generator
        self._times = torch.linspace(START_TIME, 1.0, steps + 1, dtype=photo.dtype)
        self.model_calls = 0

    @property
    def steps(self) -> int:
        return len(self._times) - 1

    def initial_state(self) -> Float[Tensor, "batch 3 height width"]:
        noise = torch.randn(
            self._photo.shape,
            generator=self._generator,
            device=self._photo.device,
            dtype=self._photo.dtype,
        )
        return START_TIME * self._photo + (1.0 - START_TIME) * noise

    def endpoint(
        self, state: Float[Tensor, "batch 3 height width"], step: StepContext
    ) -> Endpoint[Float[Tensor, "batch 3 height width"]]:
        differentiable_state = state.detach().requires_grad_(True)
        sample = self._flow_map(differentiable_state, step)
        return Endpoint(differentiable_state, _to_image(sample), sample.detach())

    def advance(
        self,
        state: Float[Tensor, "batch 3 height width"],
        step: StepContext,
        *,
        prediction: Float[Tensor, "batch 3 height width"] | None,
        steering: Float[Tensor, "batch 3 height width"] | None,
    ) -> Float[Tensor, "batch 3 height width"]:
        with torch.no_grad():
            sample = self._flow_map(state, step) if prediction is None else prediction
            if steering is not None:
                state = state + steering
            time, next_time = self._times[step.index], self._times[step.index + 1]
            # The same endpoint the reward was scored on, re-interpolated: the
            # noise the state implies against it is carried to the next time.
            noise = (state - time * sample) / (1.0 - time)
            return next_time * sample + (1.0 - next_time) * noise

    def to_image(
        self, state: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch 3 height width"]:
        return _to_image(state)

    @jaxtyped(typechecker=beartype)
    def _flow_map(
        self, state: Float[Tensor, "batch 3 height width"], step: StepContext
    ) -> Float[Tensor, "batch 3 height width"]:
        self.model_calls += 1
        time = self._times[step.index]
        velocity = torch.tanh(state + self._class_label / 1000.0)
        return state + (1.0 - time) * velocity


class ToyFlowMapBackend:
    """A weightless `FlowMapBackend`, so the real adapter can be walked without FLUX.

    The latent is the image in the VAE's ``[-1, 1]`` convention, the velocity
    field is a smooth function of it, and the schedule is a plain descent to 0 —
    everything the real backend does that a fake cannot (packed latents, a
    resolution-dependent shift, a transformer) is exactly what
    `FlowMapProcess` is not allowed to know about.

    Args:
        class_label: stands for the conditioning the process never sees.
        encode_jitter: how much of the encode is the generator's, so a test can
            tell whether the walk's seed reaches the initial state.
    """

    def __init__(self, *, class_label: int = 7, encode_jitter: float = 0.05) -> None:
        self._class_label = class_label
        self._encode_jitter = encode_jitter
        self.velocity_calls = 0

    def schedule(
        self,
        *,
        steps: int,
        start_sigma: float,
        latent: Float[Tensor, "batch 3 height width"],
    ) -> list[float]:
        del latent
        return torch.linspace(start_sigma, 0.0, steps + 1).tolist()

    def encode(
        self,
        image: Float[Tensor, "batch 3 height width"],
        *,
        generator: torch.Generator,
    ) -> Float[Tensor, "batch 3 height width"]:
        posterior = torch.randn(
            image.shape, generator=generator, device=image.device, dtype=image.dtype
        )
        return image * 2.0 - 1.0 + self._encode_jitter * posterior

    def decode(
        self, latent: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch 3 height width"]:
        return _to_image(latent)

    @jaxtyped(typechecker=beartype)
    def velocity(
        self,
        latent: Float[Tensor, "batch 3 height width"],
        *,
        sigma: float,
        target_sigma: float,
    ) -> Float[Tensor, "batch 3 height width"]:
        self.velocity_calls += 1
        # The velocity a flow map reports is the one whose jump lands on data:
        # `state - sigma * velocity` is this toy's endpoint estimate, and its
        # notion of data is "smooth", so the estimate is the state blurred. That
        # is enough to make the walk a denoiser — the arm that starts from the
        # renoised photo keeps the photo's structure and the arm that starts
        # from noise cannot recover it, which is the whole img2img claim.
        endpoint = _smoothed(latent) + self._class_label / 1000.0
        return (latent - endpoint) / max(sigma - target_sigma, _MINIMUM_SIGMA)


def _smoothed(
    latent: Float[Tensor, "batch 3 height width"],
) -> Float[Tensor, "batch 3 height width"]:
    # A 3x3 box blur: the toy's entire model of what data looks like.
    return torch.nn.functional.avg_pool2d(latent, 3, stride=1, padding=1)


def _to_image(
    sample: Float[Tensor, "batch 3 height width"],
) -> Float[Tensor, "batch 3 height width"]:
    # tanh rather than a clamp, as the pipeline fake's decoder does: no pixel
    # saturates, so a zero gradient is a real finding rather than a rail.
    return (torch.tanh(sample) + 1.0) / 2.0
