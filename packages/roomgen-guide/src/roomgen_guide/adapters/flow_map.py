"""A few-step flow map as an endpoint process, started from the photographed room."""

from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_guide.process.endpoint import Endpoint, StepContext

# The walk's state is float32 whatever the backend runs in. Steering is small by
# construction — a capped update is a fraction of a percent of the state's RMS —
# and bfloat16, the dtype every flow-map checkpoint here loads in, carries an ulp
# of about 8e-3 at O(1). Accumulating the state in it rounds any steering below
# roughly 2e-3 RMS away entirely: `state + steering` comes back bit-identical to
# `state`, the guided walk is `torch.equal` to the unguided one, and nothing
# anywhere reports a problem — the gradient was finite, the cap never bound, the
# loop ran. Only the two places the backend is actually called are cast down.
_STATE_DTYPE: Final = torch.float32


@runtime_checkable
class FlowMapBackend(Protocol):
    """A two-time flow-map sampler, reduced to the four calls a walk needs.

    The convention is the rectified-flow one the flow-map checkpoints are
    trained in: sigma 1 is noise, sigma 0 is data, and a state on the
    interpolant is ``sigma * noise + (1 - sigma) * data``. Everything the
    backend owns — the transformer, its LoRA, the prompt embeddings, the packed
    latent geometry, the shift the schedule is warped by — is invisible here.
    """

    def schedule(
        self,
        *,
        steps: int,
        start_sigma: float,
        latent: Float[Tensor, "batch *latent"],
    ) -> Sequence[float]:
        """The `steps` + 1 sigmas the walk visits, descending from `start_sigma` to 0.

        Args:
            start_sigma: where the walk begins, in ``(0, 1]``; the returned
                schedule may warp it, and the first entry — not this argument —
                is the sigma the state is actually initialised at.
            latent: what the walk is over, for a backend whose schedule depends
                on the sequence length (FLUX's resolution-dependent shift).
        """
        ...

    def encode(
        self,
        image: Float[Tensor, "batch 3 height width"],
        *,
        generator: torch.Generator,
    ) -> Float[Tensor, "batch *latent"]:
        """The photo in latent space, values in ``[0, 1]`` on the way in."""
        ...

    def decode(
        self, latent: Float[Tensor, "batch *latent"]
    ) -> Float[Tensor, "batch 3 height width"]:
        """A latent back to an image in ``[0, 1]``, **still on the autograd graph**."""
        ...

    def velocity(
        self,
        latent: Float[Tensor, "batch *latent"],
        *,
        sigma: float,
        target_sigma: float,
    ) -> Float[Tensor, "batch *latent"]:
        """The flow map's velocity at `latent`, from `sigma` towards `target_sigma`.

        `target_sigma` 0 asks the two-time map for the jump all the way to data,
        which is what makes the endpoint estimate sharp at every noise level and
        is the whole reason a flow map is here rather than an ε-predictor.
        """
        ...


class FlowMapProcess:
    """A few-step flow map over one photo and one prompt, as an `EndpointProcess`.

    The state is the sampler's latent; the endpoint estimate is the flow map's
    jump to sigma 0 decoded to pixels, so the energy is scored in image space
    and its gradient reaches the latent through the decoder alone; the step is
    the Euler step off that same jump, so a guided step pays for one model call.

    **Initialisation is the renovation setting.** The photo is encoded and
    renoised to the first sigma of the schedule —
    ``sigma * noise + (1 - sigma) * photo``, the rectified-flow SDEdit — so the
    walk starts on the room it is meant to keep and the reward's job is to hold
    geometry that is already correct. Starting from pure noise instead makes
    matching a *specific* room's depth reachable only by inventing texture the
    depth network reads as that room, which is an adversarial attack on the
    judge rather than a renovation; `start_sigma` 1 is exactly that arm, kept
    because it is the control the img2img claim is made against.

    One instance drives one sample: it holds that sample's photo, schedule and
    generator, and the prompt lives in the backend.

    **The state is float32 whatever the backend runs in**, and only the velocity
    call and the decode are cast down to it — see `_STATE_DTYPE`.

    Args:
        photo: NCHW, floating point, values in ``[0, 1]``.
        steps: how many flow-map steps the walk takes; the NFE budget, since a
            guided step reuses the model call its endpoint already made.
        start_sigma: how far up the interpolant the photo is renoised, in
            ``(0, 1]``; 1 discards the photo and generates from noise. It is the
            *pre-shift* request, not the sigma the walk starts at: a backend
            whose schedule is resolution-warped moves it, and FLUX's is —
            `start_sigma` 0.6 begins at sigma 0.826 at 1024², so the renovation
            arm keeps less of the photo than the number suggests. The first
            entry of `schedule` is the truth.
        generator: the only source of randomness in the walk — the VAE's
            encoding sample and the renoise — so a run is reproducible from its
            seed alone. It must live on the device the latents do.
    """

    def __init__(
        self,
        backend: FlowMapBackend,
        photo: Float[Tensor, "batch 3 height width"],
        *,
        steps: int,
        start_sigma: float,
        generator: torch.Generator,
    ) -> None:
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        if not 0.0 < start_sigma <= 1.0:
            raise ValueError(f"start_sigma must lie in (0, 1], got {start_sigma}")
        self._backend = backend
        self._generator = generator
        with torch.no_grad():
            encoded = backend.encode(photo, generator=generator)
        # What the backend speaks at its own call boundary, read off the latent
        # it just handed back rather than asked for: the four-call protocol has
        # no reason to name a dtype, and this is the only place one is needed.
        self._backend_dtype = encoded.dtype
        self._photo_latent = encoded.to(_STATE_DTYPE)
        self._sigmas = tuple(
            backend.schedule(steps=steps, start_sigma=start_sigma, latent=encoded)
        )
        if len(self._sigmas) != steps + 1:
            raise ValueError(
                f"the backend returned {len(self._sigmas)} sigmas for {steps} steps; "
                "a schedule names the sigma before every step and the one after the last"
            )
        if self._sigmas[-1] != 0.0:
            raise ValueError(
                f"the schedule ends at sigma {self._sigmas[-1]}, not 0: the final state "
                "is decoded as the sample, so the walk has to arrive at the data end"
            )

    @property
    def steps(self) -> int:
        return len(self._sigmas) - 1

    def initial_state(self) -> Float[Tensor, "batch *latent"]:
        latent = self._photo_latent
        noise = torch.randn(
            latent.shape,
            generator=self._generator,
            device=latent.device,
            dtype=_STATE_DTYPE,
        )
        sigma = self._sigmas[0]
        return sigma * noise + (1.0 - sigma) * latent

    def endpoint(
        self, state: Float[Tensor, "batch *latent"], step: StepContext
    ) -> Endpoint[Float[Tensor, "batch *latent"]]:
        sigma = self._sigmas[step.index]
        differentiable_latent = state.detach().requires_grad_(True)
        # The velocity enters the endpoint as a constant, so the gradient runs
        # through the flow map's arithmetic and the decoder but not the
        # transformer — the cheap pullback, and the one that keeps a guided step
        # at a single model call. Under `no_grad` rather than detached after the
        # fact so the transformer's activations are never kept at all.
        with torch.no_grad():
            velocity = self._velocity(differentiable_latent, sigma=sigma)
        endpoint_latent = differentiable_latent - sigma * velocity
        return Endpoint(differentiable_latent, self._decode(endpoint_latent), velocity)

    def advance(
        self,
        state: Float[Tensor, "batch *latent"],
        step: StepContext,
        *,
        prediction: Float[Tensor, "batch *latent"] | None,
        steering: Float[Tensor, "batch *latent"] | None,
    ) -> Float[Tensor, "batch *latent"]:
        sigma, next_sigma = self._sigmas[step.index], self._sigmas[step.index + 1]
        with torch.no_grad():
            velocity = (
                self._velocity(state, sigma=sigma) if prediction is None else prediction
            )
            if steering is not None:
                state = state + steering
            return state - (sigma - next_sigma) * velocity

    def to_image(
        self, state: Float[Tensor, "batch *latent"]
    ) -> Float[Tensor, "batch 3 height width"]:
        with torch.no_grad():
            return self._decode(state)

    def _velocity(
        self, state: Float[Tensor, "batch *latent"], *, sigma: float
    ) -> Float[Tensor, "batch *latent"]:
        # The model call is one of the two boundaries the backend's dtype exists
        # at: the transformer is handed what it runs in, and what it returns is
        # lifted straight back, so every accumulation the walk does is float32.
        velocity = self._backend.velocity(
            state.to(self._backend_dtype), sigma=sigma, target_sigma=0.0
        )
        return velocity.to(_STATE_DTYPE)

    def _decode(
        self, latent: Float[Tensor, "batch *latent"]
    ) -> Float[Tensor, "batch 3 height width"]:
        # The other boundary: the VAE runs in the backend's dtype, and the cast
        # into it stays on the graph, so the energy's gradient comes back to the
        # float32 leaf. The image's own dtype is the backend's to decide.
        return self._backend.decode(latent.to(self._backend_dtype))
