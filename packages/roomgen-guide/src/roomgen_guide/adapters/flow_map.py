"""A few-step flow map as an endpoint process, started from the photographed room."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, runtime_checkable

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

type FlowMapStepping = Literal["endpoint_reuse", "two_time"]
"""Which velocity the *step* is taken off. The endpoint's is always the jump to data.

`endpoint_reuse` reuses the ``(sigma, 0)`` velocity the endpoint estimate
already paid for, so a guided step costs one model call; it is FMRG's
`flow_map1` and algebraically the linear-interpolation shortcut of its §3.
`two_time` asks the map for the ``(sigma, sigma_next)`` velocity instead, a
second call per step and FMRG's `flow_map2`. Upstream picks by budget and says
why (§E.1): reusing the endpoint is better at low NFE, and *worse* as the budget
grows, because the shortcut's error accumulates over a finer grid. The faithful
FMRG-J operating point is `two_time`, which is what its published high-NFE
numbers and roomgen's own tuned point are measured under.
"""


@dataclass(frozen=True)
class GuidanceWindow:
    """FMRG's early stop: the grid is truncated and the tail is one jump to data.

    Not the same object as a tilt policy's `guided_steps`, and the difference is
    the whole point. Stopping the *steering* leaves the walk taking every
    remaining step of the same grid; FMRG's `early_stop` **truncates the grid**,
    so the trajectory below the window is covered by a single flow-map jump from
    the stopping sigma straight to data — the walk pays one model call for the
    tail instead of `steps - early_stop` of them, and the base model gets exactly
    one chance to re-project rather than many. Both mechanisms are called early
    stopping and they are not interchangeable; this is the one the published
    results use.

    Args:
        early_stop: how many sigmas of the resolved schedule the walk visits
            before it jumps. `steps` therefore sets only the *resolution* of the
            grid, and `early_stop / steps` is the paper's `t_stop`.
        unguided_steps: how many steps that final jump is split into, all of them
            unguided. 1 is the single jump; upstream's own sweeps use 2, which
            leaves the sample a second unguided step to settle on the manifold.

    The last visited sigma's step is never guided — it is the jump — so a window
    of `early_stop` sigmas has `early_stop - 1` guided steps while the rate
    anneals over `early_stop`, exactly as upstream's does.
    """

    early_stop: int
    unguided_steps: int = 1

    def __post_init__(self) -> None:
        if self.early_stop < 2:
            raise ValueError(
                f"early_stop must be at least 2 — one guided step and the jump — "
                f"got {self.early_stop}"
            )
        if self.unguided_steps < 1:
            raise ValueError(
                f"unguided_steps must be at least 1, got {self.unguided_steps}"
            )

    @property
    def guided_steps(self) -> int:
        """How many of the truncated walk's steps a tilt policy should steer."""
        return self.early_stop - 1

    def truncate(self, sigmas: Sequence[float]) -> tuple[float, ...]:
        """The schedule the walk actually visits: the head, then the split jump."""
        if self.early_stop > len(sigmas) - 1:
            raise ValueError(
                f"early_stop {self.early_stop} is past the end of a schedule with "
                f"{len(sigmas) - 1} steps: the window has to stop inside the walk "
                f"it truncates"
            )
        head = tuple(sigmas[: self.early_stop])
        stop = head[-1]
        tail = tuple(
            stop * (1.0 - (index + 1) / self.unguided_steps)
            for index in range(self.unguided_steps)
        )
        return head + tail


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
    jump to sigma 0 decoded to pixels, so the energy is scored in image space;
    the step is the Euler step off a velocity `stepping` chooses.

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
        window: FMRG's early stop, which truncates the schedule and covers the
            rest in one jump; `None` walks the whole grid. See
            :class:`GuidanceWindow` — the tilt policy's own window is a different
            mechanism and the two are meant to be set together.
        stepping: which velocity the step is taken off, see
            :data:`FlowMapStepping`.
        differentiate_through_velocity: keep the velocity call on the gradient
            path, so the endpoint carries the flow map's Jacobian and the reward
            gradient is pulled back through the transformer — FMRG-J rather than
            FMRG-E. It is the difference the paper's Prop. 2.5 is about: the
            Jacobian annihilates the component of the reward gradient that leaves
            the data manifold, which is what a depth network's off-manifold
            optimum (texture that reads as geometry) escapes through. It costs a
            transformer backward per evaluated endpoint, so the backend must have
            gradient checkpointing on.
    """

    def __init__(
        self,
        backend: FlowMapBackend,
        photo: Float[Tensor, "batch 3 height width"],
        *,
        steps: int,
        start_sigma: float,
        generator: torch.Generator,
        window: GuidanceWindow | None = None,
        stepping: FlowMapStepping = "endpoint_reuse",
        differentiate_through_velocity: bool = False,
    ) -> None:
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        if not 0.0 < start_sigma <= 1.0:
            raise ValueError(f"start_sigma must lie in (0, 1], got {start_sigma}")
        self._backend = backend
        self._generator = generator
        self._stepping = stepping
        self._differentiate_through_velocity = differentiate_through_velocity
        with torch.no_grad():
            encoded = backend.encode(photo, generator=generator)
        # What the backend speaks at its own call boundary, read off the latent
        # it just handed back rather than asked for: the four-call protocol has
        # no reason to name a dtype, and this is the only place one is needed.
        self._backend_dtype = encoded.dtype
        self._photo_latent = encoded.to(_STATE_DTYPE)
        scheduled = tuple(
            backend.schedule(steps=steps, start_sigma=start_sigma, latent=encoded)
        )
        if len(scheduled) != steps + 1:
            raise ValueError(
                f"the backend returned {len(scheduled)} sigmas for {steps} steps; "
                "a schedule names the sigma before every step and the one after the last"
            )
        # The window truncates the grid the backend resolved, so `steps` is the
        # resolution of that grid and `self.steps` is what the walk actually takes.
        self._sigmas = scheduled if window is None else window.truncate(scheduled)
        if self._sigmas[-1] != 0.0:
            raise ValueError(
                f"the schedule ends at sigma {self._sigmas[-1]}, not 0: the final state "
                "is decoded as the sample, so the walk has to arrive at the data end"
            )

    @property
    def steps(self) -> int:
        return len(self._sigmas) - 1

    def step_span(self, index: int) -> float:
        return self._sigmas[index] - self._sigmas[index + 1]

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
        # Left attached, the velocity call puts the transformer in the graph and
        # the pullback carries the flow map's Jacobian (FMRG-J). Held constant —
        # under `no_grad` rather than detached after the fact, so its activations
        # are never kept at all — the gradient runs through the flow map's
        # arithmetic and the decoder alone (FMRG-E), which is a step's cheapest
        # possible pullback and the wrong one for a reward whose optimum is off
        # the manifold.
        # `and torch.is_grad_enabled()`: a caller that only wants the endpoint's
        # value — the warmup selection scoring a particle — asks for it under
        # `no_grad`, and a Jacobian path that overrode that would keep a
        # transformer's activations alive for a number nothing differentiates.
        with torch.set_grad_enabled(
            self._differentiate_through_velocity and torch.is_grad_enabled()
        ):
            velocity = self._velocity(differentiable_latent, sigma=sigma)
        endpoint_latent = differentiable_latent - sigma * velocity
        # The prediction is what `advance` steps off, and it is detached whatever
        # the endpoint did: a graph carried out of here would be held alive for
        # the whole step for a call that is taken under `no_grad` anyway.
        return Endpoint(
            differentiable_latent, self._decode(endpoint_latent), velocity.detach()
        )

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
            if self._stepping == "two_time":
                # A second call, at (sigma, sigma_next): what the endpoint
                # already predicted is the jump to data, which is a different
                # velocity and only an approximation of this one.
                velocity = self._velocity(state, sigma=sigma, target_sigma=next_sigma)
            else:
                velocity = (
                    self._velocity(state, sigma=sigma)
                    if prediction is None
                    else prediction
                )
            # The velocity is read at the state the walk arrived at, and the
            # steering is added after it: the guidance is a correction to the
            # trajectory, not a displacement the model is then asked to explain.
            if steering is not None:
                state = state + steering
            return state - (sigma - next_sigma) * velocity

    def to_image(
        self, state: Float[Tensor, "batch *latent"]
    ) -> Float[Tensor, "batch 3 height width"]:
        with torch.no_grad():
            return self._decode(state)

    def _velocity(
        self,
        state: Float[Tensor, "batch *latent"],
        *,
        sigma: float,
        target_sigma: float = 0.0,
    ) -> Float[Tensor, "batch *latent"]:
        # The model call is one of the two boundaries the backend's dtype exists
        # at: the transformer is handed what it runs in, and what it returns is
        # lifted straight back, so every accumulation the walk does is float32.
        velocity = self._backend.velocity(
            state.to(self._backend_dtype), sigma=sigma, target_sigma=target_sigma
        )
        return velocity.to(_STATE_DTYPE)

    def _decode(
        self, latent: Float[Tensor, "batch *latent"]
    ) -> Float[Tensor, "batch 3 height width"]:
        # The other boundary: the VAE runs in the backend's dtype, and the cast
        # into it stays on the graph, so the energy's gradient comes back to the
        # float32 leaf. The image's own dtype is the backend's to decide.
        return self._backend.decode(latent.to(self._backend_dtype))
