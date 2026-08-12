# The endpoint-process adapter contract

How a sampler is plugged into roomgen's reward steering. Implement the four
Protocols below and the geometric reward — or any weighted sum containing it —
steers your backend through the same driver Stable Diffusion runs under, with
no change to the reward, the mask, or the eval.

Everything lives in `packages/roomgen-guide/src/roomgen_guide/process/`;
`roomgen_guide.sampling.guided.LatentDiffusionProcess` is the worked example.
`roomgen_guide/adapters/` is where a third-party sampler meets this contract in
either direction: `flow_map.FlowMapProcess` is a few-step flow map brought under
the driver, and `mfm_reward.MfmGeometricReward` is the opposite direction, the
reward presented to a sampler that keeps its own loop.

## The four pieces

| Piece | Where | Contract |
|---|---|---|
| `EndpointProcess[Prediction]` | `process/endpoint.py` | your backend: a state, a differentiable endpoint estimate of it, a step |
| `Guidance` | `guidance/base.py` | the reward: predicted image in `[0, 1]` NCHW → one **energy per batch element**, low is good |
| `TiltPolicy` | `process/tilt.py` | reward gradients → the displacement the state takes |
| `sample_endpoint_process` | `process/driver.py` | the walk over the three; you do not write a loop |

The driver is the whole of the control flow:

```python
state = process.initial_state()
for index in range(process.steps):
    step = StepContext(index=index, total=process.steps)
    steering = tilt(energy_gradient_at(step), state=state, step=step)
    state = process.advance(state, step, prediction=..., steering=steering)
return process.to_image(state)
```

where `energy_gradient_at(step)(candidate)` is one call to
`process.endpoint(candidate, step)`, one call to the reward on
`Endpoint.image`, and one backward to `Endpoint.state`.

## The reward is per batch element, and the caller reduces it

A `Guidance` returns `Float[Tensor, "batch"]` and never reduces it. The driver
takes a single backward over the batch, so the driver is where the `.mean()`
happens — the same mean `strength` has always been calibrated against, which
divides each element's pull by the batch size.

The reduction belongs to the caller because it is not always a mean. MFM-G
weights `mc_samples` posterior draws against *each other* by their own scores
(`mfm/utils/steering.py` takes a `logsumexp` over them), and MFM-Search
resamples particles by reward: an energy that collapsed the batch would hand
every draw an identical weight and degenerate the estimator into a plain
average, while every test still passed and every sample still looked healthy.
`masked_depth_error` has always been per batch element; the reward boundary is
what used to throw that away.

## What an adapter must provide

One `EndpointProcess` instance drives **one sample**. It is constructed with
everything specific to that sample — the photo, the conditioning, the schedule,
the `torch.Generator` — and the protocol names none of them: a class-conditional
backend and a text-conditional one are the same thing here.

- **`steps`** — how many times the state is advanced. A few-step flow map
  returns 4; SD1.5 returns the length of its walked timestep tail.
- **`initial_state()`** — pure noise, or a photo mixed with noise. Draw from the
  generator you were constructed with; never the global RNG.
- **`endpoint(state, step) -> Endpoint`** — the sharp endpoint estimate as a
  **differentiable function of the state**, in pixel space, values in `[0, 1]`.
  Build a leaf with `state.detach().requires_grad_(True)`, form the endpoint
  from it, and return `Endpoint(leaf, image, prediction)`. Nothing on the path
  from the leaf to the image may detach: a detached graph yields a zero gradient
  and sampling proceeds looking perfectly normal.
- **`advance(state, step, *, prediction, steering)`** — add `steering` to the
  state (it is a displacement in state space, already signed to descend the
  energy) and take the step. `prediction` is the model output of the first
  endpoint this step evaluated, so reuse it rather than calling the model again;
  it is `None` when the policy left the step alone, and then the step must
  produce its own.
- **`to_image(state)`** — the finished sample in `[0, 1]`.

`Prediction` is yours: ε̂ for SD1.5, the endpoint sample or the velocity for a
flow map. The driver only carries it from `endpoint` to `advance`.

## Where the non-uniform bits go

| Concern | Where it lives |
|---|---|
| Few-step schedules, sigma/timestep spacing, `t_start` | the process's constructor |
| Text prompt, class id, CFG scale, negative prompts | the process's constructor, opaque to the protocol |
| Guidance window, early stop, per-step strength schedule | the tilt policy, off `StepContext.index` / `.total` |
| Inner optimisation (FMRG's `n_opt`) | the tilt policy, by calling `energy_gradient` more than once — `ClippedGradientTilt(iterations=n)`, which caps each of the n applied updates rather than the step they sum to |
| Jacobian of the map in the pullback | the process's `endpoint`: keep the model output attached to the leaf and autograd carries it; detach it and you get the cheap pullback (this is exactly SD1.5's `differentiate_through_unet` knob) |
| Step clipping / renormalisation | the tilt policy — `process/tilt.py`'s `ClippedGradientTilt` is the raw step with an RMS cap and is not SD-specific |
| Latent-space models | the process: decode inside `endpoint`, since the reward is always scored in pixel space |

A policy that evaluates candidate states must evaluate the step's own `state`
**first**: the first evaluation's model output is the one `advance` reuses.

## The FMRG adapter, concretely

Built, in `adapters/flow_map.py` and `adapters/fmrg_flux.py`.

- `FlowMapProcess` holds the photo, the schedule and the generator; everything
  FLUX-shaped — the transformer and its LoRA, the packed-latent geometry, the
  prompt embeddings, the resolution-dependent sigma shift — is behind the
  four-call `FlowMapBackend`, which `FluxFlowMapBackend` implements over FMRG's
  sampler.
- `initial_state`: the photo encoded and renoised to the first sigma,
  `sigma * noise + (1 - sigma) * photo`. `start_sigma` 1 discards the photo and
  is the text-to-image control, not the renovation arm — **matching a specific
  room's depth from noise is reachable mainly by texture the depth network
  misreads, which is an attack on the judge rather than a renovation.**
- `endpoint`: the two-time flow map's jump to sigma 0 from the leaf,
  VAE-decoded. The endpoint is sharp at every noise level, which is the whole
  reason for the move — no `(1-t)/t` blow-up in the high-noise window. The
  velocity enters as a constant, so a guided step costs one transformer call
  however many inner iterations the policy takes.
- `advance`: the Euler step off that same velocity to the next sigma.
- Tilt: `ClippedGradientTilt(strength, clip_step_rms, iterations, guided_steps)`
  — FMRG's `n_opt` inner loop under v1's RMS cap, inside FMRG's headline
  window: guide early, stop early. Upstream's loop has no cap at all, and its
  inner iterations are unconstrained ascent on a frozen judge; the cap is what
  separates guidance from a PGD attack on the depth network.

## An MFM adapter, concretely

- Constructor: the `SIModelWrapper`, the ImageNet VAE decode, the class label and
  CFG scale, `t_start`, the step count and the generator.
- `endpoint`: `model.X(0, 1, eps, model.v(...))` at the leaf, then
  `inverse_scaler` — the same call `mfm.utils.steering` makes, kept attached to
  the leaf instead of detached.
- `advance`: the Euler step (or the deterministic interpolation) at the next
  time, with `steering` added to the state first.
- **Sign.** MFM's `reward_fn(images)` and `reward_fn.score(images, prompts)`
  return per-image rewards where **higher is better**; a `Guidance` returns
  per-image energies where lower is better. The shapes already agree — it is
  only the sign and the temperature that do not. Under this driver no shim is
  needed —
  `GeometricGuidance` goes in unchanged. It is only the other direction, driving
  MFM's *own* sampler or its exp-tilted fine-tuning loss, that needs the sign and
  temperature converted, and that shim already exists:
  `roomgen_guide.adapters.mfm_reward.MfmGeometricReward`.

## Composing rewards

`WeightedGuidance((WeightedTerm(geometric, 1.0), WeightedTerm(extra, 0.3)))` is a
`Guidance` too, so extra rewards ride alongside the geometric one with no
sampler change. Weights carry unit conversion as well as relative pull: two
energies in different units are not comparable at weight 1.

## Proving an adapter

`packages/roomgen-guide/tests/roomgen_guide_fakes/flow_map.py` holds two
weightless doubles — `ToyFlowMapProcess`, a second `EndpointProcess` in pixel
space, and `ToyFlowMapBackend`, a `FlowMapBackend` whose model of data is a
blur, which is enough to walk the *real* `FlowMapProcess` with no FLUX in the
room. `tests/test_process.py` and `tests/test_flow_map.py` run both through the
driver with the real geometric reward. A new adapter's first test is the same
one: the energy of the guided sample is materially below the unguided sample's
at the same seed. If it is not, the gradient is detached somewhere.
