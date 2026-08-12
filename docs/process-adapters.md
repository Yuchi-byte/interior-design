# The endpoint-process adapter contract

How a sampler is plugged into roomgen's reward steering. Implement the four
Protocols below and the geometric reward — or any weighted sum containing it —
steers your backend through the same driver Stable Diffusion runs under, with
no change to the reward, the mask, or the eval.

Everything lives in `packages/roomgen-guide/src/roomgen_guide/process/`;
`roomgen_guide.sampling.guided.LatentDiffusionProcess` is the worked example.
The reverse direction — roomgen's reward presented to a third-party sampler's
own interface, rather than a third-party sampler brought under this driver —
lives in `roomgen_guide/adapters/`.

## The four pieces

| Piece | Where | Contract |
|---|---|---|
| `EndpointProcess[Prediction]` | `process/endpoint.py` | your backend: a state, a differentiable endpoint estimate of it, a step |
| `Guidance` | `guidance/base.py` | the reward: predicted image in `[0, 1]` NCHW → scalar **energy**, low is good |
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
| Inner optimisation (FMRG's `n_opt`) | the tilt policy, by calling `energy_gradient` more than once |
| Jacobian of the map in the pullback | the process's `endpoint`: keep the model output attached to the leaf and autograd carries it; detach it and you get the cheap pullback (this is exactly SD1.5's `differentiate_through_unet` knob) |
| Step clipping / renormalisation | the tilt policy — `ClippedGradientTilt` is the raw step with an RMS cap and is not SD-specific |
| Latent-space models | the process: decode inside `endpoint`, since the reward is always scored in pixel space |

A policy that evaluates candidate states must evaluate the step's own `state`
**first**: the first evaluation's model output is the one `advance` reuses.

## An FMRG adapter, concretely

- Constructor: the FLUX transformer and its VAE, the packed-latent geometry, the
  prompt embeddings, the sigma schedule and the generator.
- `initial_state`: the photo encoded and renoised to the first sigma, or pure
  noise.
- `endpoint`: the two-time flow map's jump to t=1 from the leaf, VAE-decoded.
  The endpoint is sharp at every noise level, which is the whole reason for the
  move — no `(1-t)/t` blow-up in the high-noise window.
- `advance`: the interpolation off that same endpoint at the next sigma.
- Tilt: a policy with `n_opt` inner iterations and a high-noise window
  (`step.index < window * step.total`), which is FMRG's headline prescription —
  guide early, stop early.

## An MFM adapter, concretely

- Constructor: the `SIModelWrapper`, the ImageNet VAE decode, the class label and
  CFG scale, `t_start`, the step count and the generator.
- `endpoint`: `model.X(0, 1, eps, model.v(...))` at the leaf, then
  `inverse_scaler` — the same call `mfm.utils.steering` makes, kept attached to
  the leaf instead of detached.
- `advance`: the Euler step (or the deterministic interpolation) at the next
  time, with `steering` added to the state first.
- **Sign.** MFM's `reward_fn(images)` and `reward_fn.score(images, prompts)`
  return per-image rewards where **higher is better**; a `Guidance` is a
  **scalar energy** where lower is better. Under this driver no shim is needed —
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

`packages/roomgen-guide/tests/roomgen_guide_fakes/flow_map.py` is a weightless
second implementation — pixel-space state, flow-map endpoint, interpolated step,
no VAE — and `tests/test_process.py` runs it through the driver with the real
geometric reward. A new adapter's first test is the same one: the energy of the
guided sample is materially below the unguided sample's at the same seed. If it
is not, the gradient is detached somewhere.
