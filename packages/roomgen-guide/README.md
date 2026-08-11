# roomgen-guide

Guided sampling over a `diffusers` img2img pipeline, plus the experiment harness
shared with `experiments/`: the pydantic config base, the `ROOMGEN_*` settings
contract, verdict scoring, the logger protocol, and eval metrics.

## The guidance loop

`sampling/guided.py` wraps the denoising loop. At each step it forms the
predicted clean image x̂₀ from the current latent, decodes it through the VAE,
and passes it to a `Guidance` (`guidance/base.py`) which returns a scalar
energy; `guidance/geometric.py` is the one that re-runs the depth model and
scores against a `GeometricReference` from `roomgen-geometry`. The gradient of
that energy with respect to the latent steers the sample, scaled by
`guidance/schedule.py` — strong early, while the layout is still being decided,
decayed late so guidance does not fight texture detail. No weights are updated.

`renovate.py` is the end-to-end entry point: photo + prompt in, reference built
once, guided sample out under `ROOMGEN_WDIR`.

`eval/` scores a run on two axes, because they trade off: masked depth error
against the reference, and a prompt-adherence / perceptual metric. A sampler
that simply returns the input photo wins the first and fails the second, which
is why `verdict.py`'s gate is two-sided (see `experiments/README.md`).

## Settings

`RoomgenSettings` (`settings.py`) is the one env-var contract in the repo. Plain
`WANDB_MODE` and `HF_HOME` are not read.

| Variable | Default | Meaning |
|---|---|---|
| `ROOMGEN_WDIR` | `outputs/` at the repo root (gitignored) | where every local artifact lands |
| `ROOMGEN_WANDB_MODE` | `disabled` | `disabled` / `offline` / `online` |
| `ROOMGEN_HF_HOME` | the platform Hub cache | where model weights are cached |

Never hardcode any of these in a script — set them per machine.

## GPU

This is a torch project and guidance is expensive. The gradient backpropagates
through the depth network **and** the VAE decoder, whose activations at sample
resolution must be retained for the backward pass at every guided step, so a
guided run costs several times the VRAM of the same pipeline sampled unguided.
Budget accordingly at 512px, and expect 1024px SDXL guidance to need the levers
below.

- Gradient-checkpoint the VAE decoder.
- Evaluate the energy at a reduced depth-input resolution.
- Guide on a subset of steps (the schedule already decays; skipping late steps
  entirely is close to free).

CPU torch is the default install, which is enough for the fake-backed tests but
not for real sampling. Layer CUDA wheels over a synced venv:

```bash
uv pip install --torch-backend=auto torch torchvision
```

Seeds are explicit: pass a `torch.Generator`, never touch the global RNG. See
the repo `CLAUDE.md` for the checks gate and the differentiability rules.
