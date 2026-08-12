# roomgen-guide

Guided sampling over a `diffusers` img2img pipeline, plus the experiment harness
shared with `experiments/`: the pydantic config base, the `ROOMGEN_*` settings
contract, run metrics, and eval metrics.

## The guidance loop

`sampling/guided.py` wraps the denoising loop. At each step it forms the
predicted clean image x̂₀ from the current latent, decodes it through the VAE,
and passes it to a `Guidance` (`guidance/base.py`) which returns a scalar
energy; `guidance/geometric.py` is the one that re-runs the depth model and
scores against a `GeometricReference` from `roomgen-geometry`. The gradient of
that energy with respect to the latent steers the sample, multiplied by one
`strength` held constant over the walk; `clip_step_rms` caps how far any single
step may move the latent. No weights are updated.

`renovate.py` is the end-to-end entry point: photo + prompt in, reference built
once, guided sample out under `ROOMGEN_WDIR`.

`eval/` scores a finished render on two axes, because they trade off: masked
depth error against the reference, and a prompt-adherence metric. A sampler that
simply returns the input photo wins the first and fails the second, and so does
one that fills the room with depth-satisfying slabs, which is why a framework
reads the two together. `shell_coverage` sits beside them as a diagnostic — how
much shell a render's own segmentation left to score, and therefore how
comparable the two depth errors are — not as a second guard (see
`experiments/README.md`).

## Settings

`RoomgenSettings` (`settings.py`) is the one env-var contract in the repo. Plain
`HF_HOME` is not read.

| Variable | Default | Meaning |
|---|---|---|
| `ROOMGEN_WDIR` | `outputs/` at the repo root (gitignored) | where every local artifact lands |
| `ROOMGEN_HF_HOME` | `.cache/huggingface/` at the repo root (gitignored) | where model weights are cached |

Never hardcode any of these in a script — set them per machine. Both defaults
are anchored to the enclosing repo root; started outside a clone, a process
must set the variable it needs rather than inherit the working directory.

## GPU

This is a torch project and guidance is expensive. The gradient backpropagates
through the depth network **and** the VAE decoder, whose activations at sample
resolution must be retained for the backward pass at every guided step, so a
guided run costs several times the VRAM of the same pipeline sampled unguided.

Measured at SD1.5's native 512px, fp32, 18 walked steps, on an RTX PRO 4500
Blackwell: **~6 s per guided sample against ~1.4 s unguided**, ~23 GiB peak
allocated with all four checkpoints resident. `differentiate_through_unet` adds
a UNet backward per guided step on top of that.

Three levers buy headroom above 512px, none of them wired to a config knob yet:
gradient-checkpoint the VAE decoder, evaluate the energy at a reduced
depth-input resolution, or guide on a subset of steps.

CPU torch is the default install, which is enough for the fake-backed tests but
not for real sampling. Layer CUDA wheels over a synced venv:

```bash
uv pip install --torch-backend=auto torch torchvision
```

Seeds are explicit: pass a `torch.Generator`, never touch the global RNG. See
the repo `CLAUDE.md` for the checks gate and the differentiability rules.
