# Architecture

A map of the two packages and how `experiments/` uses them. Dependency
direction: `roomgen-geometry` is consumed by `roomgen-guide`; the geometry
package never imports the guide package, knows nothing about diffusion, and is
usable on its own as a depth/segmentation/reward library. One uv workspace, one
py3.12 venv, fully declared in the package `pyproject.toml`s (`uv sync`).

## Packages

- `packages/roomgen-geometry`: the geometric grounding stack — monocular depth,
  semantic segmentation, the architectural mask, the affine alignment, and the
  reward; torch + transformers only, no diffusers.
- `packages/roomgen-guide`: guided sampling over a `diffusers` img2img pipeline
  plus the experiment harness (config base, `ROOMGEN_*` settings, verdict
  scoring, logging, eval).
- `experiments/`: numbered frameworks (not single runs) that orchestrate the
  packages and assert pass/fail verdicts in code.

## `roomgen-geometry`

Every model backend sits behind a `Protocol`, so a backend swap is a config
change. Layered `depth`/`segmentation` -> `masking` -> `alignment` -> `reward`
-> `reference`, each importing only from lower layers.

- `depth/base.py`: `DepthEstimator`, the `Protocol` mapping an image tensor to
  affine-invariant inverse depth.
- `depth/depth_anything.py`: the Depth Anything V2 backend
  (`depth-anything/Depth-Anything-V2-Small-hf`, DPT head on a DINOv2 backbone;
  the Small tier is the only Apache-2.0 one), kept differentiable end to end
  including preprocessing.
- `segmentation/base.py`: `Segmenter`, the `Protocol` mapping an image to
  per-pixel ADE20K class ids.
- `segmentation/eomt.py`: the EoMT backend
  (`tue-mps/ade20k_semantic_eomt_large_512`, MIT-licensed DINOv2 weights), run
  once per photo under `no_grad` — segmentation is never on the gradient path.
- `segmentation/ade20k.py`: the 150-class label table and the
  `SHELL`/`CONTENTS` partition (wall, floor, ceiling, window, door, column,
  stairs, ... vs. furniture, furnishings, decor) — the domain-knowledge file.
- `masking.py`: `architectural_mask`, the boolean shell mask, plus the boundary
  erosion that drops the few pixels either side of a segment edge where
  monocular depth is unreliable.
- `alignment.py`: `align_scale_shift`, the closed-form weighted least-squares
  fit of a prediction onto a reference over the masked pixels.
- `reward.py`: `masked_depth_error`, the scale-and-shift-invariant masked
  objective, plus an optional multi-scale gradient-matching term that penalises
  disagreement in depth *gradients* across image pyramid levels.
- `reference.py`: `GeometricReference`, the frozen (masked depth, mask) pair
  computed once from the input photo and detached.
- `config.py`: the pydantic v2 configs for the backends and the reward.

## `roomgen-guide`

The guided sampler and the experiment harness shared with `experiments/`.

- `guidance/base.py`: `Guidance`, the `Protocol` mapping a predicted clean
  image x̂₀ to a scalar energy.
- `guidance/geometric.py`: `GeometricGuidance`, the energy that decodes
  x̂₀, re-runs the depth model, and scores it against a
  `GeometricReference`.
- `guidance/schedule.py`: guidance-strength schedules over the timestep — strong
  early while layout is still being decided, decayed late so texture detail is
  not fought.
- `sampling/pipeline.py`: `Img2ImgPipeline` and the component Protocols under
  it — the structural view of a diffusers img2img pipeline, so a weightless fake
  can stand in for one.
- `sampling/guided.py`: the guided denoising loop, stepping the pipeline's UNet,
  VAE and scheduler itself because a diffusers `__call__` runs under `no_grad`
  and would strand the guidance gradient.
- `renovate.py`: `renovate`, the end-to-end img2img entry point (photo + prompt
  -> reference -> guided sample -> outputs), over Stable Diffusion 1.5
  (`stable-diffusion-v1-5/stable-diffusion-v1-5`, the canonical Hub mirror).
- `config.py`: `Config`, the pydantic v2 `extra="forbid"` base every typed
  experiment config extends, with OmegaConf-backed YAML + dotlist loading.
- `experiment.py`: `ExperimentConfig`, the run config a framework builds from
  its `conf/` (name, explicit seeds, and the two-sided gate's thresholds).
- `settings.py`: `RoomgenSettings`, the one `ROOMGEN_*` env-var contract in the
  repo (`ROOMGEN_WDIR`, `ROOMGEN_WANDB_MODE`, `ROOMGEN_HF_HOME`).
- `verdict.py`: `score_run` and the `Verdict` it returns — one guided run's
  depth error and prompt adherence judged against its unguided control's, both
  sides at once, on the `VerdictGateConfig` its framework declares.
- `logging/`: `Logger` protocol plus its `console`/`null`/`wandb`
  implementations, so sampler code never imports wandb directly.
- `eval/`: masked depth error against the reference, plus a prompt-adherence /
  perceptual metric that catches the degenerate win where guidance succeeds by
  simply not changing the image.

## Test doubles

- `packages/roomgen-geometry/tests/roomgen_fakes/`: weightless `DepthEstimator`
  and `Segmenter` implementations (a floor-shaped disparity ramp coupled to
  pixel content; a ceiling/wall/floor room with a window and a sofa), the
  shared gradient assertion, and the fixtures exposing them. Both suites import
  it via the root pytest `pythonpath`, so the default suite needs no network,
  no GPU and no Hub cache.
- `packages/roomgen-guide/tests/roomgen_guide_fakes/`: the matching doubles for
  the diffusion side — a UNet, VAE and DDIM scheduler satisfying
  `Img2ImgPipeline` on the SD1.5 latent geometry — kept here rather than beside
  the geometry doubles, which know nothing about diffusion.
- The root `conftest.py` installs jaxtyping's runtime type-check hook over
  `roomgen_geometry`, so every shape annotation in that package is enforced for
  the duration of the suite.

## The reward

Depth Anything V2 predicts **affine-invariant inverse depth**: the output is
determined only up to an unknown scale and shift, and that pair is not stable
between two different images. Comparing two raw predictions is therefore
meaningless, and the reward must remove the affine degree of freedom first.

Let `d` be the prediction on the current sample, `r` the reference, and `w` the
shell mask (0/1, or soft weights). `align_scale_shift` solves

    min_{s,t}  Σ_i w_i (s·d_i + t − r_i)²

whose normal equations are the 2x2 system

    [ Σw d²   Σw d ] [ s ]   [ Σw d r ]
    [ Σw d    Σw   ] [ t ] = [ Σw r   ]

with the closed-form solution (writing `S_dd = Σ w d²`, `S_d = Σ w d`,
`S_1 = Σ w`, `S_dr = Σ w d r`, `S_r = Σ w r`, `det = S_dd·S_1 − S_d²`):

    s = (S_1·S_dr − S_d·S_r) / det
    t = (S_dd·S_r  − S_d·S_dr) / det

`det` degenerates when the masked region is tiny or has near-constant depth;
the solve guards on it rather than propagating a NaN into the gradient.
`masked_depth_error` is then the masked mean squared residual of the aligned
prediction,

    E = ( Σ_i w_i (s·d_i + t − r_i)² ) / Σ_i w_i

which is invariant to any affine rescaling of `d` by construction. This is the
scale-and-shift-invariant loss of MiDaS (arXiv:1907.01341) used as a reward
rather than a training objective. The optional gradient-matching term adds the
same masked mean over `|∇(s·d + t) − ∇r|` at several pyramid levels, which
sharpens planarity and edge alignment that a pure value error tolerates.

Everything downstream depends on this alignment: it is what makes the reward
comparable across two different images at all.

## Guidance

At each denoising step the sampler forms the predicted clean image x̂₀
from the current latent and the model's noise prediction, decodes it through the
VAE, runs the same `DepthEstimator` on the result, and evaluates the energy `E`
above. The steering direction is the gradient of that energy **with respect to
the latent** at that step, and it is applied to the latent before the scheduler
takes its step, scaled by `guidance/schedule.py`. No weights are updated — this
is training-free guidance in the universal-guidance / FreeDoM family.

The gradient therefore backpropagates through the depth network *and* the VAE
decoder — but not the UNet, whose noise prediction enters x̂₀'s closed form as a
constant. The VAE decode dominates memory: its decoder activations at full sample
resolution must be retained for the backward pass, and this cost is paid once
per guided step. Gradient checkpointing on the decoder, computing the energy at a
reduced depth-input resolution, and guiding on a subset of steps are the three
levers; expect guided sampling to cost several times the VRAM of the same
unguided pipeline. See `packages/roomgen-guide/README.md` for the practical
figures.

## Outputs

Everything (renders, masked depth maps, config snapshots, wandb runs) defaults
under the gitignored `outputs/` at the repo root via
`RoomgenSettings`/`ROOMGEN_WDIR`.
