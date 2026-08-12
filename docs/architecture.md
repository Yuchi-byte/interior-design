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
  plus the experiment harness (config base, `ROOMGEN_*` settings, run
  metrics, eval).
- `experiments/`: numbered frameworks (not single runs) that orchestrate the
  packages and report what they measured.
- `packages/mfm`: vendored third-party source — the official Meta Flow Maps
  implementation (MIT, `packages/mfm/NOTICE.md` records upstream and commit),
  read and trained on the GPU pod for the track-2 flow-map direction. Not a
  workspace member, not imported by our packages, outside the gate.

## `roomgen-geometry`

Every model backend sits behind a `Protocol`, so a backend swap is a config
change. Layered `depth`/`segmentation` -> `masking` -> `alignment` -> `reward`
-> `reference`, each importing only from lower layers.

- `depth/base.py`: `DepthEstimator`, the `Protocol` mapping an image tensor to
  affine-invariant inverse depth.
- `depth/depth_anything.py`: the Depth Anything V2 backend
  (`depth-anything/Depth-Anything-V2-Small-hf`, DPT head on a DINOv2 backbone;
  the Small tier is the only Apache-2.0 one), frozen and in eval but never
  under `no_grad` — preprocessing is tensor ops, so the gradient reaches the
  input image.
- `segmentation/base.py`: `Segmenter`, the `Protocol` mapping an image to
  per-pixel ADE20K class ids.
- `segmentation/eomt.py`: the EoMT backend
  (`tue-mps/ade20k_semantic_eomt_large_512`, MIT-licensed DINOv2 weights), run
  once per photo under `no_grad` — segmentation is never on the gradient path —
  with a tensor-op resize/normalise and ids decoded back onto the input frame
  from the mask-query logits.
- `segmentation/ade20k.py`: the 150-class label table and the
  `ARCHITECTURAL_CLASSES`/`CONTENTS_CLASSES` partition (wall, floor, ceiling,
  door, column, stairs, ... vs. furniture, furnishings, decor; the glazing
  itself is in neither, since depth behind glass reads the view, not the room)
  — the domain-knowledge file.
- `masking.py`: `scorable_mask`, the one place a segmentation is turned into the
  pixels a reward may score, under either `MaskMode` polarity —
  `intersect_shell` keeps only affirmative shell, `subtract_contents` exempts
  only affirmative contents, so an id the partition does not name stays scored;
  `architectural_mask` is that function at `intersect_shell`, which is how the
  reference photo's shell is always read. Plus the boundary erosion, applied
  to the kept side in either mode, that drops the few pixels either side of a
  segment edge where monocular depth is unreliable — a max-pool over the
  inverted mask (Chebyshev radius in pixels); the image border is not treated as
  a segment edge.
- `alignment.py`: `align_scale_shift`, the closed-form weighted least-squares
  fit of a prediction onto a reference over the masked pixels.
- `reward.py`: `masked_depth_error`, the scale-and-shift-invariant masked
  objective.
- `reference.py`: `GeometricReference`, the frozen (masked depth, mask) pair
  computed once from the input photo and detached.

## `roomgen-guide`

The guided sampler and the experiment harness shared with `experiments/`.

- `guidance/base.py`: `Guidance`, the `Protocol` mapping a predicted clean
  image x̂₀ to a scalar energy.
- `guidance/geometric.py`: `GeometricGuidance`, the energy that decodes x̂₀,
  re-runs the depth model, and scores it against a `GeometricReference` over the
  reference's shell — intersected, whenever a segmenter is configured, with the
  shell of that segmenter's (detached, `no_grad`) view of the same x̂₀, so
  furniture the renovation puts in front of a wall is free and a wall that has
  moved is not; an optional `on_mask` hook reports the effective mask of each
  step to a caller that wants to record it.
- `sampling/pipeline.py`: `Img2ImgPipeline` and the component Protocols under
  it — the structural view of a diffusers img2img pipeline, so a weightless fake
  can stand in for one.
- `sampling/guided.py`: the guided denoising loop, stepping the pipeline's UNet,
  VAE and scheduler itself because a diffusers `__call__` runs under `no_grad`
  and would strand the guidance gradient.
- `renovate.py`: `renovate`, the end-to-end img2img entry point (photo + prompt
  -> reference -> guided sample -> image; a caller scoring against the same
  geometry hands its own reference in), and `load_pipeline`, which brings up
  the Stable Diffusion 1.5 it runs on
  (`stable-diffusion-v1-5/stable-diffusion-v1-5`, the canonical Hub mirror)
  without the safety checker the seeded cache leaves out.
- `config.py`: `Config`, the pydantic v2 `extra="forbid"` base every typed
  experiment config extends, with OmegaConf-backed YAML + dotlist loading.
- `experiment.py`: `ExperimentConfig`, the run config a framework builds from
  its `conf/` (name and the explicit seeds it runs).
- `settings.py`: `RoomgenSettings`, the one `ROOMGEN_*` env-var contract in the
  repo (`ROOMGEN_WDIR`, `ROOMGEN_HF_HOME`).
- `prefetch.py`: `roomgen-prefetch`, the CLI that seeds the Hub cache with the
  four pretrained checkpoints so later loads are instant and offline; `--check`
  re-reads it offline and requires each snapshot to hold weights and a config,
  which is a floor on the cache rather than a proof it is complete.
- `metrics.py`: `RunMetrics`, one guided run's measurements beside its
  control's at one seed, with the depth-error reduction and adherence ratio
  read off them — `None` rather than 0 where the control scored zero and the
  ratio is undefined. It records numbers; nothing here passes or fails a run.
- `determinism.py`: `enable_determinism`, the best-effort narrowing of
  run-to-run variance (deterministic algorithms `warn_only`, cuDNN
  deterministic, `CUBLAS_WORKSPACE_CONFIG`), called before the first CUDA
  allocation.
- `eval/`: what a finished render is measured on, therefore under `no_grad`:
  `depth.py`'s masked depth error against the reference and the signed
  per-pixel residual map behind it, both computed in float32 so the map still
  squares to the score — both take an evaluation mask, the
  reference shell narrowed by a segmentation of a finished render, and a
  framework comparing two arms narrows by *both* so the pair is scored on one
  pixel set, with `shell_coverage` reporting how much of the shell that left to
  score because a render scored on nothing scores zero — and the
  `adherence.py` `PromptAdherence` boundary whose
  `clip.py` implementation scores the render against the prompt — the metric
  that catches the degenerate win where guidance succeeds by not renovating.

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

    s = clamp( (S_1·S_dr − S_d·S_r) / (det + λ·S_1·S_dd),  ±1000 )
    t = (S_r − s·S_d) / S_1

`det` degenerates when the masked region is tiny or has near-constant depth.
The ridge `λ = 1e-12` and the cap on `s` are what carry the solve through that
continuously — `s` falls to 0 and the fit becomes mean-matching, rather than
jumping to an unaligned comparison — and `t` is back-substituted from the capped
`s` so the fit still matches the two masked means. Both are slack by orders on a
conditioned mask. An empty mask alone falls back to the identity `(1, 0)`.
Pixels outside the mask are *replaced* with zero rather than weighted by it,
since `0 × NaN` is NaN in both directions. The whole solve runs in float64 and
the energy in float32, whatever dtype the maps arrive in.
`masked_depth_error` is then the masked mean squared residual of the aligned
prediction,

    E = ( Σ_i w_i (s·d_i + t − r_i)² ) / Σ_i w_i

which is invariant to any affine rescaling of `d` by construction. This is the
scale-and-shift-invariant loss of MiDaS (arXiv:1907.01341) used as a reward
rather than a training objective.

Everything downstream depends on this alignment: it is what makes the reward
comparable across two different images at all.

## Guidance

At each denoising step the sampler forms the predicted clean image x̂₀
from the current latent and the model's noise prediction, decodes it through the
VAE, runs the same `DepthEstimator` on the result, and evaluates the energy `E`
above. The steering direction is the gradient of that energy **with respect to
the latent** at that step, and it is applied to the latent before the scheduler
takes its step, multiplied by one `strength` held constant over the walk and
optionally capped at a fixed step RMS (`clip_step_rms`). No weights are updated
— this is training-free guidance in the universal-guidance / FreeDoM family.

The gradient therefore backpropagates through the depth network *and* the VAE
decoder. The UNet sits outside that path, its noise prediction entering x̂₀'s
closed form as a constant, unless `differentiate_through_unet` puts it back in:
the step then also carries the denoiser's Jacobian (DPS), which damps a push the
denoiser would only undo, for one UNet backward more per guided step. The VAE
decode dominates memory: its decoder activations at full sample
resolution must be retained for the backward pass, and this cost is paid once
per guided step. Gradient checkpointing on the decoder, computing the energy at a
reduced depth-input resolution, and guiding on a subset of steps are the three
levers; expect guided sampling to cost several times the VRAM of the same
unguided pipeline. See `packages/roomgen-guide/README.md` for the practical
figures.

## Outputs

Everything (renders, masked depth maps, config snapshots) defaults
under the gitignored `outputs/` at the repo root via
`RoomgenSettings`/`ROOMGEN_WDIR`.
