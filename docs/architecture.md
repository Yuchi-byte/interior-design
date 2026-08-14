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
- `packages/fmrg`: vendored third-party source — the official Flow Map Reward
  Guidance implementation (Apache-2.0, `packages/fmrg/NOTICE.md` records
  upstream and commit), read and run on the GPU pod for the track-1 FLUX
  flow-map direction. Not a workspace member, not imported by our packages,
  outside the gate.

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
  per-pixel ADE20K class ids, and `ClassLogitSegmenter`, the one that also
  exposes the per-class scores those ids are the argmax of — the mask path is a
  discrete decision and stays a constant, the score path is smooth and is the
  only thing on the segmentation side a reward can differentiate.
- `segmentation/eomt.py`: the EoMT backend
  (`tue-mps/ade20k_semantic_eomt_large_512`, MIT-licensed DINOv2 weights), with
  a tensor-op resize/normalise and a tensor-op semantic decode (upstream's, in
  four lines, rather than the image processor's — which returns only ids and
  whose handling of a directly-called model varies by transformers version)
  feeding its two paths: `class_logits` returns the per-class scores *with*
  gradient, at the cost of a forward and a backward through the encoder per
  guided step it is evaluated on, and `__call__` is their argmax under
  `no_grad`. Weights are frozen, so that backward accumulates nothing.
- `segmentation/ade20k.py`: the 150-class label table and the
  `ARCHITECTURAL_CLASSES`/`CONTENTS_CLASSES` partition (wall, floor, ceiling,
  door, column, stairs, ... vs. furniture, furnishings, decor; the glazing
  itself is in neither, since depth behind glass reads the view, not the room)
  — the domain-knowledge file. `STRUCTURAL_CLASSES` is the shell *plus* the
  glazing: the class term asks a different question of a glazed pixel than the
  depth term does, and identity behind glass is the room's even though depth
  behind it is the view's.
- `masking.py`: `scorable_mask`, the one place a segmentation is turned into the
  pixels a reward may score, under either `MaskMode` polarity —
  `intersect_shell` keeps only affirmative shell, `subtract_contents` exempts
  only affirmative contents, so an id the partition does not name stays scored;
  `architectural_mask` is that function at `intersect_shell`, which is how the
  reference photo's shell is always read. Plus the boundary erosion, applied
  to the kept side in either mode, that drops the few pixels either side of a
  segment edge where monocular depth is unreliable — a max-pool over the
  inverted mask (Chebyshev radius in pixels); the image border is not treated as
  a segment edge. `structural_mask` is the class term's reference side, read the
  same way and eroded the same way over `STRUCTURAL_CLASSES`.
- `alignment.py`: `align_scale_shift`, the closed-form weighted least-squares
  fit of a prediction onto a reference over the masked pixels, at a
  non-negative scale.
- `reward.py`: `masked_depth_error`, the scale-and-shift-invariant masked
  objective, and `masked_class_error`, the class-consistency one — a squared
  hinge on the share of a pixel's class mass the reference's own class holds,
  bounded and with a dead zone above the margin, rather than a cross-entropy
  whose `-log p` is unbounded exactly where the term fires and whose gradient is
  largest where the segmenter is least worth believing.
- `reference.py`: `GeometricReference`, the frozen (masked depth, mask) pair
  computed once from the input photo and detached, and `StructuralReference`,
  the frozen (class map, structural mask) pair beside it — the room may not
  move, and its surfaces and openings may not become something else.

## `roomgen-guide`

The guided sampler and the experiment harness shared with `experiments/`.

- `process/`: the backend-agnostic shape of reward-steered sampling, so a
  sampler swap is a dependency injection rather than a second loop —
  `endpoint.py`'s `EndpointProcess` (a state, a differentiable pixel-space
  endpoint estimate of it, a step that advances it) with `Endpoint` and
  `StepContext`, `tilt.py`'s `TiltPolicy` (reward gradients in, the state's
  displacement out) with the `EnergyGradient` it evaluates through and
  `ClippedGradientTilt`, the raw step under an RMS cap — applied to every inner
  iteration, since an uncapped inner loop against a frozen judge is projected
  gradient descent on it — inside an optional early-stopping window, and
  `driver.py`'s `sample_endpoint_process`, the one walk every backend is
  sampled by. `docs/process-adapters.md` is what a new backend is written
  against.
- `adapters/`: where a third-party sampler and roomgen meet, in both
  directions. `mfm_reward.py`'s `MfmGeometricReward` is the depth energy as the
  `score(images, prompts)` Meta Flow Maps ascends; `flow_map.py`'s
  `FlowMapProcess` is a few-step flow map as an `EndpointProcess` (the photo
  encoded and renoised to a partial sigma, the map's jump to sigma 0 as the
  endpoint, an Euler step off that same jump, the state carried in float32 and
  cast down only at the backend's two call boundaries) over the four-call
  `FlowMapBackend` behind it; `fmrg_flux.py` is that backend for FMRG's FLUX
  sampler, the one module that reaches the vendored tree — by deferred import
  only, and never at test time. `mfm_composite.py`'s `MfmWeightedReward` sums
  several of those `score(images, prompts)` rewards per image, so the depth
  energy rides beside one of MFM's own preference models (HPSv2) as a single
  reward, with `gradient_scale` for sizing the weight between two terms whose
  units have nothing to do with each other.
- `guidance/base.py`: `Guidance`, the `Protocol` mapping a predicted clean
  image x̂₀ to one energy per batch element — the reward boundary the process
  package steers by, reducing it with `.mean()` for its single backward — plus
  the `direction` convention that separates a descended energy from an ascended
  score (structurally identical), and the `require_descended_energy` /
  `require_per_element` guards the driver and `WeightedGuidance` enforce it with.
- `guidance/composite.py`: `WeightedGuidance`, several energies summed into the
  one a sampler descends, so an extra reward rides alongside the geometric one.
- `guidance/structural.py`: `StructuralGuidance`, the class-consistency energy
  that rides beside the depth one — the reference photo's class held over the
  structural pixels the prediction has not affirmatively covered with contents,
  scored through EoMT's `class_logits` and differentiated back to the latent
  while both masks stay detached constants. It is the term that charges for a
  window painted over as flat wall, which the depth energy scores as perfect.
- `guidance/clip.py`: `ClipTextGuidance`, the differentiable text energy — a
  frozen CLIP scoring the decoded x̂₀ against a prompt embedding with the
  gradient kept alive through tensor-op preprocessing — and `ClipTextEnergies`,
  the loaded-once model that turns each prompt into one.
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
- `sampling/guided.py`: Stable Diffusion img2img as an `EndpointProcess` —
  `LatentDiffusionProcess` (SDEdit renoise, Tweedie x̂₀ through the VAE decode, a
  scheduler step) and `sample_guided`, its composition with the process
  package's `ClippedGradientTilt` over the driver. It steps the pipeline's
  UNet, VAE and scheduler itself because a diffusers `__call__` runs under
  `no_grad` and would strand the guidance gradient.
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
  control's at one seed — both depth errors on the common mask, both adherence
  scores, the reconstruction floor, the common and per-arm shell coverages —
  with the depth-error reduction and adherence ratio read off them, `None`
  rather than 0 where the control scored zero and the ratio is undefined. It
  records numbers; nothing here passes or fails a run.
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

- `packages/roomgen-geometry/tests/roomgen_fakes/`: weightless `DepthEstimator`,
  `Segmenter` and `ClassLogitSegmenter` implementations (a floor-shaped
  disparity ramp coupled to pixel content; a ceiling/wall/floor room with a
  window and a sofa; a segmenter that reads class off brightness, so its scores
  are a differentiable function of the pixels and its ids are their argmax), the
  shared gradient assertion, and the fixtures exposing them. Both suites import
  it via the root pytest `pythonpath`, so the default suite needs no network,
  no GPU and no Hub cache.
- `packages/roomgen-guide/tests/roomgen_guide_fakes/`: the matching doubles for
  the diffusion side — a UNet, VAE and DDIM scheduler satisfying
  `Img2ImgPipeline` on the SD1.5 latent geometry, plus `ToyFlowMapProcess`, a
  second `EndpointProcess` with no latent space and no epsilon parameterisation,
  which is what proves the driver takes an injected backend rather than this one
  — kept here rather than beside the geometry doubles, which know nothing about
  diffusion.
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
`S_1 = Σ w`, `S_dr = Σ w d r`, `S_r = Σ w r`, `det = S_dd·S_1 − S_d²`,
`cov = S_1·S_dr − S_d·S_r`):

    s = clamp( cov / (det + λ·S_1·S_dd),  0, 1000 )
    t = (S_r − s·S_d) / S_1
    v = min(cov, 0)² / (S_1² · (det + λ·S_1·S_dd))

`det` degenerates when the masked region is tiny or has near-constant depth.
The ridge `λ = 1e-12` and the cap on `s` are what carry the solve through that
continuously — `s` falls to 0 and the fit becomes mean-matching, rather than
jumping to an unaligned comparison — and `t` is back-substituted from the capped
`s` so the fit still matches the two masked means. Both are slack by orders on a
conditioned mask. An empty mask alone falls back to the identity `(1, 0)`.
Pixels outside the mask are *replaced* with zero rather than weighted by it,
since `0 × NaN` is NaN in both directions. The whole solve runs in float64 and
the energy in float32, whatever dtype the maps arrive in.

**`s` is floored at 0: the fit may rescale the prediction but never reflect it.**
Near/far ordering is geometry, not part of the affine ambiguity the backend
leaves undetermined — measured, an unconstrained fit gave depth-inverted renders
negative scales and scored them as near-perfect matches. The floor is the
constrained optimum (`s = max(0, unconstrained s)`), and `v` — the refused
variance, `cov²/Var(d)` and so bounded by `Var(r)` — is the residual reduction
that reflection would have bought, returned beside the fit rather than
discarded. `masked_depth_error` is then the masked mean squared residual of the
aligned prediction, plus that:

    E = ( Σ_i w_i (s·d_i + t − r_i)² ) / Σ_i w_i  +  v

`v` is identically zero wherever the unconstrained fit was already positive, so
this is the scale-and-shift-invariant loss of MiDaS (arXiv:1907.01341) — used as
a reward rather than a training objective — on every prediction that shares the
room's orientation, and it is invariant to any *positive* affine rescaling of
`d` by construction. On a reversed one, where the floored fit alone would score
the flat `Var(r)` with no gradient at all, `v` restores both the size (`E ≥
Var(r)`, no sign-flip discount) and a descent direction back toward the room's
own ordering. The two branches meet with matching slope at `cov = 0`, so nothing
jumps as the optimum crosses the floor. Eval's `signed_depth_residual_map` fits
through the same function and inherits all of it; being a per-pixel map it
carries the floored fit's residual only, which is the one case where the score
exceeds its mean square.

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
per guided step, so guided sampling costs several times the VRAM of the same
unguided pipeline. See `packages/roomgen-guide/README.md` for the measured
wall-clock and VRAM, and for the levers that would buy headroom above 512px.

## Outputs

Everything (renders, masked depth maps, config snapshots) defaults
under the gitignored `outputs/` at the repo root via
`RoomgenSettings`/`ROOMGEN_WDIR`.
