# 0002_geometric_guidance

The v1 framework: `GeometricGuidance` over the same Stable Diffusion 1.5 img2img
sampler `0001_unguided_baseline` measures, on real photographs, gated by the
two-sided verdict of `experiments/README.md`. This is the project's central
claim, so the control is not a remembered number from another run: every guided
sample is paired with an unguided one at the **same photo, prompt, seed, sampler
budget and trajectory**, produced in the same process.

## Hypothesis

Steering each denoising step by the gradient of a masked, scale-and-shift-
invariant depth error — taken on the predicted clean image x̂₀, restricted to the
eroded architectural shell — **keeps the photographed room while the prompt
restyles its contents**. Concretely, against an unguided control at the same
seed:

- **(a)** masked depth error against the frozen `GeometricReference` falls by at
  least the configured margin, on every scene and every seed; and
- **(b)** prompt adherence does not drop below the configured fraction of the
  control's — guidance restyles a room rather than declining to renovate one.

The claim is only interesting because (b) is a live constraint. The degenerate
way to win (a) is to return the input photo, which scores a perfect geometry
error; over-guidance approaches exactly that failure, and gate (b) is what
catches it. Any run reporting (a) alone has proved nothing.

Two premises this framework leans on, both measured rather than assumed
(numbers in Setup below): that the depth reward sees **geometry** and not
appearance, and that the reward is exactly zero between a photo and itself, so a
non-zero number is drift and not affine noise.

## Setup

Pinned in `conf/`; `conf/benchmark.yaml` is the claim, `conf/smoke.yaml` is the
same flow on weightless doubles for plumbing. `run.py` is glue: it loads the
selected variant, ensures the photos, builds the stack, and drives
`roomgen_guide.renovate.renovate` twice per (scene, seed).

- **Scenes.** The eight Wikimedia/Unsplash interiors of
  `experiments/assets/photos.json` — three living rooms, three bedrooms, two
  kitchens — downloaded and sha256-verified into
  `$ROOMGEN_WDIR/benchmark_photos/`, centre-cropped square and resized to
  512x512 (SD1.5's training resolution, a multiple of 64) as `[0, 1]` NCHW
  tensors, decoded with torchvision and never round-tripped through PIL.
- **Prompts.** Three renovation prompts, one per room type, paired with the
  photos in `conf/benchmark.yaml` and held identical between the guided run and
  its control.
- **Seeds.** `[0, 1]`, listed rather than drawn, threaded as a
  `torch.Generator` created on the sampling device. The gate must hold on each.
- **Reference.** `GeometricReference` per scene: Depth Anything V2 Small inverse
  depth, masked to the EoMT ADE20K shell with `erosion_radius=2`, computed once
  and detached. Measured shell coverage on these photos: 49-70% of the frame.
- **Sampler.** SD1.5 img2img, 30 steps, `noise_strength=0.6`,
  `text_guidance_scale=7.5`, DDIM (the checkpoint's PNDM is replaced: DDIM's
  update is the deterministic closed form the guided loop's x̂₀ correction
  assumes).
- **Guidance.** `ConstantStrength`, default 150.0 (derivation below).
  `ConstantStrength` rather than `DecayingStrength` because the latter is a
  `NotImplementedError` upstream — a decayed schedule belongs to the variant
  that can evaluate it, not to a default.
- **Metrics.** `roomgen_guide.eval`: gate (a) is `depth_error` against the
  reference, gate (b) the CLIP prompt score
  (`openai/clip-vit-base-patch32`); the smoke variant substitutes
  `framework/metrics.py`'s checkpoint-free contents-change metric so it can run
  offline.
  Each scene also records a **reconstruction floor**: the same depth error on
  the photo passed through the VAE alone, which is what a *perfect*
  geometry-preserving renovation still pays.
- **Also recorded.** Wall-clock per sample, peak VRAM, and the rendered photo,
  shell mask, guided and control images per (scene, seed).

### Where the guidance strength comes from

There is no scale-free choice: the energy is in the depth backend's own squared
disparity units. The number is anchored on `test_sampling.py` /
`test_renovate.py`, which both drive the loop at `ConstantStrength(500.0)`, and
then corrected for the fact that the real stack's energy is not the fakes'.
Measured on these photos with Depth Anything V2 at 512x512 (CPU, float32):

| image scored against the reference | masked depth error | mean abs. dE/dimage |
|---|---|---|
| the photo itself | 0.0 (exact) | 0.0 |
| photometric only (`0.6*x + 0.2`) | 2.0e-4 | 1.1e-5 |
| additive noise, sigma 0.1 | 2.1e-2 | 7.8e-5 |
| shifted 32 px horizontally | 8.1e-1 | 8.9e-4 |
| a different room entirely | 1.4e0 | 8.1e-4 |

Three things follow. The reward is exactly zero on an unchanged photo, so the
affine alignment is doing its job. It separates geometry from appearance by
three to four orders of magnitude — a wall shift costs ~4000x what a global
tone change does — which is the premise of the whole approach. And the real
gradients are one to two orders larger than the fake stack's (whose energies run
~5e-3 with `|dE/dimage| ~1e-5`), so the tests' 500 would over-steer. Hence 150
as the **starting point**, with the sweep
`--set guidance.strength=50|150|500|1500` as this framework's first job.

### Verdict gate

Asserted in code (`roomgen_guide.verdict.score_run`), on every (scene, seed),
both sides at once:

- **(a)** `guided_depth_error <= (1 - depth_error_margin) * control_depth_error`
- **(b)** `guided_prompt_adherence >= prompt_adherence_floor * control_prompt_adherence`

`conf/benchmark.yaml` sets `depth_error_margin=0.2` and
`prompt_adherence_floor=0.9`. **These are provisional.** 0001 has not run, so
the constants it is supposed to deliver do not exist yet; re-derive both from
its numbers before this framework's verdict is quoted as a finding.

### Running it

```bash
# plumbing, CPU, offline, ~10s — weightless doubles, real photos, real gate
uv run --no-sync python experiments/0002_geometric_guidance/run.py --config smoke

# one scene, one seed, 10 steps on the real stack: does this pod run it, at what cost
ROOMGEN_HF_HOME=/workspace/hf-cache uv run --no-sync \
  python experiments/0002_geometric_guidance/run.py --config pilot

# the claim: GPU, needs the prefetched checkpoints (+ CLIP on first run)
ROOMGEN_HF_HOME=/workspace/hf-cache uv run --no-sync \
  python experiments/0002_geometric_guidance/run.py --config benchmark
```

`run.py` exits non-zero if any run fails either side of the gate. Artifacts land
under `$ROOMGEN_WDIR/<config name>/`: `results.json` (config snapshot, per-run
numbers, verdict) and `images/`.

## Results

**Not yet run.** No run of this framework has been performed. Every cell below is
a placeholder — no number in this section is measured, estimated, or otherwise
real. The magnitudes in Setup are the *only* measured numbers in this document,
and they are properties of the reward, not results of a sampling run.

### Strength sweep (pilot scene, seed 0, 10 steps, 2026-08-12)

The planned 50–1500 grid was wrong by an order of magnitude: the first pilot at
150 *destroyed* the geometry (E_guided 1.49 vs control 0.040, adherence 56%),
so the sweep ran downward instead.

| strength | E_guided | E_control | reduction | adherence ratio | gate |
|---|---|---|---|---|---|
| 0.1 | 0.0397 | 0.0403 | 1.6% | 100.0% | fail (geometry) |
| 0.5 | 0.0375 | 0.0403 | 7.0% | 100.1% | fail (geometry) |
| 2 | 0.0313 | 0.0403 | 22.4% | 100.2% | pass |
| **10** | **0.0146** | 0.0403 | **63.8%** | 100.1% | pass |
| 50 | 0.0199 | 0.0403 | 50.8% | 100.9% | pass |
| 150 | 1.4913 | 0.0403 | −3597% | 55.6% | fail (both) |

Monotone improvement up to ~10, past the peak by 50, catastrophic by 150.
**Pinned: 10.0** in both `pilot.yaml` and `benchmark.yaml`.

**TODO after the full benchmark run:** the per-scene table at strength 10, the
cost table, and the decision. Record the run's `results.json` path.

### Per (scene, seed) at the pinned strength

| scene | seed | E_recon (floor) | E_control | E_guided | reduction | adherence guided/control | gate (a) | gate (b) |
|---|---|---|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### Cost

| quantity | unguided control | guided | ratio |
|---|---|---|---|
| wall-clock per sample | TODO | TODO | TODO |
| peak VRAM | TODO | TODO | TODO |

## Decision

**TODO — pending the first run.** Nothing is settled. What this framework will
settle:

1. Whether depth guidance on the shell mask beats the unguided control by the
   calibrated margin on **every** scene and seed while clearing the adherence
   floor — the project's central claim, and the promotion gate for v2.
2. What the guidance strength is, and whether one value transfers across room
   types or has to be set per scene family (if per-scene, the method is much
   less useful — an open question in
   `docs/research/2026-08-11-geometric-guidance.md`).
3. What guidance costs in wall-clock and VRAM against the same sampler unguided.

Failure modes to read the numbers for: gate (b) failing while (a) passes is
over-guidance freezing the image; both failing with a near-zero depth-error
reduction is a dead gradient rather than weak guidance — check whether the
control and guided images are bit-identical before reaching for more strength.

## Gaps in `roomgen_guide` this framework had to work around

What is left under `framework/` is this benchmark's own glue plus the
checkpoint-free smoke metric; the shared machinery is in `roomgen_guide`.

1. **`roomgen_guide.logging` does not exist**, so there is no `Logger` to pass
   the verdict through and no wandb tracking: this framework prints and writes
   `results.json`. `ROOMGEN_WANDB_MODE` is therefore not honoured by 0002 yet,
   which `experiments/README.md` documents as the tracking contract.
2. **The reward's dtype is the input image's.** `DepthAnythingV2` casts its
   prediction back to `images.dtype`, so a float16 sampler drags the energy, the
   alignment residual and the guidance gradient into float16 with it. The
   backend's own `dtype` is configurable (and surfaced as `backends.depth_dtype`
   here), but there is no knob to keep the *energy* in float32 above a
   half-precision VAE without a cast inside `packages/`. `conf/benchmark.yaml`
   therefore runs float32 throughout.
3. **`guidance/schedule.DecayingStrength.__call__` is a
   `NotImplementedError`**, so the strong-early/decayed-late schedule the
   research note recommends cannot be selected. `constant` is the only value
   `guidance.schedule` accepts.
4. **`reward.gradient_matching_error` is a `NotImplementedError`** — expected,
   that is v2's business, noted so the sequencing stays explicit.
