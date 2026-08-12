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
- **Guidance.** `ConstantStrength` 10.0 (pinned by the sweep below).
  `DecayingStrength` (cosine, `strength` → `final_strength`) is selectable via
  `guidance.schedule=decaying` and is the next variant to measure.
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

### Per (scene, seed) at strength 10, constant (2026-08-12, 16/16 PASSED)

Artifacts: `outputs/0002_constant_s10/` (results.json + per-run images),
RTX PRO 4500 Blackwell, fp32 throughout, 30 configured steps (18 walked after
SDEdit truncation at `noise_strength=0.6`).

| scene | seed | E_recon (floor) | E_control | E_guided | reduction | adherence guided/control | gate (a) | gate (b) |
|---|---|---|---|---|---|---|---|---|
| hotel_suite_living_room | 0 | 0.0011 | 0.2089 | 0.0520 | 75.1% | 98.2% | pass | pass |
| hotel_suite_living_room | 1 | 0.0011 | 0.0387 | 0.0115 | 70.4% | 99.1% | pass | pass |
| scandi_living_room | 0 | 0.0008 | 0.0469 | 0.0093 | 80.2% | 98.2% | pass | pass |
| scandi_living_room | 1 | 0.0008 | 0.0561 | 0.0214 | 61.8% | 101.6% | pass | pass |
| lakehouse_living_room | 0 | 0.0138 | 0.1917 | 0.0546 | 71.5% | 93.2% | pass | pass |
| lakehouse_living_room | 1 | 0.0138 | 0.8443 | 0.1564 | 81.5% | 97.8% | pass | pass |
| bedroom_warm_ensuite | 0 | 0.0007 | 0.0722 | 0.0283 | 60.8% | 103.3% | pass | pass |
| bedroom_warm_ensuite | 1 | 0.0007 | 0.0671 | 0.0362 | 46.1% | 99.6% | pass | pass |
| bedroom_twin_blue | 0 | 0.0004 | 0.0453 | 0.0051 | 88.8% | 102.0% | pass | pass |
| bedroom_twin_blue | 1 | 0.0004 | 0.0968 | 0.0234 | 75.8% | 98.7% | pass | pass |
| ny_loft_bedroom | 0 | 0.0006 | 0.2472 | 0.0271 | 89.0% | 98.4% | pass | pass |
| ny_loft_bedroom | 1 | 0.0006 | 0.2507 | 0.0167 | 93.4% | 99.6% | pass | pass |
| hostel_kitchen | 0 | 0.0001 | 0.0265 | 0.0107 | 59.7% | 100.4% | pass | pass |
| hostel_kitchen | 1 | 0.0001 | 0.0405 | 0.0232 | 42.6% | 100.2% | pass | pass |
| blue_white_kitchen | 0 | 0.0021 | 0.0346 | 0.0138 | 60.0% | 98.6% | pass | pass |
| blue_white_kitchen | 1 | 0.0021 | 0.5373 | 0.0074 | 98.6% | 101.1% | pass | pass |

Mean reduction 72.2% (range 42.6–98.6%); adherence 93.2–103.3% of control,
never near the 90% floor. One value of the strength transfers across all three
room families — no per-scene tuning was needed.

### Cost

| quantity | unguided control | guided | ratio |
|---|---|---|---|
| wall-clock per sample (512², 18 walked steps) | 1.39 s | 4.32 s | 3.1× |

The 3× is the per-step VAE decode + depth forward + backward the guidance
adds; model loading from the network volume (~7 min cold) dominates short
sessions instead.

## Decision

**The central claim holds at SD1.5/512.** 2026-08-12, constant strength 10:

1. Depth guidance on the shell mask beat the unguided control on **all 16**
   (scene, seed) pairs — mean depth-error reduction 72%, minimum 42.6%, well
   past the 20% margin — while adherence stayed within noise of control
   (93–103%). The promotion gate for v2 is met on this benchmark.
2. One strength (10.0) transfers across all three room families with no
   per-scene tuning. The usable range is wide (2–50 passes the gate on the
   pilot scene); the cliff at ~150 is far from the operating point.
3. Guidance costs 3.1× the unguided sampler in wall-clock (1.4 s → 4.3 s per
   512² sample) — cheap enough that cost is not a factor in what to run next.

### Follow-up arms (2026-08-12, same seeds)

**Decaying schedule (strength 10 → 0, cosine): settled, keep constant.**
14/16 — mean reduction halves (72.2% → 50.2%) for zero adherence gain
(99.3% vs 99.6%); the two failures are the geometry margin, not adherence.
At noise 0.6 the late steps it gives up were doing useful geometric work.

**noise_strength=0.8 at constant 10: adversarial collapse, gate caught it.**
12/16. The four failures split two ways. Two scenes miss the geometry margin
(bedroom_warm_ensuite again — the persistently hardest scene). Worse, the
seeming star results (lakehouse −87%, blue_white_kitchen −98% depth error) are
**adversarial textures**: the guided image is depth-satisfying noise, not a
room (verified by eye, `outputs/noise08_inspect/`). With 24 walked steps of
nearly-pure noise there is too little image prior to resist a constant-10
pull, and sampling runs away into a texture that games the depth network.
The CLIP floor (46–52% of control) is what caught it — the two-sided gate
earned its keep. High-noise restyle needs the guidance step bounded relative
to the gradient's scale (FreeDoM-style normalization), not a raw
energy-gradient multiple.

Next: gradient-normalized guidance step in `sample_guided` (strength becomes
a step size, robust across noise levels), re-sweep at 0.6 and 0.8, then v2
sequencing (gradient-matching term, newer/flow-matching base models).

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
3. **`reward.gradient_matching_error` is a `NotImplementedError`** — expected,
   that is v2's business, noted so the sequencing stays explicit.
