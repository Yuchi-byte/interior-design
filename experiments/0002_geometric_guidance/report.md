# 0002_geometric_guidance

The v1 framework: `GeometricGuidance` over Stable Diffusion 1.5 img2img, on real
photographs. This is the project's central claim, so the control is not a
remembered number from another run: every guided sample is paired with an
unguided one at the **same photo, prompt, seed, sampler budget and trajectory**,
produced in the same process.

## Hypothesis

Steering each denoising step by the gradient of a masked, scale-and-shift-
invariant depth error — taken on the predicted clean image x̂₀, restricted to the
eroded architectural shell — **keeps the photographed room while the prompt
restyles its contents**. Concretely, against an unguided control at the same
seed: masked depth error against the frozen `GeometricReference` falls, while
prompt adherence stays level with the control's.

Both halves matter. The degenerate way to lower depth error is to return the
input photo, which scores a perfect geometry error; over-guidance approaches
exactly that failure, and adherence is what exposes it. The other degenerate way
is depth-satisfying planar slabs, which adherence — or a preference model beside
the energy — is what exposes. Coverage is reported as a diagnostic: it says how
much shell each arm's own segmentation left to score, and so how comparable the
two depth errors are, but it does not point one way under a slab render, which
the segmenter reads as *more* shell. A depth number read alone proves nothing.

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
- **Seeds.** Listed rather than drawn, threaded as a `torch.Generator` created
  on the sampling device. `conf/benchmark.yaml` pins `[0, 1, 2, 3]`, which is
  what the reference numbers were measured over; seeds 2 and 3 were never used
  for tuning.
- **Reference.** `GeometricReference` per scene: Depth Anything V2 Small inverse
  depth, masked to the EoMT ADE20K shell with `erosion_radius=2`, computed once
  and detached. Measured shell coverage on these photos: 49-70% of the frame.
- **Sampler.** SD1.5 img2img, 30 steps, `noise_strength=0.6`,
  `text_guidance_scale=7.5`, DDIM (the checkpoint's PNDM is replaced: DDIM's
  update is the deterministic closed form the guided loop's x̂₀ correction
  assumes). `conf/benchmark.yaml` is the **frozen v1 claim config**: every
  number in this report rests on its 0.6, which is why it is pinned there and
  not raised. New comparisons run at σ₀ 0.9 or from pure noise, where the
  unguided control is off the metric's noise floor
  (`docs/research/geometric-guidance.md`).
- **Guidance.** Constant strength 10.0 with the step capped at `clip=0.02` in
  latent-RMS units; `resegment` on, so the reward scores reference-shell ∧
  prediction-shell at every step under `mask_mode=subtract_contents`, the
  polarity the sweep below settled on; `through_unet` off (the DPS arm is the
  A/B a new architecture's Jacobian question is asked against, not the default).
- **Metrics.** `roomgen_guide.eval`, both arms scored on the **one common
  mask** — each arm's own reference-shell-narrowed-by-its-own-segmentation,
  intersected — so the two depth errors share a pixel set and neither arm
  profits from hiding what it moved: `depth_error` against the reference,
  `shell_coverage` for the common mask and each arm's own, and the CLIP prompt
  score (`openai/clip-vit-base-patch32`) for both arms and their ratio. The
  smoke variant substitutes `framework/metrics.py`'s checkpoint-free
  contents-change metric so it can run offline. Each scene also records a
  **reconstruction floor**: the same depth error on the photo passed through the
  VAE alone, on the reference's own shell, which is what a *perfect*
  geometry-preserving renovation still pays.
- **No thresholds.** The run reports these numbers and renders the gallery;
  nothing in the code passes or fails a run.
- **Also recorded.** Wall-clock per sample, peak VRAM, the per-step guidance
  mask at the ends and middle of the walk, the per-arm and common evaluation
  masks, the signed depth-residual maps, and the photo, shell mask, guided and
  control images per (scene, seed).

### Where the guidance strength comes from

There is no scale-free choice: the energy is in the depth backend's own squared
disparity units. The number is anchored on `test_sampling.py` /
`test_renovate.py`, which both drive the loop at strength 500.0, and
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
~5e-3 with `|dE/dimage| ~1e-5`), so the tests' 500 over-steers the real stack.
The sweep below lands the real operating point at 10.

### Verdict gate (retired 2026-08-12)

The framework asserted a two-sided gate in code
(`depth_error_margin=0.2`, `prompt_adherence_floor=0.9`) and every "PASS"/"FAIL"
below is that gate's. It has since been removed: a run now reports its metrics
and its gallery, and judging them is the owner's. The Results sections stand as
the record of what was measured under it.

### Running it

```bash
# plumbing, CPU, offline, ~10s — weightless doubles, real photos, real metrics
uv run --no-sync python experiments/0002_geometric_guidance/run.py --config smoke

# one scene, one seed, 10 steps on the real stack: does this pod run it, at what cost
ROOMGEN_HF_HOME=/workspace/hf-cache uv run --no-sync \
  python experiments/0002_geometric_guidance/run.py --config pilot

# the claim: GPU, needs the prefetched checkpoints (+ CLIP on first run)
ROOMGEN_HF_HOME=/workspace/hf-cache uv run --no-sync \
  python experiments/0002_geometric_guidance/run.py --config benchmark
```

`run.py` exits non-zero only on an error. Artifacts land under
`$ROOMGEN_WDIR/<config name>/`: `results.json` (config snapshot, provenance,
per-run numbers), `qualitative.html` and `images/`.

## Results

All numbers below are measured, from GPU runs on 2026-08-12 (RTX PRO 4500
Blackwell, fp32). Every run's raw artifacts live under `outputs/` (gitignored);
the arm-name ↔ directory mapping is stated per table. Caveats that postdate
these measurements: the guided arm was later found run-to-run nondeterministic
on CUDA (~0.6pp at the mean, up to ~11pp on a single run), so sub-point
arm-vs-arm orderings here are indicative, not resolved; and every number in
this section was scored on the reference mask alone, before the common-mask
eval landed — comparable within this section, not with later runs.

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
   (scene, seed) pairs — mean depth-error reduction 72%, minimum 42.6% — while
   adherence stayed within noise of control (93–103%).
2. One strength (10.0) transfers across all three room families with no
   per-scene tuning. The usable range is wide (2–50 on the pilot scene); the
   cliff at ~150 is far from the operating point.
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

### Normalized-step sweep (2026-08-12, full benchmark per cell)

`normalize_gradient` divides the gradient by its own RMS, making `strength` a
per-step latent-RMS displacement (total push ≤ steps × strength regardless of
the energy).

| strength | noise 0.6 | noise 0.8 |
|---|---|---|
| 0.02 | **16/16**, mean red 71.5%, adh ≥96.0% | — |
| 0.03 | 14/16 (two kitchen geometry near-misses) | **15/16**, mean red 71.5%, adh ≥91.2% |
| 0.04 | **16/16**, mean red 71.3%, adh ≥95.3% | — |
| 0.05 | — | 13/16 (adherence 87–89%) |
| 0.1 | 5/16 — overshoot, depth error *worse* than control | 4/16 |
| 0.3 | 0/16 | 0/16 |

The 0.02–0.04 plateau matches raw-10's quality at 0.6 (the 0.03 misses are
seed-noise at the 20% margin: −12%/−19% with adherence intact). The decisive
result is 0.8: normalized 0.03 gets the same mean −71.5% with adherence
≥91% — where the raw step collapsed two scenes into adversarial noise — and
its one failure is an honest geometry near-miss. The guided lakehouse image
at 0.8 is a coherent aggressive restyle preserving the window-wall structure
(verified by eye, `outputs/noise08_inspect/lakehouse_norm003_guided.png`).

**Pinned defaults at the time: `normalize: true, strength: 0.04`** (16/16 at
the default noise 0.6). *(Superseded the same day: the held-out seeds below
exposed the fixed-RMS step's overshoot, the clipped raw step replaced it, and
the `normalize` flag has since been removed from the code entirely.)*

### Held-out seeds (2, 3 — never used for tuning) at the pinned default

13/16. All twelve high-drift runs pass (reductions 20–91%, adherence ≥92.6%).
The three failures are kitchens whose control barely drifted
(E_control 0.016–0.051, the lowest in the set) where the fixed 0.04-RMS push
made depth *worse* (−24% to −55%) — overshoot of an already-good geometry.
The hard-scene pattern across every arm is the same one: the relative 20%
margin is hardest exactly where the control is already good (kitchens,
bedroom_warm_ensuite: large excluded-furniture area, low-drift shell), so the
gate partly conflates "guidance weak" with "nothing to fix".

The two step rules fail in complementary regimes: the raw step scales with
the energy (adapts down when error is tiny; unbounded above — collapses at
noise 0.8), the normalized step is bounded above (no collapse) but never
shrinks (overshoots when there is nothing to fix).

### Clipped raw step (raw scaling, RMS-capped) — seeds 0–3, full benchmark

| cap | noise 0.6 | noise 0.8 |
|---|---|---|
| 0.04 | **32/32**, mean red 73.5%, adh ≥96.2% | 28/32 |
| 0.02 | **32/32**, mean red 72.7%, adh ≥96.6% | **30/32**, mean red 64.9%, adh ≥95.3% |

The synthesis dominates both parents: at 0.6 it recovers the low-drift
kitchens that broke the fixed-RMS step (the raw step decays there) while
matching its quality everywhere else; at 0.8 the cap prevents the collapse
and every remaining failure is an honest geometry near-miss (13–20%) on
bedroom_warm_ensuite — the persistently hardest scene, not a new mode.

**Pinned default: `strength: 10, clip: 0.02`** — one configuration across noise
levels and across all four seeds, two of which were never used for tuning.

### Current reference numbers (2026-08-12, current semantics)

**Every arm above this line was measured under semantics that no longer hold.**
These five were run under the current ones — common-mask eval, windowpane
excluded from the shell, `backends.deterministic`, seeds 0–3, strength 10 with
`clip: 0.02` — and are the numbers any new arm is compared against. Controls:
`outputs/0002_clean_{n06,n09}_{intersect,subtract}`, `outputs/0002_dps_n09`.

| arm | E_control → E_guided (mean) | depth reduction | adherence ratio | common coverage | guided s/sample |
|---|---|---|---|---|---|
| noise 0.6, intersect_shell | 0.0668 → 0.0148 | 73.9% | 100.0% | 84% | 6.0 |
| noise 0.6, subtract_contents | 0.0674 → 0.0129 | 74.0% | 99.9% | 85% | 6.0 |
| noise 0.9, intersect_shell | 0.549 → 0.253 | 53.6% | 99.6% | 61% | 9.0 |
| noise 0.9, subtract_contents | 0.559 → 0.260 | 55.4% | 99.6% | 67% | 9.0 |
| noise 0.9, DPS (`through_unet`) | 0.543 → 0.272 | 53.3% | 98.7% | 56% | 12.3 (27.5 GiB peak) |

**2026-08-12:** the alignment's scale is now floored at zero (orientation is
geometry, not affine freedom). The arms in this table are unaffected — every one
of them fits at a positive scale, and the energy is bit-identical there — but
the **pre-fix MFM v2 kitchen-cell depth numbers were sign-flip-flattered**:
those cells fit at negative scales, so their ~95% "reductions" were the
alignment reflecting a degenerate render, not the render matching the room. Any
MFM depth number measured before this date needs re-running.

The central claim holds under the honest metric: at noise 0.6 guidance removes
**~74%** of the control's depth error with adherence at parity, over four seeds
and eight scenes, on a pixel set both arms segmented for themselves. At 0.9 —
where the control drifts to E≈0.55, the regime in which the room is very nearly
regenerated — the honest number is **~54%**: real, and more modest than the
pre-common-mask eval suggested. `subtract_contents` edges `intersect_shell` at
0.9, +1.8pp of reduction while scoring *more* of the room (67% vs 61% common
coverage), and ties it at 0.6; it is the candidate default. **DPS is settled:**
keeping the UNet on the gradient path is indistinguishable on geometry (53.3%
vs 53.6%), slightly worse on adherence and coverage, and costs +37% wall-clock
and ~27.5 GiB — the cheap constant-ε̂ gradient is the right one for SD1.5. What
caps the high-noise arm is therefore not a missing Jacobian but the
ε-prediction pivot itself, whose endpoint estimate is amplified by (1−t)/t
(arXiv:2607.00647) exactly where guidance has the most leverage. That is the v2
flow-map argument (`docs/research/geometric-guidance.md`).

## Where this framework stands relative to `roomgen_guide`

What is left under `framework/` is this benchmark's own glue — config, photos,
backends, run loop, gallery, residual maps — plus the checkpoint-free smoke
metric; every shared piece (config base, settings, run metrics, eval,
determinism) is in `roomgen_guide`.

There is no experiment tracker in the repo and none is wanted: a run's record is
its `results.json`, its gallery and this report, all under `ROOMGEN_WDIR`.

`DepthAnythingV2` returns its prediction in the input image's dtype, but the
reward's reduction and the guided step's RMS sizing are float32 by contract, so
a float16 sampler still gets a finite, correctly-directed gradient.
`conf/benchmark.yaml` stays float32 throughout for comparability of the recorded
numbers, not out of necessity.

## Study: DA3 alignment modes (`conf/da3_alignment.yaml`, 2026-08-14)

### Hypothesis

The comment above `AlignmentMode` claimed — from a measurement never written
down as an experiment — that when guiding with the DA3 *metric* backend,
shift-only alignment beats raw residuals: "raw metric residuals score best of
any arm from a renoised photo and collapse from pure noise". The owner doubts
it. This study puts the renoised-photo half of that claim on the bench: per
(scene, seed), one unguided control and one guided arm per mode (`none` — the
raw masked residual — and `shift`), all three from the same seed and prompt.
No arm is graded by its own objective alone: every render is scored under all
three DA3 alignment modes **and** by the affine-invariant DA2 judge under the
full fit, beside CLIP adherence and shell coverage.

### Setup

Four scenes of the benchmark grid (hotel_suite_living_room,
scandi_living_room, bedroom_warm_ensuite, blue_white_kitchen), the benchmark's
own prompts, seeds [0, 1, 2, 3]. SD1.5 img2img at 512², 30 steps,
`noise_strength 0.9` (the recorded regime decision; 0.6 sits at the metric's
noise floor), `text_guidance_scale 7.5`, DDIM, deterministic, float32
throughout. Guidance steers by DA3METRIC-LARGE with `resegment: true`,
`erosion_radius 2`, clipped raw step at `clip 0.02`. Strength was swept on the
pilot scene at seed 0 (10 / 30 / 100 / 300 — see `conf/da3_alignment.yaml`)
and pinned at **300**, where the two guided arms clearly separate while
adherence holds and both renders stay clean rooms. All arms of a (scene, seed)
are scored on one common mask (the shell all three renders left visible);
every render and reference photo leaves a DA3 and a DA2 depth map, the DA3
maps painted on one shared scale per scene so an absolute offset stays
visible, the DA2 maps fitted onto the photo's frame first. The run also
surfaced and fixed a driver bug that this study was the first to trip at
scale: the finished render was returned still carrying the final VAE decode's
autograd graph (5.6 GiB at 512² per held render), which is what ran the first
full grid out of memory — see `process/driver.py`.

### Results

Means over all 16 (scene, seed) tuples, each arm scored on the tuple's common
mask (69% of the shell in the mean). DA3/DA2 errors are in each backend's own
squared-disparity units and compare within a column only.

| arm          | DA3 none | DA3 shift | DA3 ssi  | DA2 ssi | adherence | coverage | sec |
|--------------|----------|-----------|----------|---------|-----------|----------|-----|
| control      | 0.023532 | 0.014002  | 0.009033 | 0.826   | 0.3345    | 69%      | 1.3 |
| guided_none  | 0.005386 | 0.004276  | 0.003033 | 0.501   | 0.3351    | 69%      | 8.0 |
| guided_shift | 0.019123 | 0.002613  | 0.001706 | 0.426   | 0.3327    | 69%      | 8.0 |

Per scene (mean over the four seeds):

| scene                   | arm          | none    | shift   | ssi     | DA2    | adh    | cov |
|-------------------------|--------------|---------|---------|---------|--------|--------|-----|
| hotel_suite_living_room | control      | 0.03438 | 0.00820 | 0.00466 | 0.4946 | 0.3114 | 56% |
|                         | guided_none  | 0.00150 | 0.00075 | 0.00072 | 0.2610 | 0.3129 | 59% |
|                         | guided_shift | 0.02061 | 0.00090 | 0.00087 | 0.2243 | 0.3136 | 61% |
| scandi_living_room      | control      | 0.01271 | 0.00988 | 0.00660 | 0.4197 | 0.3200 | 61% |
|                         | guided_none  | 0.00361 | 0.00250 | 0.00200 | 0.2174 | 0.3237 | 62% |
|                         | guided_shift | 0.00964 | 0.00202 | 0.00185 | 0.2060 | 0.3164 | 63% |
| bedroom_warm_ensuite    | control      | 0.01345 | 0.01090 | 0.00251 | 0.1030 | 0.3576 | 85% |
|                         | guided_none  | 0.00512 | 0.00392 | 0.00203 | 0.0899 | 0.3553 | 83% |
|                         | guided_shift | 0.01184 | 0.00226 | 0.00183 | 0.0854 | 0.3516 | 83% |
| blue_white_kitchen      | control      | 0.03359 | 0.02702 | 0.02236 | 2.2878 | 0.3491 | 72% |
|                         | guided_none  | 0.01132 | 0.00993 | 0.00738 | 1.4362 | 0.3484 | 71% |
|                         | guided_shift | 0.03441 | 0.00528 | 0.00227 | 1.1867 | 0.3492 | 69% |

What the numbers show, without judging them:

- Each guided arm wins its own objective, as it must: the raw arm scores best
  under `none` (0.0054 vs the shift arm's 0.0191), the shift arm best under
  `shift` (0.0026 vs 0.0043).
- Under the two judges neither arm optimised, the shift arm scores better on
  both: DA3 `scale_shift` 0.00171 vs 0.00303, and the DA2 second judge 0.426
  vs 0.501 (control 0.826). The pattern holds in every scene's mean for DA3
  ssi, and in 3 of 4 scenes for DA2.
- Scored raw, the shift arm (0.0191) sits barely below the control (0.0235):
  shift-guided renders drift in absolute disparity level, which the shared-
  scale DA3 maps make directly visible (e.g. blue_white_kitchen seed 0). The
  raw arm is the only one that also pins the absolute level, by construction.
- Adherence is level across all three arms (0.333–0.335 in the mean; the
  per-scene spread is a point or two either way), and common coverage is
  identical by construction. No arm degenerated: all 48 renders are plausible
  rooms, none is the input photo back, none is depth-satisfying slabs.
- Reconstruction floors sit 2–3 orders below every measured error (DA3 ssi
  floors 5e-6 – 4e-5; DA2 floors 7e-4 – 2e-3), so nothing here is reading VAE
  round-trip noise.
- Cost: 8.0 s per guided arm, 1.3 s per control at 27 walked steps on the RTX
  5090, peak 15.16 GiB after the driver fix.

Run record: `outputs/0002_da3_alignment/` (results.json + `images/`), commit
`7d184d5`.

### Decision

The renoised-photo half of the old comment's claim is now a measurement with
its arms crossed over both independent judges: raw wins only the raw metric,
shift-only wins DA3 ssi and DA2 ssi, and adherence separates nothing. The
pure-noise half ("raw collapses from pure noise") remains unmeasured — this
study starts every walk from the renoised photo by design. Judging whether
shift-only's absolute-level drift matters — it is the one thing the metric
backend was brought in to hold — is the owner's call, and the `AlignmentMode`
comment now points here rather than asserting an answer.

## Study: prompt framing (`conf/framing.yaml`, 2026-08-14)

### Hypothesis

Every prompt this repo has sampled with is a plain style description ("a
scandinavian style living room, pale oak furniture, ..."), and the photograph
constrains layout only through the depth reward. The owner's question: does an
explicit *renovation* framing — the same style content behind one fixed
instruction prefix, "renovate this furnished room into ..." — produce (1)
better geometric constraint, (2) better style adherence, (3) a more preferred
image? A 2x2 per (scene, seed) at matched seeds: {style description,
renovation instruction} x {unguided, guided}, so each axis varies exactly the
thing it names. The owner's two comparisons are old-vs-new prompt among the
unguided arms and among the guided ones.

### Setup

Five scenes over the benchmark's three room types (hotel_suite_living_room,
scandi_living_room, bedroom_warm_ensuite, ny_loft_bedroom, hostel_kitchen),
seeds [0, 1]. `style_prompt` is the exact prompt `benchmark.yaml` has always
run for that room type; `renovation_prompt` is that text verbatim behind the
one prefix (a test asserts the containment, so the prompt axis cannot drift
into a content change). SD1.5 img2img at 512², 30 steps, `noise_strength 0.9`
(the recorded regime decision; 0.6 sits at the metric's noise floor),
`text_guidance_scale 7.5`, DDIM, deterministic; sampler dtype **float16**
because the card is shared — the fp32 guided VAE backward peaks ~24 GiB and
OOMs beside a concurrent run — with depth and segmentation float32, and the
reward's reduction float32 by contract. Guidance is `benchmark.yaml`'s
clipped-raw arm (strength 10, `clip 0.02`, `resegment: true`,
`erosion_radius 2`); unguided arms are strength 0 on the same trajectory.

All four arms of a (scene, seed) are scored on one common mask — the reference
shell narrowed by each arm's own segmentation, intersected across all four —
59% of the shell in the mean. Adherence (CLIP-B/32) and preference
(**PickScore v1**, CLIP-H fine-tuned on human choices, eval-only and loaded
after sampling so its ~4 GB never sits beside the guided backward) are both
scored against the **same neutral style text** for every arm — the style
description itself, never the arm's own prompt, which would grade the two
framings on different text. Every render and every reference photo leaves an
absolute self-normalised disparity map beside the signed residual maps, all
four residual panels of a grid on the baseline (style-unguided) arm's scale.

### Results

Means over all 10 (scene, seed) grids. Depth error is DA2 scale/shift-
invariant masked MSE on the common mask; shell is the share of the reference
shell the arm's own render left visible; preference is PickScore's
logit-scaled score against the neutral text (higher preferred).

| arm                 | depth error | adherence | preference | shell visible |
|---------------------|-------------|-----------|------------|---------------|
| style, unguided     | 0.3083      | 0.3373    | 21.48      | 68%           |
| renovation, unguided| 0.3085      | 0.3248    | 20.90      | 84%           |
| style, guided       | 0.0952      | 0.3389    | 21.39      | 67%           |
| renovation, guided  | 0.0761      | 0.3280    | 20.93      | 81%           |

The owner's two comparisons (renovation minus style):

- **Unguided:** depth error +0.0003 (+0.1%), adherence −0.0125 (ratio 0.963),
  preference −0.58.
- **Guided:** depth error −0.0191 (−20.1%), adherence −0.0109 (ratio 0.968),
  preference −0.47.

Read against the guidance axis: guidance removes 69.1% of the style arm's
depth error and 75.4% of the renovation arm's, i.e. the framing's whole
geometric gain appears only in combination with the geometric term, and is a
fifth of what the term itself contributes at either prompt.

Per scene (mean over both seeds; E = depth error, adh = adherence, pref =
preference):

| scene                   | arm                  | E      | adh    | pref  | shell |
|-------------------------|----------------------|--------|--------|-------|-------|
| hotel_suite_living_room | style, unguided      | 0.5725 | 0.3075 | 20.93 | 64%   |
|                         | renovation, unguided | 0.7197 | 0.2987 | 20.52 | 78%   |
|                         | style, guided        | 0.0386 | 0.3097 | 20.78 | 65%   |
|                         | renovation, guided   | 0.0410 | 0.2998 | 20.44 | 68%   |
| scandi_living_room      | style, unguided      | 0.2863 | 0.3215 | 22.25 | 65%   |
|                         | renovation, unguided | 0.2150 | 0.2901 | 21.07 | 85%   |
|                         | style, guided        | 0.0853 | 0.3179 | 21.98 | 67%   |
|                         | renovation, guided   | 0.0499 | 0.2969 | 21.22 | 84%   |
| bedroom_warm_ensuite    | style, unguided      | 0.1169 | 0.3549 | 20.87 | 79%   |
|                         | renovation, unguided | 0.1126 | 0.3441 | 20.16 | 97%   |
|                         | style, guided        | 0.0742 | 0.3550 | 20.86 | 76%   |
|                         | renovation, guided   | 0.0834 | 0.3425 | 20.25 | 93%   |
| ny_loft_bedroom         | style, unguided      | 0.3154 | 0.3513 | 21.79 | 61%   |
|                         | renovation, unguided | 0.3086 | 0.3548 | 21.90 | 81%   |
|                         | style, guided        | 0.2485 | 0.3603 | 21.88 | 61%   |
|                         | renovation, guided   | 0.1743 | 0.3638 | 22.16 | 83%   |
| hostel_kitchen          | style, unguided      | 0.2503 | 0.3512 | 21.55 | 70%   |
|                         | renovation, unguided | 0.1869 | 0.3364 | 20.87 | 79%   |
|                         | style, guided        | 0.0293 | 0.3514 | 21.46 | 68%   |
|                         | renovation, guided   | 0.0317 | 0.3369 | 20.55 | 78%   |

What the numbers show, without judging them:

- **Geometry, unguided:** in the mean the two framings tie (+0.1%), and the
  per-scene signs disagree — the renovation prompt is better on
  scandi_living_room and hostel_kitchen, worse on hotel_suite_living_room,
  a wash on the bedrooms. The framing alone is not a geometric constraint at
  this noise level.
- **Geometry, guided:** the renovation framing lowers the mean by 20.1%
  relative, but the gap is carried almost entirely by two grids
  (scandi_living_room and ny_loft_bedroom seed 0); on four of ten grids the
  sign reverses. Both framings' guided arms land far below either unguided
  arm.
- **Shell visible is the framing's one large, consistent effect:** the
  renovation prompt's renders leave 81–84% of the photographed shell visible
  against the style prompt's 67–68%, in every scene, guided or not — the
  instruction framing renovates *less*, keeping more of the furnished photo.
  The depth tie above is therefore read over what each arm left scoreable,
  with the common mask making the pairs commensurable.
- **Adherence:** the renovation framing scores 3–4% below the style prompt on
  the same neutral text in every scene but ny_loft_bedroom, guided or not.
- **Preference:** PickScore prefers the style-prompt render in the mean
  (−0.58 unguided, −0.47 guided) and in four of five scenes;
  ny_loft_bedroom again dissents.

### Decision

Measured, not judged: the explicit renovation framing did not buy geometric
constraint on its own, bought a modest and scene-dependent depth improvement
under guidance, and paid a consistent 3–4% of adherence and about half a
PickScore point for it — while changing noticeably less of the room, which is
visible directly in the shell-coverage column and in the galleries. Whether
"renovates less but holds a little more geometry when guided" is the trade
the product wants is the owner's call. The run, its `results.json`, the
depth map of every render and reference photo, and the 2x2 gallery are under
`$ROOMGEN_WDIR/0002_prompt_framing/`.

## Study: FMRG flow-map head-to-head (`conf/fmrg.yaml`, 2026-08-14)

### Hypothesis

The geometric reward is sampler-agnostic by construction (`docs/process-adapters.md`);
the overnight pod runs showed it steering FMRG's FLUX flow map on single scenes.
The question this study answers: on the benchmark's own eight scenes and four
seeds, does FMRG + `GeometricGuidance` hold the room the way the SD1.5 stack
does, at a cost the local 32GB card can pay — i.e. can the FLUX flow map be the
repo's *default* guided method rather than a pod-only track?

### Setup

The benchmark's eight scenes, three prompts and seeds [0-3], verbatim (a test
asserts the identity). FMRG's FLUX.1-dev + 512-res FlowMap LoRA behind
`FluxFlowMapBackend`, walked by `FlowMapProcess` at 512², bf16 transformer,
float32 state, depth and segmentation float32. Sampler: 208-point sigma grid,
`GuidanceWindow(early_stop=51, unguided_steps=1)` — 50 guided steps, one
uncontrolled jump to data — `start_sigma 0.9` (the walk actually starts at
sigma 0.944 after FLUX's 512² shift), `endpoint_reuse` stepping, the cheap
pullback (`differentiate_through_velocity: false`: the gradient traverses the
VAE decoder and the flow-map arithmetic, never the transformer). Guidance:
`AnnealedSignTilt` at FMRG's `eta 5.0`, `n_opt 1`, annealed over the truncated
grid. Control is `eta 0` on the identical trajectory. Reference, resegmenting
mask, common-mask eval and CLIP-B/32 adherence are the benchmark's, unchanged.

Two fidelity caveats, recorded rather than resolved:

- **This operating point is the overnight-validated variant, not upstream's
  exact `fmrgj_nfe100`.** Upstream's sweep entry is `flow_map2`, window 26 with
  2 unguided steps, 3 warmup particles over 11 warmup steps, Jacobian pullback;
  ours keeps its grid and eta but widens the window (51/1), reuses the endpoint
  velocity, runs no warmup restarts, and takes the cheap pullback.
- **The adapter conditions every velocity call on `(t, 0)`** where upstream's
  sampler uses `(t, t_next)`, so the control arm is not a bit-comparable FMRG
  baseline — it is the *same adapter unguided*, which is the comparison the
  guidance claim actually needs.

### Results

32 (scene, seed) pairs, both arms scored on the common mask (65% of the
reference shell in the mean). Mean masked depth error **0.3658 control →
0.0895 guided (−78.1%)** at adherence ratio **100.5%** of control; FLUX VAE
reconstruction floor 0.0003. Guided 13.1s, control 8.2s per sample; peak
allocated VRAM **28.72 GiB guided / 25.24 GiB control** on the RTX 5090
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), one ~3-minute sampler
load serving all eight scenes through the prompt-embedding cache.

Per scene (means over the four seeds; E = masked depth error):

| scene                   | E ctl  | E gd   | reduction | adher. | coverage |
|-------------------------|--------|--------|-----------|--------|----------|
| hotel_suite_living_room | 0.4593 | 0.0802 | −82.5%    | 100.1% | 50%      |
| scandi_living_room      | 0.2984 | 0.0382 | −87.2%    | 101.9% | 58%      |
| lakehouse_living_room   | 0.4032 | 0.0740 | −81.6%    | 99.6%  | 67%      |
| bedroom_warm_ensuite    | 0.0766 | 0.0223 | −70.9%    | 98.4%  | 75%      |
| bedroom_twin_blue       | 0.1019 | 0.0243 | −76.2%    | 101.5% | 83%      |
| ny_loft_bedroom         | 0.1594 | 0.0246 | −84.6%    | 102.9% | 55%      |
| hostel_kitchen          | 0.3718 | 0.0454 | −87.8%    | 99.5%  | 67%      |
| blue_white_kitchen      | 1.0555 | 0.4071 | −61.4%    | 99.8%  | 67%      |

Weakest pairs, for the reader who wants the tails: blue_white_kitchen seed 1
(0.97 guided vs 1.30 control, −25.0%, the one pair guidance did not mostly
close), and ny_loft_bedroom seed 3, whose common mask fell to 4% of the shell —
a comparison over a sliver, flagged by the coverage column doing its job.

**SD1.5 context, same scenes and metric, not a controlled A/B** (different
sampler, different noise space at a given seed, SD numbers from today's
prompt-framing study's style arms — noise 0.9, strength 10, clip 0.02 — read
from `outputs/0002_prompt_framing/results.json`, not re-run): over the ten
(scene, seed) pairs the two studies share, SD1.5 removes 62.0% of a 0.3083
mean control error at adherence ratio 100.5% in 3.4s guided / 0.5s control per
sample (~8 GiB); FMRG on those same ten pairs removes 81.3% of a 0.2308 mean
control error at adherence ratio 101.5%, in 13.1s / 8.2s at ~29 GiB. The
sampler is ~4x slower and ~4x hungrier, and its guided walk lands materially
lower depth error at par adherence.

### Decision

FMRG is usable as the local default: it fits the card with headroom, runs the
full benchmark in under half an hour, and the guided-vs-control claim holds on
every scene. `run.py` now defaults to `--config fmrg`; the SD1.5 claim config
remains `conf/benchmark.yaml`, unchanged, and every recorded SD number above it
stands. What runs next is the owner's call; the natural candidates are the
faithful upstream operating point (flow_map2 window 26/2 + warmup particles +
Jacobian pullback, all already expressible in `conf/`) and an eta sweep at
matched NFE.
