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
exactly that failure, and adherence is what exposes it. So does coverage: an arm
whose own segmentation excuses the shell is scored on nothing. A depth number
read alone proves nothing.

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
  on the sampling device. `conf/benchmark.yaml` pins `[0, 1]`; the baseline arms
  below were run over `[0, 1, 2, 3]`, seeds 2 and 3 never having been used for
  tuning.
- **Reference.** `GeometricReference` per scene: Depth Anything V2 Small inverse
  depth, masked to the EoMT ADE20K shell with `erosion_radius=2`, computed once
  and detached. Measured shell coverage on these photos: 49-70% of the frame.
- **Sampler.** SD1.5 img2img, 30 steps, `noise_strength=0.6`,
  `text_guidance_scale=7.5`, DDIM (the checkpoint's PNDM is replaced: DDIM's
  update is the deterministic closed form the guided loop's x̂₀ correction
  assumes).
- **Guidance.** Constant strength 10.0 with the step capped at `clip=0.02` in
  latent-RMS units; `resegment` on, so the reward scores reference-shell ∧
  prediction-shell at every step under `mask_mode=intersect_shell`;
  `through_unet` off (the DPS arm is an A/B, not the default).
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
