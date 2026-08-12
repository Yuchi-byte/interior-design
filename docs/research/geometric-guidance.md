# Geometric guidance

The living research record: what roomgen's method is, what it has measured,
what is settled, and where it goes next. Updated in place — this file describes
the current direction, not the path taken to it.

**TL;DR.** Steering a frozen sampler by the gradient of a masked,
scale-and-shift-invariant depth error keeps a photographed room's architectural
shell fixed while the prompt restyles its furniture. On SD1.5 img2img at 512²
this removes ~74% of the unguided control's depth error at unchanged prompt
adherence, for ~4x the wall-clock. The step rule that works is the **raw
energy gradient with an RMS cap**; the reward is re-masked every step against a
fresh segmentation of the prediction; the affine alignment is non-negotiable.
The limitation is the ε-prediction pivot, whose x̂₀ estimate degrades exactly in
the high-noise window where guidance has the most leverage — which is what v2's
move to flow-map endpoints is for.

---

## 1. The method

Training-free guidance: a **frozen** pretrained sampler is steered at each
denoising step by the gradient of a differentiable energy evaluated on the
predicted clean image, in the classifier-guidance → DPS → FreeDoM /
Universal-Guidance line. No weights are updated and no adapter is trained.

The energy is the masked, scale-and-shift-invariant depth error between the
current prediction and the input photo, restricted to the **architectural
shell** — walls, floor, ceiling, doors — and blind to the contents, which is
exactly what renovation is allowed to change. The central bet is that a
pretrained monocular depth network already encodes planarity, room layout and
relative scale, so error in its output space is a usable proxy for "same room,
different furniture" with no 3D reconstruction to build, tune, or fail silently.

Two positioning facts, one line each:

- **The mask is the differentiator.** A ControlNet-depth hint conditions on the
  whole image, which pins the furniture too; a guided energy takes a mask
  natively — pixels outside it contribute zero to the energy and zero to the
  gradient.
- **Depth is 2.5D and view-dependent.** The objective constrains "the shell
  looks the same *from this camera*", which is the fixed-viewpoint renovation
  requirement and nothing more; it does not extend for free to novel views.

### The affine alignment is not optional

Depth Anything V2 predicts affine-invariant inverse depth: determined only up to
an unknown scale and shift, and that pair is **not stable between two images**.
A raw MSE between two depth maps measures mostly affine mismatch. Every
comparison first solves `min_{s,t} Σ w (s·d + t − r)²` over the masked pixels in
closed form and scores the aligned prediction — MiDaS's SSI loss
(arXiv:1907.01341) used as a reward, differentiated through (`s` and `t` carry
gradient). The mask is **eroded** at segment boundaries, because monocular depth
is unreliable within a few pixels of an edge and those pixels otherwise dominate
the residual. Derivation and degenerate-`det` guard: `docs/architecture.md`.

### The model stack

| Role | Checkpoint | Licence |
|---|---|---|
| Depth | Depth Anything V2 Small, `depth-anything/Depth-Anything-V2-Small-hf` (arXiv:2406.09414) | Apache-2.0 — the larger tiers are CC-BY-NC |
| Segmentation | EoMT, `tue-mps/ade20k_semantic_eomt_large_512` (arXiv:2503.19108) | MIT — the DINOv3-backed variants are not |
| Base sampler | Stable Diffusion 1.5 img2img, `stable-diffusion-v1-5/stable-diffusion-v1-5` | CreativeML OpenRAIL-M |

Licences are load-bearing: nothing here caps the project at non-commercial use.
The SD1.5 id is the canonical mirror — the original `runwayml` repo is gone from
the Hub. EoMT decodes mask queries inside a plain ViT rather than through a
separate pixel decoder, and runs once per image off the gradient path, so its
cost barely matters.

---

## 2. v1, and what it measured

v1 is `experiments/0002_geometric_guidance`: SD1.5 img2img (SDEdit renoise,
DDIM, 30 steps, 512², fp32), guided by the gradient of the shell-masked depth
energy taken on Tweedie's x̂₀ through the VAE decode, against an unguided control
at the **same photo, prompt, seed, sampler budget and trajectory** produced in
the same process.

- **The step.** Raw ∂E/∂latent × `strength` 10, capped at `clip` 0.02 in
  latent-RMS units. Sized in float32 whatever the latent's dtype.
- **The mask.** Re-segmented every step: the reward scores reference-shell ∧
  prediction-scorable, so furniture the renovation puts in front of a wall is
  free while a wall that moved is not. Both polarities exist (`mask_mode`):
  `intersect_shell` keeps only affirmative shell, `subtract_contents` exempts
  only affirmative contents.
- **The eval.** Both arms are scored on one common mask — each arm's reference
  shell narrowed by a segmentation of *that* render, intersected — with per-arm
  coverage reported beside it, so an arm cannot win by hiding the geometry it
  moved.
- **The reporting.** Metrics and a gallery. There are no thresholds in the
  code: a framework measures, the owner judges.

Measured (seeds 0–3, eight scenes, deterministic mode, RTX PRO 4500 Blackwell,
fp32; full table in the framework's `report.md`):

| noise strength | mean depth-error reduction vs. control | adherence (guided/control) | shell scored in common |
|---|---|---|---|
| 0.6 | 73.9% (`intersect_shell`) / 74.0% (`subtract_contents`) | 100% | 84–85% |
| 0.9 | 53.6% / 55.4% | 99.6% | 61–67% |

Cost: ~6 s per guided 512² sample at noise 0.6 against 1.4 s unguided — the
per-step VAE decode plus depth forward and backward. Cheap enough that cost is
not what decides the next step.

`subtract_contents` ties at 0.6 and beats `intersect_shell` by 1.8pp at 0.9
while scoring more of the room, so it is the candidate default.

The whole knob surface is `strength`, `clip`, `erosion_radius`, `mask_mode`,
`resegment`, `through_unet` (the DPS A/B, settled against as a default) and
`backends.deterministic`.

---

## 3. Settled

- **Clipped raw step.** The raw gradient scales with the energy, so it decays
  by itself on a scene whose geometry is already close; the RMS cap is what
  stops a large energy running away into depth-satisfying texture at high
  noise. A fixed-RMS (normalized) step is bounded above but never shrinks and
  overshoots low-drift scenes; the clipped raw step dominates both.
- **Constant strength.** A decaying schedule gives up the late steps that were
  doing useful geometric work, for no adherence gain.
- **Per-step re-segmentation.** Scoring the reference shell alone charges a
  renovation for every surface it legitimately covers up.
- **Windowpane is excluded from the shell.** Depth behind glass reads the view,
  not the room. The ADE20K partition is three-way — shell / affirmative
  contents / unclaimed — so an unnamed id stays scored rather than being
  silently excused.
- **The step arithmetic is float32 by contract**, so a half-precision sampler
  keeps a finite, correctly-directed gradient.
- **Honest eval.** One common mask for both arms, coverage always reported
  beside the score, adherence always reported beside depth error: the
  degenerate way to win on geometry is not to renovate.
- **The cheap gradient is the right one on SD1.5.** DPS (`through_unet`) is
  indistinguishable on geometry at noise 0.9, slightly worse on adherence and
  coverage, and costs +37% wall-clock and ~27.5 GiB. Keep it as an A/B knob,
  not a default.
- **The limitation: the ε-prediction pivot.** Endpoint-estimate error scales as
  (1−t)/t for ε-prediction, (1−t) for v-prediction and 1 for a flow map
  (arXiv:2607.00647) — so the x̂₀ that guidance is evaluated on is worst exactly
  in the high-noise window where the trajectory is still free enough to be
  steered. The DPS null result localises the high-noise ceiling here rather
  than in a missing denoiser Jacobian, which is what makes v2 a
  parameterisation change and not a tuning exercise.

---

## 4. v2: flow-map endpoints

The direction is to guide on a model whose endpoint estimate does not degrade
with noise. Two tracks, run in parallel.

**Track 1 — inference, existing models.** Port the guided loop to a
rectified-flow base: **FLUX.2-klein-base-4B** (Apache-2.0) primary, **Z-Image
base** as fallback. The Tweedie pivot is replaced by the velocity closed form
for the endpoint, and then by a k-step ODE lookahead per FMRG
(arXiv:2604.27147), which is the formal derivation of what roomgen already does
and whose headline prescription is to **guide inside the high-noise window and
stop early**. Everything else — the reward, the mask, the eval — is unchanged.

**Track 2 — training.** Vendor the meta-flow-maps implementation
(`github.com/adh1s/mfm`, already under `packages/mfm`, and
`github.com/Yuchi-byte/physics-informed-flow-map`) and distill/train on the
second pod, so the endpoint estimate is a learned flow map rather than a
one-step approximation.

Standing hard case for either track: `bedroom_warm_ensuite` — large excluded
furniture area, low-drift shell, so there is little for guidance to fix and
every arm finds it hardest.

**Regime (owner, 2026-08-12): experiments run at σ₀ 0.9 or pure noise, never
0.6** — at 0.6 the unguided control already sits at the metric's noise floor,
so nothing measured there can separate methods.

---

## 5. Open questions

- **Does the depth prior see small wall shifts?** Monocular depth is smooth. A
  wall translated modestly may cost less error than a plausible furniture
  change costs outside the mask. Measurable without any sampling: perturb a
  room's geometry synthetically and plot aligned masked error against
  perturbation magnitude.
- **Is the mask reliable enough?** ADE20K mislabels built-in cabinetry, mirrors
  and large wall art. A mirror is the worst case: segmented as wall, but its
  depth is the reflected scene's.
- **Does guidance through the VAE decode fight the decoder?** Part of the
  signal may be spent on decoder-reconstruction artefacts rather than geometry.
  Isolating it needs a latent-space depth proxy.
- **Is CLIP the right adherence metric?** It is only as trustworthy as its
  ability to fail a frozen image; a perceptual distance (LPIPS) measures
  something different.
- **How much of the headroom is the base sampler's?** The drift guidance closes
  is a property of SD1.5 and must be re-measured after the v2 migration.

---

## 6. References (arXiv)

- **Depth Anything V2** — arXiv:2406.09414. *(the depth backbone; affine-invariant inverse depth)*
- **MiDaS** — arXiv:1907.01341. *(the scale-and-shift-invariant loss — the crux)*
- **EoMT** — Your ViT is Secretly an Image Segmentation Model. arXiv:2503.19108.
- **FreeDoM** — Training-Free Energy-Guided Conditional Diffusion Model. arXiv:2303.09833. *(closest published match to v1)*
- **DPS** — Diffusion Posterior Sampling for General Noisy Inverse Problems. arXiv:2209.14687. *(`through_unet`)*
- **Universal Guidance** — arXiv:2302.07121.
- **FMRG** — arXiv:2604.27147. *(the formal derivation of this approach; guide in the high-noise window, stop early)*
- **Endpoint-estimate error scaling** — arXiv:2607.00647. *((1−t)/t for ε, (1−t) for v, 1 for a flow map)*
- **Meta Flow Maps** — arXiv:2601.14430. *(the track-2 training code, vendored at `packages/mfm`)*
- **ControlNet** — arXiv:2302.05543. *(the trained-adapter alternative)*
- **Classifier guidance** — arXiv:2105.05233. **Classifier-free guidance** — arXiv:2207.12598.
- **Latent Diffusion / Stable Diffusion** — arXiv:2112.10752. **SDEdit** — arXiv:2108.01073.
- **DINOv2** — arXiv:2304.07193. **CLIP** — arXiv:2103.00020. **LPIPS** — arXiv:1801.03924.
