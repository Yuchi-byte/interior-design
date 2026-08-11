# Depth-Guided Sampling for Geometry-Preserving Room Renovation

**Date:** 2026-08-11 · **Scope:** how to hold a photographed room's architectural
shell fixed while a diffusion model restyles its contents — the training-free
guidance approach, the model stack it runs on, and the affine-invariance problem
that any depth-space objective must solve first.

**TL;DR.** Build the **unguided baseline and the eval harness first**, then climb
a 4-rung ladder: **v0** unguided img2img → **v1** depth guidance on the
architectural shell mask (FreeDoM-style, gradient of a masked
scale-and-shift-invariant depth error taken on x̂₀ at every step) → **v2** add
the multi-scale gradient-matching term and tune the strength schedule → **v3**
benchmark against **ControlNet-depth conditioning**, which is the honest strong
baseline and may simply win. Guidance's real advantages over ControlNet are that
it needs no trained adapter, applies to any pipeline, and — the point of this
project — can be **restricted to a mask**, conditioning on the shell while
leaving the furniture unconstrained, which a whole-image ControlNet hint does
not naturally do. Its real disadvantages are per-step cost, VRAM through the VAE
decode, and a strength schedule that must be tuned per scene family. The model
stack is fixed (§3): **Stable Diffusion 1.5** as the base sampler, **Depth
Anything V2 Small** for depth, **EoMT** for the ADE20K segmentation the shell
mask is cut from. **The single non-negotiable
technical detail is the affine alignment**: Depth Anything V2 is
affine-invariant, so a raw depth MSE is meaningless and every comparison goes
through a per-step closed-form weighted least-squares alignment over the masked
pixels (MiDaS's SSI loss, arXiv:1907.01341), differentiated through.

---

## 1. The approach and its baseline

Training-free guidance means steering a *frozen* pretrained sampler with the
gradient of a differentiable objective on the sample; the alternative is a
conditioning model that must be trained. roomgen takes the first, in the
FreeDoM line (classifier guidance arXiv:2105.05233 → DPS arXiv:2209.14687 →
FreeDoM arXiv:2303.09833 / Universal Guidance arXiv:2302.07121), and measures
itself against the second. Cost is per denoising step, relative to one unguided
step.

| Approach | Core idea | Cost | Needs training |
|---|---|---|---|
| **FreeDoM** (arXiv:2303.09833) — what roomgen does | Take ∇ of an **arbitrary off-the-shelf energy** (here a depth net) through Tweedie's x̂₀, on a time-dependent strength schedule. | ~2-4x (energy net fwd+bwd) | No |
| **ControlNet-depth** (arXiv:2302.05543) — the v3 baseline | *Not guidance.* A trained side network injects a depth hint into the UNet, so geometry is respected by construction at **no per-step gradient cost**. | ~1.2-1.4x (extra branch fwd only) | **Yes** — but SD1.5 checkpoints are public |

Two cross-cutting notes. First, **the mask is the differentiator.**
ControlNet-depth conditions on a whole-image depth hint, which also pins the
furniture — precisely what renovation must be free to change. Getting
shell-only behaviour out of ControlNet means synthesising a hint with the
furniture inpainted or the conditioning spatially attenuated; that is possible
but is itself a research question. Guidance takes the mask natively: pixels
outside it contribute zero to the energy and zero to the gradient. Second,
**they compose**: ControlNet-depth for coarse layout plus a light geometric
guidance term for shell-only refinement is a legitimate v4 and should not be
framed as an either/or.

---

## 2. Why depth rather than explicit 3D

The alternative framing is to reconstruct the room — planes, a layout cuboid,
camera intrinsics, a mesh — and enforce consistency in that representation.
Reasons to prefer depth space:

- **Differentiability end to end.** A monocular depth network is one
  differentiable function from pixels to a geometry-bearing field. Plane
  fitting, RANSAC, layout-corner extraction and mesh reconstruction are not, or
  are only through awkward relaxations. Guidance needs a gradient at every step;
  depth gives one for the cost of a forward+backward pass.
- **No metric calibration required.** Explicit 3D from a single photo needs
  intrinsics and an absolute scale, neither of which a user's photograph
  carries. The depth-space objective sidesteps this by being *defined* up to the
  affine ambiguity (§4) rather than needing it resolved.
- **The prior is already trained.** The central bet is that Depth Anything V2
  (arXiv:2406.09414), a DPT head on a DINOv2 backbone trained on very large
  mixed data, has already internalised planarity, room layout, and relative
  scale — so error in its output space is a usable proxy for "same room" with no
  3D pipeline to build, tune, or fail silently.
- **Failure modes are legible.** A masked depth-error map is an image; it can be
  logged next to the render and eyeballed. A layout-estimation failure is a
  wrong cuboid with no obvious signal.

The honest counterargument: depth is a **2.5D, view-dependent** field, so the
objective constrains "the shell looks the same from this camera", not "the room
is the same room". For a fixed-viewpoint renovation task that is exactly the
requirement — but it means the approach does **not** extend for free to
novel-view renovation, and the note should not imply it does. The other
counterargument is that monocular depth is smooth and can be tolerant of exactly
the small wall shifts we care about; that is what the gradient-matching term
(§5, v2) exists to test.

---

## 3. The model stack

| Role | Checkpoint | Licence |
|---|---|---|
| Depth | Depth Anything V2 Small, `depth-anything/Depth-Anything-V2-Small-hf` (arXiv:2406.09414) | Apache-2.0 — the larger tiers are CC-BY-NC |
| Segmentation | EoMT, `tue-mps/ade20k_semantic_eomt_large_512` (arXiv:2503.19108) | MIT — the DINOv3-backed variants are not |
| Base sampler | Stable Diffusion 1.5 img2img, `stable-diffusion-v1-5/stable-diffusion-v1-5` | CreativeML OpenRAIL-M |

Licences are load-bearing: this stack is chosen so nothing caps the project at
non-commercial use. The SD1.5 id is the canonical mirror — the original
`runwayml` repo is gone from the Hub.

EoMT decodes mask queries inside a plain ViT encoder rather than through a
separate pixel decoder, so it emits the same ADE20K-150 label space the shell
mask is cut from at a fraction of the per-image cost — and it is run once per
photo, off the gradient path, so its cost barely matters.

Future work, once the method is validated: migrate the sampler to a
flow-matching model (SDXL, FLUX), where x̂₀ is recovered from the velocity
field rather than the noise prediction and the guidance schedule must be
re-tuned.

---

## 4. The affine-invariance problem

Depth Anything V2, like MiDaS before it, predicts **affine-invariant inverse
depth**: its output is determined only up to an unknown scale `s` and shift `t`,
and — critically — **that pair is not stable across two different images**. Two
renders of the same room can differ by an arbitrary affine transform of the
predicted field. A raw MSE between two predictions therefore measures mostly the
affine mismatch, not the geometry: it can be huge for identical geometry and
small for genuinely drifted geometry. Any depth-space objective must remove this
degree of freedom before it measures anything.

**The handling: per-step closed-form alignment, weights = the eroded shell
mask, differentiated through.** Solve `min_{s,t} Σ w (s·d + t − r)²` over the
masked pixels as a 2x2 system, then take the masked MSE of the aligned
prediction — MiDaS's scale-and-shift-invariant loss (arXiv:1907.01341) used as a
reward. It costs five weighted sums and a 2x2 solve, and it is invariant by
construction rather than approximately (the alignment coefficients depend on the
prediction, so `s` and `t` carry gradient — do not stop-gradient them without
measuring the effect; treating them as constants is a defensible ablation, not
an obvious simplification). Two practical guards: the 2x2 determinant
degenerates when the mask is small or the masked depth is near-constant, and the
mask must be **eroded at segment boundaries** because monocular depth is
unreliable within a few pixels of an edge and those pixels otherwise dominate
the residual.

---

## 5. Staged path

Every promotion is gated on the two-sided verdict of `experiments/README.md`:
masked depth error down by a margin **and** prompt adherence not below a floor.
Gate (a) alone is trivially won by not renovating.

| Stage | What | Effort | Risk | Promotion gate |
|---|---|---|---|---|
| **v0 unguided baseline** | Plain img2img over a fixed scene/prompt/seed set; both eval metrics wired; strength swept. Establishes the drift and calibrates the thresholds. | ~2-3 days | Low. Mostly harness. Real risk is metric choice, not modelling. | Not a research result. Ships the harness and the two calibrated constants; must show unguided depth error is well clear of the VAE-reconstruction floor. |
| **v1 depth guidance on the shell mask** | FreeDoM-style guidance: x̂₀ → VAE decode → Depth Anything V2 → aligned masked error → ∇ w.r.t. latent, on a decaying schedule. The core claim. | ~1-2 weeks | Medium. Failure modes: over-guidance freezing the image (caught by gate (b)), zero gradient from an accidentally detached graph, VRAM through the decode. | Beat v0 on gate (a) by the calibrated margin on every scene and seed **while** clearing gate (b). This is the project's central claim. |
| **v2 gradient matching + schedule tuning** | Add the multi-scale gradient-matching term (masked, on the aligned prediction) and tune the strength schedule; ablate stop-gradient on `(s,t)`, guidance-step subsets, energy resolution. | ~1 week | Low-med. Incremental; the risk is spending the week for a null result. | Beat v1 by a real margin on gate (a) at equal or better gate (b) and no worse wall-clock. A tie means ship v1 and record the ablation. |
| **v3 vs. ControlNet-depth** | Published ControlNet-depth for the same pipeline, conditioned on the same shell-masked depth reference, on identical scenes/prompts/seeds. The strong baseline. | ~1 week | Med. **Genuine risk that ControlNet simply wins on quality-per-second.** That would be a real finding, not a failure. | Not a promotion — a comparison. Report the Pareto (both metrics vs. wall-clock and VRAM). If ControlNet dominates, the defensible remaining claims are shell-only masking and adapter-free portability; test the composition of the two as v4. |

**Primary metrics.** Masked depth error against the frozen reference (gate a);
prompt adherence / perceptual change (gate b, and the guard against the
degenerate win); wall-clock and peak VRAM per sample; and a fixed-seed
determinism check. Report the Pareto over both metrics, never gate (a) alone.

---

## 6. Open questions

- **Does the depth prior actually see wall shifts?** Monocular depth is smooth,
  and a wall translated by a modest amount may cost less error than a plausible
  furniture change costs *outside* the mask. Measurable directly, before any
  sampling: synthetically perturb a room's geometry and plot aligned masked
  error vs. perturbation magnitude. **This experiment should probably run before
  v1 — it can invalidate the premise cheaply.**
- **Is the mask reliable enough?** ADE20K segmentation mislabels built-in
  cabinetry, mirrors, and large wall art. A mirror is the worst case:
  segmented as wall, but its depth is the reflected scene's. How much does gate
  (a) move if the mask is wrong on 5% of pixels?
- **Does guidance through the VAE decode fight the decoder?** The gradient
  passes through a lossy autoencoder; part of the signal may be spent on
  decoder-reconstruction artefacts rather than geometry. Comparing against a
  latent-space depth proxy would isolate this, at the cost of needing one.
- **Schedule transfer.** Strong-early/decayed-late is the FreeDoM
  recommendation, but does one schedule transfer across room types and denoising
  strengths, or is it per-scene? If per-scene, the method is much less useful.
- **What is the right adherence metric?** A CLIP-style prompt score
  (arXiv:2103.00020) and a perceptual-distance metric (LPIPS, arXiv:1801.03924)
  measure different things, and gate (b) is only as trustworthy as this choice.
  A metric that a frozen image can pass would silently void the whole gate.
- **How much headroom does SD1.5 leave?** A stronger prior would drift less to
  begin with, so the margin v1 has to close is a property of the base sampler
  and must be re-measured after any migration.

---

## 7. References (arXiv)

- **Depth Anything V2** — arXiv:2406.09414. *(the depth backbone; affine-invariant inverse depth)*
- **MiDaS** — Towards Robust Monocular Depth Estimation. arXiv:1907.01341. *(the scale-and-shift-invariant loss, §4 — the crux)*
- **EoMT** — Your ViT is Secretly an Image Segmentation Model. arXiv:2503.19108. *(the segmentation backend)*
- **DPS** — Diffusion Posterior Sampling for General Noisy Inverse Problems. arXiv:2209.14687.
- **FreeDoM** — Training-Free Energy-Guided Conditional Diffusion Model. arXiv:2303.09833. *(closest published match to v1)*
- **Universal Guidance** — Universal Guidance for Diffusion Models. arXiv:2302.07121.
- **ControlNet** — Adding Conditional Control to Text-to-Image Diffusion Models. arXiv:2302.05543. *(the v3 baseline)*
- **Classifier guidance** — Diffusion Models Beat GANs on Image Synthesis. arXiv:2105.05233.
- **Classifier-free guidance** — arXiv:2207.12598.
- **Latent Diffusion / Stable Diffusion** — arXiv:2112.10752. *(the base sampler)*
- **SDEdit** — guided image synthesis and editing with SDE. arXiv:2108.01073. *(the img2img recipe v0 uses)*
- **DINOv2** — arXiv:2304.07193. *(the features both backbones are built on)*
- **Depth Anything V1** — arXiv:2401.10891.
- **CLIP** — arXiv:2103.00020. **LPIPS** — arXiv:1801.03924. *(candidate gate-(b) metrics)*
