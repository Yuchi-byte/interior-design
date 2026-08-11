# Depth-Guided Sampling for Geometry-Preserving Room Renovation

**Date:** 2026-08-11 · **Scope:** how to hold a photographed room's architectural
shell fixed while a diffusion model restyles its contents — training-free
guidance vs. conditioning alternatives, and the affine-invariance problem that
any depth-space objective must solve first. Decision-oriented.

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
decode, and a strength schedule that must be tuned per scene family. **The
single non-negotiable technical detail is the affine alignment**: Depth Anything
V2 is affine-invariant, so a raw depth MSE is meaningless and the recommended
handling is a per-step closed-form weighted least-squares alignment over the
masked pixels (MiDaS's SSI loss, arXiv:1907.01341), differentiated through.

---

## 1. Landscape

Training-free guidance means steering a *frozen* pretrained sampler with the
gradient of a differentiable objective on the sample; the alternatives are
conditioning models that must be trained. Cost is given per denoising step
relative to one unguided step.

| Approach | Core idea | Cost | Needs training | Implementation weight |
|---|---|---|---|---|
| **Classifier guidance** (arXiv:2105.05233) | Shift the score by ∇ log p(y\|x_t) from a classifier trained on **noisy** x_t. The original recipe; the noisy-input requirement is what everything below removes. | ~1.2-2x (one classifier fwd+bwd) | Yes — a noise-aware classifier | Light *if* such a classifier exists; here it does not |
| **DPS** (arXiv:2209.14687) | Approximate the posterior for a known forward operator A by taking ∇_{x_t} ‖y − A(x̂₀(x_t))‖² through Tweedie's x̂₀. Turns any differentiable measurement into guidance, no noisy-input model needed. Principled, well-studied for linear inverse problems. | ~2-3x (backprop through the denoiser) | No | Medium — needs x̂₀ exposed and a clean A |
| **FreeDoM** (arXiv:2303.09833) | The same x̂₀ trick with an **arbitrary off-the-shelf energy** (a CLIP score, a face-ID net, a depth net), plus a time-travel/resampling step and a time-dependent strength. Closest published match to what roomgen does. | ~2-4x (energy net fwd+bwd; more with time-travel) | No | Medium — energy is a black box, schedule needs tuning |
| **Universal Guidance** (arXiv:2302.07121) | Forward guidance on x̂₀ + a **backward** correction that solves a small inner optimisation to make x̂₀ satisfy the constraint, then per-step recurrence. Stronger constraint satisfaction than plain forward guidance, at real extra cost. | ~3-8x (inner loop) | No | Medium-heavy |
| **ΠGDM / manifold-constrained variants** (e.g. manifold constraints, arXiv:2206.00941) | Correct the guidance direction using the denoiser's Jacobian or a manifold projection so the step stays on the data manifold. Mainly matters for hard linear inverse problems. | ~2-4x | No | Heavy; unclear payoff for a soft geometric prior |
| **ControlNet-depth** (arXiv:2302.05543) | *Not guidance.* A trained side network injects a depth hint into the UNet, so geometry is respected by construction at **no per-step gradient cost**. The strong baseline this project must beat or justify itself against. | ~1.2-1.4x (extra branch fwd only) | **Yes** — but checkpoints are public for SD1.5/SDXL | Light to use, zero to train (use published weights) |

Two cross-cutting notes for the decision. First, **the mask is the differentiator.**
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
  affine ambiguity (§3) rather than needing it resolved.
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
(§4, v2) exists to test.

---

## 3. The affine-invariance problem

Depth Anything V2, like MiDaS before it, predicts **affine-invariant inverse
depth**: its output is determined only up to an unknown scale `s` and shift `t`,
and — critically — **that pair is not stable across two different images**. Two
renders of the same room can differ by an arbitrary affine transform of the
predicted field. A raw MSE between two predictions therefore measures mostly the
affine mismatch, not the geometry: it can be huge for identical geometry and
small for genuinely drifted geometry. Any depth-space objective must remove this
degree of freedom before it measures anything.

| Option | What it does | Cost | Verdict |
|---|---|---|---|
| **Per-step closed-form alignment** (recommended) | Solve `min_{s,t} Σ w (s·d + t − r)²` over the masked pixels in closed form (a 2x2 system), then take the masked MSE of the aligned prediction. MiDaS's scale-and-shift-invariant loss (arXiv:1907.01341), used as a reward. | Negligible — five weighted sums and a 2x2 solve, fully differentiable | **Take this.** Exact, cheap, differentiable, and invariant by construction. |
| **Metric-depth model** | Swap in a metric predictor (ZoeDepth, arXiv:2302.12288; Depth Anything V2's metric fine-tunes; Metric3D-family models — *ID not verified*) so outputs are directly comparable. | Same forward cost | Fallback. Removes the ambiguity in principle, but metric heads are less accurate relatively, are indoor/outdoor-domain-sensitive, and their absolute scale on a *generated* image is not trustworthy. Do not build on it first. |
| **Rank / ordinal loss** | Score only the depth *ordering* of sampled pixel pairs (the DIW-style objective, arXiv:1604.03901), which is invariant to any monotone transform, not just affine. | Higher (pair sampling), noisier gradient | Interesting as a robustness check, weak as a primary signal: ordering is preserved under exactly the smooth deformations that constitute layout drift. |
| **Normalise by median/MAD** | Standardise both maps by robust statistics before comparison. | Negligible | A crude approximation of alignment. Strictly dominated by the closed form, which costs the same and is optimal. |

**Recommendation: per-step closed-form alignment, weights = the eroded shell
mask, differentiated through** (the alignment coefficients depend on the
prediction, so `s` and `t` carry gradient — do not stop-gradient them without
measuring the effect; treating them as constants is a defensible ablation, not
an obvious simplification). Two practical guards: the 2x2 determinant
degenerates when the mask is small or the masked depth is near-constant, and the
mask must be **eroded at segment boundaries** because monocular depth is
unreliable within a few pixels of an edge and those pixels otherwise dominate
the residual.

---

## 4. Recommended staged path

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

## 5. Open questions

- **Does the depth prior actually see wall shifts?** Monocular depth is smooth,
  and a wall translated by a modest amount may cost less error than a plausible
  furniture change costs *outside* the mask. Measurable directly, before any
  sampling: synthetically perturb a room's geometry and plot aligned masked
  error vs. perturbation magnitude. **This experiment should probably run before
  v1 — it can invalidate the premise cheaply.**
- **Is the mask reliable enough?** Mask2Former (arXiv:2112.01527) mislabels
  built-in cabinetry, mirrors, and large wall art. A mirror is the worst case:
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
- **SD1.5 vs. SDXL.** SDXL's stronger prior may drift less to begin with,
  shrinking the headroom v1 has to close; the baseline must be measured on
  whichever pipeline ships.

---

## 6. References (arXiv)

- **Depth Anything V2** — arXiv:2406.09414. *(the depth backbone; affine-invariant inverse depth)*
- **MiDaS** — Towards Robust Monocular Depth Estimation. arXiv:1907.01341. *(the scale-and-shift-invariant loss, §3 — the crux)*
- **Mask2Former** — Masked-attention Mask Transformer for Universal Image Segmentation. arXiv:2112.01527.
- **DPS** — Diffusion Posterior Sampling for General Noisy Inverse Problems. arXiv:2209.14687.
- **FreeDoM** — Training-Free Energy-Guided Conditional Diffusion Model. arXiv:2303.09833. *(closest published match to v1)*
- **Universal Guidance** — Universal Guidance for Diffusion Models. arXiv:2302.07121.
- **ControlNet** — Adding Conditional Control to Text-to-Image Diffusion Models. arXiv:2302.05543. *(the v3 baseline)*
- **Classifier guidance** — Diffusion Models Beat GANs on Image Synthesis. arXiv:2105.05233.
- **Classifier-free guidance** — arXiv:2207.12598.
- **Latent Diffusion / Stable Diffusion** — arXiv:2112.10752.
- **SDEdit** — guided image synthesis and editing with SDE. arXiv:2108.01073. *(the img2img recipe v0 uses)*
- **Manifold constraints for diffusion inverse problems** — arXiv:2206.00941. *(the ΠGDM-adjacent row of §1; ΠGDM itself is ICLR 2023 and I do not have a confident arXiv ID for it — verify before citing)*
- **DINOv2** — arXiv:2304.07193. *(the depth backbone's features)*
- **Depth Anything V1** — arXiv:2401.10891.
- **ZoeDepth** — metric depth. arXiv:2302.12288. *(the metric-depth fallback of §3)*
- **Marigold** — diffusion-based monocular depth. arXiv:2312.02145. *(alternative depth backend; ID believed correct but not re-verified)*
- **Depth in the Wild (ordinal depth)** — arXiv:1604.03901. *(the rank-loss option of §3)*
- **CLIP** — arXiv:2103.00020. **LPIPS** — arXiv:1801.03924. *(candidate gate-(b) metrics)*

Uncertainty note: the IDs for ΠGDM and for the Metric3D family are **not
verified** and are flagged inline above; every other ID here is one I am
confident in, but any citation load-bearing for a decision should be checked
against the abstract before it is repeated in a paper or a report.
