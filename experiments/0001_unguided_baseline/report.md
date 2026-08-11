# 0001_unguided_baseline

The v0 framework: plain `diffusers` img2img renovation with **no guidance**,
scored on the same two metrics every later framework is gated on. Its job is to
establish that the problem exists and to calibrate the thresholds of the
two-sided gate (`experiments/README.md`).

## Hypothesis

Unguided img2img renovation drifts geometrically: the architectural shell moves
(walls shift, window openings migrate, apparent room scale changes) even at
denoising strengths low enough to leave the furniture recognisably re-styled.
Concretely, the masked depth error of unguided output against the
`GeometricReference` built from the input photo is **large** — far above the
error between the reference and a round-tripped copy of the input photo, and
therefore large enough to leave real headroom for guidance to close.

Two numbers come out of this framework, and both are inputs to every later one:

1. **The headroom.** `E_unguided` (unguided output vs. reference) against
   `E_floor` (the input photo encoded and decoded through the VAE at the same
   resolution vs. reference — the error a *perfect* geometry-preserving
   renovation still pays from reconstruction alone). Gate (a)'s margin is
   derived from the gap between them, not picked.
2. **The adherence floor.** The prompt-adherence score of unguided output at
   each denoising strength, which is by construction the *unconstrained* best a
   guided sampler could hope for. Gate (b)'s floor is set as a fraction of it.

Corollary hypothesis, tested by the strength sweep: adherence and geometric
fidelity trade off monotonically in denoising strength, so no choice of
strength alone solves the problem. If a strength exists that both preserves
geometry and renovates convincingly, guidance is unnecessary and this framework
says so.

## Setup

**Not yet fixed.** Scene set, prompts, seeds and sampler budget are pinned in
`conf/` when this framework's `run.py` lands; the intended shape:

- **Scenes.** A small fixed set of interior photographs spanning room types and
  camera fields of view, each with a Mask2Former ADE20K segmentation whose
  shell mask covers a substantial fraction of the frame.
- **Reference.** `GeometricReference` per scene: Depth Anything V2 inverse
  depth, masked to the eroded architectural shell, computed once and frozen.
- **Sampler.** Stable Diffusion / SDXL img2img via `diffusers`, no guidance
  term, fixed step count, denoising strength swept over a small grid.
- **Prompts.** One fixed renovation prompt per scene ("same room, <style>
  furniture"), held identical across every variant and every later framework.
- **Seeds.** Several explicit seeds per (scene, prompt, strength), threaded as a
  `torch.Generator` rather than the global RNG — so the unguided control here is
  bit-reproducible and directly comparable to the guided runs of later
  frameworks at the same seed.
- **Metrics.** `roomgen_guide.eval`: masked depth error against the reference,
  and the prompt-adherence / perceptual metric.

### Verdict gate

This framework is **calibration, not a claim**, so it asserts only that the
headroom exists:

- **(H) headroom:** `E_unguided >= E_floor * M` for a margin `M` fixed in
  `conf/`, on every scene — unguided drift is well clear of the
  VAE-reconstruction floor.

The two-sided gate of `experiments/README.md` (depth error down by a margin,
adherence not below a floor) is what *later* frameworks assert; its two
constants are derived from this framework's numbers.

## Results

**Not yet run.** No run of this framework has been performed. Every cell below
is a placeholder — no number in this report is measured, estimated, or
otherwise real.

### Geometry vs. adherence by denoising strength

| strength | E_floor (recon) | E_unguided | headroom E_unguided / E_floor | prompt adherence |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

### Per-scene, at the selected strength

| scene | shell mask coverage | seeds | E_unguided (mean +/- sd) | prompt adherence (mean +/- sd) | gate (H) |
|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD |

### Calibrated constants (the deliverable)

| constant | meaning | value |
|---|---|---|
| gate (a) margin | required depth-error reduction of guided vs. unguided | TBD |
| gate (b) floor | minimum prompt adherence, as a fraction of unguided | TBD |
| reference strength | denoising strength later frameworks hold fixed | TBD |

## Decision

**Pending — the run has not been performed.** Nothing is settled; the TBD
constants above are the deliverable, and until they exist no later framework has
a calibrated gate to assert against.

What this framework settles once run: (1) whether unguided renovation drifts
enough to be worth correcting, (2) whether some denoising strength already
resolves the trade-off without guidance, and (3) the two constants of the
two-sided gate. If (2) comes out true, the guidance path is not justified for
this sampler and that is the finding, reported as such.

Next: `0002` adds `GeometricGuidance` on the shell mask over this same sampler —
same scenes, same prompts, same seeds — and asserts the two-sided gate against
the unguided control recorded here.
