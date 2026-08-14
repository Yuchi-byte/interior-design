# 0003_strip_and_furnish

Two stages over one stack. **Gallery A (strip):** the furnished photograph has
its contents segmented out (EoMT ADE20K → `contents_mask`) and the holes
inpainted bare, with every pixel outside the mask composited back from the
photograph itself. **Gallery B (furnish):** the stripped room is furnished back
by guided SD1.5 img2img, three arms at the same seeds — a differentiable CLIP
text energy alone, the same text energy plus the masked depth energy in one
`WeightedGuidance`, and an unguided control. The prompt sits in the
classifier-free conditioning of all three arms; only the descended energy
differs.

## Hypothesis

Two claims, one per stage.

1. **A room can be stripped without moving.** Segmentation plus true
   inpainting — regenerate only the contents mask, composite the photograph
   back everywhere else — leaves an empty room that the depth network still
   reads as the same room. The strip's depth error against the photograph
   should sit near what re-reading a de-furnished room costs on pixels that
   were never touched at all, and the untouched region must be untouched to
   the bit.
2. **Furnishing an empty room is where the depth term earns its keep.** At a
   noise strength high enough to actually furnish (0.7 here), a text energy
   alone has no reason to keep the shell; adding the depth energy — weight
   *measured* from the two terms' gradient RMS on real samples, not guessed —
   should hold the room's geometry while adherence stays level. The degenerate
   wins stay visible: not furnishing at all is exposed by adherence, and
   depth-satisfying slabs by adherence plus the galleries.

## Setup

Pinned in `conf/study.yaml`; `conf/smoke.yaml` is the same flow on weightless
doubles, `conf/pilot.yaml` one scene on the real stack.

- **Scenes.** Three of the sha256-verified benchmark photos — two living rooms
  (`scandi_living_room`, `lakehouse_living_room`) and one bedroom
  (`ny_loft_bedroom`) — centre-cropped and resized to 512², one furnishing
  prompt each.
- **Seeds.** `[0, 1]`, explicit `torch.Generator`s on the sampling device. The
  strip runs at seed 0.
- **Strip.** Contents mask dilated 8 px; up to 3 inpainting passes at
  `fill_strength=0.85`, 40 steps, CFG 12.0 on a bare-room prompt with the
  furniture refused by name in the negative prompt; every later pass repaints
  only what the segmenter still calls contents *inside the first mask*, and the
  final image composites the photograph back outside that mask.
- **Furnish.** SDEdit img2img at `noise_strength=0.7` (35 of 50 DDIM steps
  walked): 0.6 leaves the room essentially unfurnished, 0.8 furnishes a
  different room in every arm. Guidance strength 40.0 with the raw step capped
  at `clip_step_rms=0.02`, shared by both guided arms, so arm B is arm A plus a
  depth term rather than arm A at a different temperature.
- **Reference.** Stage 2's `GeometricReference` is frozen from the *stripped*
  room, not the photograph — the empty shell is the geometry a furnishing is
  asked to keep. `resegment` on, `erosion_radius=2`.
- **The depth weight is measured, not guessed.** `depth_weight: null` in the
  config: each term's gradient RMS is taken at step 17/35 of the real walk on
  each scene, and the composite's weight is the mean of the per-scene
  text/depth RMS ratios — **0.652112** this run (per-scene ratios 0.48, 0.07,
  1.41; the per-scene measurements are in `results.json` under
  `weight_sizing`).
- **The CLIP that judges is not the CLIP that steers.** Guidance descends
  `openai/clip-vit-base-patch32`; adherence is scored with
  `openai/clip-vit-large-patch14`. A config that sets the two equal is a
  `ValidationError`, not a footnote.
- **Precision.** UNet and VAE in float16 (arm B's single backward graph holds
  the VAE decode, the depth network and CLIP together, and float32 peaks past
  the 32 GB card); the depth network, segmenter and both CLIPs stay float32.
- **Metrics.** All three arms of a (scene, seed) are scored on **one common
  mask** — the reference shell intersected with every arm's own scorable
  shell — so no arm profits from hiding geometry. Depth error goes through
  `align_scale_shift` as everywhere; each scene also records the VAE
  round-trip floor. No thresholds anywhere in the code.

### Running it

```bash
# plumbing, CPU, offline, seconds — weightless doubles end to end
uv run --no-sync python experiments/0003_strip_and_furnish/run.py --config smoke

# the study: GPU, prefetched checkpoints + the two CLIPs
uv run --no-sync python experiments/0003_strip_and_furnish/run.py --config study
```

Artifacts land under `$ROOMGEN_WDIR/0003_strip_and_furnish/`: `results.json`,
`gallery_a_strip/`, `gallery_b_furnish/`.

## Results

Run of 2026-08-14, commit `2eb0ce0`, RTX 5090, torch 2.13.0+cu132, peak
23.51 GiB. Raw numbers: `outputs/0003_strip_and_furnish/results.json`.

### Stage 1 — strip

| scene | contents cut | shell | residual contents | E vs photo | E on never-repainted shell (coverage) | E floor (VAE) | max abs. diff outside mask | seconds |
|---|---|---|---|---|---|---|---|---|
| scandi_living_room | 41% | 54% | 4.4% | 0.0214 | 0.0212 (90%) | 0.0009 | 0.0 | 2.1 |
| lakehouse_living_room | 33% | 49% | 0.0% | 0.0183 | 0.0175 (97%) | 0.0011 | 0.0 | 0.7 |
| ny_loft_bedroom | 59% | 38% | 0.0% | 0.0665 | 0.0637 (82%) | 0.0007 | 0.0 | 0.7 |

- **The compositing claim holds exactly:** outside the dilated contents mask
  the strip differs from the photograph by 0.0 on every scene, against a raw
  (uncomposited) inpainter output that differs by up to 0.75. The stripped
  room is the photograph everywhere the mask did not reach.
- **Residual contents after up to 3 passes:** 4.4% / 0.0% / 0.0% of the cut
  region still segments as furniture (one pass alone had left ~49% on the
  pilot scene).
- The strip's depth error vs the photo (0.018–0.067) is dominated by the
  *re-reading* cost, not the fill: the never-repainted shell — pixels
  identical to the photograph — scores 0.0175–0.0637 on its own, within a few
  percent of the full-shell number, while the VAE floor is ~0.001. Removing
  the furniture changes what the monocular network infers about the untouched
  walls; the fill itself adds almost nothing on top.

### Stage 2 — furnish (3 scenes × 2 seeds, one common mask per pair)

Per (scene, seed), depth error E on the common mask (A = text_only,
B = text_and_depth, ctl = control; floor = VAE round trip of the empty room):

| scene | seed | floor | E ctl | E A | E B | B vs ctl | B vs A | adh A | adh B | adh ctl | common cov |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scandi_living_room | 0 | 0.0005 | 0.0821 | 0.0371 | 0.0088 | +89.3% | +76.3% | 0.2836 | 0.2810 | 0.2734 | 55% |
| scandi_living_room | 1 | 0.0005 | 0.1362 | 0.1561 | 0.0110 | +91.9% | +92.9% | 0.2867 | 0.2916 | 0.2835 | 72% |
| lakehouse_living_room | 0 | 0.0023 | 0.3649 | 0.3897 | 0.0134 | +96.3% | +96.6% | 0.2819 | 0.2734 | 0.2562 | 68% |
| lakehouse_living_room | 1 | 0.0023 | 0.3870 | 0.7971 | 0.1621 | +58.1% | +79.7% | 0.2758 | 0.2648 | 0.2938 | 52% |
| ny_loft_bedroom | 0 | 0.0006 | 0.0350 | 0.0369 | 0.0130 | +63.0% | +64.9% | 0.2829 | 0.2846 | 0.2865 | 81% |
| ny_loft_bedroom | 1 | 0.0006 | 1.3709 | 1.0462 | 0.2188 | +84.0% | +79.1% | 0.3048 | 0.2918 | 0.2887 | 63% |

Means over the six pairs:

| arm | depth error | Δ vs ctl | adherence | vs ctl | own shell coverage | s/sample |
|---|---|---|---|---|---|---|
| text_only (A) | 0.4105 | −9.0% | 0.2860 | 102.2% | 71% | 2.5 |
| text_and_depth (B) | 0.0712 | +80.4% | 0.2812 | 100.5% | 70% | 4.6 |
| control | 0.3960 | — | 0.2804 | 100.0% | 71% | 0.7 |

Common mask coverage: 65% mean. B vs A directly: mean depth-error reduction
+81.6% at a mean adherence ratio of 98.4%.

- **The text energy alone does not hold the room:** arm A's depth error is
  worse than the unguided control's on 4 of 6 pairs and −9.0% at the mean.
  Descending CLIP moves adherence (+2.2% over control) and leaves the geometry
  wherever the sampler takes it.
- **Adding the measured depth term recovers the geometry on every pair:**
  arm B beats the control on 6/6 (range +58% to +96%) and beats arm A on 6/6
  (+65% to +97%), with adherence within a few percent of both (100.5% of
  control, 98.4% of A at the mean). The furnishing this buys is on view in
  `gallery_b_furnish/` beside the depth map of every render.
- Arm B's absolute error (0.0088–0.2188) sits one to two orders above the VAE
  floor (0.0005–0.0023): the shell is held, not returned verbatim — consistent
  with 35 of 50 steps rewalked and whole-room furnishing visible in the
  gallery.
- Shell coverage is a diagnostic, not a score: the three arms' own coverages
  agree at the mean (71/70/71%), so the common mask (52–81% per pair) is not
  systematically an arm hiding shell.
- Cost: the composite's step (VAE decode + depth + CLIP forward and backward)
  is 4.6 s per sample against 2.5 s for CLIP alone and 0.7 s unguided.

## Decision

What this run settles, as measured:

1. **Strip-by-segmentation works as specified.** The untouched region is
   bit-identical to the photograph (max |Δ| 0.0 outside the mask, all scenes),
   at most 4.4% of the cut region still reads as furniture after three passes,
   and the depth cost of the strip is dominated by the network re-reading a
   de-furnished room, not by the fill.
2. **On furnishing empty rooms, the depth term is what holds the geometry.**
   Text-only guidance leaves depth error at control level (−9.0% at the mean,
   worse on 4/6 pairs); text+depth reduces it on every pair (+80.4% mean vs
   control) with adherence level with both other arms — under an adherence
   judge that is a different CLIP from the one the guidance descends, and a
   depth weight read off the measured gradient RMS ratio (0.652) rather than
   tuned.
3. Open next: the per-scene RMS ratios spread 0.07–1.41 around the mean weight,
   so a per-scene (or per-step) weight is a natural follow-up arm; and
   `lakehouse_living_room` seed 1 — the weakest B-vs-ctl pair (+58.1%) at the
   smallest common mask (52%) — is the pair to inspect qualitatively first.
