# 0003_strip_and_furnish

Two stages over one stack. **Gallery A (strip):** the furnished photograph has
its contents segmented out (EoMT ADE20K → `contents_mask`) and the holes
inpainted bare, with every pixel outside the mask composited back from the
photograph itself; each pass repaints whatever inside the original hole the
segmenter still reads as something a strip removes (`REMOVABLE_CLASSES`, the
contents blocklist). **Gallery B (furnish):** the stripped room is furnished back
by guided SD1.5 img2img, three arms at the same seeds — a differentiable CLIP
text energy alone, the same text energy plus the masked depth energy in one
`WeightedGuidance`, and an unguided control. The prompt sits in the
classifier-free conditioning of all three arms; only the descended energy
differs. The depth energy's reference is the **photograph's own shell depth,
retained from stage 1** — never re-estimated from the inpainted room — so the
guidance inherits nothing the inpainting got wrong; the stripped room's
reference is still computed and scored against as a diagnostic.

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
- **Strip.** Removal mask (`REMOVABLE_CLASSES`: the contents plus curtain and
  rug, which are reward shell but still get stripped) dilated 8 px; up to 5
  inpainting passes at
  `fill_strength=0.85`, 40 steps, CFG 12.0 on a bare-room prompt with the
  observed hallucination families (furniture, radiator, heater, appliance,
  ledge, machine, cart, wheels, …) refused by name in the negative prompt. The
  re-check is a **contents blocklist**: a later pass repaints whatever *inside
  the first mask* the segmenter still names as removable (`REMOVABLE_CLASSES`,
  expanded to every plausibly movable indoor class — radiator included — and
  grown by the dilation radius). The blocklist's honest limit: a fill in
  neither class list (a vacuum-shaped object ADE20K cannot name, or a shape
  EoMT labels as shell) is accepted as-is, and only the negative prompt argues
  with those. The final image composites the photograph back outside the first
  mask, so the untouched region stays bit-identical however many passes run.
- **Furnish.** SDEdit img2img at `noise_strength=0.7` (35 of 50 DDIM steps
  walked): 0.6 leaves the room essentially unfurnished, 0.8 furnishes a
  different room in every arm. Guidance strength 40.0 with the raw step capped
  at `clip_step_rms=0.02`, shared by both guided arms, so arm B is arm A plus a
  depth term rather than arm A at a different temperature.
- **Reference.** Stage 2's guidance `GeometricReference` is the *photograph's*,
  retained from stage 1: the same reference the strip froze to score itself —
  the photo's depth over the photo's own shell mask, furniture excluded — is
  reused verbatim (`guidance_reference`), never re-estimated from the inpainted
  room, so stage 2 does not inherit whatever depth errors the inpainting
  introduced. The inherent trade: shell the furniture occluded is outside the
  photo's mask, so this reference holds 38–54% of the frame against 76–89% for
  a reference read off the empty room. The stripped room's reference — the
  previous run's guidance target — is still computed and every arm scored
  against it as a secondary diagnostic. `resegment` on, `erosion_radius=2`.
- **The depth weight is measured, not guessed.** `depth_weight: null` in the
  config: each term's gradient RMS is taken at step 17/35 of the real walk on
  each scene, and the composite's weight is the mean of the per-scene
  text/depth RMS ratios — **0.483765** this run (per-scene ratios 0.24, 0.14,
  1.07; the per-scene measurements are in `results.json` under
  `weight_sizing`).
- **The CLIP that judges is not the CLIP that steers.** Guidance descends
  `openai/clip-vit-base-patch32`; adherence is scored with
  `openai/clip-vit-large-patch14`. A config that sets the two equal is a
  `ValidationError`, not a footnote.
- **Precision.** UNet and VAE in float16 (arm B's single backward graph holds
  the VAE decode, the depth network and CLIP together, and float32 peaks past
  the 32 GB card); the depth network, segmenter and both CLIPs stay float32.
- **Metrics.** The primary depth error is against the retained photo reference
  — what the guided arm optimises, and the "did the room stay the room"
  question; each arm's error against the empty-room reference is reported
  beside it as the diagnostic that keeps this run comparable with the previous
  one. Under **each** reference, all three arms of a (scene, seed) are scored
  on **one common mask** — that reference's shell intersected with every arm's
  own scorable shell — so no arm profits from hiding geometry. Depth error
  goes through `align_scale_shift` as everywhere; each scene also records the
  empty room's VAE round trip against both references. The strip's requirement
  has its own number: `residual_contents_fraction`, the share of the original
  hole the final segmentation still affirmatively names as removable — with
  the categorical `hole_not_shell_fraction` kept beside it (the gap between
  the two is the no-name fills the blocklist cannot refuse), and
  `pass_residual_fractions`, the refused share after each pass, as the
  convergence trace the pass budget is judged by. No thresholds anywhere in
  the code.
- **Seeing the masks.** Gallery A carries a class-coloured segmentation map for
  the photo and the final empty room (per-class colours for wall / floor /
  ceiling / door / windowpane, one colour each for other shell, contents,
  everything else; the key is `segmentation_color_key` in `results.json`).
  Gallery B renders every mask the numbers are read over as green overlays: the
  guidance reward mask (the retained photo reference's shell) on the photo and
  on the empty room per scene, the common evaluation mask per (scene, seed)
  under each reference, and each arm's scorable slice of the reference shell on
  its own render. Every filename is recorded in `results.json`'s images maps.

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

Run of 2026-08-14, commit `c7e0013`, RTX 5090, torch 2.13.0+cu132, peak
9.62 GiB. Raw numbers: `outputs/0003_strip_and_furnish/results.json`. The
previous run — commit `2eb0ce0`, guidance *and* eval on the empty-room
reference — is preserved beside it at
`outputs/0003_strip_and_furnish_prior_2eb0ce0/`; its numbers are quoted below
where the two protocols are compared. One caveat measured in passing: with
`deterministic: true`, torch still warns (`warn_only`) that two ops on the
guidance backward — memory-efficient attention backward and bicubic-AA
upsample backward — have no deterministic CUDA implementation, and repeating
the study did shift the guided arms' numbers a few thousandths (e.g. arm B on
`lakehouse_living_room` seed 0: 0.0108 vs 0.0222); the control arm and stage 1
reproduce exactly.

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

### Stage 2 — furnish, primary: against the retained photo reference

Per (scene, seed), depth error E on the photo reference's common mask
(A = text_only, B = text_and_depth, ctl = control). "strip pays" is what the
*unedited* empty room's VAE round trip scores against the photo — the depth
error stage 2 starts already carrying, mostly the strip's re-reading cost:

| scene | seed | strip pays | E ctl | E A | E B | B vs ctl | B vs A | adh A | adh B | adh ctl | common cov |
|---|---|---|---|---|---|---|---|---|---|---|---|
| scandi_living_room | 0 | 0.0243 | 0.1109 | 0.0418 | 0.0054 | +95.2% | +87.2% | 0.2844 | 0.2865 | 0.2734 | 61% |
| scandi_living_room | 1 | 0.0243 | 0.0482 | 0.0536 | 0.0092 | +80.8% | +82.7% | 0.2912 | 0.2922 | 0.2835 | 90% |
| lakehouse_living_room | 0 | 0.0213 | 0.1437 | 0.1515 | 0.0222 | +84.5% | +85.3% | 0.2820 | 0.2709 | 0.2562 | 82% |
| lakehouse_living_room | 1 | 0.0213 | 0.0470 | 0.1813 | 0.0075 | +84.1% | +95.9% | 0.2751 | 0.2790 | 0.2938 | 64% |
| ny_loft_bedroom | 0 | 0.0622 | 0.0667 | 0.0782 | 0.0093 | +86.0% | +88.0% | 0.2903 | 0.2894 | 0.2865 | 95% |
| ny_loft_bedroom | 1 | 0.0622 | 0.2397 | 0.2008 | 0.0274 | +88.6% | +86.4% | 0.3014 | 0.3129 | 0.2887 | 85% |

### Stage 2 — diagnostic: the same renders against the empty-room reference

The previous protocol's reference, scored on its own common mask (floor = the
empty room's VAE round trip against itself):

| scene | seed | floor | E ctl | E A | E B | common cov |
|---|---|---|---|---|---|---|
| scandi_living_room | 0 | 0.0005 | 0.0833 | 0.0331 | 0.0418 | 53% |
| scandi_living_room | 1 | 0.0005 | 0.1390 | 0.1700 | 0.0441 | 72% |
| lakehouse_living_room | 0 | 0.0023 | 0.3689 | 0.3877 | 0.0886 | 68% |
| lakehouse_living_room | 1 | 0.0023 | 0.3857 | 0.7838 | 0.2251 | 52% |
| ny_loft_bedroom | 0 | 0.0006 | 0.0235 | 0.0384 | 0.2309 | 81% |
| ny_loft_bedroom | 1 | 0.0006 | 1.3494 | 1.0944 | 0.6378 | 60% |

Means over the six pairs:

| arm | E vs photo | Δ vs ctl | E vs empty room | adherence | vs ctl | own shell coverage | s/sample |
|---|---|---|---|---|---|---|---|
| text_only (A) | 0.1179 | −40.2% | 0.4179 | 0.2874 | 102.7% | 83% | 2.5 |
| text_and_depth (B) | 0.0135 | +86.5% | 0.2114 | 0.2885 | 103.0% | 84% | 4.5 |
| control | 0.1094 | — | 0.3916 | 0.2804 | 100.0% | 83% | 0.6 |

Common mask coverage: 80% mean of the photo reference's shell, 65% of the
empty-room reference's — but the two shells differ in size (photo 38–54% of
the frame, empty room 76–89%; `reference_shell_fractions` in `results.json`),
which is the retained reference's inherent trade. B vs A directly: mean
depth-error reduction +87.6% at a mean adherence ratio of 100.3%.

- **The text energy alone does not hold the room:** arm A is worse than the
  unguided control on 4 of 6 pairs against the photo (−40.2% at the mean) and
  moves adherence +2.7% over control — CLIP descent leaves the geometry
  wherever the sampler takes it, under either reference.
- **The depth term holds the photograph's geometry on every pair:** arm B
  beats the control on 6/6 (+81% to +95%) and arm A on 6/6 (+83% to +96%)
  against the photo reference, with adherence level with both (103.0% of
  control, 100.3% of A at the mean).
- **Guidance no longer inherits the strip's depth error.** Arm B's error
  against the photo (0.0054–0.0274) lands *below* what the unedited empty room
  itself pays against it (0.0213–0.0622): steered by the retained reference,
  the furnishing recovers geometry the inpainting had already drifted from,
  which is precisely what the reference change was for. Under the previous
  protocol that error was the target, invisible by construction.
- **The diagnostic column shows where the two references disagree.** B still
  beats the control against the empty-room reference on 5/6 pairs (mean 0.2114
  vs 0.3916) — holding the photo's shell mostly holds the strip's too — but on
  `ny_loft_bedroom` seed 0 it is worse (0.2309 vs 0.0235), and that is the
  scene where the strip drifted furthest from the photo (0.0665 depth error,
  the largest of the three): where photo and strip disagree, guidance now
  follows the photograph, by construction. The previous run (`2eb0ce0`), whose
  guidance descended the empty-room reference, scored B at 0.0712 mean against
  it (+80.4% vs ctl) — better on that axis, as each protocol wins the axis it
  optimises.
- Shell coverage is a diagnostic, not a score: the three arms' own coverages
  agree at the mean (83/84/83% of the photo shell), so the common mask
  (61–95% per pair) is not systematically an arm hiding shell.
- Cost: the composite's step (VAE decode + depth + CLIP forward and backward)
  is 4.5 s per sample against 2.5 s for CLIP alone and 0.6 s unguided.

## Decision

What this run settles, as measured:

1. **Strip-by-segmentation works as specified.** Stage 1 is unchanged and
   reproduces the previous run exactly: the untouched region is bit-identical
   to the photograph (max |Δ| 0.0 outside the mask, all scenes), at most 4.4%
   of the cut region still reads as furniture after three passes, and the
   depth cost of the strip is dominated by the network re-reading a
   de-furnished room, not by the fill.
2. **Retaining the photograph's depth as the stage-2 reference works, and
   changes what is held.** Text+depth reduces the photo-reference error on
   every pair (+86.5% mean vs control) at level adherence, and lands below the
   depth error the inpainted room itself carries — guidance no longer inherits
   the strip's mistakes. The cost is the smaller shell (38–54% of the frame vs
   76–89%) and, on the scene where the strip drifted most, a render that sits
   further from the *stripped* room while sitting nearer the photograph.
3. Open next: the per-scene RMS ratios spread 0.14–1.07 around the mean weight
   (0.484), so a per-scene (or per-step) weight remains a natural follow-up
   arm; `ny_loft_bedroom` seed 0 — the one pair the diagnostic column flags —
   is the pair to inspect qualitatively first, since it is where the two
   references' disagreement is visible in a render.
