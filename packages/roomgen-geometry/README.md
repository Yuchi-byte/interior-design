# roomgen-geometry

The geometric grounding stack. `depth/` and `segmentation/` wrap pretrained
models behind `DepthEstimator` / `Segmenter` protocols, `masking.py` turns a
segmentation into the architectural shell mask, `alignment.py` removes the
depth model's affine ambiguity, and `reward.py` scores one image's geometry
against another's. No diffusers dependency: this package knows nothing about
sampling.

`reference.py` is the entry point a caller usually wants — `GeometricReference`
runs both models once on the input photo and freezes the (masked depth, mask)
pair that everything downstream compares against.

## Affine-invariant depth

Depth Anything V2 predicts **inverse depth up to an unknown scale and shift**,
and that pair is not stable between two images — the same room rendered twice
can score an enormous raw MSE while being geometrically identical, and a real
layout drift can score a small one. **A raw MSE between two depth maps is
meaningless.**

`align_scale_shift` fixes this by solving the 2x2 weighted least-squares system
for `(s, t)` minimising `Σ w (s·d + t − r)²` over the masked pixels in closed
form, and the error is taken on the aligned prediction. The result is invariant
to any *positive* affine rescaling of the prediction — which is exactly the
invariance the model's output has. It stops there: `s` is floored at 0, since
which surfaces are near and which are far is the geometry being scored and not
an unknown of the backend, and a prediction the fit could only match by
reflecting is charged for it instead of being credited. The derivation, the
degenerate-`det` guard and the refused-variance term are in
`docs/architecture.md`; the rule is in `CLAUDE.md`: nothing compares depth
without aligning first.

Two further details matter numerically. The shell mask is **eroded** at segment
boundaries — monocular depth is unreliable within a few pixels of an edge, and
those pixels otherwise dominate the residual. And the whole path stays
differentiable: preprocessing is tensor ops, never a PIL round-trip, and no
`.detach()` / `no_grad` / in-place op appears between input and reward.
Segmentation is the exception and runs under `no_grad` — the mask is a constant.

## Models

Both are pulled from the Hugging Face Hub on first use and cached under
`ROOMGEN_HF_HOME`. Weights are never committed.

| Role | Checkpoint | Licence | Approx. download |
|---|---|---|---|
| Depth | `depth-anything/Depth-Anything-V2-Small-hf` (DPT head on DINOv2-S) | Apache-2.0 | ~0.1 GB |
| Segmentation | `tue-mps/ade20k_semantic_eomt_large_512` (150 ADE20K classes) | MIT | ~1.3 GB |

The licence column is load-bearing: the larger Depth Anything V2 tiers are
CC-BY-NC, and the DINOv3-backed EoMT checkpoints carry their own restrictive
terms, so neither is a free swap.

Both sit behind protocols, so a backend swap — a metric-depth model, a
different segmentation head — is a config change that touches neither
`masking.py`, `alignment.py`, nor `reward.py`.

## Testing

Tests needing these checkpoints carry the `models` marker and are excluded from
CI (no GPU, no Hub cache there). Everything else — masking, alignment, the
reward's invariances, gradient flow — runs against small fakes satisfying the
protocols, on CPU, in seconds.
