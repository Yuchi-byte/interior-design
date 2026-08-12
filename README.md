# roomgen

roomgen renovates a photographed room with a diffusion model without letting the
room move: sampling is steered by a differentiable monocular-depth reward
restricted to the architectural shell, so walls, windows and scale hold while
the furniture is free to change. On Stable Diffusion 1.5 img2img at 512² it
removes ~74% of an unguided control's shell depth error at unchanged prompt
adherence (`experiments/0002_geometric_guidance`), for ~4x the wall-clock.

## Layout

```
pyproject.toml            # virtual uv workspace root: tooling + dependency groups only
packages/
  roomgen-geometry/       # the geometric grounding stack
    src/roomgen_geometry/
      depth/                #   DepthEstimator protocol + Depth Anything V2 backend
      segmentation/         #   Segmenter protocol + EoMT, ADE20K shell/contents split
      masking.py            #   architectural_mask + boundary erosion
      alignment.py          #   align_scale_shift (closed-form weighted least squares)
      reward.py             #   masked_depth_error, the scale/shift-invariant objective
      reference.py          #   GeometricReference, frozen from the input photo
  roomgen-guide/          # guided sampling + experiment harness
    src/roomgen_guide/
      guidance/             #   Guidance protocol + the geometric depth energy
      sampling/guided.py    #   the guided denoising loop over a diffusers pipeline
      eval/                 #   what a finished render is scored on: depth, adherence
      renovate.py           #   end-to-end img2img renovation entry point (SD1.5)
      settings.py           #   RoomgenSettings (ROOMGEN_* env vars)
      metrics.py            #   per-run measurements a framework reports
      determinism.py        #   best-effort deterministic kernels, before first CUDA use
  mfm/                    # vendored third-party flow-map source (see NOTICE.md)
experiments/              # numbered experiment frameworks (see experiments/README.md)
docs/
  architecture.md         # package/module map (kept current)
  research/               # the research direction (owner-reviewed)
```

One uv workspace, one py3.12 venv, fully declared in the package
`pyproject.toml`s. torch is pinned to the CPU index, so a bare `uv sync` stays
light; layer CUDA wheels on top with
`uv pip install --torch-backend=auto torch torchvision` (see
`packages/roomgen-guide`).

## Getting started

```bash
git clone <repo-url>
cd interior-design
uv sync
uv run pre-commit install
```

Checks (also run in CI): see `CLAUDE.md`'s Checks section.
