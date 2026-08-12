# Vendored third-party code — not roomgen code

Everything else in this directory is upstream's, copied verbatim and held to
upstream's standards, not ours (see "Gate" below).

| | |
| --- | --- |
| Upstream | https://github.com/jrrhuang/fmrg |
| Commit | `be485ba40f1d1c163b95ad61112aca6baa13ed22` (2026-06-10) |
| Paper | Huang, Lin, Shah, Nair, Boffi — *How to Guide Your Flow: Few-Step Alignment via Flow Map Reward Guidance*, ICML 2026, [arXiv:2604.27147](https://arxiv.org/abs/2604.27147) |
| License | Apache-2.0 (`LICENSE`, kept intact) — permits our use and modification with attribution |

Upstream's root `LICENSE` states Apache-2.0 and carries the standard grant
paragraph, but abbreviates the body with a note that the full text is "to be
expanded before public release". The complete Apache-2.0 text does ship in the
tree, at `flux_two_timestep/LICENSE`. Both files are kept as they are.

## What was changed in the copy

Only two things: the `.git` directory is stripped, and `assets/` (3MB of paper
figures the README links, two of them over the 512kB `check-added-large-files`
ceiling) is not vendored. No source file is edited. Re-sync by re-extracting the
upstream tarball at a newer commit over this directory and updating the hash
above — do not hand-patch it, or the provenance line stops being true.

## Not a workspace member

Upstream ships no `pyproject.toml` at all — its environment is a conda
`environment.yml` — so the root `packages/*` glob matching this directory is a
hard `uv` error ("missing a `pyproject.toml`") rather than a resolution
problem. The root `[tool.uv.workspace]` therefore excludes this path.

Its pins could not join our resolution anyway: `environment.yml` takes
`torch==2.5.1+cu121` / `torchvision==0.20.1+cu121` from the PyTorch CUDA index,
against the root's declarative pin of torch to `pytorch-cpu`. (Those two pins
are also dead on our hardware — cu121 has no `sm_120` kernels, so it cannot run
on the RTX PRO 4500 Blackwell pod at all; the pod venv is built on a
Blackwell-capable torch instead.) The remaining pins are milder —
`diffusers==0.34.0` against `roomgen-guide`'s `diffusers>=0.31`, and
`transformers>=4.41.2`, which `roomgen-geometry`'s `>=4.54` floor satisfies —
but an exact `diffusers` pin applied workspace-wide is still not something
`roomgen-guide` should inherit.

So this package is vendored *source*, not an installed dependency: nothing under
`packages/roomgen-*` imports it, and `uv.lock` is untouched by it. Its
environment is built separately — on the FLUX pod that is `/root/fmrg-venv`,
deliberately outside the roomgen venv and on pod-local disk.

## Checkpoints are gated

`checkpoints_download.sh` fetches two flow-map LoRAs from the ungated
`gabeguofanclub/flux-1-dev-flowmap-lsd`, but the base model it adapts,
`black-forest-labs/FLUX.1-dev`, is a **gated** Hub repo (`gated: "auto"`):
running anything here needs the license accepted on the model page and an
`HF_TOKEN` in the environment. Weights are never committed here.

## Gate

Vendored code is excluded from ruff (root `[tool.ruff] extend-exclude`), and
mypy never sees it because `files` names our two packages explicitly. `make
check` therefore stays green without anyone rewriting upstream's style.
