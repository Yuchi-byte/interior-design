# Vendored third-party code — not roomgen code

Everything else in this directory is upstream's, copied verbatim and held to
upstream's standards, not ours (see "Gate" below).

| | |
| --- | --- |
| Upstream | https://github.com/adh1s/mfm |
| Commit | `53c0f60db695cad88bbace8fb26469614e9e7d7d` (2026-05-14) |
| Paper | Potaptchik, Saravanan, Mammadov, Prat, Albergo, Teh — *Meta Flow Maps enable scalable reward alignment*, ICML 2026, [arXiv:2601.14430](https://arxiv.org/abs/2601.14430) |
| License | MIT (`LICENSE`, kept intact) — permits our use and modification with attribution |

## What was changed in the copy

Only two things: the `.git` directory is stripped, and `assets/` (20MB of paper
figures the README links) is not vendored. No source file is edited. Re-sync by
re-extracting the upstream tarball at a newer commit over this directory and
updating the hash above — do not hand-patch it, or the provenance line stops
being true.

## Not a workspace member

`packages/mfm/pyproject.toml` is upstream's and is deliberately **not** part of
our uv workspace (the root `[tool.uv.workspace]` excludes this path, which the
`packages/*` glob would otherwise pick up). Its pins are irreconcilable with
ours in two places:

- `transformers==4.50.1` against `roomgen-geometry`'s `transformers>=4.54`,
  which is the first release shipping `EomtForUniversalSegmentation`. A uv
  workspace resolves one `transformers` for every member, so admitting this one
  breaks segmentation.
- `diffusers @ git+https://github.com/huggingface/diffusers@af28ae2d`, a direct
  URL, which as a `tool.uv.source` would apply workspace-wide and pin
  `roomgen-guide` to that commit instead of a release.

So this package is vendored *source*, not an installed dependency: nothing under
`packages/roomgen-*` imports `mfm`, and `uv.lock` is untouched by it. Its
environment is built separately from `requirements.txt` — on the flow pod that
is `/workspace/flowmaps-venv`, deliberately outside the roomgen venv.

## Gate

Vendored code is excluded from ruff (root `[tool.ruff] extend-exclude`), and
mypy never sees it because `files` names our two packages explicitly. `make
check` therefore stays green without anyone rewriting upstream's style.
