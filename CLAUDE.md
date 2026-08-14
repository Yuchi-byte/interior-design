# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## What this is

roomgen renovates a photographed room with a diffusion model without letting
the room move: sampling is steered by a differentiable geometric reward derived
from monocular depth, restricted to the architectural shell (walls, floor,
ceiling, doors) and blind to the furniture, which is exactly what
renovation is allowed to change. The central hypothesis is that a pretrained
monocular depth network's features already encode the structure that matters —
planarity, room layout, relative scale — so a depth-space reconstruction error
is a usable proxy for "same room, different furniture" with no explicit 3D
reconstruction and no retraining (training-free guidance, the
universal-guidance / FreeDoM family). `docs/architecture.md` is the high-level
package map — **keep it updated** whenever the package layout or module
responsibilities change (one sentence max per module).

## Conventions

Coding style is governed by [docs/coding-guidelines.md](docs/coding-guidelines.md):
clear abstractions, strict typing IS the documentation, comments minimal and
only on user-facing API or genuine *why*.

- **uv workspace**, Python 3.12, one venv for the whole repo. The root
  `pyproject.toml` is a virtual workspace coordinator (tooling + dependency
  groups only, no `[project]` table) and pins torch/torchvision to the
  `pytorch-cpu` index so a bare `uv sync` never pulls CUDA wheels.
  `packages/roomgen-geometry` is the geometric grounding stack (depth,
  segmentation, masking, alignment, reward); `packages/roomgen-guide` is guided
  sampling plus the experiment harness
  (`config`/`experiment`/`settings`/`metrics`/`eval`/`determinism`).
  The geometry package never imports the guide package. `packages/mfm` and
  `packages/fmrg` are vendored third-party source (see each `NOTICE.md`): not
  workspace members, not imported by our packages, outside every gate — never
  edit or reformat them.
- All config objects are pydantic v2 (`extra="forbid"`) via the `Config` base
  in `packages/roomgen-guide/src/roomgen_guide/config.py` — a typo'd field is a
  `ValidationError`, not a silent no-op.
- **Commit frequently.** After any coherent set of related changes, commit
  without waiting to be asked. Never `git add -A` — stage specific files.
  Commit messages explain *why*, not what the diff already shows.

## Environment

One venv for the whole repo (py3.12), fully declared in the two package
`pyproject.toml`s — a plain `uv sync` builds it, and `uv.lock` is committed:

```bash
uv sync                                              # CPU torch (~200MB wheels)
uv pip install --torch-backend=auto torch torchvision # layer CUDA wheels on top
```

## Checks

```bash
uv run ruff check conftest.py packages/roomgen-geometry/src packages/roomgen-geometry/tests packages/roomgen-guide/src packages/roomgen-guide/tests experiments
uv run ruff format --check conftest.py packages/roomgen-geometry/src packages/roomgen-geometry/tests packages/roomgen-guide/src packages/roomgen-guide/tests experiments
uv run mypy packages/roomgen-geometry/src packages/roomgen-guide/src
uv run mypy packages/roomgen-geometry/src packages/roomgen-guide/src experiments/NNNN_slug  # one per framework
uv run pytest -q                    # CI runs `-m "not models"`; locally run it unfiltered
```

`experiments/` is gated as a whole directory, not per framework: experiment
code is real code — it loads the checkpoints and computes the numbers a report
quotes — so a new `NNNN_slug/` is inside the gate the day it is created. mypy is
the one check that takes the frameworks one at a time rather than the directory
at once: each framework owns a top-level `run.py`, a numbered directory is not a
package name mypy can qualify one with, and a single process therefore sees
several modules called `run` and refuses them all. `make typecheck` walks
`experiments/[0-9]*` itself, so the loop is written down once.

`make check` runs exactly those four (see the `Makefile` for the rest of the
wrappers). `uv run pre-commit install` wires all four into a pre-commit hook; CI
(`.github/workflows/ci.yml`) runs the same gate with the `models` tests
deselected.

## CI and its constraints

CI is a single job mirroring local setup: `uv sync` (py3.12), then
ruff/format/mypy/pytest. The runner has **no GPU, no Hugging Face cache, and no
model weights**.

**Model weights are never committed and never downloaded in CI.** Tests that
need real checkpoints carry the `models` marker (declared in the root
`pyproject.toml`) and are deselected by CI's `pytest -q -m "not models"`.
Everything else must run against small fakes that satisfy the `DepthEstimator`
/ `Segmenter` / pipeline Protocols — a fake depth backend returning a synthetic
disparity field is enough to test masking, alignment, the reward, and the
guidance loop end to end. Run the unfiltered suite locally before merge.

## The reward must stay differentiable

The single easiest way to silently break this project. Nothing on the path from
the latent to the scalar reward may call `torch.no_grad()`, `.detach()`,
`.item()`, `.numpy()`, or an in-place op (`x += ...`, `x.clamp_()`,
`tensor[mask] = v`) on a tensor carrying gradient. The depth backend in
`depth/depth_anything.py` is kept differentiable end to end, including its
preprocessing (resize/normalise as tensor ops, never a PIL round-trip).

The `GeometricReference` is computed once from the input photo and **is**
detached — it is a constant. The per-step prediction never is. A detached graph
yields a zero gradient, guidance becomes a no-op, and sampling proceeds looking
perfectly normal: tests must assert the gradient is non-zero and finite, not
that the loop runs.

## Never compare raw depth maps

Depth Anything V2 predicts *affine-invariant inverse depth*: determined only up
to an unknown scale and shift, and that scale/shift is not stable between two
different images. A raw MSE between two depth maps is meaningless. **Every
depth comparison goes through `align_scale_shift` first** (`reward.py` is the
only place that should need to; see `docs/architecture.md` for the closed form).
This is a hard rule — a PR comparing unaligned depth is wrong regardless of the
numbers it produces. The invariance stops at a *positive* scale: the fit is
floored at zero because near/far ordering is the geometry being scored, and
un-flooring it scores a depth-inverted render as a perfect match.

## Determinism

Seeds are explicit and configured, never implicit. Pass a `torch.Generator`
through the call chain; never rely on the global RNG, and never call
`torch.manual_seed` inside library code. Two runs at one seed on one device
must produce the same image and the same numbers.

## Experiments

ML experiments live in `experiments/` (contract: `experiments/README.md`).
Numbered `NNNN_slug/` directories are frameworks, not single runs. A
framework **measures and reports** — the rendered comparisons plus the numbers
behind them (both arms' depth error and its reduction, both arms' prompt
adherence and its ratio, and how much shell the pair was scored over) — and
asserts no pass/fail: there are no thresholds in the code, and judging the
evidence is the owner's. Depth error still never travels alone: winning it by
not renovating, or by filling the room with depth-satisfying slabs, is what the
adherence (or preference) score is there to expose — shell coverage is a
diagnostic beside it, and a slab-filled render reads as *more* shell, not less.
`experiments/JOURNAL.md` is **owner-managed**: the owner writes entries
or dictates them verbatim — never append to it unprompted.

## Plans are temporary; docs/research is owner-reviewed

Plan documents are temporary objects: once the work they describe is complete,
delete the plan (and any references to it) rather than archiving it — current
state belongs in code, tests, and `docs/architecture.md`. `docs/research` is
the owner's curated research record: propose changes (new findings, scope
shifts, corrected assumptions) rather than silently rewriting, and let the
owner confirm before editing the file itself.

## No hardcoded paths

Never hardcode a working directory, output path or checkpoint cache. Use
`RoomgenSettings` (`packages/roomgen-guide/src/roomgen_guide/settings.py`),
overridable via `ROOMGEN_*` environment variables — `ROOMGEN_WDIR` (default
`outputs/` at the repo root, gitignored) and `ROOMGEN_HF_HOME` (default
`.cache/huggingface/` at the repo root, also gitignored). Those two are the
whole env-var contract; plain `HF_HOME` is not read by our code. Both defaults
are anchored to the enclosing repo root, and a process started outside a clone
gets an error naming the variable to set rather than a cache silently rooted at
its working directory.

## Parallel sessions

Multiple agents may work this repo at once. Never run `git reset --hard`,
`git checkout -- .`, or `git clean -fd` to tidy your own working tree — it can
destroy another session's uncommitted edits. Use `git stash` on your own paths
only, and commit green checkpoints promptly.
