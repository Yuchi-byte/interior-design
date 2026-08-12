# Coding guidelines

Binding for all code in `src/`, `tests/`, and `experiments/`.

## Types are documentation

- Strict typing is the primary documentation. A precise signature replaces a
  docstring that restates it.
- Never write a comment or docstring that repeats what the name and type
  already say.
- Prefer narrowing types (`Literal`, discriminated unions, jaxtyping shapes like
  `Float[Tensor, "batch h w"]`) over prose describing valid values.

## Comments

- Comments are for the user-facing API surface and for *why*, never *what*.
- Public entry points (things an experiment or another module imports) may
  carry a short docstring: one line, plus args only when a contract is not
  expressible in the type (units, disparity vs. depth convention, whether a
  tensor is expected to carry gradient, side effects).
- Internal helpers get no docstrings. If an internal needs explaining, rename
  or restructure it first.
- Inline comments only where the code is surprising (workarounds, API churn,
  numerical subtleties) — and they state the reason, ideally with a reference.
  The degenerate-mask guard in `align_scale_shift` is the archetype: a
  near-singular 2x2 normal-equation solve needs one line saying so.

## Docs describe purpose and current state, not history

- A docstring states what the abstraction is *for*, at the level of the
  abstraction itself — the role it plays for its caller, not how it is built.
- Never cite design history: no decision numbers, plan/spec section references,
  upstream-project comparisons, or "this replaces X". Git history is the
  record; a docstring reader is a user, not a reviewer.
- All documentation — docstrings, comments, `docs/*.md` — describes the code
  as it exists now. Never the pattern "previously X, now Y": no "now uses",
  "moved from", "no longer", "instead of the old". Edit docs in place to the
  current reality; git history carries the transition.
- Implementation specifics (the closed-form least-squares derivation, the
  boundary-erosion radius, ADE20K class-id quirks) belong as inline comments
  next to the code they explain — and only where that code is surprising —
  never in the docstring.

## Naming

- A name must read correctly at the call site, without the surrounding
  context that named it (`masked_depth_error`, not `error`; `architectural_mask`,
  not `mask`).
- Spell out domain words (`depth`, `disparity`, `latent`, `segmentation` — not
  `d`, `disp`, `lat`, `seg`). Single letters only for tight math indices that
  mirror a formula (the `s`, `t` of a scale/shift fit are fine).
- If an internal needs a comment to explain its role, rename it instead.

## Abstractions

- Clear, small, single-purpose modules; one concept per file.
- Depend on interfaces (`Protocol`) at boundaries; concrete classes inside.
  `DepthEstimator`, `Segmenter`, `Guidance` and `PromptAdherence` are the
  boundaries — a backend swap must never reach past them.
- No speculative generality: build the abstraction when the second consumer
  exists, not before.
- No dummy/placeholder configs, no hardcoded paths, no dead code.

## Tests

- Test names state the behavior; no docstrings in tests.
- Assert real behavior, not "runs without raising".
- Few, sharp tests. No trivial tests (constructor-runs, is-callable,
  shape-only when a value assertion is available).
- Prefer invariant tests (an affine-rescaled depth map scores the same error,
  masks are idempotent, a reward gradient is non-zero and finite, a fixed seed
  reproduces a sample) over smoke tests. At most one end-to-end smoke test per
  subsystem; everything else exercises one behavior.
- Tests requiring real pretrained checkpoints carry the `models` marker and are
  excluded from CI. Everything else runs against small fakes satisfying the
  Protocols, so the default suite needs no network, no GPU, and no Hub cache.
