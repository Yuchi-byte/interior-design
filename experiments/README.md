# Experiments

Each numbered directory `NNNN_slug/` is an experiment **framework** — a class
of related runs sharing machinery, not a single run. Prefer adding a variant
to an existing framework over scaffolding a new number.

**A new study is a new `conf/` variant set in an existing framework, not a new
directory — unless the experiment SHAPE is genuinely new.** Shared machinery
(the `Config` base, the `ROOMGEN_*` settings, verdict scoring, eval metrics)
lives in `roomgen_guide`; a framework is only a `conf/`, a thin `run.py` of
glue, and a `report.md`.

## Layout

```
experiments/
├── README.md              # this contract
├── JOURNAL.md             # owner-managed verdict log (see below)
└── NNNN_slug/
    ├── run.py               # entry point
    ├── conf/                # config groups (variants)
    └── report.md            # Hypothesis -> Setup -> Results -> Decision
```

## Anatomy of a framework

- **`run.py`** builds a typed `Config` (the `roomgen_guide.config.Config` base,
  `extra="forbid"`) from `conf/`, then drives the run: build the
  `GeometricReference` from the input photo -> sample (guided and unguided
  control, same seed) -> eval both -> assert the verdict.
- **`conf/`** holds config groups; select a variant explicitly rather than
  patching top-level keys ad hoc.
- **`report.md`** records the finding: **Hypothesis** (what claim is being
  tested) -> **Setup** (scenes, prompts, seeds, guidance schedule, sampler
  budget) -> **Results** (numbers, pointers to the run) -> **Decision** (what
  this settles, and what runs next).

## Verdicts are asserted in code, never by eye

Every framework gates its claim on thresholds checked in code and passes the
resulting verdict through the run's finish/summary step. A report without an
asserted threshold is a plot, not a finding.

**The gate is two-sided.** Guided sampling must

- **(a) reduce masked depth error** against the `GeometricReference` by at
  least a threshold margin versus the unguided control at the same seed and
  prompt, **and**
- **(b) not regress prompt adherence** below a floor (the
  prompt-adherence / perceptual metric in `roomgen_guide.eval`).

Both, on every seed and scene. Gate (b) is not a formality: the degenerate way
to win on depth error is **not to renovate at all** — a sampler that returns
the input photo scores a perfect geometry error. Any framework reporting (a)
alone has proved nothing. This two-sided shape is the heart of the experimental
design; a new framework inherits it rather than inventing its own.

## Where outputs land

All local artifacts (renders, masked depth maps, mask overlays, config
snapshots) go under `ROOMGEN_WDIR` (`RoomgenSettings`, default `outputs/` at the
repo root — gitignored, so runs never dirty the tree; override via the
`ROOMGEN_WDIR` env var) — set it per-machine, not hardcoded in a script. Model
weights are cached under `ROOMGEN_HF_HOME` and are never committed.

Tracking (config, metrics, media, verdict) goes to **Weights & Biases**,
controlled by `ROOMGEN_WANDB_MODE` (plain `WANDB_MODE` is not honoured —
`RoomgenSettings` reads only `ROOMGEN_*` variables):

```bash
ROOMGEN_WANDB_MODE=disabled uv run --no-sync python experiments/NNNN_slug/run.py   # fast plumbing check, no tracking
ROOMGEN_WANDB_MODE=offline  uv run --no-sync python experiments/NNNN_slug/run.py   # record locally, `wandb sync` later
ROOMGEN_WANDB_MODE=online   uv run --no-sync python experiments/NNNN_slug/run.py   # requires `wandb login`
```

## JOURNAL.md

**Owner-managed.** The owner writes journal entries, or dictates them
verbatim; agents never append on their own initiative. One line per concluded
finding: framework, one-line verdict, pointer to the run/report. Don't journal
in package docs (`CLAUDE.md` cites experiment numbers instead) and don't
rewrite past entries.

## Planned frameworks

See `docs/research/2026-08-11-geometric-guidance.md` for the staged path and
promotion gates.
