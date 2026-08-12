# Experiments

Each numbered directory `NNNN_slug/` is an experiment **framework** — a class
of related runs sharing machinery, not a single run. Prefer adding a variant
to an existing framework over scaffolding a new number.

**A new study is a new `conf/` variant set in an existing framework, not a new
directory — unless the experiment SHAPE is genuinely new.** Shared machinery
(the `Config` base, the `ROOMGEN_*` settings, run metrics, eval metrics)
lives in `roomgen_guide`; a framework is only a `conf/`, a thin `run.py` of
glue, and a `report.md`.

## Layout

```
experiments/
├── README.md              # this contract
├── JOURNAL.md             # owner-managed findings log (see below)
└── NNNN_slug/
    ├── run.py               # entry point
    ├── conf/                # config groups (variants)
    ├── framework/           # the run's own machinery, if it outgrows run.py
    ├── tests/               # its weightless surface, inside the gate
    └── report.md            # Hypothesis -> Setup -> Results -> Decision
```

## Anatomy of a framework

- **`run.py`** builds a typed `Config` (the `roomgen_guide.config.Config` base,
  `extra="forbid"`) from `conf/`, then drives the run: build the
  `GeometricReference` from the input photo -> sample (guided and unguided
  control, same seed) -> eval both -> report the numbers and render the
  gallery.
- **`conf/`** holds config groups; select a variant explicitly rather than
  patching top-level keys ad hoc.
- **`report.md`** records the finding: **Hypothesis** (what claim is being
  tested) -> **Setup** (scenes, prompts, seeds, guidance strength, sampler
  budget) -> **Results** (numbers, pointers to the run) -> **Decision** (what
  this settles, and what runs next).

## A framework measures; the owner judges

A run reports qualitative and quantitative results — the rendered comparisons
and the numbers behind them — and asserts nothing about them. There are no
thresholds in the code and no pass/fail in the output; reading the evidence is
the owner's job.

What every framework must therefore report, because a single number decides
nothing on its own:

- **masked depth error** against the `GeometricReference`, guided and unguided
  control at the same seed and prompt, and the reduction between them;
- **prompt adherence** for both arms and their ratio, because the degenerate
  way to lower depth error is **not to renovate at all** — a sampler that
  returns the input photo scores a perfect geometry error;
- **how much shell the pair was scored over**, since both arms are scored on
  one common evaluation mask and a comparison read over a sliver of the room
  says little about the room;
- **the qualitative pairs themselves**, which the gallery renders beside the
  numbers.

## Where outputs land

All local artifacts (renders, masked depth maps, mask overlays, config
snapshots) go under `ROOMGEN_WDIR` (`RoomgenSettings`, default `outputs/` at the
repo root — gitignored, so runs never dirty the tree; override via the
`ROOMGEN_WDIR` env var) — set it per-machine, not hardcoded in a script. Model
weights are cached under `ROOMGEN_HF_HOME` and are never committed.

There is no experiment tracker: a framework's record is the `results.json`,
the rendered gallery and the report beside them, all under `ROOMGEN_WDIR`.
`ROOMGEN_WDIR` and `ROOMGEN_HF_HOME` are the whole environment contract
(`RoomgenSettings` reads only `ROOMGEN_*` variables, so plain `HF_HOME` is not
honoured).

## JOURNAL.md

**Owner-managed.** The owner writes journal entries, or dictates them
verbatim; agents never append on their own initiative. One line per concluded
finding: framework, one-line conclusion, pointer to the run/report. Don't journal
in package docs (`CLAUDE.md` cites experiment numbers instead) and don't
rewrite past entries.

## Planned frameworks

See `docs/research/2026-08-11-geometric-guidance.md` for the staged path.
