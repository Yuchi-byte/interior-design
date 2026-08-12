"""The run's qualitative gallery: one self-contained HTML page built from `results.json`.

    uv run --no-sync python experiments/0002_geometric_guidance/framework/gallery.py \
        outputs/<run>/results.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Final

from PIL import Image

if __package__ in (None, ""):  # run as a script: `framework` is not yet importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.residual_maps import RESIDUAL_CLIP

IMAGE_SUBDIRECTORY: Final = "images"
GALLERY_FILENAME: Final = "qualitative.html"
THUMBNAIL_EDGE: Final = 416
JPEG_QUALITY: Final = 80

# Rendered per scene as its own strip cell; anything the file names but the run
# did not write is simply skipped, so a results.json from a run that emitted no
# depth maps still produces a complete page.
_STRIP: Final[tuple[tuple[str, str], ...]] = (
    ("{scene}_photo.png", "Photograph"),
    ("{scene}_seed{seed}_control.png", "Unguided control"),
    ("{scene}_seed{seed}_guided.png", "Guided"),
)
_HERO_CAPTION: Final = "Guided"
_SHELL_MASK: Final = "{scene}_shell_mask.png"
_SHELL_MASK_CAPTION: Final = "Eroded shell mask (scored region in white)"
_DEPTH_MAP_CAPTIONS: Final = (
    "Unguided \N{MINUS SIGN} reference",
    "Guided \N{MINUS SIGN} reference",
)

# Top-level keys of results.json that are structure rather than run parameters,
# and so are rendered by name elsewhere on the page instead of echoed.
_STRUCTURAL_KEYS: Final = frozenset(
    {"config", "runs", "results", "verdict", "settings"}
)


@dataclass(frozen=True)
class _Gate:
    depth_error_margin: float | None
    prompt_adherence_floor: float | None


@dataclass(frozen=True)
class _Run:
    scene: str
    prompt: str
    seed: int
    guided_depth_error: float
    control_depth_error: float
    depth_error_reduction: float
    adherence_ratio: float
    reconstruction_depth_error: float | None
    mean_abs_control_residual: float | None
    depth_map_control: str | None
    depth_map_guided: str | None

    def geometry_passed(self, gate: _Gate) -> bool:
        margin = gate.depth_error_margin
        return margin is None or self.depth_error_reduction >= margin

    def adherence_passed(self, gate: _Gate) -> bool:
        floor = gate.prompt_adherence_floor
        return floor is None or self.adherence_ratio >= floor

    def passed(self, gate: _Gate) -> bool:
        return self.geometry_passed(gate) and self.adherence_passed(gate)


def write_gallery(results_path: Path, output_path: Path | None = None) -> Path:
    """Render the run's scenes, images and per-seed numbers to a standalone HTML page.

    Args:
        results_path: a benchmark `results.json`; its rendered images are read
            from the `images/` directory beside it and embedded as data URIs,
            so the page needs no network and no sibling files.
        output_path: defaults to `qualitative.html` beside `results_path`.
    """
    document = _mapping(json.loads(results_path.read_text(encoding="utf-8")))
    config = _mapping(document.get("config"))
    runs = _runs(document)
    if not runs:
        raise ValueError(f"{results_path} contains no run entries to render")

    page = _page(
        document,
        config=config,
        runs=runs,
        image_dir=results_path.parent / IMAGE_SUBDIRECTORY,
    )
    destination = output_path or results_path.parent / GALLERY_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The page carries no <head>, so it cannot declare a charset: escaping every
    # non-ASCII character to a numeric reference makes it render the same
    # whatever encoding the viewer guesses.
    destination.write_text(
        page.encode("ascii", "xmlcharrefreplace").decode(), encoding="ascii"
    )
    return destination


def _page(
    document: dict[str, object],
    *,
    config: dict[str, object],
    runs: tuple[_Run, ...],
    image_dir: Path,
) -> str:
    gate = _gate(config)
    name = _text(config.get("name")) or image_dir.parent.name
    scenes = _by_scene(runs)
    sections = "".join(
        _scene_section(scene_runs, image_dir=image_dir, gate=gate)
        for scene_runs in scenes.values()
    )
    return (
        f"<title>{escape(name)} — qualitative comparisons</title>\n"
        f"<style>\n{_STYLE}</style>\n"
        "<main>\n"
        f'  <p class="eyebrow">roomgen &middot; {escape(name)} &middot; '
        f"{len(scenes)} scenes &times; {len(_seeds(runs))} seeds</p>\n"
        "  <h1>Same room, different furniture: qualitative comparisons</h1>\n"
        f'  <p class="lede">{_LEDE}</p>\n'
        f"{_pinned(document, config=config)}"
        f"{_verdict_note(runs, gate=gate)}"
        f"{sections}"
        f"{_FOOTER}"
        "</main>\n"
    )


def _pinned(document: dict[str, object], *, config: dict[str, object]) -> str:
    # Echoed, never restated: every scalar the run recorded is rendered by its
    # own key, so a config field added upstream appears here without a change.
    rows = [
        _pinned_row(section, _mapping(value))
        for section, value in config.items()
        if isinstance(value, dict)
    ]
    rows.insert(0, _pinned_row("config", _scalars(config)))
    rows.append(_pinned_row("run", _scalars(document, skip=_STRUCTURAL_KEYS)))
    body = "".join(row for row in rows if row)
    return f'  <p class="pinned">{body}</p>\n'


def _pinned_row(label: str, values: dict[str, object]) -> str:
    if not values:
        return ""
    fields = " &middot; ".join(
        f"{escape(key)} <b>{escape(_scalar(value))}</b>"
        for key, value in values.items()
    )
    return f'<span class="pinned-row"><i>{escape(label)}</i>{fields}</span>'


def _verdict_note(runs: tuple[_Run, ...], *, gate: _Gate) -> str:
    passing = sum(1 for run in runs if run.passed(gate))
    reduction = _mean(run.depth_error_reduction for run in runs)
    adherence = _mean(run.adherence_ratio for run in runs)
    worst = min(runs, key=lambda run: run.depth_error_reduction)
    gate_text = (
        "depth error down at least "
        f"{_percent(gate.depth_error_margin)} against the control with adherence at "
        f"{_percent(gate.prompt_adherence_floor)} of control or better"
        if gate.depth_error_margin is not None
        else "the run's two-sided gate"
    )
    return (
        "  <h2>The benchmark rooms</h2>\n"
        f'  <p class="section-note">Every strip is the lowest seed of the grid; '
        f"across all {len(runs)} (scene, seed) pairs "
        f"<b>{passing}/{len(runs)}</b> pass {gate_text}. "
        f"Mean depth error {_signed_percent(-reduction)}, mean adherence "
        f"{_percent(adherence, decimals=1)} of control, weakest scene "
        f"{escape(_title(worst.scene))} at "
        f"{_signed_percent(-worst.depth_error_reduction)}.</p>\n"
    )


def _scene_section(runs: tuple[_Run, ...], *, image_dir: Path, gate: _Gate) -> str:
    shown = runs[0]
    cells = [
        _figure(
            image_dir / pattern.format(scene=shown.scene, seed=shown.seed),
            caption=caption,
            scene=shown.scene,
            hero=caption == _HERO_CAPTION,
        )
        for pattern, caption in _STRIP
    ]
    strip = [cell for cell in cells if cell]
    mask = _figure(
        image_dir / _SHELL_MASK.format(scene=shown.scene),
        caption=_SHELL_MASK_CAPTION,
        scene=shown.scene,
    )
    return (
        '  <section class="scene">\n'
        '    <header class="scene-head">\n'
        f"      <h3>{escape(_title(shown.scene))}</h3>\n"
        f'      <p class="prompt">&ldquo;{escape(shown.prompt)}&rdquo;</p>\n'
        "    </header>\n"
        f'    <div class="strip" style="--columns: {len(strip)}">{"".join(strip)}</div>\n'
        f"{_depth_row(shown, image_dir=image_dir)}"
        f"{_stats(runs, gate=gate)}"
        f"{_detail_block(mask)}"
        "  </section>\n"
    )


def _depth_row(run: _Run, *, image_dir: Path) -> str:
    figures = [
        _figure(image_dir / Path(name).name, caption=caption, scene=run.scene)
        for name, caption in zip(
            (run.depth_map_control, run.depth_map_guided),
            _DEPTH_MAP_CAPTIONS,
            strict=True,
        )
        if name
    ]
    if not any(figures):
        return ""
    return (
        '    <div class="depth-row">\n'
        '      <p class="row-label">Depth vs reference</p>\n'
        f'      <div class="duo">{"".join(figures)}</div>\n'
        f'      <p class="map-note">{_map_note(run)}</p>\n'
        "    </div>\n"
    )


def _map_note(run: _Run) -> str:
    # The saturation point is imported rather than restated: the maps and their
    # caption drift apart silently otherwise.
    scale = (
        f" ({_energy(run.mean_abs_control_residual)})"
        if run.mean_abs_control_residual is not None
        else ""
    )
    return (
        "Signed deviation from the photo's depth, both panels on one shared "
        f"scale saturating at &plusmn;{RESIDUAL_CLIP:g}&times; the control's mean "
        f"absolute masked residual{scale}: blue reads further from the camera "
        "than the photo, red reads closer, pale is geometry left where it was. "
        "Grey is outside the shell mask, which scores nothing."
    )


def _detail_block(figure: str) -> str:
    if not figure:
        return ""
    return (
        "    <details>\n"
        "      <summary>What the reward sees</summary>\n"
        f'      <div class="duo">{figure}</div>\n'
        "    </details>\n"
    )


def _figure(path: Path, *, caption: str, scene: str, hero: bool = False) -> str:
    uri = _data_uri(path)
    if uri is None:
        return ""
    alt = f"{_title(scene)}: {caption[0].lower() + caption[1:]}"
    return (
        f'<figure class="{"hero-cell" if hero else "cell"}">'
        f'<img src="{uri}" alt="{escape(alt)}" loading="lazy">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def _stats(runs: tuple[_Run, ...], *, gate: _Gate) -> str:
    shown = runs[0]
    fields = [
        (
            f"depth error {_energy(shown.control_depth_error)} &rarr; "
            f"{_energy(shown.guided_depth_error)} "
            f'<b class="{_verdict_class(shown.geometry_passed(gate))}">'
            f"{_signed_percent(-shown.depth_error_reduction)}</b>"
        ),
        (
            f'adherence <b class="{_verdict_class(shown.adherence_passed(gate))}">'
            f"{_percent(shown.adherence_ratio)}</b> of control"
        ),
    ]
    if len(runs) > 1:
        per_seed = "/".join(_signed_percent(-run.depth_error_reduction) for run in runs)
        fields.append(f"{_seed_label(runs)}: {per_seed}")
    else:
        fields.append(f"seed {shown.seed}")
    if shown.reconstruction_depth_error is not None:
        fields.append(f"VAE floor {_energy(shown.reconstruction_depth_error)}")
    if shown.mean_abs_control_residual is not None:
        fields.append(
            f"control mean |deviation| {_energy(shown.mean_abs_control_residual)}"
        )
    spans = "".join(f'<span class="metric">{field}</span>' for field in fields)
    return f'    <p class="stats">{spans}</p>\n'


def _seed_label(runs: tuple[_Run, ...]) -> str:
    seeds = [run.seed for run in runs]
    contiguous = seeds == list(range(seeds[0], seeds[0] + len(seeds)))
    span = (
        f"{seeds[0]}&ndash;{seeds[-1]}"
        if contiguous
        else ", ".join(str(seed) for seed in seeds)
    )
    return f"seeds {span}"


def _data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    buffer = io.BytesIO()
    with Image.open(path) as opened:
        thumbnail = opened.convert("RGB")
        # `thumbnail` caps the longest edge and keeps the ratio, so a
        # non-square depth map is embedded undistorted.
        thumbnail.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), Image.Resampling.LANCZOS)
        thumbnail.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _runs(document: dict[str, object]) -> tuple[_Run, ...]:
    rows = document.get("runs") or document.get("results")
    if not isinstance(rows, list):
        return ()
    return tuple(_run(_mapping(row)) for row in rows if isinstance(row, dict))


def _run(row: dict[str, object]) -> _Run:
    control = _number(row.get("control_depth_error")) or 0.0
    guided = _number(row.get("guided_depth_error")) or 0.0
    reduction = _number(row.get("depth_error_reduction"))
    adherence = _number(row.get("adherence_ratio"))
    return _Run(
        scene=_text(row.get("scene")) or "scene",
        prompt=_text(row.get("prompt")) or "",
        seed=int(_number(row.get("seed")) or 0),
        guided_depth_error=guided,
        control_depth_error=control,
        depth_error_reduction=(
            reduction
            if reduction is not None
            else (1.0 - guided / control if control else 0.0)
        ),
        adherence_ratio=adherence if adherence is not None else 0.0,
        reconstruction_depth_error=_number(row.get("reconstruction_depth_error")),
        mean_abs_control_residual=_number(row.get("mean_abs_control_residual")),
        depth_map_control=_text(row.get("depth_map_control")),
        depth_map_guided=_text(row.get("depth_map_guided")),
    )


def _by_scene(runs: tuple[_Run, ...]) -> dict[str, tuple[_Run, ...]]:
    scenes: dict[str, list[_Run]] = {}
    for run in runs:
        scenes.setdefault(run.scene, []).append(run)
    return {
        scene: tuple(sorted(scene_runs, key=lambda run: run.seed))
        for scene, scene_runs in scenes.items()
    }


def _seeds(runs: tuple[_Run, ...]) -> set[int]:
    return {run.seed for run in runs}


def _gate(config: dict[str, object]) -> _Gate:
    gate = _mapping(config.get("gate"))
    return _Gate(
        depth_error_margin=_number(gate.get("depth_error_margin")),
        prompt_adherence_floor=_number(gate.get("prompt_adherence_floor")),
    )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _scalars(
    values: dict[str, object], *, skip: frozenset[str] = frozenset()
) -> dict[str, object]:
    return {
        key: value
        for key, value in values.items()
        if key not in skip
        and (
            not isinstance(value, dict | list)
            or (isinstance(value, list) and not any(isinstance(x, dict) for x in value))
        )
        and value is not None
    }


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, list):
        return ", ".join(_scalar(item) for item in value)
    return str(value)


def _energy(value: float) -> str:
    return f"{value:.3f}" if abs(value) >= 0.001 else f"{value:.1e}"


def _percent(value: float | None, *, decimals: int = 0) -> str:
    return "n/a" if value is None else f"{value * 100:.{decimals}f}%"


def _signed_percent(value: float) -> str:
    return f"{'&minus;' if value < 0 else '+'}{abs(value) * 100:.0f}%"


def _verdict_class(passed: bool) -> str:
    return "pass" if passed else "fail"


def _title(scene: str) -> str:
    words = scene.replace("_", " ")
    return words[:1].upper() + words[1:]


_LEDE: Final = (
    "Each strip shows the input photograph, the unguided SDEdit control, and "
    "the geometrically guided sample from the same seed. The guided column is "
    "steered by a differentiable masked depth error against the photo's "
    "architectural shell &mdash; walls, floor, ceiling, windows, doors &mdash; "
    "and blind to the furniture, which the renovation is allowed to change. "
    "Depth error is scale/shift-aligned inverse depth MSE on the shell; "
    "adherence is the prompt metric relative to the control."
)

_FOOTER: Final = (
    "  <footer>Generated from the run's <span "
    'class="mono">results.json</span> and the images beside it &mdash; every '
    "number, scene and prompt on this page is read from the run, none typed by "
    "hand. Benchmark photos are CC0 / public-domain (manifest: <span "
    'class="mono">experiments/assets/photos.json</span>).</footer>\n'
)

_STYLE: Final = """:root {
  --bg: #fafaf8; --ink: #1d1f21; --ink-2: #55595e; --line: #e3e2dc;
  --accent: #3a6b9b; --pass: #2e7d5b; --fail: #b0492f; --card: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #141619; --ink: #e8e6e1; --ink-2: #9aa0a6; --line: #2a2d31;
    --accent: #7aa7d0; --pass: #5cb890; --fail: #d97b62; --card: #1b1e22; }
}
:root[data-theme="dark"] { --bg: #141619; --ink: #e8e6e1; --ink-2: #9aa0a6;
  --line: #2a2d31; --accent: #7aa7d0; --pass: #5cb890; --fail: #d97b62;
  --card: #1b1e22; }
:root[data-theme="light"] { --bg: #fafaf8; --ink: #1d1f21; --ink-2: #55595e;
  --line: #e3e2dc; --accent: #3a6b9b; --pass: #2e7d5b; --fail: #b0492f;
  --card: #ffffff; }
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1080px; margin: 0 auto; padding: 3rem 1.25rem 5rem; }
.eyebrow {
  font-family: "Avenir Next", Futura, system-ui, sans-serif;
  text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.72rem;
  color: var(--accent); font-weight: 600; margin: 0 0 0.5rem;
}
h1 {
  font-family: "Avenir Next", Futura, system-ui, sans-serif;
  font-size: clamp(1.6rem, 3.4vw, 2.3rem); font-weight: 600;
  letter-spacing: -0.01em; margin: 0 0 0.6rem; text-wrap: balance;
}
.lede { max-width: 62ch; color: var(--ink-2); margin: 0 0 1.4rem; }
.pinned {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; font-size: 0.8rem; color: var(--ink-2);
  border: 1px solid var(--line); border-radius: 6px; padding: 0.65rem 0.9rem;
  background: var(--card); margin: 0; display: grid; gap: 0.3rem;
}
.pinned b { color: var(--ink); font-weight: 600; }
.pinned-row { display: block; }
.pinned-row i {
  display: inline-block; min-width: 6.5em; font-style: normal;
  color: var(--accent); text-transform: uppercase; letter-spacing: 0.08em;
  font-size: 0.72rem;
}
h2 {
  font-family: "Avenir Next", Futura, system-ui, sans-serif;
  font-size: 1.25rem; font-weight: 600; margin: 3.2rem 0 0.4rem;
  padding-top: 1.6rem; border-top: 1px solid var(--line); text-wrap: balance;
}
.section-note { max-width: 66ch; color: var(--ink-2); margin: 0 0 1.6rem; }
.section-note b { color: var(--ink); }
.scene { margin: 0 0 2.6rem; }
.scene-head { display: flex; flex-wrap: wrap; align-items: baseline;
  gap: 0.35rem 1rem; margin-bottom: 0.7rem; }
.scene-head h3 {
  font-family: "Avenir Next", Futura, system-ui, sans-serif;
  font-size: 1.02rem; font-weight: 600; margin: 0;
}
.prompt { margin: 0; color: var(--ink-2); font-size: 0.86rem; font-style: italic; }
.strip { display: grid; grid-template-columns: repeat(var(--columns, 3),
  minmax(0, 1fr)); gap: 0.6rem; }
figure { margin: 0; }
figure img { width: 100%; height: auto; display: block; border-radius: 4px; }
.hero-cell img { outline: 2px solid var(--accent); outline-offset: -2px; }
figcaption {
  font-family: "Avenir Next", Futura, system-ui, sans-serif;
  text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.64rem;
  color: var(--ink-2); margin-top: 0.4rem;
}
.hero-cell figcaption { color: var(--accent); font-weight: 600; }
.stats {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; font-size: 0.8rem; color: var(--ink-2);
  margin: 0.55rem 0 0; display: flex; flex-wrap: wrap; gap: 0.3rem 1.4rem;
}
.stats b { font-weight: 600; }
.pass { color: var(--pass); }
.fail { color: var(--fail); }
.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
.depth-row { margin-top: 0.9rem; }
.row-label {
  font-family: "Avenir Next", Futura, system-ui, sans-serif;
  text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.64rem;
  color: var(--ink-2); margin: 0 0 0.4rem;
}
.map-note { max-width: 72ch; color: var(--ink-2); font-size: 0.76rem;
  margin: 0.45rem 0 0; }
details { margin-top: 0.7rem; }
summary {
  font-family: "Avenir Next", Futura, system-ui, sans-serif;
  text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.64rem;
  color: var(--accent); cursor: pointer;
}
details .duo { margin-top: 0.6rem; }
.duo { display: grid; grid-template-columns: repeat(2, minmax(0, 420px));
  gap: 0.6rem; }
@media (max-width: 720px) {
  .strip, .duo { grid-template-columns: 1fr; }
}
footer { margin-top: 3.5rem; padding-top: 1.2rem; border-top: 1px solid var(--line);
  color: var(--ink-2); font-size: 0.8rem; max-width: 70ch; }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="a benchmark results.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"destination HTML (default: {GALLERY_FILENAME} beside the results)",
    )
    arguments = parser.parse_args()
    written = write_gallery(arguments.results, arguments.output)
    print(f"[gallery] {written} ({written.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
