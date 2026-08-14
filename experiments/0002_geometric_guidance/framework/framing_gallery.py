"""The framing study's qualitative gallery: one self-contained HTML page from `results.json`.

    uv run --no-sync python experiments/0002_geometric_guidance/framework/framing_gallery.py \
        outputs/<run>/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Final

if __package__ in (None, ""):  # run as a script: `framework` is not yet importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.gallery import (
    _STYLE,
    IMAGE_SUBDIRECTORY,
    _data_uri,
    _energy,
    _mapping,
    _mean,
    _number,
    _percent,
    _text,
    _title,
)
from framework.residual_maps import RESIDUAL_CLIP

FRAMING_GALLERY_FILENAME: Final = "qualitative.html"

_ARM_CAPTIONS: Final = {
    "style_unguided": "Style prompt, unguided",
    "renovation_unguided": "Renovation prompt, unguided",
    "style_guided": "Style prompt, guided",
    "renovation_guided": "Renovation prompt, guided",
}
_ARM_ORDER: Final = (
    "style_unguided",
    "renovation_unguided",
    "style_guided",
    "renovation_guided",
)

_LEDE: Final = (
    "Each scene runs a 2&times;2 at matched seeds: the plain style description "
    "against an explicit renovation instruction carrying the same style "
    "content, each sampled unguided and under the masked depth guidance. "
    "Depth error is scale/shift-aligned inverse-depth MSE on the photographed "
    "shell, read for all four arms over the one pixel set all four left "
    "visible; adherence and preference are scored against the one neutral "
    "style text under every arm, never an arm's own prompt. Every panel's "
    "absolute depth map sits beneath it, self-normalised, since "
    "affine-invariant disparity has no scale two images could share."
)

_JUDGEMENT_NOTE: Final = (
    "The page reports what the run measured and asserts nothing about it: "
    "there are no thresholds here."
)


def write_framing_gallery(results_path: Path, output_path: Path | None = None) -> Path:
    """Render the study's scenes, 2x2 grids, depth maps and numbers to one HTML page."""
    document = _mapping(json.loads(results_path.read_text(encoding="utf-8")))
    config = _mapping(document.get("config"))
    runs = [_mapping(row) for row in _rows(document)]
    if not runs:
        raise ValueError(f"{results_path} contains no run entries to render")

    image_dir = results_path.parent / IMAGE_SUBDIRECTORY
    name = _text(config.get("name")) or results_path.parent.name
    scenes = _by_scene(runs)
    sections = "".join(
        _scene_section(scene_runs, image_dir=image_dir)
        for scene_runs in scenes.values()
    )
    commit = _text(document.get("git_commit"))
    provenance = (
        f" &middot; commit {escape(commit[:8])}"
        if commit and commit != "unknown"
        else ""
    )
    seeds = {int(_number(run.get("seed")) or 0) for run in runs}
    page = (
        f"<title>{escape(name)} — prompt framing 2x2</title>\n"
        f"<style>\n{_STYLE}{_EXTRA_STYLE}</style>\n"
        "<main>\n"
        f'  <p class="eyebrow">roomgen &middot; {escape(name)} &middot; '
        f"{len(scenes)} scenes &times; {len(seeds)} seeds &times; 4 arms"
        f"{provenance}</p>\n"
        "  <h1>Does saying &ldquo;renovate&rdquo; change the renovation?</h1>\n"
        f'  <p class="lede">{_LEDE}</p>\n'
        f"{_summary(document)}"
        f"{sections}"
        '  <footer>Generated from the run\'s <span class="mono">results.json'
        "</span> and the images beside it &mdash; every number, scene and "
        "prompt on this page is read from the run, none typed by hand."
        "</footer>\n"
        "</main>\n"
    )
    destination = output_path or results_path.parent / FRAMING_GALLERY_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        page.encode("ascii", "xmlcharrefreplace").decode(), encoding="ascii"
    )
    return destination


def _summary(document: dict[str, object]) -> str:
    comparisons = _mapping(document.get("comparisons"))
    if not comparisons:
        return ""
    blocks = []
    for label, title in (
        ("unguided_style_vs_renovation", "Unguided, style vs renovation framing"),
        ("guided_style_vs_renovation", "Guided, style vs renovation framing"),
    ):
        comparison = _mapping(comparisons.get(label))
        if not comparison:
            continue
        fields = []
        for metric in ("depth_error", "adherence", "preference"):
            cell = _mapping(comparison.get(metric))
            style = _number(cell.get("style"))
            renovation = _number(cell.get("renovation"))
            if style is None or renovation is None:
                continue
            fields.append(
                f"{escape(metric.replace('_', ' '))} "
                f"{_energy(style)} &rarr; {_energy(renovation)}"
            )
        if fields:
            spans = "".join(f'<span class="metric">{field}</span>' for field in fields)
            blocks.append(
                f'    <p class="stats"><span class="metric"><b>{escape(title)}'
                f"</b></span>{spans}</p>\n"
            )
    if not blocks:
        return ""
    return (
        "  <h2>The two comparisons, in the mean</h2>\n"
        f'  <p class="section-note">Style-framing mean &rarr; renovation-framing '
        f"mean over every (scene, seed), per metric. {_JUDGEMENT_NOTE}</p>\n"
        + "".join(blocks)
    )


# Column layout the benchmark gallery has no use for: one vertical stack per
# arm, render above depth above deviation.
_EXTRA_STYLE: Final = """
.stack { display: grid; gap: 0.6rem; align-content: start; }
"""


def _scene_section(runs: list[dict[str, object]], *, image_dir: Path) -> str:
    shown = runs[0]
    scene = _text(shown.get("scene")) or "scene"
    seed = int(_number(shown.get("seed")) or 0)
    arms = {
        _text(arm.get("arm")) or "": arm
        for arm in (_mapping(entry) for entry in _arm_rows(shown))
    }

    reference_column = _column(
        image_dir,
        cells=(
            (_text(shown.get("photo_image")), "Photograph"),
            (_text(shown.get("photo_depth_map")), "Photograph, depth"),
        ),
        scene=scene,
    )
    arm_columns = [
        _column(
            image_dir,
            cells=(
                (_text(arms.get(arm, {}).get("render")), _ARM_CAPTIONS[arm]),
                (_text(arms.get(arm, {}).get("depth_map")), "Depth"),
                (
                    _text(arms.get(arm, {}).get("residual_map")),
                    f"Deviation (&plusmn;{RESIDUAL_CLIP:g}&times; baseline)",
                ),
            ),
            scene=scene,
        )
        for arm in _ARM_ORDER
    ]
    columns = [column for column in (reference_column, *arm_columns) if column]
    stats = "".join(_arm_stats(arms.get(arm, {})) for arm in _ARM_ORDER)
    extras = _scene_stats(runs)
    return (
        '  <section class="scene">\n'
        '    <header class="scene-head">\n'
        f"      <h3>{escape(_title(scene))}</h3>\n"
        "    </header>\n"
        f'    <p class="prompt">style: &ldquo;{escape(_text(shown.get("style_prompt")) or "")}&rdquo;</p>\n'
        f'    <p class="prompt">renovation: &ldquo;{escape(_text(shown.get("renovation_prompt")) or "")}&rdquo;</p>\n'
        f'    <p class="prompt">scored against: &ldquo;{escape(_text(shown.get("eval_prompt")) or "")}&rdquo; (seed {seed} shown)</p>\n'
        f'    <div class="strip" style="--columns: {len(columns)}">{"".join(columns)}</div>\n'
        f"{stats}{extras}"
        "  </section>\n"
    )


def _column(
    image_dir: Path, *, cells: tuple[tuple[str | None, str], ...], scene: str
) -> str:
    figures = []
    for name, caption in cells:
        if not name:
            continue
        uri = _data_uri(image_dir / Path(name).name)
        if uri is None:
            continue
        alt = f"{_title(scene)}: {caption}"
        figures.append(
            f'<figure class="cell"><img src="{uri}" alt="{escape(alt)}" '
            f'loading="lazy"><figcaption>{escape(caption)}</figcaption></figure>'
        )
    if not figures:
        return ""
    return f'<div class="stack">{"".join(figures)}</div>'


def _arm_stats(arm: dict[str, object]) -> str:
    arm_id = _text(arm.get("arm"))
    if arm_id is None:
        return ""
    fields = [f"<b>{escape(_ARM_CAPTIONS.get(arm_id, arm_id))}</b>"]
    depth = _number(arm.get("depth_error"))
    if depth is not None:
        fields.append(f"depth error {_energy(depth)}")
    adherence = _number(arm.get("adherence"))
    if adherence is not None:
        fields.append(f"adherence {_energy(adherence)}")
    preference = _number(arm.get("preference"))
    if preference is not None:
        fields.append(f"preference {_energy(preference)}")
    coverage = _number(arm.get("shell_coverage"))
    if coverage is not None:
        fields.append(f"shell visible {_percent(coverage)}")
    spans = "".join(f'<span class="metric">{field}</span>' for field in fields)
    return f'    <p class="stats">{spans}</p>\n'


def _scene_stats(runs: list[dict[str, object]]) -> str:
    shown = runs[0]
    fields = []
    coverage = _mean([_number(run.get("common_coverage")) for run in runs])
    if coverage is not None:
        fields.append(f"shell scored in common {_percent(coverage)} (mean over seeds)")
    floor = _number(shown.get("reconstruction_depth_error"))
    if floor is not None:
        fields.append(f"VAE floor {_energy(floor)}")
    if len(runs) > 1:
        fields.append(
            "seeds "
            + ", ".join(str(int(_number(run.get("seed")) or 0)) for run in runs)
        )
    if not fields:
        return ""
    spans = "".join(f'<span class="metric">{field}</span>' for field in fields)
    return f'    <p class="stats">{spans}</p>\n'


def _rows(document: dict[str, object]) -> list[object]:
    rows = document.get("runs")
    return rows if isinstance(rows, list) else []


def _arm_rows(run: dict[str, object]) -> list[object]:
    arms = run.get("arms")
    return arms if isinstance(arms, list) else []


def _by_scene(runs: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    scenes: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        scenes.setdefault(_text(run.get("scene")) or "scene", []).append(run)
    for rows in scenes.values():
        rows.sort(key=lambda row: int(_number(row.get("seed")) or 0))
    return scenes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="a framing-study results.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"destination HTML (default: {FRAMING_GALLERY_FILENAME} beside the results)",
    )
    arguments = parser.parse_args()
    written = write_framing_gallery(arguments.results, arguments.output)
    print(f"[gallery] {written} ({written.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
