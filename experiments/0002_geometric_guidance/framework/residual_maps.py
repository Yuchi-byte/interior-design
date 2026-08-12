"""Presentation of the per-pixel depth residual: where guidance won the geometry, in colour.

At the noise strengths this benchmark runs, a guided render and its control are
both plausible rooms and the difference is invisible in RGB. It is not invisible
in depth, so every pair also leaves these two maps behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torchvision.io import write_png

# Both maps are denominated in multiples of the control's mean masked residual,
# so 1.0 reads as "the control's average error". The saturation points are what
# make the two panels comparable at a glance; anything beyond them is clipped.
RESIDUAL_CLIP: Final = 3.0
DIFFERENCE_CLIP: Final = 2.0

_OUTSIDE_SHELL: Final = (0.45, 0.45, 0.45)
_PANEL_SEPARATOR: Final = (1.0, 1.0, 1.0)
_SEPARATOR_WIDTH: Final = 6

# Piecewise-linear RGB lookups over uniformly spaced stops, in place of a
# plotting dependency. Sequential runs dark (no error) to bright (saturated).
_RESIDUAL_STOPS: Final = (
    (0.00, 0.00, 0.02),
    (0.24, 0.04, 0.35),
    (0.58, 0.15, 0.40),
    (0.87, 0.29, 0.22),
    (0.99, 0.62, 0.04),
    (0.99, 0.99, 0.75),
)
# Diverging, red through neutral to blue: red is guidance losing to its control,
# blue is guidance winning, and the pale centre is the two tying.
_DIFFERENCE_STOPS: Final = (
    (0.40, 0.00, 0.12),
    (0.84, 0.38, 0.30),
    (0.99, 0.80, 0.71),
    (0.97, 0.97, 0.97),
    (0.70, 0.87, 0.90),
    (0.21, 0.55, 0.73),
    (0.02, 0.24, 0.42),
)


def mean_masked_residual(
    residual: Float[Tensor, "1 height width"], mask: Bool[Tensor, "1 height width"]
) -> float:
    """The scale both maps are read against: the control's average shell error."""
    return float(residual.sum() / mask.to(residual.dtype).sum().clamp(min=1.0))


def write_residual_comparison(
    control: Float[Tensor, "1 height width"],
    guided: Float[Tensor, "1 height width"],
    *,
    mask: Bool[Tensor, "1 height width"],
    normalizer: float,
    path: Path,
) -> None:
    """Control beside guided on one shared scale, in units of `normalizer`.

    Args:
        control, guided: per-pixel squared residuals from
            `roomgen_guide.eval.depth_residual_map`.
        normalizer: what 1.0 on the colour scale means, in those residual units.
    """
    panels = [
        _colorise(
            (residual[0] / _safe(normalizer)).clamp(0.0, RESIDUAL_CLIP) / RESIDUAL_CLIP,
            _RESIDUAL_STOPS,
            mask[0],
        )
        for residual in (control, guided)
    ]
    separator = _constant_column(
        panels[0], width=_SEPARATOR_WIDTH, color=_PANEL_SEPARATOR
    )
    _write(torch.cat([panels[0], separator, panels[1]], dim=-1), path)


def write_residual_difference(
    control: Float[Tensor, "1 height width"],
    guided: Float[Tensor, "1 height width"],
    *,
    mask: Bool[Tensor, "1 height width"],
    normalizer: float,
    path: Path,
) -> None:
    """`(control - guided) / normalizer`, blue where guidance helped and red where it hurt."""
    difference = ((control[0] - guided[0]) / _safe(normalizer)).clamp(
        -DIFFERENCE_CLIP, DIFFERENCE_CLIP
    )
    _write(
        _colorise(
            (difference + DIFFERENCE_CLIP) / (2.0 * DIFFERENCE_CLIP),
            _DIFFERENCE_STOPS,
            mask[0],
        ),
        path,
    )


def _colorise(
    unit: Float[Tensor, "height width"],
    stops: tuple[tuple[float, float, float], ...],
    mask: Bool[Tensor, "height width"],
) -> Float[Tensor, "3 height width"]:
    palette = torch.tensor(stops, dtype=torch.float32, device=unit.device)
    position = unit.to(torch.float32).clamp(0.0, 1.0) * (palette.shape[0] - 1)
    lower = position.floor().clamp(max=palette.shape[0] - 2)
    weight = (position - lower).unsqueeze(-1)
    index = lower.to(torch.long)
    colour = torch.lerp(palette[index], palette[index + 1], weight)
    # Everything the shell mask excludes is furniture the renovation was allowed
    # to move, and scores nothing: it gets a flat neutral rather than a zero,
    # which would otherwise read as a perfect region.
    outside = torch.tensor(_OUTSIDE_SHELL, dtype=colour.dtype, device=colour.device)
    return torch.where(mask.unsqueeze(-1), colour, outside).permute(2, 0, 1)


def _constant_column(
    panel: Float[Tensor, "3 height width"], *, width: int, color: tuple[float, ...]
) -> Float[Tensor, "3 height width"]:
    return torch.tensor(color, dtype=panel.dtype, device=panel.device)[
        :, None, None
    ].expand(-1, panel.shape[1], width)


def _safe(normalizer: float) -> float:
    # A control that scored zero everywhere has no scale to normalise by; 1.0
    # leaves the residuals in their own units instead of dividing by nothing.
    return normalizer if normalizer > 0.0 else 1.0


def _write(image: Float[Tensor, "3 height width"], path: Path) -> None:
    write_png((image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu(), str(path))
