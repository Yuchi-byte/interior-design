"""Presentation of the per-pixel depth residual: where each arm bent the room, in colour.

At the noise strengths this benchmark runs, a guided render and its control are
both plausible rooms and the difference is invisible in RGB. It is not invisible
in depth, so each arm also leaves one of these maps behind, the two of them on a
single shared scale so the pair can be read side by side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torchvision.io import write_png

# Both maps are denominated in multiples of the control's mean absolute masked
# residual, so 1.0 reads as "the control's average deviation" and the colour at
# a pixel means the same thing in either map. The scale saturates here.
RESIDUAL_CLIP: Final = 3.0

_OUTSIDE_SHELL: Final = (0.45, 0.45, 0.45)

# A piecewise-linear RGB lookup over uniformly spaced stops, in place of a
# plotting dependency. Diverging blue through neutral to red: blue is a surface
# reading further from the camera than the photo, red is one reading closer,
# and the pale centre is geometry the render left where it found it.
_RESIDUAL_STOPS: Final = (
    (0.02, 0.24, 0.42),
    (0.21, 0.55, 0.73),
    (0.70, 0.87, 0.90),
    (0.97, 0.97, 0.97),
    (0.99, 0.80, 0.71),
    (0.84, 0.38, 0.30),
    (0.40, 0.00, 0.12),
)


def mean_absolute_residual(
    residual: Float[Tensor, "1 height width"], mask: Bool[Tensor, "1 height width"]
) -> float:
    """The scale a scene's maps are read against: the control's average shell deviation."""
    return float(residual.abs().sum() / mask.to(residual.dtype).sum().clamp(min=1.0))


def write_signed_residual_map(
    residual: Float[Tensor, "1 height width"],
    *,
    mask: Bool[Tensor, "1 height width"],
    normalizer: float,
    path: Path,
) -> None:
    """One arm's signed depth deviation from the photo, in units of `normalizer`.

    Args:
        residual: from `roomgen_guide.eval.signed_depth_residual_map` — positive
            where the render reads closer to the camera than the photo.
        normalizer: what 1.0 on the colour scale means, in those residual units;
            pass the same value for both arms of a pair or the maps do not compare.
    """
    scaled = (residual[0] / _safe(normalizer)).clamp(-RESIDUAL_CLIP, RESIDUAL_CLIP)
    _write(
        _colorise((scaled + RESIDUAL_CLIP) / (2.0 * RESIDUAL_CLIP), mask[0]),
        path,
    )


def _colorise(
    unit: Float[Tensor, "height width"], mask: Bool[Tensor, "height width"]
) -> Float[Tensor, "3 height width"]:
    palette = torch.tensor(_RESIDUAL_STOPS, dtype=torch.float32, device=unit.device)
    position = unit.to(torch.float32).clamp(0.0, 1.0) * (palette.shape[0] - 1)
    lower = position.floor().clamp(max=palette.shape[0] - 2)
    weight = (position - lower).unsqueeze(-1)
    index = lower.to(torch.long)
    colour = torch.lerp(palette[index], palette[index + 1], weight)
    # Everything the shell mask excludes is furniture the renovation was allowed
    # to move, and scores nothing: it gets a flat neutral rather than a zero,
    # which would otherwise read as geometry held perfectly still.
    outside = torch.tensor(_OUTSIDE_SHELL, dtype=colour.dtype, device=colour.device)
    return torch.where(mask.unsqueeze(-1), colour, outside).permute(2, 0, 1)


def _safe(normalizer: float) -> float:
    # A control that deviated nowhere has no scale to normalise by; 1.0 leaves
    # the residuals in their own units instead of dividing by nothing.
    return normalizer if normalizer > 0.0 else 1.0


def _write(image: Float[Tensor, "3 height width"], path: Path) -> None:
    write_png((image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu(), str(path))
