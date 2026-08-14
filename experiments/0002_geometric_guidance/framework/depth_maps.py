"""Absolute disparity renders: what the depth backend saw in a photo or a finished render.

The residual maps (`residual_maps.py`) show where a render *bent* the room;
these show the depth field itself, one map per image, which is what makes a
reference photo's geometry and each arm's readable side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torchvision.io import write_png

# The same neutral the residual maps paint outside the shell: pixels the mask
# excludes are furniture the renovation may move, and score nothing.
_OUTSIDE_MASK: Final = (0.45, 0.45, 0.45)

# A magma-like piecewise-linear ramp over uniform stops, in place of a plotting
# dependency: near surfaces (high disparity) render bright, far ones dark.
_DEPTH_STOPS: Final = (
    (0.001, 0.000, 0.014),
    (0.159, 0.068, 0.352),
    (0.445, 0.122, 0.507),
    (0.717, 0.215, 0.475),
    (0.945, 0.377, 0.365),
    (0.996, 0.624, 0.427),
    (0.988, 0.998, 0.645),
)


def write_disparity_map(disparity: Float[Tensor, "1 height width"], path: Path) -> None:
    """One image's inverse depth as a full-frame colour map.

    Self-normalised over its own min..max: affine-invariant disparity carries no
    scale two images could share, so each map spans its own range and only the
    *shape* of the field compares across maps — which is exactly the invariance
    the reward scores under. A constant field renders flat rather than dividing
    by zero.
    """
    field = disparity[0].to(torch.float32)
    low, high = field.min(), field.max()
    span = (high - low).clamp(min=torch.finfo(torch.float32).eps)
    unit = (field - low) / span
    write_png(
        (_colorise(unit).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu(),
        str(path),
    )


def write_masked_disparity_map(
    disparity: Float[Tensor, "1 height width"],
    *,
    mask: Bool[Tensor, "1 height width"],
    path: Path,
) -> None:
    """Inverse depth over the scored pixels alone, neutral grey everywhere else.

    Normalised over the mask's own min..max rather than the frame's, so the
    shell's structure spends the whole colour range instead of being compressed
    by whatever near furniture the mask excluded. An empty mask paints the
    whole frame neutral rather than normalising over nothing.
    """
    field = disparity[0].to(torch.float32)
    selected = field[mask[0]]
    if selected.numel() == 0:
        unit = torch.zeros_like(field)
    else:
        low, high = selected.min(), selected.max()
        span = (high - low).clamp(min=torch.finfo(torch.float32).eps)
        unit = (field - low) / span
    colour = _colorise(unit)
    outside = torch.tensor(
        _OUTSIDE_MASK, dtype=colour.dtype, device=colour.device
    ).view(3, 1, 1)
    masked = torch.where(mask, colour, outside)
    write_png(
        (masked.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu(),
        str(path),
    )


@dataclass(frozen=True)
class DisparityRange:
    """One near/far span shared by every map of a scene, so colours compare across maps.

    Self-normalisation (`write_disparity_map`) is right for an affine-invariant
    backend, whose maps share no scale to begin with; a *metric* backend's maps
    do share one, and the absolute offset or rescale a render introduced is
    then exactly what a per-map normalisation would erase. The range is taken
    once — from the reference photo's map — and every other map of the scene is
    painted against it, saturating rather than rescaling where it escapes.
    """

    low: float
    high: float

    @classmethod
    def of(cls, disparity: Float[Tensor, "1 height width"]) -> DisparityRange:
        finite = disparity[torch.isfinite(disparity)].to(torch.float32)
        return cls(low=float(finite.min()), high=float(finite.max()))


def write_scaled_disparity_map(
    disparity: Float[Tensor, "1 height width"],
    *,
    scale: DisparityRange,
    path: Path,
) -> None:
    """One image's inverse depth painted against a scene's shared `scale`."""
    field = disparity[0].to(torch.float32)
    span = max(scale.high - scale.low, float(torch.finfo(torch.float32).eps))
    unit = (field - scale.low) / span
    write_png(
        (_colorise(unit).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu(),
        str(path),
    )


def _colorise(unit: Float[Tensor, "height width"]) -> Float[Tensor, "3 height width"]:
    palette = torch.tensor(_DEPTH_STOPS, dtype=torch.float32, device=unit.device)
    position = unit.clamp(0.0, 1.0) * (palette.shape[0] - 1)
    lower = position.floor().clamp(max=palette.shape[0] - 2)
    weight = (position - lower).unsqueeze(-1)
    index = lower.to(torch.long)
    colour = torch.lerp(palette[index], palette[index + 1], weight)
    return colour.permute(2, 0, 1)
