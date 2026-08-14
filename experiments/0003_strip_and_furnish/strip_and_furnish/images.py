"""What the two galleries are made of: renders, the mask that was cut, and depth in colour."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from torchvision.io import write_png

from roomgen_geometry.alignment import align_scale_shift, apply_scale_shift

# What the mask overlay paints the region the inpainter was handed, and how
# strongly: strong enough to read at a glance, transparent enough that the
# furniture under it is still identifiable as what was removed.
_MASK_TINT: Final = (0.95, 0.15, 0.45)
_MASK_OPACITY: Final = 0.55

# A piecewise-linear RGB lookup over uniformly spaced stops, in place of a
# plotting dependency: dark blue far, pale yellow near, which is how a
# monocular inverse-depth map is conventionally read.
_DEPTH_STOPS: Final = (
    (0.05, 0.03, 0.28),
    (0.30, 0.06, 0.44),
    (0.58, 0.15, 0.40),
    (0.83, 0.32, 0.24),
    (0.96, 0.60, 0.11),
    (0.99, 0.91, 0.15),
)


@dataclass(frozen=True)
class DisparityScale:
    """The near/far range a scene's depth maps are all painted against.

    Two depth maps only compare if one colour means one disparity in both, and
    affine-invariant inverse depth carries no scale of its own — so the scale is
    taken once, from the photograph, and every other map in the scene is fitted
    onto that frame before it is painted.
    """

    reference: Float[Tensor, "1 height width"]
    low: float
    high: float

    @classmethod
    def of(cls, disparity: Float[Tensor, "1 height width"]) -> DisparityScale:
        finite = disparity[torch.isfinite(disparity)].to(torch.float32)
        return cls(
            reference=disparity, low=float(finite.min()), high=float(finite.max())
        )


def write_image(image: Float[Tensor, "1 3 height width"], path: Path) -> None:
    """One render, quantised to 8-bit PNG."""
    quantised = (image[0].detach().to(torch.float32).clamp(0.0, 1.0) * 255.0).round()
    write_png(quantised.to(torch.uint8).cpu(), str(path))


def write_mask(mask: Bool[Tensor, "1 height width"], path: Path) -> None:
    """One mask, white where True."""
    write_png((mask[0].to(torch.uint8) * 255).unsqueeze(0).cpu(), str(path))


def write_mask_overlay(
    photo: Float[Tensor, "1 3 height width"],
    mask: Bool[Tensor, "1 height width"],
    path: Path,
) -> None:
    """The photograph with the region the inpainter was handed tinted over it."""
    tint = torch.tensor(_MASK_TINT, dtype=torch.float32, device=photo.device).view(
        3, 1, 1
    )
    frame = photo[0].detach().to(torch.float32).clamp(0.0, 1.0)
    blended = torch.lerp(frame, tint.expand_as(frame), _MASK_OPACITY)
    write_png(
        (torch.where(mask[0], blended, frame) * 255.0).round().to(torch.uint8).cpu(),
        str(path),
    )


def write_disparity_map(
    disparity: Float[Tensor, "1 height width"],
    *,
    scale: DisparityScale,
    mask: Bool[Tensor, "1 height width"],
    path: Path,
) -> None:
    """One inverse-depth map, fitted onto `scale`'s frame and painted in it.

    Args:
        mask: where the affine fit is taken — the shell the two maps are meant
            to be comparable over. The whole frame is painted; only the fit is
            masked, since a room's geometry is what the fit has to agree on and
            its furniture is not.

    A raw prediction is meaningless against another prediction: Depth Anything
    V2 leaves scale and shift undetermined and they are not stable between two
    images, so every map here goes through the same non-negative fit the reward
    does before a colour is assigned to it.
    """
    fitted = apply_scale_shift(
        disparity.to(torch.float32),
        align_scale_shift(
            disparity.to(torch.float32), scale.reference.to(torch.float32), mask
        ),
    )
    span = scale.high - scale.low
    unit = (fitted[0] - scale.low) / (span if span > 0.0 else 1.0)
    write_png((_colorise(unit) * 255.0).round().to(torch.uint8).cpu(), str(path))


def _colorise(unit: Float[Tensor, "height width"]) -> Float[Tensor, "3 height width"]:
    palette = torch.tensor(_DEPTH_STOPS, dtype=torch.float32, device=unit.device)
    position = unit.to(torch.float32).clamp(0.0, 1.0) * (palette.shape[0] - 1)
    lower = position.floor().clamp(max=palette.shape[0] - 2)
    weight = (position - lower).unsqueeze(-1)
    index = lower.to(torch.long)
    return torch.lerp(palette[index], palette[index + 1], weight).permute(2, 0, 1)
