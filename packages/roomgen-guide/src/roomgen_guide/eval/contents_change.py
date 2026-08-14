"""Adherence without a checkpoint: how far the render moved where it was free to move."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from roomgen_geometry.reference import GeometricReference


@dataclass(frozen=True)
class ContentsChangeAdherence:
    """Mean absolute change from the photo outside the shell mask.

    It cannot tell a scandinavian sofa from a baroque one, so it is not a prompt
    score — but it still catches the degenerate win mechanically: a sampler that
    lowers depth error by returning the photo unchanged scores zero here. It is
    what a run with no network and no weights is judged on.
    """

    def __call__(
        self,
        image: Float[Tensor, "batch 3 height width"],
        *,
        prompt: str,
        photo: Float[Tensor, "batch 3 height width"],
        reference: GeometricReference,
    ) -> float:
        with torch.no_grad():
            # The contents are everything the shell mask excludes — the furniture
            # the renovation was allowed to restyle.
            weight = (~reference.mask).to(image.dtype).unsqueeze(1).expand_as(image)
            difference = (image - photo.to(image.dtype)).abs() * weight
            changed = difference.flatten(1).sum(-1)
            counted = weight.flatten(1).sum(-1).clamp(min=1.0)
        return float((changed / counted).mean())
