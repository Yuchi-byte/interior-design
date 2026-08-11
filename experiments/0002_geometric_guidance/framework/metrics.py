"""The gate-(b) metric a variant selects, and the checkpoint-free stand-in for it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from framework.config import AdherenceConfig
from roomgen_geometry.reference import GeometricReference
from roomgen_guide.eval import ClipPromptAdherence, PromptAdherence


@dataclass(frozen=True)
class ContentsChangeAdherence:
    """How far the render moved from the photo outside the shell.

    It cannot tell a scandinavian sofa from a baroque one, so it is not a prompt
    score — but it is the half of gate (b) that has teeth mechanically: a
    sampler that wins on depth error by returning the photo unchanged scores
    zero here and fails. Used by the smoke variant, which must run offline.
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


def build_adherence(
    config: AdherenceConfig, *, device: torch.device, cache_dir: Path | None
) -> PromptAdherence:
    """The gate-(b) metric `config` selects."""
    match config.kind:
        case "clip":
            return ClipPromptAdherence(
                config.checkpoint, device=device, cache_dir=cache_dir
            )
        case "contents_change":
            return ContentsChangeAdherence()
