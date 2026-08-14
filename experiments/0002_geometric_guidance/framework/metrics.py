"""The prompt-adherence metric a variant selects, and the instruction-compliance readings."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from framework.config import AdherenceConfig
from roomgen_guide.eval import (
    ClipPromptAdherence,
    ContentsChangeAdherence,
    PromptAdherence,
)


def build_adherence(
    config: AdherenceConfig, *, device: torch.device, cache_dir: Path | None
) -> PromptAdherence:
    """The prompt-adherence metric `config` selects."""
    match config.kind:
        case "clip":
            return ClipPromptAdherence(
                config.checkpoint, device=device, cache_dir=cache_dir
            )
        case "contents_change":
            return ContentsChangeAdherence()


def class_area_fraction(
    segmentation: Int[Tensor, "batch height width"], classes: Iterable[int]
) -> float:
    """The share of the frame the segmenter assigns to `classes`.

    The removal-compliance reading: counted on a render it says how much of the
    frame still reads as the object the instruction removed, and counted on the
    photo it is the baseline that share is judged against. A fraction, not a
    mask intersection — an object the renovation moved elsewhere in the frame
    still counts as not removed.
    """
    identifiers = torch.tensor(
        sorted(set(classes)), dtype=segmentation.dtype, device=segmentation.device
    )
    return float(torch.isin(segmentation, identifiers).to(torch.float32).mean())


def masked_mean_square(
    field: Float[Tensor, "batch height width"],
    mask: Bool[Tensor, "batch height width"],
) -> float | None:
    """The mean square of `field` over `mask`; `None` when the mask is empty.

    The windows-kept reading takes it over a signed depth-residual map
    restricted to the photo's windowpane pixels: the map was aligned over the
    scene's whole common shell, so this is the window's share of the very error
    the depth score integrates — not a fresh fit over glass alone, which a few
    ambiguous panes could not support.
    """
    if not bool(mask.any()):
        return None
    return float(field[mask].to(torch.float32).square().mean())
