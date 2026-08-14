"""The prompt-adherence metric a variant selects."""

from __future__ import annotations

from pathlib import Path

import torch

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
