"""Seeds the Hub cache with the four pretrained checkpoints, once, offline after."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from roomgen_geometry.depth.depth_anything import (
    DEFAULT_CHECKPOINT as DEPTH_CHECKPOINT,
)
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide.eval.clip import DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT
from roomgen_guide.renovate import DEFAULT_CHECKPOINT as DIFFUSION_CHECKPOINT
from roomgen_guide.settings import RoomgenSettings


@dataclass(frozen=True)
class _Checkpoint:
    repo_id: str
    ignore_patterns: tuple[str, ...] = ()


# Stable Diffusion 1.5's repo carries ~25GB of redundant single-file
# checkpoints (the `.ckpt`/non-diffusers weights); only the diffusers-format
# shards under each pipeline component are needed.
_SD15_IGNORE_PATTERNS: tuple[str, ...] = (
    "*.ckpt",
    "*.bin",
    "*.h5",
    "*.msgpack",
    "v1-5-pruned*",
    "*.non_ema.*",
    "safety_checker/*",
)

CHECKPOINTS: tuple[_Checkpoint, ...] = (
    _Checkpoint(DEPTH_CHECKPOINT),
    _Checkpoint(SEGMENTATION_CHECKPOINT),
    _Checkpoint(DIFFUSION_CHECKPOINT, _SD15_IGNORE_PATTERNS),
    _Checkpoint(ADHERENCE_CHECKPOINT),
)


def prefetch(cache_dir: Path) -> None:
    """Downloads every checkpoint in :data:`CHECKPOINTS` into `cache_dir`."""
    for checkpoint in CHECKPOINTS:
        snapshot_download(
            checkpoint.repo_id,
            cache_dir=cache_dir,
            ignore_patterns=list(checkpoint.ignore_patterns) or None,
        )


def verify(cache_dir: Path) -> None:
    """Proves every checkpoint in :data:`CHECKPOINTS` is fully cached, offline."""
    for checkpoint in CHECKPOINTS:
        snapshot_download(
            checkpoint.repo_id,
            cache_dir=cache_dir,
            ignore_patterns=list(checkpoint.ignore_patterns) or None,
            local_files_only=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed or verify the Hugging Face Hub cache used by roomgen."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the cache is already fully populated, without network",
    )
    args = parser.parse_args()

    cache_dir = RoomgenSettings().hub_cache
    if args.check:
        verify(cache_dir)
        sys.stdout.write(f"All checkpoints verified in {cache_dir}\n")
    else:
        prefetch(cache_dir)
        sys.stdout.write(f"All checkpoints downloaded into {cache_dir}\n")


if __name__ == "__main__":
    main()
