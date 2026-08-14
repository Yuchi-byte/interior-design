"""Seeds the Hub cache with every pretrained checkpoint a run loads, once, offline after."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from huggingface_hub import snapshot_download

from roomgen_geometry.depth.depth_anything import (
    DEFAULT_CHECKPOINT as DEPTH_CHECKPOINT,
)
from roomgen_geometry.segmentation.eomt import (
    DEFAULT_CHECKPOINT as SEGMENTATION_CHECKPOINT,
)
from roomgen_guide.eval.clip import DEFAULT_CHECKPOINT as ADHERENCE_CHECKPOINT
from roomgen_guide.renovate import DEFAULT_CHECKPOINT as DIFFUSION_CHECKPOINT
from roomgen_guide.sampling.inpaint import DEFAULT_CHECKPOINT as INPAINTING_CHECKPOINT
from roomgen_guide.settings import RoomgenSettings


@dataclass(frozen=True)
class Checkpoint:
    repo_id: str
    ignore_patterns: tuple[str, ...] = ()


# What a snapshot must hold for `verify` to call it cached at all.
WEIGHT_SUFFIXES: Final = (".safetensors", ".bin")
CONFIG_FILENAME: Final = "config.json"

# Stable Diffusion 1.5's repo carries ~25GB of redundant single-file
# checkpoints (the `.ckpt`/non-diffusers weights); only the diffusers-format
# shards under each pipeline component are needed. `*.fp16.*` goes with them:
# `load_pipeline` passes no `variant`, so diffusers reads the full-precision
# shards and casts, and the half-precision copies are never opened.
_SD15_IGNORE_PATTERNS: Final = (
    "*.ckpt",
    "*.bin",
    "*.h5",
    "*.msgpack",
    "*.fp16.*",
    "v1-5-pruned*",
    "*.non_ema.*",
    "safety_checker/*",
)

# EoMT ships the same 1.2GB of weights twice; safetensors is what transformers
# prefers, so the pickle copy is 1.2GB of cache for nothing.
_EOMT_IGNORE_PATTERNS: Final = ("*.bin",)

# CLIP ships the same weights four ways, but *not* as safetensors on `main`
# (the safetensors conversion lives on a separate revision), so `pytorch_model.
# bin` is the copy that must stay; the TensorFlow and Flax ones are 1.1GB that
# nothing in this repo can load.
_CLIP_IGNORE_PATTERNS: Final = ("*.h5", "*.msgpack")

CHECKPOINTS: Final[tuple[Checkpoint, ...]] = (
    Checkpoint(DEPTH_CHECKPOINT),
    Checkpoint(SEGMENTATION_CHECKPOINT, _EOMT_IGNORE_PATTERNS),
    Checkpoint(DIFFUSION_CHECKPOINT, _SD15_IGNORE_PATTERNS),
    Checkpoint(INPAINTING_CHECKPOINT, _SD15_IGNORE_PATTERNS),
    Checkpoint(ADHERENCE_CHECKPOINT, _CLIP_IGNORE_PATTERNS),
)


def prefetch(cache_dir: Path) -> None:
    """Downloads every checkpoint in :data:`CHECKPOINTS` into `cache_dir`.

    Goes through the Hub online, which is also what writes the repo listing
    `verify` later reads offline: a cache seeded by copying blobs in by hand has
    no listing, and an offline `snapshot_download` over it can only report the
    files it can see.
    """
    for checkpoint in CHECKPOINTS:
        _snapshot(checkpoint, cache_dir, local_files_only=False)


def verify(cache_dir: Path) -> None:
    """Checks each checkpoint's snapshot holds weights and a config, offline.

    A floor, not a proof of completeness. `snapshot_download(local_files_only=
    True)` returns a snapshot path rather than raising even when that directory
    is empty — with no cached repo listing there is nothing for it to compare
    against — so on its own it proves only that the cache has heard of the repo.
    Each snapshot is therefore additionally required to contain at least one
    weights file (``*.safetensors`` or ``*.bin``) and a ``config.json``, at any
    depth. That catches an empty, half-deleted or listing-less cache; it does
    not enumerate what `from_pretrained` will ask for, so a cache missing one
    component of a multi-folder pipeline can still pass.

    Raises:
        RuntimeError: if a checkpoint's snapshot is missing, empty, or holds no
            weights/config pair. The Hub client's own offline miss is re-raised
            the same way, so one failure mode reaches the caller, not two.
    """
    for checkpoint in CHECKPOINTS:
        try:
            directory = _snapshot(checkpoint, cache_dir, local_files_only=True)
        except OSError as error:
            # The Hub client's own miss: no snapshot for the revision at all.
            raise RuntimeError(
                f"{checkpoint.repo_id} is not in the cache at {cache_dir} ({error}). "
                f"{_REMEDY}"
            ) from error
        _assert_populated(checkpoint, directory)


def _snapshot(
    checkpoint: Checkpoint, cache_dir: Path, *, local_files_only: bool
) -> Path:
    return Path(
        snapshot_download(
            checkpoint.repo_id,
            cache_dir=cache_dir,
            ignore_patterns=list(checkpoint.ignore_patterns) or None,
            local_files_only=local_files_only,
        )
    )


def _assert_populated(checkpoint: Checkpoint, directory: Path) -> None:
    files = [path for path in directory.rglob("*") if path.is_file()]
    if not any(path.suffix in WEIGHT_SUFFIXES for path in files):
        raise RuntimeError(
            f"{checkpoint.repo_id} is not cached in {directory}: no "
            f"{' or '.join(WEIGHT_SUFFIXES)} file under it. {_REMEDY}"
        )
    if not any(path.name == CONFIG_FILENAME for path in files):
        raise RuntimeError(
            f"{checkpoint.repo_id} is cached in {directory} without a "
            f"{CONFIG_FILENAME}, so loading it will still reach for the Hub. {_REMEDY}"
        )


_REMEDY: Final = "Run roomgen-prefetch on a machine with network access."


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
