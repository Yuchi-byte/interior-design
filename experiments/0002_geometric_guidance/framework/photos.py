"""The benchmark photos: fetched if missing, decoded straight into ``[0, 1]`` NCHW tensors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import torch
from jaxtyping import Float
from torch import Tensor
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.v2.functional import center_crop, resize

_ASSETS: Final = Path(__file__).resolve().parents[2] / "assets"
_FETCH_SCRIPT: Final = _ASSETS / "fetch_photos.py"
_MANIFEST: Final = _ASSETS / "photos.json"


def _fetcher() -> ModuleType:
    # `experiments/assets` is a directory of scripts, not an importable package,
    # and the sha256-verified download logic in it is the single definition of
    # what "the benchmark photos" are; loading the file is how this framework
    # reuses it rather than restating it.
    spec = importlib.util.spec_from_file_location("roomgen_fetch_photos", _FETCH_SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not load the photo fetcher at {_FETCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the script defines dataclasses, and
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_photos(photo_dir: Path) -> dict[str, Path]:
    """Every photo of the manifest, present and sha256-verified under `photo_dir`.

    Downloads what is missing and fails loudly — with the command to run on a
    machine that has network — rather than quietly benchmarking a subset.
    """
    fetcher = _fetcher()
    photo_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for entry in fetcher.load_manifest(_MANIFEST):
        destination = fetcher.target_path(photo_dir, entry)
        if destination.exists() and fetcher.sha256_of(destination) == entry.sha256:
            paths[entry.id] = destination
            continue
        try:
            verified = fetcher.fetch_one(entry, photo_dir)
        except OSError as error:
            raise RuntimeError(_unavailable(entry.id, photo_dir, error)) from error
        if not verified:
            raise RuntimeError(_unavailable(entry.id, photo_dir, "sha256 mismatch"))
        paths[entry.id] = destination
    return paths


def load_photo(
    path: Path, *, resolution: int, device: torch.device, dtype: torch.dtype
) -> Float[Tensor, "1 3 resolution resolution"]:
    """Decode one photo into the ``[0, 1]`` NCHW convention every backend expects.

    The centre crop comes before the resize so the room is never anamorphically
    squashed: the reward compares the render against a depth map of *this*
    framing, so any distortion is baked into both sides and invisible to it.
    """
    decoded = read_image(str(path), mode=ImageReadMode.RGB)
    _, height, width = decoded.shape
    side = min(height, width)
    # Resampling in float, not uint8: the quantisation of an 8-bit intermediate
    # is the sort of thing that shows up in a depth error at the third decimal.
    square = center_crop(decoded.to(torch.float32) / 255.0, [side, side])
    resized = resize(square, [resolution, resolution], antialias=True)
    return resized.clamp(0.0, 1.0).unsqueeze(0).to(device=device, dtype=dtype)


def _unavailable(photo_id: str, photo_dir: Path, reason: object) -> str:
    return (
        f"benchmark photo {photo_id!r} is missing from {photo_dir} and could not be "
        f"downloaded ({reason}). On a machine with network access run:\n"
        f"    uv run --no-sync python experiments/assets/fetch_photos.py "
        f"--output-dir {photo_dir}\n"
        "or copy that directory over from one that has it."
    )
