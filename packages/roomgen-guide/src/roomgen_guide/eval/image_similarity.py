"""The eval-side image similarity: how close the render stayed to the photo, to an independent judge."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cosine_similarity, interpolate
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

DEFAULT_IMAGE_SIMILARITY_CHECKPOINT: Final = "openai/clip-vit-large-patch14"


class _VisionTower(Protocol):
    def __call__(self, *, pixel_values: Tensor) -> object: ...


class ClipImageSimilarity:
    """Raw cosine similarity between the render's and the photo's CLIP image embeddings.

    The judge must be independent of any encoder a guidance energy descends —
    a different checkpoint, or the steered arm is graded on the very model it
    optimised. Unclamped and in ``[-1, 1]``: a copy of the photo scores near 1,
    so this number is read *beside* prompt adherence, never alone — high
    similarity with low adherence is the copying failure, not a win.

    Args:
        checkpoint: Hugging Face Hub id; weights are fetched on demand, never
            vendored.
        cache_dir: Hub snapshot cache (`RoomgenSettings.hub_cache`); `None`
            follows the Hub client's own environment.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_IMAGE_SIMILARITY_CHECKPOINT,
        *,
        device: torch.device,
        cache_dir: Path | None = None,
    ) -> None:
        model: Module = CLIPVisionModelWithProjection.from_pretrained(
            checkpoint, cache_dir=cache_dir
        )
        tower = model.to(device).eval().requires_grad_(False)
        self._tower: _VisionTower = lambda *, pixel_values: tower(
            pixel_values=pixel_values
        )
        # Statistics only; the preprocessing itself is tensor ops, as everywhere
        # else in this repo, so no image round-trips through PIL.
        processor = CLIPImageProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
        self._input_size = (
            int(processor.crop_size["height"]),
            int(processor.crop_size["width"]),
        )
        self._image_mean = torch.tensor(processor.image_mean, device=device).view(
            1, 3, 1, 1
        )
        self._image_std = torch.tensor(processor.image_std, device=device).view(
            1, 3, 1, 1
        )
        self._device = device

    def __call__(
        self,
        image: Float[Tensor, "batch 3 height width"],
        *,
        photo: Float[Tensor, "batch 3 height width"],
    ) -> float:
        with torch.no_grad():
            similarity = cosine_similarity(
                self._embed(image), self._embed(photo.to(image.dtype))
            )
        return float(similarity.mean())

    def _embed(
        self, image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch embedding"]:
        resized = interpolate(
            image.to(device=self._device, dtype=torch.float32),
            size=self._input_size,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        normalised = (resized.clamp(0.0, 1.0) - self._image_mean) / self._image_std
        embedding = getattr(self._tower(pixel_values=normalised), "image_embeds", None)
        if not isinstance(embedding, Tensor):
            raise TypeError("CLIP's vision tower returned no image_embeds")
        return embedding
