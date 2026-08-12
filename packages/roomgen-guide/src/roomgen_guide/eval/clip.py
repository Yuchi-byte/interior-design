"""The CLIP prompt score: adherence read against the prompt itself."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol, cast

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cosine_similarity, interpolate
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

from roomgen_geometry.reference import GeometricReference

DEFAULT_CHECKPOINT: Final = "openai/clip-vit-base-patch32"


class _ClipEncoders(Protocol):
    def get_image_features(self, *, pixel_values: Tensor) -> object: ...
    def get_text_features(self, **tokens: Tensor) -> object: ...


class ClipPromptAdherence:
    """Cosine similarity between the render and its prompt (arXiv:2103.00020).

    Args:
        checkpoint: Hugging Face Hub id; weights are fetched on demand, never
            vendored.
        device: where the weights live and where the render is scored.
        cache_dir: Hub snapshot cache the weights are read from and downloaded
            into (`RoomgenSettings.hub_cache`); `None` follows the Hub client's
            own environment.

    Reading the prompt is what gives this metric its teeth: a render that
    changed a great deal in the wrong direction scores no better than one that
    changed nothing.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: torch.device,
        cache_dir: Path | None = None,
    ) -> None:
        model: Module = CLIPModel.from_pretrained(checkpoint, cache_dir=cache_dir)
        # `Module` for the placement and the freeze, which transformers leaves
        # to torch, then the two encoder calls this metric makes: transformers
        # wraps `to` in a decorator that erases its bound signature.
        self._model = cast(_ClipEncoders, model.to(device).eval().requires_grad_(False))
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
        # The processor is read for its input size and normalisation statistics
        # only; the preprocessing itself is tensor ops, as everywhere else in
        # this repo, so no image round-trips through PIL.
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
        prompt: str,
        photo: Float[Tensor, "batch 3 height width"],
        reference: GeometricReference,
    ) -> float:
        with torch.no_grad():
            resized = interpolate(
                image.to(device=self._device, dtype=torch.float32),
                size=self._input_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            normalised = (resized.clamp(0.0, 1.0) - self._image_mean) / self._image_std
            image_features = _projected(
                self._model.get_image_features(pixel_values=normalised)
            )
            tokens = self._tokenizer(
                [prompt] * image.shape[0], padding=True, return_tensors="pt"
            ).to(self._device)
            text_features = _projected(self._model.get_text_features(**tokens))
            similarity = cosine_similarity(image_features, text_features)
        # Clamped at zero the way CLIPScore defines it: a negative cosine is
        # "unrelated", and adherence is read as a ratio against the control,
        # which a sign flip would make nonsense of.
        return float(similarity.clamp(min=0.0).mean())


def _projected(features: object) -> Tensor:
    # transformers 4.x returns the projected embedding as a tensor;
    # transformers 5.x returns the whole encoder output and carries the
    # projection in `pooler_output`. The declared floor is 4.54, so both.
    if isinstance(features, Tensor):
        return features
    pooled = getattr(features, "pooler_output", None)
    if isinstance(pooled, Tensor):
        return pooled
    raise TypeError(f"CLIP returned no usable embedding, got {type(features).__name__}")
