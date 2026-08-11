"""Both sides of the gate, measured on a finished image.

This is what `roomgen_guide.eval` is meant to hold; it lives here only because
that module does not exist yet (see `report.md`, "Gaps").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import cosine_similarity, interpolate
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

from framework.config import AdherenceConfig
from roomgen_geometry.depth.base import DepthEstimator
from roomgen_geometry.reference import GeometricReference
from roomgen_geometry.reward import masked_depth_error


class PromptAdherence(Protocol):
    """Gate (b): higher means the render is more of a renovation and less of a copy.

    Args:
        image: the render, NCHW in ``[0, 1]``.
        prompt: what the render was asked for.
        photo: the input photo the render started from, same shape.
        reference: the frozen geometry, whose mask separates the shell that must
            not move from the contents that were free to change.
    """

    def __call__(
        self,
        image: Float[Tensor, "batch 3 height width"],
        *,
        prompt: str,
        photo: Float[Tensor, "batch 3 height width"],
        reference: GeometricReference,
    ) -> float: ...


def depth_error(
    image: Float[Tensor, "batch 3 height width"],
    *,
    reference: GeometricReference,
    depth_estimator: DepthEstimator,
) -> float:
    """Gate (a): masked, scale-and-shift-invariant depth error against the photo's geometry."""
    with torch.no_grad():
        disparity = depth_estimator(image)
        error = masked_depth_error(disparity, reference.disparity, reference.mask)
    return float(error.mean())


class ClipPromptAdherence:
    """CLIP cosine similarity between the render and its prompt (arXiv:2103.00020).

    The metric gate (b) is meant to be: it reads the prompt, so a render that
    changed a great deal in the wrong direction scores no better than one that
    changed nothing. Costs a fourth checkpoint, which `roomgen-prefetch` does
    not seed.
    """

    def __init__(
        self, checkpoint: str, *, device: torch.device, cache_dir: Path | None = None
    ) -> None:
        model = CLIPModel.from_pretrained(checkpoint, cache_dir=cache_dir)
        self._model = model.to(device).eval().requires_grad_(False)
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
        # "unrelated", and the gate reads adherence as a ratio against the
        # control, which a sign flip would make nonsense of.
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


@dataclass(frozen=True)
class ContentsChangeAdherence:
    """Checkpoint-free stand-in: how far the render moved from the photo outside the shell.

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
