"""Human-preference scoring of finished renders: PickScore, and a weightless stand-in.

Both satisfy `roomgen_guide.guidance.preference.ImagePreferenceModel`, so the
same scorer could ride beside the energies as guidance; this framework uses it
purely as an eval metric, wrapping every call in `no_grad` at the call site —
nothing here detaches, because the protocol promises a gradient to callers that
want one.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import interpolate, normalize
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

from framework.config import PreferenceConfig
from roomgen_guide.guidance.preference import ImagePreferenceModel


class _ClipScorer(Protocol):
    logit_scale: Tensor

    def get_image_features(self, *, pixel_values: Tensor) -> object: ...
    def get_text_features(self, **tokens: Tensor) -> object: ...


class PickScorePreference:
    """PickScore v1 (arXiv:2305.01569): CLIP-H fine-tuned to rank images by human choice.

    One score per (image, prompt), higher preferred, in the model's own
    logit-scaled cosine units — comparable between arms scored against the same
    prompt, which is how this framework reads it.

    Args:
        checkpoint: Hub id of the fine-tuned CLIP; the weights are fetched on
            demand, never vendored.
        processor_checkpoint: Hub id the tokenizer and image statistics are read
            from, since the PickScore repo ships none of its own.
        device: where the weights live and where renders are scored.
        cache_dir: Hub snapshot cache (`RoomgenSettings.hub_cache`); `None`
            follows the Hub client's own environment.
    """

    def __init__(
        self,
        checkpoint: str,
        processor_checkpoint: str,
        *,
        device: torch.device,
        cache_dir: Path | None = None,
    ) -> None:
        model: Module = CLIPModel.from_pretrained(checkpoint, cache_dir=cache_dir)
        self._model = cast(_ClipScorer, model.to(device).eval().requires_grad_(False))
        self._tokenizer = AutoTokenizer.from_pretrained(
            processor_checkpoint, cache_dir=cache_dir
        )
        # Statistics only; the preprocessing itself is tensor ops, as everywhere
        # in this repo, so no image round-trips through PIL.
        processor = CLIPImageProcessor.from_pretrained(
            processor_checkpoint, cache_dir=cache_dir
        )
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

    def score(
        self,
        images: Float[Tensor, "batch 3 height width"],
        prompts: Sequence[str] | None,
        /,
    ) -> Float[Tensor, "batch"]:
        if prompts is None:
            raise ValueError("PickScore is prompt-conditioned: pass one per image")
        resized = interpolate(
            images.to(device=self._device, dtype=torch.float32),
            size=self._input_size,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        normalised = (resized.clamp(0.0, 1.0) - self._image_mean) / self._image_std
        image_features = normalize(
            _projected(self._model.get_image_features(pixel_values=normalised)), dim=-1
        )
        tokens = self._tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self._device)
        text_features = normalize(
            _projected(self._model.get_text_features(**tokens)), dim=-1
        )
        return self._model.logit_scale.exp() * (image_features * text_features).sum(
            dim=-1
        )


@dataclass(frozen=True)
class ContrastPreference:
    """Checkpoint-free stand-in: per-image pixel spread, blind to the prompt.

    Exists so the offline smoke variant exercises the preference plumbing —
    shape, batching, recording — with a deterministic number that separates two
    different renders. It says nothing about beauty and is never a claim.
    """

    def score(
        self,
        images: Float[Tensor, "batch 3 height width"],
        prompts: Sequence[str] | None,
        /,
    ) -> Float[Tensor, "batch"]:
        return images.flatten(start_dim=1).std(dim=1)


def build_preference(
    config: PreferenceConfig, *, device: torch.device, cache_dir: Path | None
) -> ImagePreferenceModel:
    """The preference scorer `config` selects."""
    match config.kind:
        case "pickscore":
            return PickScorePreference(
                config.checkpoint,
                config.processor_checkpoint,
                device=device,
                cache_dir=cache_dir,
            )
        case "contrast":
            return ContrastPreference()


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
