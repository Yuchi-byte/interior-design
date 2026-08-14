"""The text energy: how far the render is from the prompt, in CLIP's embedding space."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cosine_similarity, interpolate
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

DEFAULT_CHECKPOINT: Final = "openai/clip-vit-base-patch32"


@runtime_checkable
class ClipImageEncoder(Protocol):
    """CLIP's image tower: normalised pixels in, one projected embedding per image out.

    Args:
        pixel_values: already resized to the tower's input size and normalised by
            its own statistics, and still attached to the latent it came from —
            the embedding must carry the gradient back, so nothing here may
            detach.
    """

    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> Float[Tensor, "batch embedding"]: ...


@dataclass(frozen=True)
class ClipPreprocessing:
    """What the image tower expects in front of it, as tensor ops rather than a PIL round trip.

    Args:
        size: side of the square the render is resized to.
        image_mean: per-channel mean, broadcastable over NCHW.
        image_std: per-channel standard deviation, broadcastable over NCHW.
    """

    size: int
    image_mean: Float[Tensor, "1 3 1 1"]
    image_std: Float[Tensor, "1 3 1 1"]

    def __call__(
        self, image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch 3 size size"]:
        resized = interpolate(
            image.to(self.image_mean.dtype),
            size=(self.size, self.size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return (resized - self.image_mean) / self.image_std


@dataclass(frozen=True)
class ClipTextGuidance:
    """Negative cosine similarity between the render and its prompt (arXiv:2103.00020).

    The energy a sampler descends to *make* the renovation the prompt asks for,
    beside the geometric one that holds the room still. The prompt is already in
    the sampler's classifier-free conditioning; this term is what puts it on the
    guidance side too, where it can be traded off against the geometry by weight.

    Args:
        encoder: frozen and in eval, and never under `no_grad` — the gradient
            reaches the latent through it.
        text_features: the prompt's embedding, frozen once: a constant, so
            construction rejects one still carrying gradient.

    Bounded in ``[-1, 1]`` and lowest where the render matches the prompt.
    Unclamped, unlike the CLIPScore the eval reports: a negative cosine is a
    render pointing away from the prompt, which is exactly where the energy has
    the most to pull against.
    """

    encoder: ClipImageEncoder
    text_features: Float[Tensor, "1 embedding"]
    preprocessing: ClipPreprocessing

    def __post_init__(self) -> None:
        if self.text_features.requires_grad:
            raise ValueError(
                "the prompt embedding is a constant of the walk: detach it before "
                "freezing it into a ClipTextGuidance"
            )

    def __call__(
        self, predicted_image: Float[Tensor, "batch 3 height width"]
    ) -> Float[Tensor, "batch"]:
        image_features = self.encoder(self.preprocessing(predicted_image))
        return -cosine_similarity(
            image_features, self.text_features.to(image_features.dtype)
        )


class _ClipEncoders(Protocol):
    def get_image_features(self, *, pixel_values: Tensor) -> object: ...
    def get_text_features(self, **tokens: Tensor) -> object: ...


@dataclass(frozen=True)
class _ImageTower:
    model: _ClipEncoders

    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> Float[Tensor, "batch embedding"]:
        return _projected(self.model.get_image_features(pixel_values=pixel_values))


def load_clip_text_guidance(
    checkpoint: str = DEFAULT_CHECKPOINT,
    *,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cache_dir: Path | None = None,
) -> ClipTextGuidance:
    """The text energy on `prompt`, from a Hub CLIP checkpoint.

    Args:
        checkpoint: Hugging Face Hub id; weights are fetched on demand, never
            vendored. A run that also *scores* adherence with CLIP must score it
            with a different checkpoint from this one, or it is grading the
            sampler on the very model the sampler optimised.
        dtype: parameter dtype of the tower and the dtype the render is cast to
            before it is encoded.
        cache_dir: Hub snapshot cache the weights are read from and downloaded
            into (`RoomgenSettings.hub_cache`); `None` follows the Hub client's
            own environment.
    """
    model: Module = CLIPModel.from_pretrained(checkpoint, cache_dir=cache_dir)
    # `Module` for the placement and the freeze, which transformers leaves to
    # torch, then the two encoder calls this energy makes: transformers wraps
    # `to` in a decorator that erases its bound signature.
    encoders = cast(
        _ClipEncoders, model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
    tokens = tokenizer([prompt], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = _projected(encoders.get_text_features(**tokens))
    # The processor is read for its input size and normalisation statistics
    # only; the preprocessing itself is tensor ops, so no image round-trips
    # through PIL and the gradient survives the resize.
    processor = CLIPImageProcessor.from_pretrained(checkpoint, cache_dir=cache_dir)
    return ClipTextGuidance(
        encoder=_ImageTower(encoders),
        text_features=text_features.detach(),
        preprocessing=ClipPreprocessing(
            size=int(processor.crop_size["height"]),
            image_mean=torch.tensor(
                processor.image_mean, device=device, dtype=dtype
            ).view(1, 3, 1, 1),
            image_std=torch.tensor(
                processor.image_std, device=device, dtype=dtype
            ).view(1, 3, 1, 1),
        ),
    )


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
