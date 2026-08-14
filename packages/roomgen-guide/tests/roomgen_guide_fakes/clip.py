"""A weightless stand-in for CLIP's image tower, and the text energy built on it."""

from dataclasses import dataclass
from typing import Final

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.nn.functional import adaptive_avg_pool2d

from roomgen_guide.guidance.clip import ClipPreprocessing, ClipTextGuidance

# Small enough to keep the fixed projection cheap, large enough that pooling
# still carries where in the frame a change happened.
_POOLED_SIDE: Final = 4
INPUT_SIZE: Final = 32
EMBEDDING_WIDTH: Final = 8


@dataclass(frozen=True)
class FakeClipImageEncoder:
    """A `ClipImageEncoder` that needs no checkpoint: a fixed random projection of the pixels.

    Smooth in every pixel and deterministic in `seed`, which is what a gradient
    test of the text energy needs and what a checkpoint-free run of a framework
    can be steered by.
    """

    seed: int = 0

    @jaxtyped(typechecker=beartype)
    def __call__(
        self, pixel_values: Float[Tensor, "batch 3 size size"]
    ) -> Float[Tensor, "batch embedding"]:
        pooled = adaptive_avg_pool2d(pixel_values, _POOLED_SIDE).flatten(1)
        projection = torch.randn(
            (pooled.shape[-1], EMBEDDING_WIDTH),
            generator=torch.Generator().manual_seed(self.seed),
        )
        return pooled @ projection.to(device=pooled.device, dtype=pooled.dtype)


def fake_clip_text_guidance(
    prompt: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> ClipTextGuidance:
    """The text energy on `prompt` with no checkpoint behind it.

    The prompt's embedding is a deterministic function of the string, so two
    prompts pull in different directions and one prompt pulls the same way twice.
    """
    digest = torch.Generator().manual_seed(sum(ord(character) for character in prompt))
    return ClipTextGuidance(
        encoder=FakeClipImageEncoder(),
        text_features=torch.randn((1, EMBEDDING_WIDTH), generator=digest).to(
            device=device, dtype=dtype
        ),
        preprocessing=ClipPreprocessing(
            size=INPUT_SIZE,
            image_mean=torch.full((1, 3, 1, 1), 0.5, device=device, dtype=dtype),
            image_std=torch.full((1, 3, 1, 1), 0.25, device=device, dtype=dtype),
        ),
    )
