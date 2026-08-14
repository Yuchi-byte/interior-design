"""A weightless stand-in for the diffusers inpainting pipeline a strip is driven by."""

from dataclasses import dataclass

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Generator, Tensor


@dataclass(frozen=True)
class _Images:
    images: Tensor


@dataclass(frozen=True)
class FakeInpaintPipeline:
    """An `InpaintPipeline` that needs no checkpoint: a flat repaint under a whole-frame round trip.

    Faithful in the one way removal depends on: the batch it returns differs from
    the input *everywhere*, exactly as a latent pipeline's VAE decode does, so a
    caller that forgets to put the photograph back outside the mask fails rather
    than merely drifting. What it paints inside the mask is a deterministic
    function of the prompt and the generator, so two seeds strip differently and
    one seed strips reproducibly.

    Args:
        round_trip_drift: how far the untouched part of the frame moves, in
            image units.
    """

    round_trip_drift: float = 0.01

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        *,
        prompt: str,
        image: Float[Tensor, "batch 3 height width"],
        mask_image: Float[Tensor, "batch 1 height width"],
        num_inference_steps: int,
        guidance_scale: float,
        generator: Generator,
        output_type: str,
    ) -> _Images:
        if output_type != "pt":
            raise ValueError(
                f"this double returns tensors, so output_type must be 'pt', got "
                f"{output_type!r}"
            )
        # A stable digest of the prompt: `hash` is salted per process, which
        # would make a fixed-seed run reproducible only within one interpreter.
        tone = sum(ord(character) for character in prompt) % 97 / 97.0
        texture = torch.rand(
            image.shape, generator=generator, dtype=image.dtype, device=image.device
        )
        painted = (tone + 0.1 * texture).clamp(0.0, 1.0)
        round_tripped = (image + self.round_trip_drift).clamp(0.0, 1.0)
        return _Images(torch.where(mask_image > 0.5, painted, round_tripped))
